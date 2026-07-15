# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functional grouped-linear compute helpers.

The functions in this module intentionally avoid depending on either
``TransformerEngineBaseModule`` autograd contexts or operation-fuser contexts.
Callers are responsible for saving/restoring the returned cache in the format
that their autograd surface expects.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Optional, Sequence

import torch

import transformer_engine_torch as tex

from ..cpp_extensions import general_grouped_gemm, general_grouped_gemm_for_grouped_tensor
from ..module.base import _2X_ACC_FPROP, quantize_weight
from ..ops._common import is_quantized_tensor, maybe_dequantize
from ..quantized_tensor import QuantizedTensorStorage
from ..tensor import Float8CurrentScalingQuantizer, GroupedTensor, GroupedTensorStorage
from ..tensor import MXFP8Quantizer, NVFP4Quantizer, Quantizer
from ..triton.grouped_dbias_dscales import compute_grouped_dbias, compute_grouped_dbias_dscales
from ..utils import get_device_compute_capability


@dataclass(slots=True)
class GroupedLinearForwardResult:
    """Grouped-linear forward output and backward cache."""

    output: torch.Tensor
    cache: dict[str, Any]
    new_weight_workspaces: list[Optional[QuantizedTensorStorage]]


def maybe_dequantize_to_dtype(
    tensor: torch.Tensor | QuantizedTensorStorage,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize quantized tensors or cast regular tensors to ``dtype``."""

    return maybe_dequantize(tensor, dtype)


def canonicalize_split_sizes(
    split_sizes: torch.Tensor | Sequence[int],
    *,
    num_groups: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return split sizes as an int64 tensor.

    ``device`` is optional because legacy split-list paths still need CPU-visible
    values, while graph-safe grouped-tensor paths need GPU-resident metadata.
    """

    if not isinstance(split_sizes, torch.Tensor):
        split_sizes = torch.tensor(split_sizes, dtype=torch.int64, device=device or "cpu")
    elif split_sizes.dtype != torch.int64:
        split_sizes = split_sizes.to(dtype=torch.int64)
    if device is not None and split_sizes.device != device:
        split_sizes = split_sizes.to(device=device)
    if split_sizes.size() != (num_groups,):
        raise ValueError(
            f"Shape of splits tensor ({tuple(split_sizes.size())}) "
            f"does not match number of GEMMs ({num_groups})."
        )
    return split_sizes


def grouped_tensor_path_is_supported(
    *,
    with_quantized_compute: bool,
    input_quantizers: Sequence[Optional[Quantizer]],
    dtype: torch.dtype,
    output_quantizers: Optional[Sequence[Optional[Quantizer]]] = None,
    single_grouped_weight: bool = False,
    require_env: bool = False,
    fp8_calibration: bool = False,
    debug: bool = False,
    cpu_offloading: bool = False,
    backward_override: Optional[str] = None,
    save_original_input: bool = False,
) -> bool:
    """Whether the graph-safe GroupedTensor/cuBLASLt path can be used."""

    if require_env and not bool(int(os.getenv("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM", "0"))):
        return False
    if (
        debug
        or cpu_offloading
        or fp8_calibration
        or backward_override is not None
        or save_original_input
    ):
        return False
    if output_quantizers is not None and any(q is not None for q in output_quantizers):
        return False

    device_capability = get_device_compute_capability()
    if not (9, 0) <= device_capability <= (11, 0):
        return False

    cublaslt_version = tex.get_cublasLt_version()
    if cublaslt_version < 130300:
        return False
    if device_capability < (10, 0) and cublaslt_version < 130400:
        return False

    if with_quantized_compute:
        if all(isinstance(q, Float8CurrentScalingQuantizer) for q in input_quantizers):
            if device_capability < (10, 0) and cublaslt_version < 130500:
                return False
            return True
        if not (10, 0) <= device_capability <= (11, 0):
            return False
        if all(isinstance(q, MXFP8Quantizer) for q in input_quantizers):
            return True
        if all(isinstance(q, NVFP4Quantizer) and q.with_rht for q in input_quantizers):
            return not single_grouped_weight
        return False

    return dtype in (torch.bfloat16, torch.float16)


def make_grouped_tensor_from_2d_buffer(
    data: torch.Tensor,
    *,
    num_groups: int,
    split_sizes: torch.Tensor,
    base_split_offsets: torch.Tensor,
    last_dim: int,
    dtype: torch.dtype,
) -> GroupedTensorStorage:
    """Wrap a packed 2D buffer as a varying-first-dimension GroupedTensor."""

    return GroupedTensorStorage(
        shape=(data.size(0), last_dim),
        dtype=dtype,
        num_tensors=num_groups,
        quantizer=None,
        data=data.reshape(-1),
        first_dims=split_sizes,
        tensor_offsets=base_split_offsets * last_dim,
    )


def make_grouped_bias(
    biases: Sequence[torch.Tensor | QuantizedTensorStorage],
    *,
    num_groups: int,
    out_features: int,
    dtype: torch.dtype,
) -> GroupedTensorStorage:
    """Pack per-group biases into the GroupedTensor GEMM bias format."""

    bias_data = torch.stack([maybe_dequantize_to_dtype(bias, dtype) for bias in biases], dim=0)
    bias_data = bias_data.contiguous()
    return GroupedTensorStorage(
        shape=(num_groups, out_features),
        dtype=dtype,
        num_tensors=num_groups,
        shapes=[(1, out_features)] * num_groups,
        quantizer=None,
        data=bias_data.reshape(-1),
    )


def get_grouped_tensor_members(grouped: GroupedTensor) -> list[torch.Tensor]:
    """Return per-group tensor views from a ``GroupedTensor`` parameter."""

    members = grouped.quantized_tensors
    if members is None:
        members = grouped.split_into_quantized_tensors()
    return list(members)


def prepare_discrete_weights_for_grouped_gemm(
    weights: Sequence[torch.Tensor | QuantizedTensorStorage],
    weight_quantizers: Sequence[Optional[Quantizer]],
    *,
    with_quantized_compute: bool,
    columnwise_usage: bool,
    dtype: torch.dtype,
    is_first_microbatch: Optional[bool] = None,
    weight_workspaces: Optional[Sequence[Optional[QuantizedTensorStorage]]] = None,
    cache_weight: bool = False,
    skip_fp8_weight_update: Optional[torch.Tensor] = None,
    use_quantize_weight: bool = True,
    optimize_for_gemm: bool = True,
) -> tuple[list[torch.Tensor | QuantizedTensorStorage], list[Optional[QuantizedTensorStorage]]]:
    """Prepare a list of per-group weights for grouped GEMM."""

    new_workspaces: list[Optional[QuantizedTensorStorage]] = [None] * len(weights)
    if not with_quantized_compute:
        return [maybe_dequantize_to_dtype(weight, dtype) for weight in weights], new_workspaces

    out: list[torch.Tensor | QuantizedTensorStorage] = []
    update_ws = is_first_microbatch is None or is_first_microbatch
    for idx, weight in enumerate(weights):
        if is_quantized_tensor(weight):
            out.append(weight)
            continue
        quantizer = weight_quantizers[idx]
        if quantizer is None:
            raise ValueError("Missing quantizer for grouped-linear weight tensor")
        quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
        if optimize_for_gemm:
            quantizer.optimize_for_gemm = True
        if not use_quantize_weight:
            out.append(quantizer(weight))
            continue
        weight_fp8, new_workspaces[idx] = quantize_weight(
            tensor=weight,
            quantizer=quantizer,
            workspace=weight_workspaces[idx] if weight_workspaces else None,
            update_workspace=update_ws,
            skip_update_flag=skip_fp8_weight_update,
            workspace_dtype=dtype,
            cache=cache_weight,
        )
        out.append(weight_fp8)
    return out, new_workspaces


def prepare_grouped_weight_for_grouped_gemm(
    weight: GroupedTensor,
    weight_quantizers: Sequence[Optional[Quantizer]],
    *,
    with_quantized_compute: bool,
    columnwise_usage: bool,
    dtype: torch.dtype,
) -> GroupedTensorStorage:
    """Prepare one grouped weight parameter for GroupedTensor GEMM."""

    is_weight_quantized = weight.quantizer is not None
    if is_weight_quantized and with_quantized_compute:
        return weight
    if is_weight_quantized and not with_quantized_compute:
        weight_parts = get_grouped_tensor_members(weight)
        weight_data = torch.stack([maybe_dequantize_to_dtype(w, dtype) for w in weight_parts])
        weight_data = weight_data.contiguous()
        return GroupedTensorStorage(
            shape=(weight.num_tensors * weight.tensor_shapes[0][0], weight.tensor_shapes[0][1]),
            dtype=dtype,
            num_tensors=weight.num_tensors,
            shapes=list(weight.tensor_shapes),
            quantizer=None,
            data=weight_data.reshape(-1),
        )
    if not with_quantized_compute:
        if weight.rowwise_data.dtype == dtype:
            return weight
        weight_data = weight.rowwise_data.to(dtype=dtype)
        return GroupedTensorStorage(
            shape=weight.logical_shape,
            dtype=dtype,
            num_tensors=weight.num_tensors,
            shapes=list(weight.tensor_shapes),
            quantizer=None,
            data=weight_data.reshape(-1),
        )

    quantizer = weight_quantizers[0]
    if quantizer is None:
        raise ValueError("Missing quantizer for grouped-linear grouped weight")
    quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
    return tex.group_quantize(
        weight.rowwise_data.view(weight.logical_shape),
        quantizer,
        weight.num_tensors,
        None,
    )


def grouped_linear_forward_grouped_tensor(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    weights: Sequence[torch.Tensor | QuantizedTensorStorage] | GroupedTensor,
    split_sizes: torch.Tensor,
    *,
    biases: Optional[Sequence[torch.Tensor | QuantizedTensorStorage]] = None,
    scales: Optional[torch.Tensor] = None,
    input_quantizers: Sequence[Optional[Quantizer]],
    weight_quantizers: Sequence[Optional[Quantizer]],
    output_quantizers: Optional[Sequence[Optional[Quantizer]]] = None,  # pylint: disable=unused-argument
    with_quantized_compute: bool,
    dtype: torch.dtype,
    input_requires_grad: bool,
    weight_requires_grad: bool,
    is_first_microbatch: Optional[bool] = None,
    weight_workspaces: Optional[Sequence[Optional[QuantizedTensorStorage]]] = None,
    cache_weight: bool = False,
    skip_fp8_weight_update: Optional[torch.Tensor] = None,
    use_quantize_weight: bool = True,
    optimize_weight_for_gemm: bool = True,
) -> GroupedLinearForwardResult:
    """Graph-safe grouped-linear forward using GroupedTensor metadata."""

    if isinstance(weights, GroupedTensor):
        num_groups = weights.num_tensors
        out_features, in_features = weights.tensor_shapes[0]
        device = weights.device
    else:
        num_groups = len(weights)
        out_features, in_features = weights[0].size()
        device = weights[0].device

    split_sizes = canonicalize_split_sizes(split_sizes, num_groups=num_groups, device=device)
    base_split_offsets = tex.splits_to_offsets(split_sizes, 1)
    split_points = base_split_offsets[1:].to(dtype=torch.int)

    original_shape = input.shape
    x = maybe_dequantize_to_dtype(input, dtype).reshape(-1, in_features)
    total_tokens = x.size(0)

    if with_quantized_compute:
        input_quantizer = input_quantizers[0]
        if input_quantizer is None:
            raise ValueError("Missing quantizer for grouped-linear input tensor")
        input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
        input_quantizer.optimize_for_gemm = True
        grouped_x = tex.group_quantize(x, input_quantizer, num_groups, split_sizes)
    else:
        grouped_x = make_grouped_tensor_from_2d_buffer(
            x,
            num_groups=num_groups,
            split_sizes=split_sizes,
            base_split_offsets=base_split_offsets,
            last_dim=in_features,
            dtype=dtype,
        )

    if isinstance(weights, GroupedTensor):
        grouped_weights: GroupedTensorStorage | list[torch.Tensor | QuantizedTensorStorage]
        grouped_weights = prepare_grouped_weight_for_grouped_gemm(
            weights,
            weight_quantizers,
            with_quantized_compute=with_quantized_compute,
            columnwise_usage=input_requires_grad,
            dtype=dtype,
        )
        new_workspaces = [None]
    else:
        grouped_weights, new_workspaces = prepare_discrete_weights_for_grouped_gemm(
            weights,
            weight_quantizers,
            with_quantized_compute=with_quantized_compute,
            columnwise_usage=input_requires_grad,
            dtype=dtype,
            is_first_microbatch=is_first_microbatch,
            weight_workspaces=weight_workspaces,
            cache_weight=cache_weight,
            skip_fp8_weight_update=skip_fp8_weight_update,
            use_quantize_weight=use_quantize_weight,
            optimize_for_gemm=optimize_weight_for_gemm,
        )

    out = torch.empty((*original_shape[:-1], out_features), dtype=dtype, device=device)
    grouped_out = make_grouped_tensor_from_2d_buffer(
        out.reshape(-1, out_features),
        num_groups=num_groups,
        split_sizes=split_sizes,
        base_split_offsets=base_split_offsets,
        last_dim=out_features,
        dtype=dtype,
    )

    grouped_bias = None
    bias_scale = None
    if biases is not None:
        grouped_bias = make_grouped_bias(
            biases,
            num_groups=num_groups,
            out_features=out_features,
            dtype=dtype,
        )
        if scales is not None:
            bias_scale = scales.reshape(-1)
            if bias_scale.dtype != torch.float32:
                bias_scale = bias_scale.to(dtype=torch.float32)

    general_grouped_gemm_for_grouped_tensor(
        grouped_weights,
        grouped_x,
        grouped_out,
        layout="TN",
        bias=grouped_bias,
        bias_scale=bias_scale,
        use_split_accumulator=_2X_ACC_FPROP,
    )

    if not input_requires_grad:
        grouped_weights = None if isinstance(weights, GroupedTensor) else [None] * num_groups
    if not weight_requires_grad:
        grouped_x = None
    if grouped_x is not None and with_quantized_compute and grouped_x.columnwise_data is not None:
        grouped_x.rowwise_data = None
        grouped_x.scale_inv = None

    saved: list[Any] = [split_sizes, base_split_offsets, split_points]
    if scales is not None:
        saved.append(scales)
    saved.append(grouped_x)
    if isinstance(weights, GroupedTensor):
        saved.append(grouped_weights)
    else:
        saved.extend(grouped_weights)

    cache = {
        "path": "grouped_tensor",
        "saved_tensors": tuple(saved),
        "has_scales": scales is not None,
        "num_groups": num_groups,
        "in_features": in_features,
        "out_features": out_features,
        "input_shape": original_shape,
        "dtype": dtype,
        "device": device,
        "single_grouped_weight": isinstance(weights, GroupedTensor),
        "has_bias": biases is not None,
    }
    return GroupedLinearForwardResult(out, cache, new_workspaces)


def grouped_linear_forward_split(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    weights: Sequence[torch.Tensor | QuantizedTensorStorage],
    split_sizes: torch.Tensor | Sequence[int],
    *,
    biases: Optional[Sequence[torch.Tensor | QuantizedTensorStorage]] = None,
    scales: Optional[torch.Tensor] = None,
    input_quantizers: Sequence[Optional[Quantizer]],
    weight_quantizers: Sequence[Optional[Quantizer]],
    output_quantizers: Optional[Sequence[Optional[Quantizer]]] = None,
    with_quantized_compute: bool,
    dtype: torch.dtype,
    input_requires_grad: bool,
    weight_requires_grad: bool,
    is_first_microbatch: Optional[bool] = None,
    weight_workspaces: Optional[Sequence[Optional[QuantizedTensorStorage]]] = None,
    cache_weight: bool = False,
    skip_fp8_weight_update: Optional[torch.Tensor] = None,
    debug_quantize_fn: Optional[Any] = None,
    cpu_offloading: bool = False,
    use_quantize_weight: bool = True,
    optimize_weight_for_gemm: bool = True,
) -> GroupedLinearForwardResult:
    """Legacy grouped-linear forward using split tensors and grouped GEMM."""

    num_groups = len(weights)
    out_features, in_features = weights[0].size()
    device = weights[0].device
    split_sizes = canonicalize_split_sizes(split_sizes, num_groups=num_groups)
    split_sizes_int = split_sizes.tolist()

    x = maybe_dequantize_to_dtype(input, dtype).reshape(-1, in_features)
    if with_quantized_compute:
        if debug_quantize_fn is not None:
            xs = debug_quantize_fn(x, input_quantizers, split_sizes_int, dtype)
        else:
            xs = tex.split_quantize(x, split_sizes_int, input_quantizers)
    else:
        xs = torch.split(x, split_sizes_int)

    if cpu_offloading:
        from ..cpu_offload import start_offload  # pylint: disable=import-outside-toplevel

        start_offload(*xs)

    ws, new_workspaces = prepare_discrete_weights_for_grouped_gemm(
        weights,
        weight_quantizers,
        with_quantized_compute=with_quantized_compute,
        columnwise_usage=input_requires_grad,
        dtype=dtype,
        is_first_microbatch=is_first_microbatch,
        weight_workspaces=weight_workspaces,
        cache_weight=cache_weight,
        skip_fp8_weight_update=skip_fp8_weight_update,
        use_quantize_weight=use_quantize_weight,
        optimize_for_gemm=optimize_weight_for_gemm,
    )

    bs = None
    if biases is not None:
        bs = [maybe_dequantize_to_dtype(bias, dtype) for bias in biases]

    out = torch.empty((*input.shape[:-1], out_features), dtype=dtype, device=device)
    use_gemm_bias = bs is not None and scales is None
    general_grouped_gemm(
        ws,
        xs,
        [out],
        output_quantizers or [None] * num_groups,
        dtype,
        single_output=True,
        m_splits=split_sizes_int,
        bias=bs if use_gemm_bias else None,
        use_bias=use_gemm_bias,
        use_split_accumulator=_2X_ACC_FPROP,
    )

    if scales is not None and bs is not None:
        scale_splits = torch.split(scales, split_sizes_int)
        out_splits = torch.split(out.reshape(-1, out_features), split_sizes_int)
        for idx in range(num_groups):
            out_splits[idx].add_(bs[idx].unsqueeze(0) * scale_splits[idx].unsqueeze(-1))

    if not input_requires_grad:
        ws = [None] * num_groups
    elif with_quantized_compute:
        for w, weight in zip(ws, weights):
            if w is not weight and is_quantized_tensor(w):
                w.update_usage(rowwise_usage=False, columnwise_usage=True)

    if not weight_requires_grad:
        xs = [None] * num_groups
    elif with_quantized_compute:
        for x_i in xs:
            if is_quantized_tensor(x_i):
                x_i.update_usage(rowwise_usage=False, columnwise_usage=True)

    saved: list[Any] = [split_sizes, None, None]
    if scales is not None:
        saved.append(scales)
    saved.extend(xs)
    saved.extend(ws)

    cache = {
        "path": "split",
        "saved_tensors": tuple(saved),
        "has_scales": scales is not None,
        "num_groups": num_groups,
        "in_features": in_features,
        "out_features": out_features,
        "input_shape": input.shape,
        "dtype": dtype,
        "device": device,
        "single_grouped_weight": False,
        "has_bias": biases is not None,
        "split_sizes_int": split_sizes_int,
    }
    return GroupedLinearForwardResult(out, cache, new_workspaces)


def compute_grouped_linear_dbias(
    grad_output_2d: torch.Tensor,
    split_offsets: torch.Tensor,
    *,
    num_groups: int,
    dtype: torch.dtype,
    scales: Optional[torch.Tensor] = None,
    biases: Optional[Sequence[torch.Tensor | QuantizedTensorStorage]] = None,
) -> tuple[list[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Compute grouped bias gradients, and optional scale gradients."""

    grad_scales = None
    if scales is not None:
        if biases is None:
            raise ValueError("Grouped bias tensors are required to compute scale gradients")
        bias_packed = torch.stack([maybe_dequantize_to_dtype(bias, dtype) for bias in biases])
        dbias_packed, grad_scales = compute_grouped_dbias_dscales(
            grad_output_2d,
            scales.to(dtype=torch.float32),
            bias_packed,
            offsets=split_offsets,
        )
    else:
        dbias_packed = compute_grouped_dbias(grad_output_2d, split_offsets, num_groups)
    return [dbias_packed[idx].to(dtype=dtype) for idx in range(num_groups)], grad_scales, dbias_packed
