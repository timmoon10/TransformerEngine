# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functional implementation of grouped linear forward and backward."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...module.base import (
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ...tensor import GroupedTensor, GroupedTensorStorage


def _to_dequantized(
    tensor: torch.Tensor | QuantizedTensorStorage,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Dequantize tensor to given dtype or just convert if not a quantized tensor"""
    if is_quantized_tensor(tensor):
        return tensor.dequantize(dtype=dtype)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor

def _is_grouped_tensor_path_supported(
    *,
    with_quantized_compute: bool,
    input_quantizers: Optional[Sequence[Quantizer]],
    weight_quantizers: Optional[Sequence[Quantizer]],
    dtype: torch.dtype,
    with_single_grouped_weight: bool
) -> bool:

    ### TODO Remove
    return False

    # Graph-safe grouped GEMM is supported on Hopper and Blackwell (SM 10.x, 11.0)
    device_arch = get_device_compute_capability()
    if not (9, 0) <= device_arch <= (11, 0):
        return False

    # cuBLAS only supports BF16/FP16 output
    if dtype not in (torch.bfloat16, torch.float16):
        return False

    # Unquantized compute
    if not with_quantized_compute:
        return True

    # Assume quantizer lists are uniform
    if input_quantizers is None:
        raise ValueError("Requested quantized compute, but input quantizers are missing.")
    if weight_quantizers is None:
        raise ValueError("Requested quantized compute, but weight quantizers are missing.")
    input_quantizer = input_quantizers[0]
    weight_quantizer = weight_quantizers[0]
    quantizers = (input_quantizer, weight_quantizer)

    # FP8 current scaling
    if all(isinstance(q, Float8CurrentScalingQuantizer) for q in quantizers):
        if device_arch < (10, 0) and tex.get_cublasLt_version() < 130500:
            # cuBLAS 13.5 added Hopper support for FP8 grouped GEMM
            return False
        return True

    # MXFP8
    if all(isinstance(q, MXFP8Quantizer) for q in quantizers):
        if device_arch < (10, 0):
            # MXFP8 requires Blackwell+
            return False
        return True

    # NVFP4
    if all(isinstance(q, NVFP4) for q in quantizers):
        if device_arch < (10, 0):
            # NVFP4 requires Blackwell+
            return False
        if input_quantizer.with_rht and not with_single_grouped_weight:
            # cuBLAS supports NVFP4 with RHT and discrete weights
            return True
        return False

    # Unsupported case
    return False

def _grouped_linear_forward_with_grouped_tensor(
) -> tuple[torch.Tensor, dict[str, Any]]:
    ### TODO Implement
    ...

def _grouped_linear_forward_with_split_quantize(
    *,
    input: torch.Tensor | GroupedTensorStorage,
    weights: Sequence[torch.Tensor] | GroupedTensorStorage,
    split_sizes: torch.Tensor,
    biases: Optional[Sequence[torch.Tensor] | GroupedTensorStorage],
    bias_scales: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    out: Optional[torch.Tensor],
    with_quantized_compute: bool,
    input_quantizers: Optional[Sequence[Quantizer]],
    weight_quantizers: Optional[Sequence[Quantizer]],
    input_requires_grad: bool = True,
    weight_requires_grad: bool = True,
    with_cpu_offload: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ### TODO Implement

    # Move split sizes to CPU for split-quantize
    split_sizes = [int(s) for s in split_sizes.tolist()]

    # Get discrete weight tensors
    if isinstance(weights, GroupedTensorStorage):
        weights = ws.split_into_quantized_tensors()
    if with_quantized_compute:
        ws = [_to_dequantized(w, dtype) for w in weights]
    else:
        ws = []
        for w, quantizer in zip(weights, weight_quantizers):
            if not is_quantized_tensor(w):
                quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                w = quantizer(w)
            ws.append(w)

    # Get discrete bias tensors
    if biases is not None:
        if isinstance(biases, GroupedTensorStorage):
            biases = biases.split_into_quantized_tensors()
            biases = [b.reshape(-1) for b in biases]
        biases = [_to_dequantized(b, dtype) for b in biases]

    # Split input tensors and quantize if needed
    x = _to_dequantized(input, dtype)
    xs = None
    if with_quantized_compute:
        for quantizer in input_quantizers:
            quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
        xs = tex.split_quantize(x, split_sizes, input_quantizers)
    else:
        xs = torch.split(x, split_sizes)

    # Start CPU offloading for inputs, if enabled
    if with_cpu_offload:
        start_offload(*xs)

    # Allocate output tensor
    if out is None:
        in_shape = input.size()
        out_features, in_features = weights[0]
        out = torch.empty(shape, dtype=dtype, device=device)

    # Perform GEMMs
    with_fused_gemm_bias = biases is not None and bias_scales is None
    general_grouped_gemm(
        weights,
        xs,
        [out],
        [None] * num_groups,  # quantization_params
        dtype,
        m_splits=split_sizes,
        bias=biases if with_fused_gemm_bias else None,
        use_bias=with_fused_gemm_bias,
        use_split_accumulator=_2X_ACC_FPROP,
        single_output=True,
    )

    # Apply scaled bias if needed
    if biases is not None and not with_fused_gemm_bias:
        bias_scales_splits = torch.split(bias_scales, split_sizes)
        out_splits = torch.split(out, split_sizes)
        for i in range(num_groups):
            b = biases[i].unsqueeze(0)
            s = bias_scales_splits[i].unsqueeze(-1)
            out_splits[i].add_(b * s)

    # Prepare weights for backward pass
    if not input_requires_grad:
        ws = [None] * num_groups
    elif with_quantized_compute:
        for w, original_weight in zip(ws, weights):
            if w is not original_weight:
                w.update_usage(rowwise_usage=False, columnwise_usage=True)

    # Prepare input for backward pass
    if not weight_requires_grad:
        xs = [None] * num_groups
    elif with_quantized_compute:
        for x in xs:
            x.update_usage(rowwise_usage=False, columnwise_usage=True)

    # Saved state for backward pass
    saved = {
        "use_grouped_tensor_path": False,
        "inputs": xs,
        "weights": ws,
        "bias_scales": bias_scales,
    }

    return out, saved

def grouped_linear_forward(
    input: torch.Tensor | GroupedTensorStorage,
    weights: Sequence[torch.Tensor] | GroupedTensorStorage,
    split_sizes: torch.Tensor,
    *,
    biases: Optional[Sequence[torch.Tensor] | GroupedTensorStorage] = None,
    bias_scales: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    with_quantized_compute: bool = False,
    input_quantizers: Optional[Sequence[Quantizer]] = None,
    weight_quantizers: Optional[Sequence[Quantizer]] = None,
    output_quantizers: Optional[Sequence[Quantizer]] = None,
    input_requires_grad: bool = True,
    weight_requires_grad: bool = True,
    with_cpu_offload: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:

    # Infer device and dtype if needed
    if dtype is None and out is not None:
        dtype = out.dtype
    if isinstance(weights, GroupedTensorStorage):
        if isinstance(weights, torch.Tensor):
            if device is None:
                device = weights.device
            if dtype is None:
                dtype = weights.dtype
        else:
            if device is None:
                device = weights.rowwise_data.device
            if dtype is None:
                dtype = weights.rowwise_data.dtype
    else:
        if device is None:
            device = weights[0].device
        if dtype is None:
            dtype = weights[0].dtype

    # Check dtype
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")
    if out is not None and out.dtype != dtype:
        raise ValueError(f"Output tensor has invalid dtype (expected {dtype}, got {out.dtype})")

    # Use grouped tensor impl if supported
    use_grouped_tensor_path = _is_grouped_tensor_path_supported(
        with_quantized_compute=with_quantized_compute,
        input_quantizers=input_quantizers,
        weight_quantizers=weight_quantizers,
        dtype=dtype,
        with_single_grouped_weight=isinstance(weights, GroupedTensorStorage),
    )
    if use_grouped_tensor_path:
        ### TODO Implement
        return _grouped_linear_forward_with_grouped_tensor(
            ...
        )

    # Split-quantize impl
    return _grouped_linear_forward_with_split_quantize(
        input=input,
        weights=weights,
        split_sizes=split_sizes,
        biases=biases,
        bias_scales=bias_scales,
        device=device,
        dtype=dtype,
        out=out,
        with_quantized_compute=with_quantized_compute,
        input_quantizers=input_quantizers,
        weight_quantizer=weight_quantizers,
        input_requires_grad=input_requires_grad,
        weight_requires_grad=weight_requires_grad,
        with_cpu_offload=with_cpu_offload,
    )
