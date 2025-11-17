# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for forward GEMM + SwiGLU."""

from __future__ import annotations
from collections.abc import Callable, Iterable
import os
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...quantization import FP8GlobalStateManager
from ...tensor import Quantizer, QuantizedTensorStorage
from ...tensor.storage.float8_tensor_storage import Float8TensorStorage
from ..basic import BasicLinear, SwiGLU
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import maybe_dequantize


# Determine whether to enable GEMM + SwiGLU fusion
_FUSED_GEMM_SWIGLU_ENABLED: bool = bool(int(os.getenv("NVTE_FUSED_GEMM_SWIGLU", "0")))

# Try importing GEMM + SwiGLU kernel from cuDNN frontend
_HAS_GEMM_SWIGLU_KERNEL: bool = False
_gemm_swiglu_wrapper_sm100: Optional[Callable] = None
if torch.cuda.get_device_capability() == (10, 0):
    try:
        from cudnn import gemm_swiglu_wrapper_sm100 as _gemm_swiglu_wrapper_sm100
        _HAS_GEMM_SWIGLU_KERNEL = True
    except ImportError:
        pass


class ForwardLinearSwiGLU(FusedOperation):
    """Fused forward GEMM + SwiGLU"""

    # Whether fused GEMM + SwiGLU op is enabled
    _is_gemm_swiglu_enabled: bool = _FUSED_GEMM_SWIGLU_ENABLED and _HAS_GEMM_SWIGLU_KERNEL

    def __init__(
        self,
        *,
        linear: BasicLinear,
        swiglu: SwiGLU,
    ) -> None:
        super().__init__((linear, swiglu))

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:

        # Get basic operations
        linear_op = self.basic_ops[0]
        linear_op_ctx = basic_op_ctxs[0]
        swiglu_op = self.basic_ops[1]
        swiglu_op_ctx = basic_op_ctxs[1]

        # Check which grads are required
        input_requires_grad = linear_op_ctx.requires_grad
        weight_requires_grad = linear_op_ctx.requires_grad and linear_op.weight.requires_grad

        # Quantizers
        input_quantizer = linear_op.get_quantizer("forward", 0)
        weight_quantizer = linear_op.get_quantizer("forward", 1)
        output_quantizer = None
        grad_output_quantizer = linear_op.get_quantizer("backward", 0)
        grad_input_quantizer = prev_op_grad_output_quantizer
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()

        # Output dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = linear_op.weight.dtype

        # Tensor dims
        weight = linear_op.weight
        linear_out_size, linear_in_size = weight.size()
        if linear_out_size % 64 != 0:
            raise ValueError(f"Invalid weight dims ({tuple(weight.size())})")
        input_shape = input_.size()
        if input_shape[-1] != linear_in_size:
            raise ValueError(
                "Input and weight tensors do not match "
                f"(input shape is {tuple(input_shape)}), "
                f"weight shape is {tuple(weight.size())}"
            )

        # Tensor parallelism
        if linear_op.tensor_parallel_mode is not None:
            raise NotImplementedError("ForwardLinearSwiGLU op does not support tensor parallelism")

        # Launch GEMM + SwiGLU kernel
        input_saved = None
        weight_saved = None
        if with_quantized_compute:
            # Quantize input and weight tensors if needed
            if not isinstance(input_, QuantizedTensorStorage):
                if input_quantizer is None:
                    raise ValueError("Missing quantizer for input tensor")
                input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
                input_ = input_quantizer(input_)
            if not isinstance(input_, Float8TensorStorage):
                raise ValueError(
                    "Expected input to be quantized to Float8Tensor, "
                    f"but got {input_.__class__.__name__}"
                )
            if not isinstance(weight, QuantizedTensorStorage):
                if weight_quantizer is None:
                    raise ValueError("Missing quantizer for weight tensor")
                weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                weight = weight_quantizer(weight)
            if not isinstance(weight, Float8TensorStorage):
                raise ValueError(
                    "Expected weight to be quantized to Float8Tensor, "
                    f"but got {weight.__class__.__name__}"
                )
            input_saved = input_
            weight_saved = weight

            # Extract FP8 data
            ### TODO Avoid GPU sync for alpha
            input_data = input_._data
            weight_data = weight._data
            if input_._fp8_dtype == tex.DType.kFloat8E4M3:
                input_data = input_data.view(dtype=torch.float8_e4m3fn)
            elif input_._fp8_dtype == tex.DType.kFloat8E5M2:
                input_data = input_data.view(dtype=torch.float8_e5m2)
            if weight._fp8_dtype == tex.DType.kFloat8E4M3:
                weight_data = weight_data.view(dtype=torch.float8_e4m3fn)
            elif weight._fp8_dtype == tex.DType.kFloat8E5M2:
                weight_data = weight_data.view(dtype=torch.float8_e5m2)
            alpha = (input_._scale_inv * weight._scale_inv).item()

            # Reorder and reshape tensors
            input_data = input_data.reshape(1, -1, linear_in_size).permute(1, 2, 0)
            weight_data = weight_data.reshape(2, linear_out_size // 64, 32, linear_in_size)
            weight_data = weight_data.flip(0).permute(1, 0, 2, 3).contiguous()
            weight_data = weight_data.view(1, linear_out_size, linear_in_size).permute(1, 2, 0)

            # Launch GEMM + SwiGLU kernel
            linear_out, swiglu_out = _gemm_swiglu_wrapper_sm100(
                input_data,
                weight_data,
                alpha=alpha,
                c_major="n",
                c_dtype=dtype,
                glu_dtype=dtype,
                use_2cta_instrs=True,
                cluster_shape_mn=(2, 2),
            )
        else:
            # Make sure tensors have expected dtype
            input_ = maybe_dequantize(input_.contiguous(), dtype)
            weight = maybe_dequantize(weight.contiguous(), dtype)
            input_saved = input_
            weight_saved = weight

            # Reorder and reshape tensors
            input_ = input_.reshape(1, -1, linear_in_size).permute(1, 2, 0)
            weight = weight.reshape(2, linear_out_size // 64, 32, linear_in_size)
            weight = weight.flip(0).permute(1, 0, 2, 3).contiguous()
            weight = weight.view(1, linear_out_size, linear_in_size).permute(1, 2, 0)

            # Launch GEMM + SwiGLU kernel
            linear_out, swiglu_out = _gemm_swiglu_wrapper_sm100(
                input_.detach(),
                weight.detach(),
                c_major="n",
                c_dtype=dtype,
                glu_dtype=dtype,
                use_2cta_instrs=True,
                cluster_shape_mn=(2, 2),
            )

        # Reorder and reshape linear output tensor
        linear_out = linear_out.view(-1, linear_out_size // 64, 2, 32)
        linear_out = linear_out.permute(0, 2, 1, 3).flip(1).contiguous()
        linear_out = linear_out.reshape(*input_shape[:-1], linear_out_size)

        # Reshape SwiGLU output tensor
        swiglu_out = swiglu_out.reshape(*input_shape[:-1], linear_out_size // 2)

        # Prepare input tensor for backward pass
        if weight_requires_grad:
            if with_quantized_compute and isinstance(input_saved, QuantizedTensorStorage):
                input_saved.update_usage(rowwise_usage=False, columnwise_usage=True)

        # Save state for backward pass
        if linear_op_ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(input_saved, linear_out)
            linear_op_ctx.save_for_backward(input_saved, weight_saved)
            linear_op_ctx.with_quantized_compute = with_quantized_compute
            linear_op_ctx.input_quantizer = input_quantizer
            linear_op_ctx.weight_quantizer = weight_quantizer
            linear_op_ctx.grad_output_quantizer = grad_output_quantizer
            linear_op_ctx.grad_input_quantizer = grad_input_quantizer
            linear_op_ctx.dtype = dtype
            linear_op_ctx.input_requires_grad = input_requires_grad
            linear_op_ctx.weight_requires_grad = weight_requires_grad
            swiglu_op_ctx.save_for_backward(linear_out)
            swiglu_op_ctx.dtype = dtype
            swiglu_op_ctx.prev_op_grad_output_quantizer = grad_output_quantizer

        return swiglu_out, [() for _ in range(len(self.basic_ops))]


def fuse_forward_linear_swiglu(
    ops: list[tuple[FusibleOperation, list[int]]],
    recipe: Optional[Recipe],
) -> list[tuple[FusibleOperation, list[int]]]:
    """Fuse forward GEMM + SwiGLU

    Parameters
    ----------
    ops: list of tuples
        Forward pass operations and the indices of the corresponding
        basic operations.

    Returns
    -------
    ops: list of tuples
        Updated forward pass operations

    """

    # Check if fused kernel is available for recipe
    if not ForwardLinearSwiGLU._is_gemm_swiglu_enabled:
        return ops
    if recipe is None or recipe.delayed() or recipe.float8_current_scaling():
        pass
    else:
        return ops

    # Scan through ops, fusing if possible
    out = []
    window = []
    while len(ops) >= 2:
        out.extend(window)

        # Check if first op is linear
        window, ops = ops[:1], ops[1:]
        op, _ = window[0]
        if not isinstance(op, BasicLinear):
            continue
        if op.tensor_parallel_mode is not None:
            # Tensor parallelism is not yet supported
            continue
        linear = op

        # Check if next op is SwiGLU
        op, _ = ops[0]
        if not isinstance(op, SwiGLU):
            continue
        swiglu = op
        window.extend(ops[:1])
        ops = ops[1:]

        # Replace window with fused op
        op = ForwardLinearSwiGLU(
            linear=linear,
            swiglu=swiglu,
        )
        basic_op_idxs = [basic_op_idxs[0] for _, basic_op_idxs in window]
        window = [(op, basic_op_idxs)]

    # Return list of ops
    out.extend(window)
    out.extend(ops)
    return out
