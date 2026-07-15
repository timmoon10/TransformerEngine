# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functional Transformer Engine PyTorch APIs."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ._functional.grouped_linear import (
    GroupedLinearForwardResult,
    grouped_linear_forward_grouped_tensor,
    grouped_linear_forward_split,
    grouped_tensor_path_is_supported,
)
from .quantized_tensor import QuantizedTensorStorage
from .tensor import GroupedTensor, Quantizer

__all__ = ["grouped_linear"]


def grouped_linear(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    weights: Sequence[torch.Tensor | QuantizedTensorStorage] | GroupedTensor,
    split_sizes: torch.Tensor | Sequence[int],
    bias: Optional[Sequence[torch.Tensor | QuantizedTensorStorage]] = None,
    *,
    scales: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    input_quantizers: Optional[Sequence[Optional[Quantizer]]] = None,
    weight_quantizers: Optional[Sequence[Optional[Quantizer]]] = None,
    output_quantizers: Optional[Sequence[Optional[Quantizer]]] = None,
    with_quantized_compute: bool = False,
    use_grouped_tensor_path: Optional[bool] = None,
    return_cache: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Apply grouped linear transformations.

    This is equivalent to splitting ``input`` along its first dimension,
    applying ``torch.nn.functional.linear`` with one weight and optional bias
    per split, and concatenating the results. When ``return_cache=True``, a
    dictionary of forward state is returned for advanced callers that implement
    a paired custom backward.
    """

    if isinstance(weights, GroupedTensor):
        num_groups = weights.num_tensors
        default_dtype = weights.dtype
    else:
        num_groups = len(weights)
        default_dtype = weights[0].dtype
    dtype = default_dtype if dtype is None else dtype
    input_quantizers = input_quantizers or [None] * num_groups
    weight_quantizers = weight_quantizers or [None] * num_groups
    output_quantizers = output_quantizers or [None] * num_groups

    if use_grouped_tensor_path is None:
        use_grouped_tensor_path = grouped_tensor_path_is_supported(
            with_quantized_compute=with_quantized_compute,
            input_quantizers=input_quantizers,
            output_quantizers=output_quantizers,
            dtype=dtype,
            single_grouped_weight=isinstance(weights, GroupedTensor),
        )

    if use_grouped_tensor_path:
        result = grouped_linear_forward_grouped_tensor(
            input,
            weights,
            split_sizes,
            biases=bias,
            scales=scales,
            input_quantizers=input_quantizers,
            weight_quantizers=weight_quantizers,
            output_quantizers=output_quantizers,
            with_quantized_compute=with_quantized_compute,
            dtype=dtype,
            input_requires_grad=input.requires_grad,
            weight_requires_grad=(
                weights.requires_grad
                if isinstance(weights, GroupedTensor)
                else any(weight.requires_grad for weight in weights)
            ),
        )
    else:
        if isinstance(weights, GroupedTensor):
            weights = list(weights.quantized_tensors or weights.split_into_quantized_tensors())
        result = grouped_linear_forward_split(
            input,
            weights,
            split_sizes,
            biases=bias,
            scales=scales,
            input_quantizers=input_quantizers,
            weight_quantizers=weight_quantizers,
            output_quantizers=output_quantizers,
            with_quantized_compute=with_quantized_compute,
            dtype=dtype,
            input_requires_grad=input.requires_grad,
            weight_requires_grad=any(weight.requires_grad for weight in weights),
        )

    if return_cache:
        return result.output, result.cache
    return result.output
