# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible expert-parallel dispatch operation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from ...ep import (
    EpBuffer,
    EpConfig,
    _ep_dispatch_bwd,
    _ep_prepare_and_dispatch_fwd,
)
from ...quantization import QuantizerRole
from ...tensor import Quantizer
from .._common import (
    maybe_dequantize,
    validate_ep_buffer,
    validate_ep_comms_recipe,
)
from ..op import BasicOperation, OperationContext


class MoeDispatch(BasicOperation):
    """Route tokens to experts and distribute across expert-parallel ranks

    The extra inputs are routing indices and FP32 routing weights. The extra
    outputs are local tokens-per-expert and received routing weights.

    Currently requires NCCL EP.

    """

    num_extra_inputs: int = 2
    # tokens-per-expert and received routing weights consumed by the expert MLP.
    num_extra_outputs: int = 2

    def __init__(self, config: EpConfig, buffer: Optional[EpBuffer] = None) -> None:
        # EpBuffer is specific to NCCL EP. Fused implementations using another communication
        # backend, such as NVSHMEM, do not need it.
        super().__init__()
        if not isinstance(config, EpConfig):
            raise TypeError(f"config must be an EpConfig, got {type(config).__name__}.")
        if config.zero_copy:
            raise NotImplementedError("MoeDispatch does not support zero-copy EP.")
        self.config = config
        self.buffer = buffer

    def op_forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("MoeDispatch uses fuser_forward")

    def op_backward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("MoeDispatch uses fuser_backward")

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
        del next_op_input_quantizer, basic_op_kwargs

        # NCCL EP buffer
        buffer = validate_ep_buffer("MoeDispatch", self.config, self.buffer)

        # Make sure input tensor is in format expected by NCCL EP
        input_ = maybe_dequantize(input_, torch.bfloat16)
        if input_.device != buffer.device:
            input_ = input_.to(device=buffer.device)
        input_shape = tuple(input_.shape)
        if len(input_shape) != 2 or input_shape[-1] != buffer.hidden_dim:
            raise ValueError(
                f"MoeDispatch input must have shape (T, {buffer.hidden_dim}), got {input_shape}."
            )
        buffer.num_local_tokens = input_shape[0]

        # Make sure routing inputs are in format expected by NCCL EP
        topk_idx, topk_weights = basic_op_extra_inputs[0]
        topk_weights = maybe_dequantize(topk_weights, torch.float32)
        if not devices_match(topk_idx.device, buffer.device):
            topk_idx = topk_idx.to(device=device)
        if not devices_match(topk_weights.device, buffer.device):
            topk_weights = topk_weights.to(device=device)

        # NCCL EP dispatch forward
        output, recv_topk_weights, dispatch_state = _ep_prepare_and_dispatch_fwd(
            input_,
            topk_weights,
            topk_idx,
            buffer,
            None,
            None,
        )
        tokens_per_expert = buffer.tokens_per_expert

        # Save state for backward
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            ctx.dispatch_state = dispatch_state
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

        return output, [(tokens_per_expert, recv_topk_weights)]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[Optional[torch.Tensor], ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        ctx = basic_op_ctxs[0]

        # Only BF16 Dispatch_bwd is supported for now.
        grad_output = maybe_dequantize(grad_output, torch.bfloat16)

        # Router weight grads
        grad_recv_weights = basic_op_grad_extra_outputs[0][1]
        if grad_recv_weights is None:
            grad_recv_weights = torch.zeros(
                grad_output.shape[0],
                dtype=torch.float32,
                device=grad_output.device,
            )
        else:
            grad_recv_weights = grad_recv_weights.to(dtype=torch.float32)

        # NCCL EP dispatch backward
        grad_input, grad_topk_weights = _ep_dispatch_bwd(
            ctx.dispatch_state,
            grad_output,
            grad_recv_weights,
        )

        return grad_input, [()], [(None, grad_topk_weights)]
