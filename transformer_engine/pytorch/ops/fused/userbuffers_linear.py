# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear layer with Userbuffers communication."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpp_extensions import FP8TensorMeta, fp8_gemm, gemm
from ...distributed import get_distributed_world_size
from ...float8_tensor import Float8Tensor
from ...fp8 import FP8GlobalStateManager
from ...module.base import get_ub, get_workspace
from ..basic import BasicLinear, Bias
from ..op import (
    BasicOperation,
    FusedOperation,
    FusibleOperation,
    OperationContext,
)
from .._common import (
    canonicalize_device,
    canonicalize_dtype,
    convert_tensor,
    is_float8_tensor,
    reshape,
)

class UserbuffersLinear(FusedOperation):

    def __init__(
        self,
        *,
        linear: BasicLinear,
        bias: Optional[Bias],
    ) -> None:

        # Basic operations that comprise this fused operation
        op_idxs = dict(
            linear=0,
            bias=None,
        )
        ops = [linear]
        if bias is not None:
            op_idxs["bias"] = len(ops)
            ops.append(bias)

        # Initialize base class
        super().__init__(ops)

        # Index of each basic operations
        self._op_idxs: dict[str, Optional[int]] = op_idxs

    @staticmethod
    def _functional_forward(
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
        with_fp8_compute: bool = False,
        input_fp8_meta: Optional[dict[str, Any]] = None,
        weight_fp8_meta: Optional[dict[str, Any]] = None,
        output_fp8_meta: Optional[dict[str, Any]] = None,
        ub_comm_name: str,
    ):

        # Check device
        if device is None:
            device = weight.device if out is None else out.device
        device = canonicalize_device(device)
        if device.type != "cuda":
            raise ValueError(f"Only CUDA devices are supported (got {device})")

        # Check datatype
        if dtype is None:
            dtype = weight.dtype if out is None else out.dtype
        dtype = canonicalize_dtype(dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")

        # Check input tensor dims
        input_dims = tuple(input.size())
        weight_dims = tuple(weight.size())
        if len(weight_dims) != 2:
            raise ValueError(f"Weight tensor is not 2D (shape={weight_dims})")
        if len(input_dims) == 0 or weight_dims[1] != input_dims[-1]:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and weight tensor (shape={weight_dims}) "
                "are not compatible"
            )

        # Check output tensor dims
        output_dims: list[int]
        output_dims = list(input_dims)
        output_dims[0] = -1
        output_dims[-1] = weight_dims[0]

        # Check if FP8 is enabled
        if with_fp8_compute:
            if input_fp8_meta is None and not is_float8_tensor(input):
                raise ValueError("No FP8 metadata was provided for casting input to FP8")
            if weight_fp8_meta is None and not is_float8_tensor(weight):
                raise ValueError("No FP8 metadata was provided for casting weight to FP8")
        else:
            input_fp8_meta = None
            weight_fp8_meta = None
            output_fp8_meta = None
        with_fp8_output = with_fp8_compute and tensor_parallel_mode != "row"
        with_fp8_output = with_fp8_output and output_fp8_meta is not None

        # Check tensor parallel group
        tensor_parallel_group_size = get_distributed_world_size(tensor_parallel_group)
        if tensor_parallel_group_size == 1:
            tensor_parallel_mode = None
        if tensor_parallel_mode not in ("column", "row"):
            raise RuntimeError(
                "Invalid configuration for Userbuffers "
                f"({tensor_parallel_group_size=}, {tensor_parallel_mode=})"
            )
        if not sequence_parallel:
            raise RuntimeError(
                f"Invalid configuration for Userbuffers ({sequence_parallel=})"
            )

        # Get Userbuffers communicator
        ub_comm = get_ub(ub_comm_name + "_fprop")
        ub_local_buffer = ub_comm.get_ubuf_output(0)
        ub_global_buffer = ub_comm.get_ubuf_output(1)
        with_ub_all_gather = tensor_parallel_mode == "column"
        with_ub_reduce_scatter = tensor_parallel_mode == "row"

        # Choose Userbuffers communication algorithm
        ub_algo = None
        if with_ub_all_gather:
            if with_fp8_compute and ub_comm.is_atomic_gemm():
                ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_AG_P2P
            else:
                ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG_P2P
        elif with_ub_reduce_scatter:
            raise NotImplementedError ### TODO Implement
        else:
            raise RuntimeError("Could not choose Userbuffers communication algorithm")

        # Cast input tensor to correct dtype
        x_local = reshape(
            input,
            (-1, input_dims[-1]),
            device=device,
            dtype=dtype,
        )
        if with_fp8_compute and not is_float8_tensor(x_local):
            fp8_dtype = get_fp8_te_dtype(
                input_fp8_meta["recipe"],
                fprop_tensor=True,
            )
            if with_ub_all_gather:
                data = ub_local_buffer
            else:
                data = torch.empty_like(x_local, dtype=torch.uint8)
            x_fp8 = Float8Tensor(
                data=data,
                fp8_meta=input_fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
                fp8_scale_inv=torch.empty([1], dtype=torch.float32, device=device),
                dtype=dtype,
            )
            with_cast_transpose = weight.requires_grad
            if tensor_parallel_mode == "column" and sequence_parallel:
                with_cast_transpose = False
            if with_cast_transpose:
                x_fp8.cast_transpose_(x_local)
            else:
                x_fp8.copy_(x_local)
            x_local = x_fp8
        elif not with_fp8_compute and is_float8_tensor(x_local):
            if with_ub_all_gather:
                x_local = ub_local_buffer.copy_(x_local)
            else:
                x_local = x_local.from_float8()

        # Initialize buffers for UB all-gather if needed
        x = x_local
        if with_ub_all_gather:
            if with_fp8_compute:
                x = Float8Tensor.make_like(x_local, data=ub_global_buffer)
                if x_local._data.data_ptr() != ub_local_buffer.data_ptr():
                    ub_local_buffer.copy_(x_local._data)
                else:
                    x_local._data = torch.empty_like(x_local._data)
            else:
                x = ub_global_buffer
                if x_local.data_ptr() != ub_local_buffer.data_ptr():
                    ub_local_buffer.copy_(x_local)
                else:
                    x_local = torch.empty_like(x_local)

        # Check weight tensor
        w = convert_tensor(
            weight,
            device=device,
            dtype=dtype,
            memory_format=torch.contiguous_format,
        )
        if with_fp8_compute and not is_float8_tensor(w):
            fp8_dtype = get_fp8_te_dtype(
                weight_fp8_meta["recipe"],
                fprop_tensor=True,
            )
            w = Float8Tensor.to_float8(
                w,
                fp8_meta=weight_fp8_meta,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
            )
        elif not with_fp8_compute and is_float8_tensor(w):
            w = w.from_float8()

        # Check bias tensor
        b = None
        if bias is not None:
            b = convert_tensor(
                bias,
                device=device,
                dtype=dtype,
                memory_format=torch.contiguous_format,
            )

        # Construct output tensor
        ### TODO UB RS
        y = None
        if with_fp8_output:
            fp8_dtype = get_fp8_te_dtype(
                output_fp8_meta["recipe"],
                fprop_tensor=True,
            )
            data = torch.empty(
                (x.size(0), weight_dims[0]),
                dtype=torch.uint8,
                device=device,
            )
            y = Float8Tensor(
                data=data,
                fp8_meta=output_fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
                dtype=dtype,
            )
        else:
            y = torch.empty(
                (x.size(0), weight_dims[0]),
                dtype=dtype,
                device=device,
            )

        # Perform GEMM
        if with_fp8_compute:
            kwargs = dict(
                out=y,
                bias=b,
                use_bias=(b is not None),
                ub_algo=ub_algo,
                ub=ub_comm,
            )
            if with_ub_all_gather:
                kwargs["extra_output_tensor"] = x_local._data
            if with_fp8_output:
                if y._fp8_meta is None:
                    # Hackily create FP8TensorMeta if needed
                    fp8_meta = FP8TensorMeta()
                    fp8_meta.scale = y._scale_inv.reciprocal()
                    fp8_meta.amax_history = torch.empty(1, 1, dtype=torch.float32, device=device)
                    fp8_meta.scale_inv = y._scale_inv
                    fp8_meta_index = 0
                else:
                    # Get FP8TensorMeta from Float8Tensor
                    fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                        forward=y._fp8_meta_forward,
                    )
                    fp8_meta = y._fp8_meta[fp8_meta_key]
                    fp8_meta_index = y._fp8_meta_index
                kwargs.update(
                    dict(
                        out=y._data,
                        out_index=fp8_meta_index,
                        fp8_meta_tensor=fp8_meta,
                        D_dtype=y._fp8_dtype,
                    )
                )
            fp8_gemm(
                w._data,
                w._scale_inv,
                0,
                w._fp8_dtype,
                x._data,
                x._scale_inv,
                0,
                x._fp8_dtype,
                y.dtype,
                get_workspace(),
                **kwargs,
            )
        else:
            kwargs = dict(
                out=y,
                bias=b,
                use_bias=(b is not None),
                ub_algo=ub_algo,
                ub=ub_comm,
            )
            if with_ub_all_gather:
                kwargs["extra_output_tensor"] = x_local
            gemm(w, x, y.dtype, get_workspace(), **kwargs)

        # Reshape output tensor
        out = reshape(y, output_dims)

        return out, x_local, w

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:

        # Get basic operations
        idx = self._op_idxs["linear"]
        linear_op = self.basic_ops[idx]
        linear_op_ctx = basic_op_ctxs[idx]
        if self._op_idxs["bias"] is None:
            bias_op = None
            bias = None
        else:
            idx = self._op_idxs["bias"]
            bias_op = self.basic_ops[idx]
            bias = bias_op.bias
            if basic_op_kwargs[idx]:
                raise ValueError("Bias operation forward does not expect keyword arguments")

        # FP8 metadata
        with_fp8_compute = FP8GlobalStateManager.is_fp8_enabled()
        input_fp8_meta = None
        weight_fp8_meta = None
        output_fp8_meta = None
        grad_output_fp8_meta = None
        grad_input_fp8_meta = None
        if with_fp8_compute:
            input_fp8_meta = linear_op.get_fp8_meta("input")
            weight_fp8_meta = linear_op.get_fp8_meta("param")
            next_op = basic_op_next_ops[-1]
            if next_op is not None and next_op.num_fp8_scales("input") > 0:
                output_fp8_meta = next_op.get_fp8_meta("input")
            grad_output_fp8_meta = linear_op.get_fp8_meta("grad_output")
            prev_op = basic_op_prev_ops[0]
            if prev_op is not None and prev_op.num_fp8_scales("grad_output") > 0:
                grad_input_fp8_meta = prev_op.get_fp8_meta("grad_output")

        # Userbuffers options
        if linear_op._userbuffers_options is None:
            raise RuntimeError("Linear op is missing dict for Userbuffers options")

        # Linear forward
        output, x_local, _ = UserbuffersLinear._functional_forward(
            input=input_,
            weight=linear_op.weight,
            bias=bias,
            device=linear_op.device,
            dtype=linear_op.dtype,
            tensor_parallel_mode=linear_op.tensor_parallel_mode,
            tensor_parallel_group=linear_op.tensor_parallel_group,
            sequence_parallel=linear_op.sequence_parallel,
            with_fp8_compute=with_fp8_compute,
            input_fp8_meta=input_fp8_meta,
            weight_fp8_meta=weight_fp8_meta,
            output_fp8_meta=output_fp8_meta,
            ub_comm_name=linear_op._userbuffers_options["comm_name"],
        )

        # Save state for backward pass
        linear_op_ctx.save_for_backward(x_local)
        linear_op_ctx.with_fp8_compute = with_fp8_compute
        linear_op_ctx.weight_fp8_meta = weight_fp8_meta
        linear_op_ctx.grad_output_fp8_meta = grad_output_fp8_meta
        linear_op_ctx.grad_input_fp8_meta = grad_input_fp8_meta
        linear_op_ctx.input_dims = input_.size()
        linear_op_ctx.input_requires_grad = input_.requires_grad
        linear_op_ctx.weight_requires_grad = linear_op.weight.requires_grad
        linear_op_ctx.has_prev_op = basic_op_prev_ops[0] is not None

        return output, [() for _ in range(len(self.basic_ops))]


def fuse_forward_userbuffers_linear(
    ops: list[tuple[FusibleOperation, list[int]]],
) -> list[tuple[FusibleOperation, list[int]]]:

    # Scan through ops, fusing if possible
    out = []
    window = []
    while ops:
        out.extend(window)

        # Check if first op is linear
        op1, _ = ops[0]
        window = [ops[0]]
        ops = ops[1:]
        if not isinstance(op1, BasicLinear):
            continue
        if op1.tensor_parallel_mode not in ("column", "row"):
            continue
        if op1._userbuffers_options is None:
            continue

        # Check if second op is bias
        op2 = None
        if ops and isinstance(ops[0][0], Bias):
            op2, _ = ops[0]
            window.append(ops[0])
            ops = ops[1:]

        # Replace window with fused op
        op = UserbuffersLinear(
            linear=op1,
            bias=op2,
        )
        basic_op_idxs = [basic_op_idxs[0] for _, basic_op_idxs in window]
        window = [(op, basic_op_idxs)]

    # Return list of ops
    out.extend(window)
    return out
