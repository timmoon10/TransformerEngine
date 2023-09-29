# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations
from typing import Any, Dict, Optional

import torch
from torch.utils._pytree import tree_map
aten = torch.ops.aten
c10d = torch.ops.c10d

from .constants import TE_DType
from .fp8 import FP8GlobalStateManager, get_fp8_te_dtype
import transformer_engine_extensions as tex

class _FromFloat8Func(torch.autograd.Function):
    """Cast from FP8 to other dtype"""
    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = tensor.dtype
        data = tensor._data.contiguous().view(1,-1).detach()
        out = tex.cast_from_fp8(
            data,
            tensor._scale_inv,
            tensor._fp8_dtype,
            TE_DType[dtype],
        )
        out = out.view(tensor.size())
        return out

    @staticmethod
    def backward(ctx, grad):
        # Assume that we want gradients in full precision
        return grad, None


class _ToFloat8Func(torch.autograd.Function):
    """Cast to FP8 from other dtype"""
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        fp8_meta: Optional[Dict[str, Any]] = None,
        fp8_meta_index: Optional[int] = None,
        fp8_dtype: Optional[tex.DType] = None,
        scale: Optional[torch.Tensor] = None,
        amax: Optional[torch.Tensor] = None,
        scale_inv: Optional[torch.Tensor] = None,
    ):

        # Manually compute scale-inverse if needed
        if scale is not None and scale_inv is None:
            if isinstance(scale, torch.Tensor):
                scale_inv = scale_inv.reciprocal()
            else:
                scale_inv = 1 / scale

        # Extract data from FP8 meta tensors if provided
        if fp8_meta is not None:
            # TODO Handle bprop case
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)
            if fp8_meta_index is None:
                raise ValueError(
                    "To initialize Float8Tensor with FP8 meta tensors, "
                    "the FP8 meta tensor index must also be provided"
                )
            if scale is None:
                scale = fp8_meta[fp8_meta_key].scale[fp8_meta_index]
            if amax is None:
                amax = fp8_meta[fp8_meta_key].amax_history[0][fp8_meta_index]
            if scale_inv is None:
                scale_inv = fp8_meta[fp8_meta_key].scale_inv[fp8_meta_index]
            if fp8_dtype is None:
                # TODO Handle bprop case
                fp8_dtype = get_fp8_te_dtype(
                    fp8_meta["recipe"],
                    fprop_tensor=True,
                )

        # Check input tensor
        tensor = tensor.contiguous().cuda().detach()

        # Check scale
        if not isinstance(scale, torch.Tensor):
            scale = torch.full(
                [1],
                scale,
                dtype=torch.float32,
                device=tensor.device,
            )
        if scale.numel() != 1:
            raise ValueError(
                "Attempted to initialize Float8Tensor with invalid scale tensor"
            )
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        # Check amax
        if amax is None:
            amax = torch.empty_like(scale)
        if not (amax.numel() == 1 and amax.is_cuda and amax.dtype == torch.float32):
            raise ValueError(
                "Attempted to initialize Float8Tensor with invalid amax tensor"
            )
        if fp8_dtype is None:
            raise ValueError(
                "Attempted to initialize Float8Tensor without specifying FP8 dtype"
            )

        # Cast data to FP8
        data = tex.cast_to_fp8(
            tensor.view(1,-1),
            scale,
            amax,
            scale_inv,
            fp8_dtype,
        )
        data = data.view(tensor.size())

        # Construct FP8 tensor
        return Float8Tensor(
            data=data,
            fp8_meta=fp8_meta,
            fp8_meta_index=fp8_meta_index,
            fp8_dtype=fp8_dtype,
            fp8_scale_inv=scale_inv,
            dtype=tensor.dtype,
        )

    @staticmethod
    def backward(ctx, grad):
        # Assume that we want gradients in full precision
        return grad, None, None, None, None, None, None

class _IdentityFunc(torch.autograd.Function):
    """Identity function"""
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def backward(ctx, grad):
        return grad


class Float8Tensor(torch.Tensor):
    """Experimental tensor class with FP8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. Operations are performed by
    casting to the nominal dtype and results are cast back to FP8.

    Changes to the FP8 scaling factors, e.g. from the FP8 recipe, are
    handled outside this class.

    """

    def __new__(
        cls,
        *,
        data: torch.Tensor,
        fp8_meta: Optional[Dict[str, Any]] = None,
        fp8_meta_index: Optional[int] = None,
        fp8_dtype: Optional[tex.DType] = None,
        fp8_scale_inv: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ):

        # Check that data buffer is valid
        if data.element_size() != 1:
            raise ValueError(
                "Float8Tensor requires data buffer with 8-bit dtype "
                f"(got dtype={data.dtype})"
            )
        if data.requires_grad:
            raise ValueError(
                "Float8Tensor requires non-differentiable data buffer"
            )
        data = data.cuda()

        # Initialize tensor object
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data: torch.Tensor = data

        # FP8 meta tensors
        if fp8_meta is not None and fp8_meta_index is None:
            raise ValueError(
                "To initialize Float8Tensor with FP8 meta tensors, "
                "the FP8 meta tensor index must also be provided"
            )
        self._fp8_meta: Optional[Dict[str, Any]] = fp8_meta
        self._fp8_meta_index: Optional[int] = fp8_meta_index

        # FP8 dtype
        self._fp8_dtype: tex.DType = fp8_dtype
        if self._fp8_dtype is None and self._fp8_meta is not None:
            # TODO Handle bprop case
            self._fp8_dtype = get_fp8_te_dtype(
                self._fp8_meta["recipe"],
                fprop_tensor=True,
            )
        if self._fp8_dtype is None:
            raise ValueError(
                "Attempted to initialize Float8Tensor without specifying FP8 dtype"
            )

        # FP8 scale-inverse
        self._scale_inv: Optional[torch.Tensor] = fp8_scale_inv
        if self._scale_inv is None and self._fp8_meta is not None:
            # TODO Handle bprop case
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)
            scale_inv = self._fp8_meta[fp8_meta_key].scale_inv[self._fp8_meta_index]
            self._scale_inv = scale_inv.detach().view(1).clone()
        if self._scale_inv is None:
            raise ValueError(
                "Attempted to initialize Float8Tensor without specifying scale-inverse"
            )
        if not isinstance(self._scale_inv, torch.Tensor):
            self._scale_inv = torch.full(
                [1],
                self._scale_inv,
                dtype=torch.float32,
                device=self._data.device,
            )
        if self._scale_inv.numel() != 1:
            raise ValueError(
                "Attempted to initialize Float8Tensor with invalid scale-inverse tensor"
            )
        self._scale_inv = self._scale_inv.to(
            device=self._data.device,
            dtype=torch.float32,
        )

        # Cached transpose
        self._transpose: Optional[Float8Tensor] = None

        return self

    def __repr__(self):
        return (
            "Float8Tensor("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv={self._scale_inv.item()}, "
            f"as_float32={self.from_float8(dtype=torch.float32)}"
            ")"
        )

    def from_float8(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return _FromFloat8Func.apply(self, dtype)

    @classmethod
    def to_float8(
        cls,
        tensor: torch.Tensor,
        *,
        fp8_meta: Optional[Dict[str, Any]] = None,
        fp8_meta_index: Optional[int] = None,
        fp8_dtype: Optional[tex.DType] = None,
        scale: Optional[torch.Tensor] = None,
        amax: Optional[torch.Tensor] = None,
        scale_inv: Optional[torch.Tensor] = None,
    ):
        return _ToFloat8Func.apply(
            tensor,
            fp8_meta,
            fp8_meta_index,
            fp8_dtype,
            scale,
            amax,
            scale_inv,
        )

    def float(self) -> torch.Tensor:
        return self.from_float8(dtype=torch.float32)

    def bfloat16(self) -> torch.Tensor:
        return self.from_float8(dtype=torch.bfloat16)

    def half(self) -> torch.Tensor:
        return self.from_float8(dtype=torch.float16)

    def cpu(self) -> torch.Tensor:
        return self.from_float8().cpu()

    def clone(self) -> Float8Tensor:
        return Float8Tensor(
            data=self._data.detach().clone(),
            fp8_meta_view=self._fp8_meta,
            fp8_meta_index=self._fp8_meta_index,
            fp8_dtype=self._fp8_dtype,
            fp8_scale_inv=self._scale_inv,
            dtype=self.dtype,
        )

    def expand_as(self, other: torch.Tensor):
        if other is self:
            # Note: expand_as is hackily used to create dummy autograd nodes
            # and access the backward graph (see
            # https://github.com/pytorch/pytorch/blob/238fb660851268f44ff88127887041fea352fe48/torch/nn/parallel/distributed.py#L1026).
            # We equally hackily add a dummy function to handle this
            # case.
            return _IdentityFunc.apply(self)
        else:
            return super().expand_as(other)

    def transpose(self, dim0: int = 0, dim1: int = 1) -> Float8Tensor:
        # TODO Support differentiation
        if self.dim() != 2:
            raise RuntimeError(
                "Float8Tensor only supports transposing 2D tensors "
                f"(got ndim={self.dim()})"
            )
        if dim0 == dim1:
            return self
        if self._transpose is None:
            data_t = tex.fp8_transpose(
                self._data.contiguous().detach(),
                self._fp8_dtype,
            )
            self._transpose = Float8Tensor(
                data=data_t,
                fp8_meta=self._fp8_meta,
                fp8_meta_index=self._fp8_meta_index,
                fp8_dtype=self._fp8_dtype,
                fp8_scale_inv=self._scale_inv,
                dtype=self.dtype
            )
        return self._transpose

    @torch.no_grad()
    def reset_fp8_meta_scale_inv(self) -> None:
        """Replace FP8 meta tensor scale-inverse with cached value

        The FP8 meta tensor scale_inv entry corresponding to this
        tensor is replaced with the scale_inv value used to construct
        the tensor.

        """
        if self._fp8_meta is None:
            return
        # TODO Handle bprop case
        fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)
        scale_inv = self._fp8_meta[fp8_meta_key].scale_inv[self._fp8_meta_index]
        scale_inv.copy_(self._scale_inv)

    def to_dtype(self, dtype: torch.dtype) -> Float8Tensor:
        """Create `Float8Tensor` with given nominal dtype

        The new tensor has the same underlying FP8 data.

        """

        return Float8Tensor(
            data=self._data,
            fp8_meta=tensor._fp8_meta,
            fp8_meta_index=tensor._fp8_meta_index,
            fp8_dtype=tensor._fp8_dtype,
            fp8_scale_inv=tensor._scale_inv,
            dtype=dtype,
        )

    def _reset_caches(self) -> None:
        """Reset cached values

        Should be called after any in-place operation.

        """
        self._transpose = None

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # In-place copy op
        if func == aten.copy_.default:

            # Check tensors
            dst = args[0]
            src = args[1]
            if not isinstance(dst, Float8Tensor):
                raise RuntimeError("Expected to copy into Float8Tensor")
            if not isinstance(src, torch.Tensor):
                raise RuntimeError("Expected to copy from tensor")
            if not dst._data.is_contiguous():
                raise RuntimeError("Transformer Engine cast kernels require contiguous data")

            # Make sure input is in expected format
            if isinstance(src, Float8Tensor):
                src = src.from_float8()
            src = src.expand(dst.size())
            src = src.to(
                device=dst.device,
                memory_format=torch.contiguous_format,
            )

            # Cast to FP8
            tex.cast_to_fp8_noalloc(
                src.view(1,-1),
                dst._scale_inv.reciprocal(),
                dst._data.view(1,-1),
                torch.empty_like(dst._scale_inv),  # amax
                dst._scale_inv,
                dst._fp8_dtype,
            )

            # Nothing to return for in-place ops
            dst._reset_caches()
            return None

        # Slice op
        # TODO Keep track of master tensor so properly invalidate
        # caches?
        if func == aten.slice.Tensor:
            tensor = args[0]
            data = tensor._data
            data_slice = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return Float8Tensor(
                data=data_slice,
                fp8_meta=tensor._fp8_meta,
                fp8_meta_index=tensor._fp8_meta_index,
                fp8_dtype=tensor._fp8_dtype,
                fp8_scale_inv=tensor._scale_inv,
                dtype=tensor.dtype,
            )

        if func == aten.transpose.int:
            raise AssertionError("Transpose operation on Float8Tensor is unsupported!")
            return args[0].transpose(0, 1)

        # Detach op
        if func == aten.detach.default:
            # Simply return a new Float8Tensor with the same attrs
            tensor = args[0]
            return Float8Tensor(
                data=tensor._data.detach(),
                fp8_meta=tensor._fp8_meta,
                fp8_meta_index=tensor._fp8_meta_index,
                fp8_dtype=tensor._fp8_dtype,
                fp8_scale_inv=tensor._scale_inv,
                dtype=tensor.dtype,
            )

        # Find FP8 tensor so we can get its FP8 scaling factors
        base_fp8_tensor = None
        for t in args:
            if isinstance(t, Float8Tensor):
                base_fp8_tensor = t
                break

        def maybe_unwrap(t):
            if isinstance(t, Float8Tensor):
                return t.from_float8()
            return t


        def maybe_wrap(t):
            if not isinstance(t, Float8Tensor):
                assert base_fp8_tensor is not None, (
                    "Could not find Float8Tensor. "
                    "Unclear what scaling factors to use for FP8 casts."
                )
                return Float8Tensor.to_float8(
                    t,
                    fp8_meta=base_fp8_tensor._fp8_meta,
                    fp8_meta_index=base_fp8_tensor._fp8_meta_index,
                    fp8_dtype=base_fp8_tensor._fp8_dtype,
                    scale=base_fp8_tensor._scale_inv.reciprocal(),
                    amax=torch.empty_like(base_fp8_tensor._scale_inv),
                    scale_inv=base_fp8_tensor._scale_inv,
                )
            return t

        def maybe_update_inplace(arg, new_arg, schema_arg):
            """Update values of FP8 tensors

            Keep the same FP8 scaling factors.

            """
            if(
                isinstance(arg, Float8Tensor) and
                isinstance(new_arg, torch.Tensor) and
                hasattr(schema_arg, 'alias_info') and
                hasattr(schema_arg.alias_info, 'is_write') and
                schema_arg.alias_info.is_write
            ):
                arg.copy_(new_arg)
                arg._reset_caches()

        # In-place op
        if func._schema.is_mutable:
            # Cast to higher precision, perform op, and cast values
            # back to original FP8 buffers
            new_args = tree_map(maybe_unwrap, args)
            new_kwargs = tree_map(maybe_unwrap, kwargs)
            schema_args = func._schema.arguments
            args_len = len(args)
            out = super().__torch_dispatch__(func, types, new_args, new_kwargs)
            for arg, new_arg, schema_arg in zip(args, new_args, schema_args):
                maybe_update_inplace(arg, new_arg, schema_arg)
            for kwarg, new_kwarg, schema_arg in zip(kwargs, new_kwargs, schema_args[args_len:]):
                assert kwarg == new_kwarg == schema_arg.name, "name of the kw argument should match"
                maybe_update_inplace(kwargs[kwarg], new_kwargs[new_kwarg], schema_arg)
            return None

        # Default op
        # Note: cast to higher precision and perform op
        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
