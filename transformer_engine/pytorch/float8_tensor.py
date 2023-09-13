import torch
from .float8_utils import E4M3, E5M2, tensor_to_scale
from torch.utils._pytree import tree_map
aten = torch.ops.aten
c10d = torch.ops.c10d

from .cpp_extensions import (
    cast_to_fp8,
    cast_from_fp8,
)
from .fp8 import get_fp8_te_dtype
import transformer_engine_extensions as tex
from transformer_engine.pytorch.constants import TE_DType
import warnings

from inspect import currentframe, getframeinfo
def get_current_loc():
    cf = currentframe()
    return f"{getframeinfo(cf).filename}:{cf.f_back.f_lineno}"

class Float8DummyFunc(torch.autograd.Function):
    """
    A dummy function to create an autograd node
    """
    @staticmethod
    def forward(ctx, tensor):

        assert isinstance(tensor, Float8Tensor), ("Input can't be anything "
                                                  "other than a Float8Tensor. ")
        return torch.randn(1)

    @staticmethod
    def backward(ctx, g):
        return g

class Float8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion between fp32 and fp8
    TODO(future): split into two for cleaner code
    """
    @staticmethod
    def forward(ctx, tensor, scale: float=None, flavor=E4M3):
        if isinstance(tensor, Float8Tensor):
            ctx.inp_is_float8 = True
            # return torch.ops.aten.float8_to_float32(tensor._data, tensor._flavor) / tensor._scale
            # TODO (sudhakars): This needs to be a reference to the exact scale
            # value. This is not the case currently because internally the
            # scale factor calculation returns a new tensor for
            # `fp8_meta["scaling_fwd"].scaling` and doesn't update the existing
            # tensor in place
            return torch.ops.aten.float8_to_float32(tensor._data, tensor._flavor) / tensor._scale
        else:
            ctx.inp_is_float8 = False
            tensor_scaled = tensor * scale
            bits_fp8 = torch.ops.aten.float32_to_float8(tensor_scaled, flavor)
            return Float8Tensor(bits_fp8, scale, flavor)

    @staticmethod
    def backward(ctx, g):
        # Assume that we always want to scale the gradients
        # back to full precision. We could do something else
        if isinstance(g, Float8Tensor) and not ctx.inp_is_float8:
            return g.to_float32(), None, None
        elif ctx.inp_is_float8:
            return Float8Tensor.from_float32(g), None, None
        else:
            return g, None, None

class Float8Tensor(torch.Tensor):
    """
    A Python-only FP8 tensor.  Contains:
    * `_data`: the underlying e4m3 or e5m2 data
    * `_scale`: the scale used to scale the original fp32 tensor. We multiply
      by scale to go from fp32 range to fp8 range, and divide by scale to go
      from fp8 range to fp32 range.
    * `_flavor`: either E4M3 or E5M2
    """

    def __new__(cls, data, fp8_meta_view=None, gemm_index=1, fake_dtype=torch.float32):
        # This is a non-differentiable constructor!
        assert not data.requires_grad

        # TODO(future): make bits8 easier to work with and switch to using it
        # assert data.dtype == torch.bits8

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=fake_dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        self.fp8_meta_view = fp8_meta_view
        self.gemm_index = gemm_index

        # For now the only case, when `fp8_meta_view` is `None` is when empty
        # `Float8Tensors` are initialized in the `get_fp8_weights_scratchpad`
        # method in TE's layers.
        if self.fp8_meta_view is not None:
            self._scale_inv_cache = self.fp8_meta_view['scaling_fwd'].scale_inv[self.gemm_index]
        else:
            self._scale_inv_cache = None

        # Also saving the transpose of the underlying data as an optimization
        # self._data_transpose = self._data.detach().clone().transpose(0,1)
        self._transpose = None
        return self

    @property
    def _scale(self):
        assert self.fp8_meta_view is not None, ("Actual scale is fetched from "
                                                "fp8 meta tensor from the TE "
                                                "module.")
        # NOTE: currently hardcoding the scale location, later switch to info
        # from the `fp8_recipe`.
        return self.fp8_meta_view['scaling_fwd'].scale[self.gemm_index]

    def _flavor(self):
        # NOTE: currently hardcoding the flavor but it should be derived from
        # the `fp8_meta_view` object
        return E4M3

    def to_fake_dtype(self, dtype):
        if self.dtype is not dtype:
            converted_tensor = Float8Tensor(
                data=self._data,
                fp8_meta_view=self.fp8_meta_view,
                gemm_index=self.gemm_index,
                fake_dtype=dtype
            )
        else:
            converted_tensor = self
        return converted_tensor

    def bfloat16(self):
        higher_precision_tensor = self.upcast_from_fp8()
        if higher_precision_tensor.dtype is not torch.bfloat16:
            higher_precision_tensor = higher_precision_tensor.bfloat16()

        return higher_precision_tensor

    def float(self):
        higher_precision_tensor = self.upcast_from_fp8()
        if higher_precision_tensor.dtype is not torch.float32:
            higher_precision_tensor = higher_precision_tensor.float()

        return higher_precision_tensor

    def transpose(self, *args):
        if self._transpose is None:
            self._transpose = Float8Tensor(
                data=self._data.detach().clone().transpose(0,1),
                fp8_meta_view=self.fp8_meta_view,
                gemm_index=self.gemm_index,
                fake_dtype=self.dtype
            )

        return self._transpose

    def cpu(self):
        return self.upcast_from_fp8().cpu()

    def clone(self):
        return Float8Tensor(
            data=self._data.detach().clone(),
            fp8_meta_view=self.fp8_meta_view,
            gemm_index=self.gemm_index,
            fake_dtype=self.dtype,
        )

    def upcast_from_fp8(self):
        # For now, we need to print the weights and for printing weights, it
        # makes sense to use `fprop_tensor=True`
        fp8_dtype = get_fp8_te_dtype(
                self.fp8_meta_view["recipe"], fprop_tensor=True
        )
        assert self.dtype in TE_DType, ("Upcasting from FP8 only for supported "
                                        "dtypes in Transformer Engine")

        return cast_from_fp8(
            self._data,
            self.fp8_meta_view["scaling_fwd"],
            self.gemm_index,
            fp8_dtype,
            TE_DType[self.dtype],
        )


    # NOTE(sudhakars): This function isn't being used anymore. Remove it from
    # here once verified that it's not needed anymore.
    # def to_float32(self):
    #     return Float8ConstrFunc.apply(self)

    def expand_as(self, unused):
        # NOTE: A hack to create a dummy autograd node and then get the
        # hook to `AccumulateGrad`. (sudhakars): need to add the actual
        # `expand_as` functionality as well.
        return Float8DummyFunc.apply(self)

    def __repr__(self):
        return f"Float8Tensor(flavor={self._flavor}, scale={self._scale}, as_float32={self.upcast_from_fp8()}"

    @classmethod
    def cast_to_fp8(cls, fp8_tensor, tensor_to_cast):
        # NOTE: For now, we're casting just weights, so `fprop_tensor=True` is
        # good.
        fp8_dtype_forward = get_fp8_te_dtype(
                fp8_tensor.fp8_meta_view["recipe"], fprop_tensor=True
        )

        fp8_tensor._data = cast_to_fp8(
                tensor_to_cast,
                fp8_tensor.fp8_meta_view["scaling_fwd"],
                fp8_tensor.gemm_index,
                fp8_dtype_forward,
        )

        # Update the `_scale_inv` since we've updated the `_data` attr and
        # in the next iteration, this will change the
        fp8_tensor._scale_inv_cache = fp8_tensor.fp8_meta_view['scaling_fwd'].scale_inv[fp8_tensor.gemm_index]


    @classmethod
    def from_float32(cls, tensor, scale, flavor):
        return Float8ConstrFunc.apply(tensor, scale, flavor)

    @classmethod
    def update_inplace_fp8_tensor(cls, t, u):
        if isinstance(t, Float8Tensor) and isinstance(u, Float8Tensor):
            t._data = u._data
            t.fp8_meta_view = u.fp8_meta_view
            t.gemm_index = u.gemm_index

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        print(f"{get_current_loc()}", f"[Floa8Tensor Torch Dispatch] func: {func}\n func inplace: {func._schema.is_mutable}\n mutable args: {[a.is_out for a in func._schema.arguments]}")
        print(f"{get_current_loc()}", f"{[a.is_out for a in func._schema.arguments]}")
        print(f"{get_current_loc()}", f"{[type(a) for a in args]}")

        def maybe_unwrap(t):
            if isinstance(t, Float8Tensor):
                return t.upcast_from_fp8()
            return t

        def maybe_wrap(t):
            if not isinstance(t, Float8Tensor):
                warnings.warn("This is problematic if the input tensor and output tensor are expected to behave almost similarly but since we're converting FP8 -> FP32/BF16 -> FP8, there's some inherent loss of info. E.g. `expand_as`")
                return Float8Tensor.from_float32(t, tensor_to_scale(t, E4M3), E4M3)
            return t

        def maybe_wrap_and_update_inplace(arg, new_arg, schema_arg):
            """
            This converts the higher precision tensor to FP8 in-place in the
            sense that the original FP8 tensor's `_data` attribute gets updated.
            The "scale" values are already updated/maintained in the
            `fp8_meta_view` attr for FP8 tensor.
            """
            if(
                isinstance(arg, Float8Tensor) and
                isinstance(new_arg, torch.Tensor) and
                hasattr(schema_arg, 'alias_info') and
                hasattr(schema_arg.alias_info, 'is_write') and
                schema_arg.alias_info.is_write
            ):
                Float8Tensor.cast_to_fp8(arg, new_arg)

        if func is aten.copy_.default:
            assert isinstance(args[0], Float8Tensor) and \
                isinstance(args[1], torch.Tensor), \
                "recheck the input types for the tensor copy " \
                "operation"

            Float8Tensor.cast_to_fp8(args[0], args[1])
            # This is an inplace copy op, so nothing to return
            return None

        if func._schema.is_mutable:
            new_args = tree_map(maybe_unwrap, args)
            new_kwargs = tree_map(maybe_unwrap, kwargs)
            schema_args = func._schema.arguments
            args_len = len(args)

            out = super().__torch_dispatch__(func, types, new_args, new_kwargs)

            for arg, new_arg, schema_arg in zip(args, new_args, schema_args):
                maybe_wrap_and_update_inplace(arg, new_arg, schema_arg)

            for kwarg, new_kwarg, schema_arg in zip(kwargs, new_kwargs, schema_args[args_len:]):
                assert kwarg == new_kwarg == schema_arg.name, "name of the kw argument should match"
                maybe_wrap_and_update_inplace(kwargs[kwarg], new_kwargs[new_kwarg], schema_arg)

            return None

        if func == aten.slice.Tensor:
            raise AssertionError("Slice operation on Float8Tensor is unsupported!")

        if func == aten.transpose.int:
            raise AssertionError("Transpose operation on Float8Tensor is unsupported!")
            original_fp8_tensor = args[0]
            out = Float8Tensor(
                original_fp8_tensor._data.transpose(0,1).detach().clone(),
                original_fp8_tensor.fp8_meta_view,
                original_fp8_tensor.gemm_index,
                original_fp8_tensor.fake_dtype
            )
            return out

        if func == aten.detach.default:
            ## Simply return a new `Float8Tensor` with the `_data`,
            ## `_scale` & `_flavor` referencing the original `Float8Tensor`
            original_fp8_tensor = args[0]
            # `Float8Tensor` by default has `requires_grad=False` so this
            # should work.
            # NOTE: When initializing parameters as `Float8Tensor` inside module
            # (TE.Linear for ex.), this op `aten.detach.default` is called
            # twice. Not sure why
            out = Float8Tensor(
                original_fp8_tensor._data,
                original_fp8_tensor.fp8_meta_view,
                original_fp8_tensor.gemm_index,
                original_fp8_tensor.dtype
            )
            return out

        # TODO: (ksivaman) For all other cases, cast back to higher precision
        # weights, then convert back to FP8
        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)

        out = super().__torch_dispatch__(func, types, args, kwargs)

        # Convert the output back to FP8
        out = tree_map(maybe_wrap, out)

        return out

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
