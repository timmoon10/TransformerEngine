import torch
from .float8_utils import E4M3, E5M2, tensor_to_scale
from torch.utils._pytree import tree_map
aten = torch.ops.aten
c10d = torch.ops.c10d

class Float8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion between fp32 and fp8
    TODO(future): split into two for cleaner code
    """
    @staticmethod
    def forward(ctx, tensor, scale: float=None, flavor=E4M3):
        if isinstance(tensor, Float8Tensor):
            ctx.inp_is_float8 = True
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

    def __new__(cls, data, scale, flavor):
        # This is a non-differentiable constructor!
        assert not data.requires_grad
        # TODO(future): make bits8 easier to work with and switch to using it
        # assert data.dtype == torch.bits8
        # assert scale.dtype == torch.float32
        # assert scale.nelement() == 1

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=torch.float32,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        # self.fp8_meta_instance_view = fp8_meta_instance

        # get the scale from the `fp8_meta_instance` view (currently isn't that
        # straightforward)
        # self._scale = self.fp8_meta_instance_view["scaling_fwd"][1]

        self._scale = scale
        # TODO: get the flavor from the fp8_meta_instance as well
        # for now, hardcoding it
        self._flavor = E4M3

        return self

    @classmethod
    def from_float32(cls, tensor, scale, flavor):
        return Float8ConstrFunc.apply(tensor, scale, flavor)

    def to_float32(self):
        return Float8ConstrFunc.apply(self)

    def __repr__(self):
        return f"Float8Tensor(flavor={self._flavor}, scale={self._scale}, as_float32={self.to_float32()}"

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        print(func)

        def maybe_unwrap(t):
            if isinstance(t, Float8Tensor):
                return t.to_float32()
            return t

        def maybe_wrap(t):
            if not isinstance(t, Float8Tensor):
                return Float8Tensor.from_float32(t, tensor_to_scale(t, E4M3), E4M3)
            return t

        if func == aten.detach.default:

            ## Method 1: cast args to fp32, then recast the `out` to fp8
            # args = tree_map(maybe_unwrap, args)
            # if kwargs is not None:
            #     kwargs = tree_map(maybe_unwrap, kwargs)

            # No need to call this, just return a new tensor artificially
            # out = super().__torch_dispatch__(func, types, args, kwargs)

            # This won't work since `out` doesn't have any data if the `args`
            # were just `Float8Tensor`
            # out = tree_map(maybe_wrap, out)

            ## Method 2: simply return a new `Float8Tensor` with the `_data`,
            ## `_scale` & `_flavor` referencing the original `Float8Tensor`
            original_fp8_tensor = args[0]
            # `Float8Tensor` by default has `requires_grad=False` so this
            # should work.
            # NOTE: When initializing parameters as `Float8Tensor` inside module
            # (TE.Linear for ex.), this op `aten.detach.default` is called
            # twice. Not sure why
            out = Float8Tensor(
                original_fp8_tensor._data,
                original_fp8_tensor._scale,
                original_fp8_tensor._flavor
            )

        else:
            out = super().__torch_dispatch__(func, types, args, kwargs)

        return out

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
