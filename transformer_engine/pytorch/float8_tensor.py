import torch

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
        self._scale = scale
        self._flavor = flavor

        return self

    def __repr__(self):
        return f"Float8Tensor(flavor={self._flavor}, scale={self._scale}, as_float32={self.to_float32()}"

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
