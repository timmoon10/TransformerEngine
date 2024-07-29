# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNorm API"""
import warnings
from typing import Iterable, Optional, Union

import torch

from transformer_engine.pytorch.ops import LayerNorm as _LayerNormOp

__all__ = ["LayerNorm"]


class LayerNorm(_LayerNormOp):
    r"""Layer Normalization

    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * \gamma + \beta

    :math:`\gamma` and :math:`\beta` are learnable affine transform
    parameters that match the inner-most dimensions of the input
    tensor.

    Parameters
    ----------
    normalized_shape: int or iterable of int
        Inner dimensions of input tensor
    eps : float, default = 1e-5
        A value added to the denominator of layer normalization for
        numerical stability
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype
    zero_centered_gamma : bool, default = 'False'
        If `True`, the :math:`\gamma` parameter is initialized to zero
        and the calculation changes to

            .. math::
                y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * (1 + \gamma) + \beta

    sm_margin: int, default = 0
        Number of SMs to exclude when launching CUDA kernels. This
        helps overlap with other kernels, e.g. communication kernels.

    """

    def __init__(
        self,
        normalized_shape: Union[Iterable[int], int],
        *,
        sequence_parallel: Optional[bool] = None,  # deprecated
        params_dtype: Optional[torch.dtype] = None,  # deprecated
        **kwargs,
    ) -> None:

        # Handle deprecated options
        if params_dtype is not None:
            if "dtype" in kwargs:
                raise RuntimeError(
                    "Both `dtype` and `params_dtype` (deprecated) kwargs are provided"
                )
            kwargs["dtype"] = params_dtype

        # Initialize layer norm operation
        super().__init__(normalized_shape, **kwargs)

        # Flag for sequence parallelism (deprecated)
        self.sequence_parallel: Optional[bool] = sequence_parallel

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        warnings.warn(
            "This method will be deprecated in an upcoming release. "
            "Update your code to use LayerNorm.reset_parameters() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.reset_parameters()

    def reset_parameters(self, defer_init: Optional[bool] = None) -> None:
        """Init LayerNorm parameters"""

        # Check whether to defer init (deprecated)
        if defer_init is not None:
            warnings.warn(
                'reset_parameters kwarg is deprecated. Set device to "meta" instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            if defer_init:
                return

        # Reset parameters
        super().reset_parameters()

        # Set flag for sequence parallelism (deprecated)
        if getattr(self, "sequence_parallel", None) is not None:
            self.weight.sequence_parallel = self.sequence_parallel
            self.bias.sequence_parallel = self.sequence_parallel

    @property
    def fwd_ln_sm_margin(self) -> int:
        """Shim for backward compatibility"""
        warnings.warn("fwd_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        return self._sm_margins["fwd"]

    @fwd_ln_sm_margin.setter
    def fwd_ln_sm_margin(self, val: int) -> None:
        """Shim for backward compatibility"""
        warnings.warn("fwd_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        self._sm_margins["fwd"] = val

    @property
    def bwd_ln_sm_margin(self) -> int:
        """Shim for backward compatibility"""
        warnings.warn("bwd_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        return self._sm_margins["bwd"]

    @bwd_ln_sm_margin.setter
    def bwd_ln_sm_margin(self, val: int) -> None:
        """Shim for backward compatibility"""
        warnings.warn("bwd_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        self._sm_margins["bwd"] = val

    @property
    def inf_ln_sm_margin(self) -> int:
        """Shim for backward compatibility"""
        warnings.warn("inf_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        return self._sm_margins["inf"]

    @inf_ln_sm_margin.setter
    def inf_ln_sm_margin(self, val: int) -> None:
        """Shim for backward compatibility"""
        warnings.warn("inf_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        self._sm_margins["inf"] = val
