# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

def test_float8tensor_sanity():
    fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3, reduce_amax=False)
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        te_model = te.Linear(in_features=16, out_features=16, bias=False, params_dtype=torch.float32, primary_weights_in_fp8=True)

    scale = te_model.weight.fp8_meta_view['scaling_fwd'].scale[1]
    ref_tensor = torch.ops.aten.float8_to_float32(te_model.weight._data, te_model.weight._flavor) / scale

    assert torch.equal(ref_tensor, te_model.weight.upcast_from_fp8())
