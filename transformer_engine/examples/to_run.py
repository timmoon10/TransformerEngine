import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import numpy as np
from transformer_engine.pytorch import Float8Tensor, E4M3, tensor_to_scale

torch.manual_seed(0)
np.random.seed(0)


# Set dimensions.
in_features = 16
out_features = 16
hidden_size = 16

model = te.Linear(in_features, out_features, bias=False, params_dtype=torch.float32)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Run 1 iter
inp = torch.randn(hidden_size, in_features, device="cuda")
print(inp[0])
# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3, reduce_amax=False)

optimizer.zero_grad()
# Enable autocasting for the forward pass
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    print(model.fp8_meta)
    out = model(inp, is_first_microbatch=None)
    print(model.fp8_meta)

loss = out.sum()
loss.backward()

# Run optimizer step
optimizer.step()

# print("scale info: ", model.fp8_meta['scaling_fwd'].scale)
# a_h_flat = model.fp8_meta['scaling_fwd'].amax_history.cpu().numpy() #[1024,3]
# np.where(a_h_flat > 0.0), a_h_flat[a_h_flat > 0.0]

# Run 2 iters
for _ in range(2):
    inp = torch.randn(hidden_size, in_features, device="cuda")
    # print(inp[0])

    # Enable autocasting for the forward pass
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model(inp, is_first_microbatch=None)

    loss = out.sum()
    loss.backward()

    # print the scaling information
    print("scale info: ", model.fp8_meta['scaling_fwd'].scale)