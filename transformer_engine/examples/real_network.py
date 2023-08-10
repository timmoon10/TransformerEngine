import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
import time
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import numpy as np
# from transformer_engine.pytorch import Float8Tensor, E4M3, tensor_to_scale
from transformer_engine.pytorch.fp8 import get_global_fp8_buffer

torch.manual_seed(0)
np.random.seed(0)

tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split="train")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

class TETextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TETextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = te.Linear(embed_dim, num_class, bias=False, params_dtype=torch.float32, primary_weights_in_fp8=True)

    def forward(self, text, offsets, is_first_microbatch=None):
        embedded = self.embedding(text, offsets)
        out = self.fc(embedded, is_first_microbatch=is_first_microbatch)
        return out

# num_class = len(set([label for (label, text) in train_iter]))
num_class = 16 # forcing the class to 16 to make it compatible with fp8 training reqs
vocab_size = len(vocab)
emsize = 16
model = TETextClassificationModel(vocab_size, emsize, num_class).to(device)
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3, reduce_amax=False)

# Hyperparameters
EPOCHS = 1  # epoch
LR = 5  # learning rate
BATCH_SIZE = 16  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)


## Dataset/Dataloader prep
train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)
train_dataloader = DataLoader(
    split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

def custom_cross_entropy_loss(pred, true):
    exp = torch.exp(pred)
    log_softmax = torch.log(exp/torch.sum(exp, dim=-1, keepdims=True))
    m, n = pred.shape
    offset = torch.arange(0, m*n, step = n).reshape(m,1).cuda()
    idx = true + offset
    return -torch.mean(torch.take(log_softmax, idx))

# def loss_func(pred, true):
#     pred = torch.log(pred)
#     return torch.sum(torch.take(pred, true))

# Training loop
for idx, (label, text, offsets) in enumerate(train_dataloader):
    if idx > 10:
        break
    optimizer.zero_grad()
    print(f"--------------iter{idx}--------------")
    print("model fwd")
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        predicted_label = model(text, offsets, is_first_microbatch=None)
    # loss = criterion(predicted_label, label) # this doesn't update weight amax at all
    # loss = custom_cross_entropy_loss(predicted_label, label)
    # loss = predicted_label.sum() # This does update the weight amaxes
    # loss = torch.sum(predicted_label) # This does update the weight amaxes
    # loss = torch.sum(torch.take(torch.log(predicted_label), label)) # This updates amax history but looks like it doesn't keep track of all of history
    print("model bwd")
    loss.backward()
    print("optimizer step")
    optimizer.step()
    amax_hist = model.fc.fp8_meta['scaling_fwd'].amax_history #[1024,3]
    print("amax_history: ", torch.where(amax_hist > 0.0), amax_hist[amax_hist > 0.0])