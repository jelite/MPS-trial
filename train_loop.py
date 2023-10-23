import types

import torch.nn as nn
import torch
import torch.optim as optim
# import torchvision.models as models
# from transformers import AutoTokenizer
# from transformers import BertModel

from Utilization.util import train_warmup, get_data_by_name, get_model_by_name
import argparse 

# def get_forward_pre_hook(label):
#     def forward_pre_hook(m, input):
#         torch.cuda.nvtx.range_push(f'f_{label}')
#     return forward_pre_hook

# def get_forward_post_hook(label):
#     def forward_post_hook(m, input, output):
#         torch.cuda.nvtx.range_pop()
#     return forward_post_hook

# def get_backward_pre_hook(label):
#     def backward_pre_hook(m, input):
#         torch.cuda.nvtx.range_push(f'b_{label}')
#     return backward_pre_hook

# def get_backward_post_hook(label):
#     def backward_post_hook(m, input, output):
#         torch.cuda.nvtx.range_pop()
#     return backward_post_hook

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--batch_size', type=int)
args = parser.parse_args()

model_name = args.model_name.lower()
batch_size = args.batch_size

stream = torch.cuda.Stream()

model = get_model_by_name(model_name)
inputs = get_data_by_name("imagenet", batch_size)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), 0.1, capturable=True)

print("============================")
print(f"train")
print(f"model: {model_name}")
print(f"batch_size: {batch_size}")
print("============================\n\n")

for iter in range(100000):
    print(f"{iter}iteration start!")
    train_warmup(stream, model, inputs, criterion, optimizer, iter_num=1)
