import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Step 1: 导入所需的库和模块

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Step 5: 数据准备和编码

def encode_data(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    return input_ids

# 示例数据编码
encoded_data = encode_data("Hello, how are you?")

# Step 6: 定义训练循环

def train_loop(input_ids):
    model.train()
    optimizer.zero_grad()
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Step 7: 设置优化器和超参数

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Step 8: 进行训练迭代

num_epochs = 10
training_data = ["Example sentence 1", "Example sentence 2"]

for epoch in range(num_epochs):
    for data in training_data:
        encoded_data = encode_data(data)
        train_loop(encoded_data)
