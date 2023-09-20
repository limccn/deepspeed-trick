from torch.nn import DataParallel

# 模型初始化
model = MyModel()
model = DataParallel(model)

# 训练循环
for input, target in data_loader:
    output = model(input)
    loss = criterion(output, target)

    model.zero_grad()
    loss.backward()
    optimizer.step()
