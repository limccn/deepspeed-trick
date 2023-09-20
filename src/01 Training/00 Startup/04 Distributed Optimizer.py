from deepspeed.ops.adam import DeepSpeedCPUAdam

# 初始化分布式优化器
optimizer = DeepSpeedCPUAdam(params=model.parameters(), lr=learning_rate)

# 在训练循环中使用分布式优化器
for input, target in data_loader:
    output = model(input)
    loss = criterion(output, target)

    model.zero_grad()
    optimizer.backward(loss)
    optimizer.step()
