accumulation_steps = 3  # 累积3个小批次的梯度，bp16累计数量越大，精度丢失越多。

for i, (inputs, labels) in enumerate(data_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 梯度累积
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step