from torch.cuda.amp import autocast, GradScaler

# 在训练循环中使用混合精度
scaler = GradScaler()
for input, target in data_loader:
    optimizer.zero_grad()

    with autocast():
        output = model(input)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
