import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def train(rank, world_size):
    # 初始化分布式训练环境
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 模型和数据加载器的初始化
    model = MyModel()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # 训练循环
    for input, target in distributed_data_loader:
        output = model(input)
        loss = criterion(output, target)

        model.zero_grad()
        loss.backward()
        optimizer.step()

# 启动分布式训练
world_size = 4  # 设置总的GPU数量
mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
