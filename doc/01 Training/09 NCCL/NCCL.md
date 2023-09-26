
## 01 Training

### 01-09 NCCL Tricks

1. 如果配置了torch distributed NCCL backend但是无效，需要修改`.deepspeed_env`文件，每个节点都需要进行相应的配置，添加以下内容：
```bash
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=eth0   # 使用以太网卡
```

2. 训练使用使用nccl，python安装相应包之前，请先将CUDA和NCCL路径放到`LD_LIBRARY_PATH`变量下
```bash
#设置cuda库的目录
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
#将nccl添加到LD_LIBRARY_PATH中
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
```

1. NVIDIA Deep Learning NCCL Documentation.
URL:[https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)