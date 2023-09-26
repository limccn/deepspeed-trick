## 02 Optimization

### 02-00 Accelerate Tricks

1. Apple Silicon GPUs 配合`accelerate`命令，可以使用 `--cpu` 方式进行加速优化

```bash
accelerate launch xxxx.py --cpu +其他参数
```

使用`PyTorch >= 1.12`,同时需要安装MPS

2. accelerate与DeepSpeed集成，可以使用`bootstrap`方式将两则结合，实现config共享

```bash
#生成配置
accelerate config

#生成配置
accelerate launch my_script.py --args_to_my_script  +其他参数
    # 使用fp16
    --mixed_precision fp16

```

测试两者集合的代码
```python
from accelerate import Accelerator
from accelerate.state import AcceleratorState


def main():
    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")

```

或者简单`deepspeed_plugin`方式，使用版本需要满足 `DeepSpeed >=0.6.5`


### 参考代码
[src/02 Optimization/00 Accelerate](https://github.com/limccn/deepspeed-trick/tree/main/src/02%20Optimization/00%20Accelerate)

### 参考：
1.GPU-Acceleration Comes to PyTorch on M1 Macs. URL:[GPU-Acceleration Comes to PyTorch on M1 Macs](https://towardsdatascience.com/gpu-acceleration-comes-to-pytorch-on-m1-macs-195c399efcc1)
