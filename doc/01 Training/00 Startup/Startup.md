## Startup Trick

开始使用deepspeed之前，下面是一些使用进行模型训练的基础操作，参考代码可以到src目录

1.混合精度训练（Mixed Precision Training）：使用半精度浮点数（FP16）来减少显存使用和加速训练。

参考：[01 Mixed Precision Training.py](https://github.com/limccn/deepspeed-trick/blob/main/src/01%20Training/00%20Startup/01%20Mixed%20Precision%20Training.py)

2.分布式训练（Distributed Training）：使用多个GPU或多台机器上的多个GPU进行并行训练。

参考：[02 Distributed Training.py](https://github.com/limccn/deepspeed-trick/blob/main/src/01%20Training/00%20Startup/02%20Distributed%20Training.py)

3.数据并行处理（Data Parallelism）：将模型复制到多个GPU上，并使用每个GPU上的部分数据进行前向传播和反向传播。

参考：[03 Data Parallelism.py](https://github.com/limccn/deepspeed-trick/blob/main/src/01%20Training/00%20Startup/03%20Data%20Parallelism.py)

4.使用分布式优化器（Distributed Optimizer）：将优化器与分布式训练一起使用，以减少内存占用。

参考：[04 Distributed Optimizer.py](https://github.com/limccn/deepspeed-trick/blob/main/src/01%20Training/00%20Startup/04%20Distributed%20Optimizer.py)

5.HIDDEN

6.使用梯度累积（Gradient Accumulation）：将多个小批次的梯度累积起来进行一次参数更新，可以减少显
存的使用。这样可以使用较大的批次大小并减少显存需求，但会增加训练时间。

参考：[06 Gradient Accumulation.py](https://github.com/limccn/deepspeed-trick/blob/main/src/01%20Training/00%20Startup/04%20Distributed%20Optimizer.py)

7.使用梯度裁剪（Gradient Clipping）：训练过程中，如果梯度过大可能会导致数值不稳定和显存溢出。使用`torch.nn.utils.clip_grad_norm_()`函数可以对梯度进行裁剪，限制梯度的范数。
```python
#parameters: 网络参数
#max_norm: 该组网络参数梯度的范数上线
#norm_type: 范数类型
torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2)

```
Sample
```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(
    parameters=model.parameters(),   -- all
    max_norm=10,                     -- 10
    norm_type=2                      -- default 2
)
optimizer.step()
```

9.调整批次大小（Batch Size）：适当调整批次大小可以平衡内存消耗和训练速度。较小的批次大小会减少显存占用但可能导致训练速度下降。较大的批次大小可能需要更多的显存，但可以提高训练速度。 `--batch-size`

10.删除不必要的中间层和hidden层，删除大量临时变量和无用的中间结果，会提高显存的使用效率。