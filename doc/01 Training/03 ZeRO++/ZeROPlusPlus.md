
## 01 Training

### 01-03 ZeroPlusPlus Tricks

1. ZeRO++在Deepspeed训练时，适用在`Stage3`阶段，按官方说法，`the three optimization reduces communication volume by 4x compared to ZeRO baseline`, 减少communication volume，使用ZeRO++是不错的途径。

通过配置三个属性可以使用，按官方文档，这三个属性可以独立发挥作用。 `zero_quantized_weights`和`zero_quantized_gradients`使用开关打开即可，
`zero_hpz_partition_size`通常按节点的GPU数量来配置

```
zero_quantized_weights: Boolean indicating whether to use quantized zero weights (qwZ), default is false

zero_hpz_partition_size: number of ranks in hpZ (secondary partition) group, default is 1 meaning no hpZ, ideal is number of ranks (gpus) per node

zero_quantized_gradients: Boolean indicating whether to use quantized zero gradients (qgZ), default is false
```

这里使用的关键配置

```json
{
    ...
    "zero_quantized_weights": true,    //default = false
    "zero_hpz_partition_size": 4,       // number of gpus
    "zero_quantized_gradients": true,    //default = false
    ...
}
```

实际训练中，`zero_hpz_partition_size`,发现如果只有`4`个以下GPU，提升效果并不能达到4x，建议GPU数量达到`8`个以上再尝试打开

2. 按deepspeed官方的参考, 测试使用Zero++可以直接参考[Microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)下的`pretrain_zeropp_gpt.py `

跟Megatron相关trick可以参考“：[Megatron.md](https://github.com/limccn/deepspeed-trick/blob/main/doc/01%20Training/06%20Megatron/Megatron.md)


### 参考代码
[src/01 Training/03 ZeRO++](https://github.com/limccn/deepspeed-trick/tree/main/src/01%20Training/03%20ZeRO%2B%2B)

## 参考
1. [Microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)
