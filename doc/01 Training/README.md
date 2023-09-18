## 01 Training


### 01-01 DeepSpeed+Transformer
### Tricks

#### 1. 使用DeepSpeed自带的默认config文件，里面包括大量auto配置的内容，如果你的显存不够大，打开训练即刻OOM，所以在开始之前，先计算一下自己的可用显存大小和实际模型的情况调整优化参数。

首先,是将模型参数设置到尽可能小,
```bash
--per_device_train_batch_size 1
--max_train_samples 500 
--num_train_epochs 1 
```
然后，确认以下选项使用auto模式,首次运行成功时记录一下运行参数的结果，方便将来调整
```
"gradient_accumulation_steps": "auto",
"gradient_clipping": "auto",
"train_batch_size": "auto",
"train_micro_batch_size_per_gpu": "auto",
```


最后，根据Zero的优化参数配置，进行调整，还是由小及大原则
举例：scheduler的调整，默认时auto
```JSON
"scheduler": {
      "type": "WarmupLR",
      "params": {
          "warmup_min_lr": 0,      //auto
          "warmup_max_lr": 0.001,  //auto
          "warmup_num_steps": 1000 //auto
      }
  }
```
```JSON
"optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 11e-3,
            "betas": [0.8, 0.999],   //--adam_beta1 --adam_beta2
            "eps": 1e-8,
            "weight_decay": 0.01    //--adam_epsilon
        }
    }
```

#### 2. Zero的三个Stage，Zero-0通常不涉及内存和GPU的优化，直接从Zero-2，Zero-3阶段开始着手即可。


根据DeepSpeed文档显示，优化效率 Zero-2 > Zero-3, 所以能在Zero-2阶段优化，尽量在Zero-2进行


```
Speed-wise (left is faster than right)
Stage 0 (DDP) > Stage 1 > Stage 2 > Stage 2 + offload > Stage 3 > Stage 3 + offloads
```
```
GPU Memory usage-wise (right is more GPU memory efficient than left)
Stage 0 (DDP) < Stage 1 < Stage 2 < Stage 2 + offload < Stage 3 < Stage 3 + offloads
```

基于以上，对于训练过程中出现的OOM的问题，可以考虑以下路径方式，
原则还是`显存不够，内存来凑`模式，DeepSpeed可以实现高于1:4的显存效率，10G显存可以训练40G以上的模型。
如果每一步失败，则选择下一步：
* 1. 使用Zero-2 
* 2. 使用Zero-2 + offload_optimizer to cpu，先用auto，然后手动调整
* 3. 改为使用Zero-3
* 4. 使用Zero-3 + offload_param ，这里打开offload_param to cpu，
* 5. 使用Zero-3 + offload_param +  offload_optimizer  这里打开offload_param 和 offload_optimizer to cpu，
* 6. 降级浮点数类型，比如fp32降级为fp16，如果你使用A100系列的显卡，可以启用bf16。
* 7. 如果你的设备支持nvme磁盘缓冲，可以开启offload_optimizer to nvme。
