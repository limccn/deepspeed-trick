## 01 Training

### 01-02 DeepSpeed+GPT-2

### Tricks
1. GPT2在torch下训练几乎不需要花太多精力，你需要做的是提前找到合适的预训练模型，然后找到合适的数据集即可完成训练。

预训练模型
|  ID | 预训练模型 | 说明 | Hugging Face |
| :----: | :----: | :---- |  :---- |
|  01 | gpt2 |基础版本的GPT-2模型，具有小型参数规模。| [gpt2](https://huggingface.co/gpt2/resolve/main/pytorch_model.bin) | 
|  02 | gpt2-medium| 中等大小的GPT-2模型，具有更多的参数规模和更好的表现。| [gpt2-medium](https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin) |
|  03 | gpt2-large|大型的GPT-2模型，具有更多的参数规模和更高的生成能力。 | [gpt2-large](https://huggingface.co/gpt2-large/resolve/main/pytorch_model.bin)|

* PyTorch支持从Hugging Face HUB的transformers库中加载需要的模型


使用方法
```python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
```

2.在显存没有压力的情况下，训练GPT2，DeepSpeed只需要做ZeRO-Stage2的优化即可。

```python
ds_config = {
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 5e-5
        }
    },
    "zero_optimization": {
        "stage": 2,
    }
}

engine, model, _, _ = DeepSpeedEngine(model=model, tokenizer=tokenizer, config_params=ds_config)

```

### 参考代码
