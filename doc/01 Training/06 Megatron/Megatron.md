## 01 Training

### 01-06 Megatron Tricks

1. [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 提供了一套不错的训练框架，但是如果需要在DeepSpeed中使用，请使用[Microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) fork出来的代码。 微软调整了一部分Sample代码
[examples_deepspeed](https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples_deepspeed),支持Azure和BERT训练。

2. 微软的examples_deepspeed里提供了Megatron-LM没有的llama训练sample，这部分内容对训练自己的LLAMA会有很重要的参考意义

LLAMA的训练Shell,微软使用`pretrain_gpt.py`开始,变动的内容比较多
```bash
# LLAMA
pretrain_llama_distributed.sh

# LLAMA2
pretrain_llama2_distributed.sh
```

同时微软也提供了一套bootstrap用于启动deepspeed和Megatron，可供参考

3. HabanaAI提供了一套比较完备的bootstrap shell，可以参考
[https://github.com/HabanaAI/Model-References/blob/master/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed/scripts/run_llama13b.sh](https://github.com/HabanaAI/Model-References/blob/master/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed/scripts/run_llama13b.sh)


### 参考代码
[src/01 Training/06 Megatron](https://github.com/limccn/deepspeed-trick/tree/main/src/01%20Training/06%20Megatron)

## 参考
1. [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
2. [Microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)
3. [HabanaAI/Megatron-DeepSpeed](https://github.com/HabanaAI/Model-References/)