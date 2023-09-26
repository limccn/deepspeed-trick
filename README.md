# deepspeed-trick

## 开始
Just record my journey to advance and democratize artificial intelligence through MSOS ZeRO and DeepSpeed
只在记录使用微软开源分布式AI训练框架ZeRO and DeepSpeed过程中的问题和解决方法。

## 目录

|  ID | ID-2 | 名称 | 说明 | 文档 | 源代码  | 
| :----: | :----: | :---- | :---- | :---- | :----: |
| 00-Install | -   |  Install  |  deepspeed相关安装 |  [Install.md](https://github.com/limccn/deepspeed-trick/blob/main/doc/00%20Install/Install.md)  | [src](https://github.com/limccn/deepspeed-trick/tree/main/src/00%20Install) |
| 00-Training | -   |  -  |  模型训练相关 |  [Readme.md](https://github.com/limccn/deepspeed-trick/blob/main/doc/01%20Training/README.md)  | - |
| 01-Training | 00  |  Startup |  开始训练 | [Startup.md](https://github.com/limccn/deepspeed-trick/blob/main/doc/01%20Training/00%20Startup/Startup.md) | [src](https://github.com/limccn/deepspeed-trick/tree/main/src/01%20Training/00%20Startup)|
| 01-Training | 01  |  Transformer |  Transformer基础模型  | [Transformer.md](https://github.com/limccn/deepspeed-trick/blob/main/doc/01%20Training/01%20DeepSpeed%2BTransformer/DeepSpeed+Transformer.md) |  [src](https://github.com/limccn/deepspeed-trick/tree/main/src/01%20Training/01%20DeepSpeed%2BTransformer)|
| 01-Training | 02  |  GPT-2 |  GPT-2 基础模型  | [GPT2.md](https://github.com/limccn/deepspeed-trick/blob/main/doc/01%20Training/02%20GPT-2/GPT-2.md) |  [src](https://github.com/limccn/deepspeed-trick/tree/main/src/01%20Training/02%20GPT-2) |
| 01-Training | 03  |  ZeRO++ |  ZeRO++  | [ZeROPlusPlus.md](https://github.com/limccn/deepspeed-trick/blob/main/doc/01%20Training/03%20ZeRO%2B%2B/ZeROPlusPlus.md) |  [src](https://github.com/limccn/deepspeed-trick/tree/main/src/01%20Training/03%20ZeRO%2B%2B)|
| 01-Training | 04  |  LLAMA |  LLAMA 模型  | [LLAMA.md](https://github.com/limccn/deepspeed-trick/blob/main/doc/01%20Training/04%20LLAMA/LLAMA.md) | [src](https://github.com/limccn/deepspeed-trick/tree/main/src/01%20Training/04%20LLAMA)|
| 01-Training | 05  |  DeepSpeed-Chat |  DeepSpeed-Chat  | [DeepSpeed-Chat.md](https://github.com/limccn/deepspeed-trick/blob/main/doc/01%20Training/05%20DeepSpeed-Chat/DeepSpeed-Chat.md) |  [src](https://github.com/limccn/deepspeed-trick/tree/main/src/01%20Training/05%20DeepSpeed-Chat)|
| 01-Training | 09  |  NCCL |  NCCL相关  | [NCCL.md](https://github.com/limccn/deepspeed-trick/blob/main/doc/01%20Training/09%20NCCL/NCCL.md) |  [src](https://github.com/limccn/deepspeed-trick/tree/main/src/01%20Training/09%20NCCL)|
| 02-Optimization | 00  |  -  |  优化   | [Readme.md](#) |  -|
| 02-Optimization | 01  |  LLM Accelerating |  LLM 加速基础   | [LLM-Accelerating.md](#) |  [src](#)|
| 02-optimizations | 02  |  Inference |  推理优化   | [Inference.md](#) |  [src](#)|



## 参考
1. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. 
URL:[https://arxiv.org/pdf/1910.02054.pdf](https://arxiv.org/pdf/1910.02054.pdf)

2. ZeRO-Offload: Democratizing Billion-Scale Model Training.
URL:[https://arxiv.org/pdf/2101.06840.pdf](https://arxiv.org/pdf/2101.06840.pdf)

3. DeepSpeed: A deep learning optimization library.
URL: [https://github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)

4. 微软DeepSpeed组官方账号 URL: [https://www.zhihu.com/people/deepspeed](https://www.zhihu.com/people/deepspeed)

5. DeepSpeed Examples URL:[https://github.com/microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)

## 致谢





