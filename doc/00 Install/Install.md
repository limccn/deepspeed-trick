
## Get Start
如果只是本地测试，使用以下命令即可快速完成deepspeed的安装

安装deepspeed，这里使用pip安装
```bash
pip install deepspeed
``` 

安装transformers

```bash
pip install transformers[deepspeed]
```

实际使用过程中，需要根据设备环境进行参数式编译安装

使用DS_BUILD_OPS=1会自动选择适合本机的配置。

```bash
DS_BUILD_OPS=1 pip install deepspeed
```

如果需要本地测试开发，可以打开CPU做ADAM，另外安装一些utils
```bash
DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install
```

Available DS_BUILD options include:
```bash
DS_BUILD_OPS toggles all ops
DS_BUILD_AIO builds asynchronous (NVMe) I/O op
DS_BUILD_CCL_COMM builds the communication collective libs
DS_BUILD_CPU_ADAM builds the CPUAdam op
DS_BUILD_FUSED_ADAM builds the FusedAdam op (from apex)
DS_BUILD_CPU_ADAGRAD builds the CPUAdagrad op
DS_BUILD_FUSED_LAMB builds the FusedLamb op
DS_BUILD_QUANTIZER builds the quantizer op
DS_BUILD_RANDOM_LTD builds the random ltd op
DS_BUILD_SPARSE_ATTN builds the sparse attention op
DS_BUILD_TRANSFORMER builds the transformer op
DS_BUILD_TRANSFORMER_INFERENCE builds the transformer-inference op
DS_BUILD_STOCHASTIC_TRANSFORMER builds the stochastic transformer op
DS_BUILD_UTILS builds various optimized utilities
```


## 我这里使用下面的命令进行安装
```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

TORCH_CUDA_ARCH_LIST="8.6"的值，可以用这条命令获取
```
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

##  使用离线whl进行安装，用于集群部署
```bash
pip install dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl
```

## TRICKS

1.我的测试机有两张不同的型号的显卡，如何安装？注意物理接口的顺序和CUDA的最低版本。
安装选项里的TORCH_CUDA_ARCH_LIST可以改成这样的形式，按显卡实际获取值设置。
```bash
TORCH_CUDA_ARCH_LIST="6.1;8.6"
```

2.为什么要使用DS_BUILD_CPU_ADAM参数？
主要deepspeed会利用CPU和RAM进行缓存，显存不够，内存会分担一部分。CPU可以完成一部分GPU的工作，比如40GB的大模型，在单机10G显存下也可以完成训练。

## References
[https://www.deepspeed.ai/tutorials/advanced-install/](https://www.deepspeed.ai/tutorials/advanced-install/)。
