# Pre-Install
# 
# PyTorch must be installed before pre-compiling any DeepSpeed c++/cuda ops.
# 


CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"

git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log

# 使用whl进行安装，此处不推荐
#pip install dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl
