# Bootstrap将Megatron与DeepSpeed结合
# bootstrap.py

import os
from deepspeed.ops.adam import DeepSpeedCPUAdam, DeepSpeedFusedAdam
from deepspeed.runtime.config import DeepSpeedConfig
from megatron import initialize_megatron

def bootstrap_deepspeed(megatron_config_file, deepspeed_config_file):


    # 加载DeepSpeed配置
    deepspeed_config = DeepSpeedConfig(deepspeed_config_file)
    
    # 共享的环境变量
    os.environ["XXXXX"] = deepspeed_config.xxxxxx
    os.environ["XXXXX"] = str(deepspeed_config.xxxxx)

    # 共享的变量
    megatron_config = load_json(megatron_config_file)  #任意方式加载

    # 注意保持一致 "lr", “batch_size” 可以共享，其他配置请注意
    megatron_config.xxx = str(deepspeed_config.xxx)    #需要转换格式的地方，自行转换
    megatron_config.xxxbbb = deepspeed_config.xxxbbb   #需要转换格式的地方，自行转换
    
    # 初始化Megatron
    initialize_megatron(megatron_config)
    
    # 启动DeepSpeed训练
    with deepspeed.initialize(deepseed_config):
        # 执行Megatron训练逻辑
        run_megatron_training()

def run_megatron_training():
    # 在这里编写您的Megatron训练逻辑
    # 训练GPT这里直接导入，pretrain_gpt.py即可
    pass

if __name__ == "__main__":
    megatron_config_file = "megatron.json"
    deepspeed_config_file = "ds_config.json"
    
    bootstrap_deepspeed(megatron_config_file, deepspeed_config_file)
