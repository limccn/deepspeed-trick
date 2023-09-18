# torch.distributed pytorch launcher first, 
# torch.distributed.run --nproc_per_node=2 your_program.py <normal cl args> --deepspeed ds_config.json

# use deepspeed to boost up 
# deepspeed --num_gpus=2 your_program.py <normal cl args> --deepspeed ds_config.json


# 先在单个GPU上跑
# deployment on single GPU
deepspeed --num_gpus=1 your_program \
--deepspeed configs/ds_config_zero2.json \
--model_name_or_path t5-small \
--per_device_train_batch_size 1 \
--output_dir output_dir \
--overwrite_output_dir --fp16 \
--do_train \
--max_train_samples 500 
--num_train_epochs 1 \
--dataset_name wmt16 \
--dataset_config "zh-en" \
--source_lang en \
--target_lang zh


# 调整参数
deepspeed your_program.py \
--deepspeed configs/ds_config_zero3.json \
--model_name_or_path t5-small \
--per_device_train_batch_size 1 \ 
--output_dir output_dir \
--overwrite_output_dir --fp16 \
--do_train \
--max_train_samples 500 \
--num_train_epochs 1 \
--dataset_name wmt16 \
--dataset_config "zh-en" \
--source_lang en \
--target_lang zh
