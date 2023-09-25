# Install
pip install torch transformers deepspeed

# 调整参数
deepspeed deepspeed-gpt2.py \
--deepspeed ds_config.json \
--model_name_or_path gpt2 \
--per_device_train_batch_size 1 \ 
--output_dir output_dir \
--overwrite_output_dir \
--fp16 \
--do_train \
--max_train_samples 100 \
--num_train_epochs 1 
