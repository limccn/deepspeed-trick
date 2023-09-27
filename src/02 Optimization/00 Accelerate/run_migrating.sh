accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \    ##非常重要，用fp16
    --num_processes=2 \
    --config_file config_acc.yaml \        
    migrating.py \
    --arg1 \
    --arg2 \
    ...
