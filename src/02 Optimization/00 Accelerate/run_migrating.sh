accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \    ##非常重要，用fp16
    --num_processes=2 \         
    migrating.py \
    --arg1 \
    --arg2 \
    ...
