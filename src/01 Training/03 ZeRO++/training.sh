# Install
pip install torch transformers deepspeed


deepspeed pretrain_zeropp_gpt.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 40 \
       --hidden-size 6144 \
       --seq-length 512 \
       --num-attention-heads 32 \
       --batch-size 1 \
       --zero-stage 3 \
       --deepspeed_config ds_zeropp_config.json \
       --deepspeed-activation-checkpointing \
       --fp16 \
       --checkpoint-activations
