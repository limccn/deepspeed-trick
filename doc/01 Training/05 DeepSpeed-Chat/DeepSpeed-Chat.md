## 01 Training

### 01-05 DeepSpeed-Chat Tricks

1.微软的DeepSpeed-Chat为训练LLAMA提供了一套基本的脚手架，针对 Llama-2-7B and Llama-2-13B提供了不错的优化，训练效率可以提升7倍，DeepSpeed-Chat包含两路优化

首先，使用混合精度+Zero++，即Mixed Precision ZeRO++ (MixZ++).将精度混合和Zero++一起使用，减少训练需要的显存。

其次，使用ZeRO-Offload将一部分GPU计算的内容转为CPU（主要在Stage-2和Stage-3阶段），可以减少GPU的使用需求，提高了资源的使用效率。
```bash
deepspeed main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \   # lr
   --weight_decay 0. \
   --num_train_epochs 4  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log

```


2.微软的DeepSpeed-Chat为针对LoRA的进行了适配，在使用LoRA时，请注意：

使用`lora.Linear`替代`nn.Linear`

```python
# ===== Before =====
# layer = nn.Linear(in_features, out_features)

# ===== After ======
import loralib as lora
# Add a pair of low-rank adaptation matrices with rank r=8
layer = lora.Linear(in_features, out_features, r=8)
```

对于优化拆分的layers，需要手动进行合并
```python
# ===== Before =====
# qkv_proj = nn.Linear(d_model, 3*d_model)
# ===== After =====
# Break it up (remember to modify the pretrained checkpoint accordingly)
q_proj = lora.Linear(d_model, d_model, r=8)
k_proj = nn.Linear(d_model, d_model)
v_proj = lora.Linear(d_model, d_model, r=8)
# Alternatively, use lora.MergedLinear (recommended)、
qkv_proj = lora.MergedLinear(d_model, 3*d_model, r=8, enable_lora=[True, False, True])  # 按需合并
```

LoRA需要配合更大的`learning rate`
```
  --learning_rate 9e-6           #
  --lora_learning_rate  5e-4     # lora lr 
``````


### 参考代码
[src/01 Training/05 DeepSpeed-Chat](https://github.com/limccn/deepspeed-trick/tree/main/src/01%20Training/05%20DeepSpeed-Chat)

## 参考
1. [microsoft/LoRA](https://github.com/microsoft/LoRA)
