
## 01 Training

### 01-04 LLAMA Tricks

1. LLAMA/LLAMA2 训练需要输入高质量的RLHF数据集，实践中发现数据集质量问题会让Finetuning出来的LLAMA“变傻”，而且训练率越高越严重。下面是实际运用的几个质量不错的数据集

RLHF数据集
```
Dahoas/rm-static 
Dahoas/full-hh-rlhf 
Dahoas/synthetic-instruct-gptj-pairwise 
yitingxie/rlhf-reward-datasets
```

openai和stanfordnlp的数据集
```
openai/webgpt_comparisons
stanfordnlp/SHP
```

中文数据集,参考：`bigscience/bloom-1b1`
```
wangrui6/Zhihu-KOL
Cohere/miracl-zh-queries-22-12
Hello-SimpleAI/HC3-Chinese
mkqa-Chinese
```

日文数据集,参考：`sberbank-ai/mGPT`
```
mkqa-Japanese 
Cohere/miracl-ja-queries-22-12 
lmqg/qg_jaquad 
lmqg/qag_jaquad
```

2.按微软的DeepSpeed-Chat训练方法，将LLAMA的Finetuning分为三步完成，Step 1需要用到更精准的数据集，Step 2和Step 3更多是进行reward score进行Finetuning。
各阶段的注意点如下：

Step 1: Supervised Finetuning
```
   --weight_decay 0.        # 关闭weight_decay
   --num_train_epochs 4     # 使用epochs不要超过16
```

Step 2: Reward Model Finetuning
``` 
   # 微软的做法
   --weight_decay 0.1       # 打开weight_decay
   --num_train_epochs 1     # 因为关闭了dropout，所以跑一遍epochs即可。

   # 实际上如果不走step3的话，可以这样
   --weight_decay 0         # 打开weight_decay好像影响有限，
   --num_train_epochs 2     # 如果不关闭dropout，可以把epochs 2
```
Step 3: RLHF Finetuning
```
   # LLAMA走完Step2已经取得不错的效果了，做Step3按需做吧，Step3需要大量显存支持，请打开offload
   --offload_reference_model 
   # 微软的做法
   --actor_learning_rate 9e-6             # Actor的LR
   --critic_learning_rate 5e-6            # Critic的LR
   --actor_weight_decay 0.1               # 打开weight_decay
   --critic_weight_decay 0.1              # 打开weight_decay
   --num_train_epochs 1                   # 因为不需要dropout，只需要跑一轮epochs
   --actor_dropout 0.0                    # 关闭dropout

```


### 参考代码
[src/01 Training/04 LLAMA](https://github.com/limccn/deepspeed-trick/tree/main/src/01%20Training/04%20LLAMA)

### 参考
[Training Experiences](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/README.md)