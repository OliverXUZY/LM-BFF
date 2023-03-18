# Multi-task finetuning

### Datasets
GLUE like datasets, we use the same dataset, excluding the STS-B (regression dataset): 
- train dataset: cola, sst2, qqp, mnli, qnli, rte, snli, 
            trec, mpqa, cr, sst5, mr, subj, mrpc. The datasets contains binary classes or multi-classes.

For testing on certain dataset (say qnli), we use all other 13 datasets as training data for multi-tasks finetuning.

### Algorithms:
1. Sub-sample $K=16$ samples per class from each datasets. Form subdatasets.
2. For epoch = 1, ..., $n\\_epoch = 10$
    1. For batch = 1, ..., $n\\_batch = 100$:
        - Sample 10 subdataset, sample 1 example from each subdatasets.
        - Forward mini-batch (10 samples) into Masked LM.
        - Prompt-based finetuning: Compute loss on the masked token, backward gradient. 

The evaluation is predicting the masked token. We do not learn linear head here. There are in total $n\\_epoch \times n\\_batch$ optimization steps.
Optimization details:

__epoch 1 to 10:__
```
optimizer: adamW
Initial learning rate: 1e-05
lr decay: linear decay to 0 during 1000 steps
```
The accuracy for each epoch is averaged cross all batches.

### Result
<!-- 100,10 for testing -->
BackBone: **RoBERTa-large**
([paper's result](https://arxiv.org/pdf/2012.15723.pdf) in parenthesis)
|test dataset |Prompt-based zero-shot|Multi-task FT + zero-shot|Prompt-based FT|Multi-task FT + PB-FT|
|-------------|----------------------|-------------------------|---------------|--------|
|SST-2        |83.60(83.6)|92.89  | 92.6 +- 1.2 (92.7 +- 0.9) | 92.8 +- 0.7 |
|RTE          |51.26(51.3)|74.0072| 67.8 +- 2.7 (69.1 +- 3.6) |  73.0 +- 2.0 |
|MRPC         |61.88(61.9)|81.5873| 71.4 +- 7.9 (74.5 +- 5.3) | 77.1 +- 3.7 |
|QNLI         |50.81(50.8)|65.84  | 69.4 +- 3.6 (64.5 +- 4.2) | 71.9 +- 1.6 |
|QQP(F1)      |32.0(32.0) |63.43  |66.2 +- 4.9 (65.5 +- 5.3)|  |
|TREC (Acc)   |47.72(49.7)|_34.6_ | 86.0 +- 3.0 (84.8 +- 5.1) | |


### Examples
Please install the requirements and download data following `README.md`. This result used transformers version `4.26.1`(or stable version). The reason we use new version instead of `3.4.0` in paper is old version requires python 3.4 and invoke many decrecated functions and classes, this will produce many warnings. It would easier to debug also.

The current running performs prompt-based finetuning without demonstration.
* Multi-tasks finetune backbone ```roberta-large```, select one task ```rte``` as test dataset, other tasks as training data. Save model in ```result/meta-rte-roberta-large ```.
```
TEST_TASK=rte
MODEL=roberta-large
SEED=42
NUM_BATCH=1000

CUDA_VISIBLE_DEVICES=1  \
python metaTrain.py \
    --model_name_or_path $MODEL \
    --task_name $TEST_TASK \
    --few_shot_type prompt \
    --num_k 16 \
    --num_train_epochs 10.0 \
    --num_batch $NUM_BATCH \
    --logging_steps 50 \
    --per_device_train_batch_size 1 \
    --max_seq_length 128 \
    --gradient_accumulation_steps 10 \
    --learning_rate 1e-5 \
    --output_dir result/meta-$TEST_TASK-$MODEL \
    --seed $SEED \
    --overwrite_output_dir \
    --do_train \
    --save_at_last \
```

* zero-shot evaluation on
   - Original backbone (replicates paper's result): ```MODEL=roberta-large```
   - Multi-tasks trained backbone: ```MODEL=result/meta-rte-roberta-large```
```
CUDA_VISIBLE_DEVICES=0 TAG=exp TYPE=prompt TASK=RTE BS=2 LR=1e-5 SEED=42  MODEL=result/meta-rte-roberta-large bash run_experiment.sh "--no_train"
```

* Prompt-finetune + evaluation
  - Original backbone (replicates paper's result): ```MODEL=roberta-large```
  - Multi-tasks trained backbone: ```MODEL=result/meta-rte-roberta-large```
```
CUDA_VISIBLE_DEVICES=0 TAG=exp TYPE=prompt TASK=RTE BS=2 LR=1e-5 SEED=42  MODEL=result/meta-rte-roberta-large bash run_experiment.sh 
```
