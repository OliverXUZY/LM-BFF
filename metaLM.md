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
|SST-2        |83.60(83.6)|92.89| 93.23 (92.7 +- 0.9) | 93.81 |
|RTE          |51.26(51.3)|74.0072| 71.84 (69.1 +- 3.6) | 74.01 |
|MRPC         |61.88(61.9)|81.5873| 78.53 (74.5 +- 5.3) | 81.5972 |
|QNLI         |50.81(50.8)|65.84| 69.85 (64.5 +- 4.2) | 69.92 |


### Examples
* Multi-tasks finetune backbone
```
TEST_TASK=mrpc
MODEL=roberta-large

NUM_BATCH=1000

CUDA_VISIBLE_DEVICES=1  \
python metaTrain.py \
    --model_name_or_path $MODEL \
    --task_name $TEST_TASK \
    --few_shot_type prompt \
    --num_k 16 \
    --num_train_epochs 10.0 \
    --num_batch $NUM_BATCH \
    --logging_steps 50\
    --per_device_train_batch_size 1 \
    --max_seq_length 128 \
    --gradient_accumulation_steps 10 \
    --learning_rate 1e-5 \
    --output_dir result/meta-$TEST_TASK-$MODEL \
    --seed 42 \
    --overwrite_output_dir \
    --do_train \
    --save_at_last \
```

* zero-shot evaluation
```
CUDA_VISIBLE_DEVICES=0 TAG=exp TYPE=prompt TASK=RTE BS=2 LR=1e-5 SEED=42  MODEL=result/meta-rte-roberta-large bash run_experiment.sh "--no_train"
```

* Prompt-finetune + evaluation
```
CUDA_VISIBLE_DEVICES=0 TAG=exp TYPE=prompt TASK=RTE BS=2 LR=1e-5 SEED=42  MODEL=result/meta-rte-roberta-large bash run_experiment.sh 
```
