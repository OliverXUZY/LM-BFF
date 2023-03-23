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

The evaluation is predicting the masked token. We do not learn linear head here. There are in total $n\\_epoch \times n\\_batch$(in this case 1000) optimization steps.
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
- [paper's result](https://arxiv.org/pdf/2012.15723.pdf) in parenthesis
- number with \*: Multi-task FT for 10000 steps, others 1000 steps
- __hightlight__ results where Multi-task FT does not help

|test dataset |Prompt-based zero-shot|Multi-task FT + zero-shot|Prompt-based FT|Multi-task FT + PB-FT|
|-------------|----------------------|-------------------------|---------------|--------|
|SST-2 (Acc)  |83.60(83.6) |92.89    | 92.6 +- 1.2 (92.7 +- 0.9) | 92.8 +- 0.7 |
|RTE (Acc)    |51.26(51.3) |74.0072  | 67.8 +- 2.7 (69.1 +- 3.6) |  73.0 +- 2.0 |
|MRPC (F1)    |61.88(61.9) |81.5873  | 71.4 +- 7.9 (74.5 +- 5.3) | 77.1 +- 3.7 |
|QNLI (Acc)   |50.81(50.8) |65.84    | 69.4 +- 3.6 (64.5 +- 4.2) | 71.9 +- 1.6 |
|QQP (F1)     |32.0(32.0)  |63.43    |66.2 +- 4.9 (65.5 +- 5.3)  |  68.7 +- 3.4 |
|TREC (Acc)   |47.72(49.7) |__34.6__   | 86.0 +- 3.0 (84.8 +- 5.1) | __83.0 +- 4.1__ |
|CR (Acc)     |79.5(79.5)  |88.8     | 91.4 +- 0.9 (90.3 +- 1.0) |__90.5 +- 1.0__ \*|
|MR (Acc)     |80.8(80.8   |86.5     | 86.9 +- 1.7 (87.0 +- 1.2) | 87.8 +-1.4 \*|
|MPQA (Acc)   |67.6(61.6)  |73.9     |84.8 +- 1.6 (84.7 +- 2.2)  |__83.0 +- 1.9__ \*|
|Subj (Acc)   |51.45(51.4) |55.3     |88.8 +- 2.5 (91.2 +- 1.1)  |__88.5 +-2.8__ |
|MNLI (Acc)   |50.84(50.8) |63.2     |68.2 +-2.7 (68.3 +- 2.3)   |71.4 +- 1.3 |
|MNLI-mm (Acc)|51.72(51.7) |65.7     |70.2 +-2.6 (70.5 +- 1.9)   |74.0 +- 1.5 |
|SNLI (Acc)   |49.51(49.5) |61.8     |75.1 +-4.2 (77.2 +- 3.7)   |78.0 +- 3.8 |
|CoLA (Matt.) |2.03(2.0)   |__-0.065__ |8.6 +-5.8 (9.3 +- 7.3)   |__5.5 +- 3.8__ |
|SST-5 (Acc)  |35.0(35.0)  |36.8     |48.1 +-1.6 (47.4 +- 2.5)   |49.5 +- 0.9 |


### Examples
Please install the requirements and download data following `README.md`. This result used transformers version `4.26.1`(or stable version). The reason we use new version instead of `3.4.0` in paper is old version requires python 3.4 and invoke many decrecated functions and classes, this will produce many warnings. It would easier to debug also.

The current running performs prompt-based finetuning without demonstration.

#### Zero-shot evaluation
* zero-shot evaluation on
   - Original backbone (replicates paper's result): ```MODEL=roberta-large```
   - Multi-tasks trained backbone: ```MODEL=result/meta-rte-roberta-large```
```
CUDA_VISIBLE_DEVICES=0 TAG=exp TYPE=prompt TASK=RTE BS=2 LR=1e-5 SEED=42  MODEL=result/meta-rte-roberta-large bash run_experiment.sh "--no_train"
```

* Zero-shot evaluation for loop. choose `MODEL=roberta-large` or multitask-finetuned saved dir like below.
```
for test_task in MNLI SNLI CoLA
do  
    CUDA_VISIBLE_DEVICES=1 \
    TAG=zero_shot \
    TYPE=prompt \
    TASK=$test_task \
    BS=2 \
    LR=1e-5 \
    SEED=42 \
    MODEL="result/meta-${test_task,,}-roberta-large" \
    bash run_experiment.sh \
    "--no_train"
done
```
#### Multi-tasks finetune
* Multi-tasks finetune backbone ```roberta-large```, select one task ```rte``` as test dataset, other tasks as training data. Save model in ```result/meta-rte-roberta-large ```. 
```
TEST_TASK=rte
MODEL=roberta-large
SEED=42
NUM_BATCH=1000

mkdir result/meta-$TEST_TASK-$MODEL

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
    2>&1 | tee "result/meta-$TEST_TASK-$MODEL/${TEST_TASK}_stderr_stdout.txt"
```
* Save above to `train_meta.sh`, we can for loop tasks:
```
for test_task in cr mr mpqa subj mnli snli cola sst-5
do 
    echo $test_task
    CUDA_VISIBLE_DEVICES=1 \
    TEST_TASK=$test_task \
    bash train_meta.sh
done
```

#### Prompt-finetune + evaluation
* Prompt-finetune + evaluation single seed
  - Original backbone (replicates paper's result): ```MODEL=roberta-large```
  - Multi-tasks trained backbone: ```MODEL=result/meta-rte-roberta-large```
```
CUDA_VISIBLE_DEVICES=0 TAG=exp TYPE=prompt TASK=RTE BS=2 LR=1e-5 SEED=42  MODEL=result/meta-rte-roberta-large bash run_experiment.sh 
```
* Across 5 seeds:
```
test_task=CoLA
for seed in 13 21 42 87 100
do  
    CUDA_VISIBLE_DEVICES=0 \
    TAG="meta-${test_task,,}" \
    TYPE=prompt \
    TASK=$test_task \
    BS=2 \
    LR=1e-5 \
    SEED=$seed \
    MODEL="roberta-large" \
    bash run_experiment.sh
done
```
* Across 5 seeds for loop tasks:
```
for test_task in mpqa subj MNLI SNLI CoLA sst-5
do
    for seed in 13 21 42 87 100
    do  
        CUDA_VISIBLE_DEVICES=0 \
        TAG="base-${test_task,,}" \
        TYPE=prompt \
        TASK=$test_task \
        BS=2 \
        LR=1e-5 \
        SEED=$seed \
        MODEL="result/meta-${test_task,,}-roberta-large" \
        bash run_experiment.sh
    done
done
```
