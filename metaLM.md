# Multi-task finetuning

### Datasets
GLUE like datasets, we use the same dataset, excluding the STS-B (regression dataset): 
- train dataset: cola, sst2, qqp, mnli, qnli, rte, wnli, snli, 
            trec, mpqa, cr, sst5, mr, subj, mrpc. The datasets contains binary classes or multi-classes.
            
### Algorithms:
1. Sub-sample $K=16$ samples per class from each datasets. Form subdatasets.
2. For epoch = 1, ..., $n\\_epoch = 10$
    1. For batch = 1, ..., $n\\_batch = 100$:
        - Sample 10 subdataset, sample 1 example from each subdatasets.
        - Forward mini-batch (10 samples) into Masked LM.
        - Prompt-based finetuning: Compute loss on the masked token, backward gradient. 
        - Compute accuracy and save.

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
|test dataset |Prompt-based zero-shot|Prompt-based FT      |Multi-task FT + PB-FT|
|-------------|-----|---------------------|--------|
|SST-2        |83.60(83.6)| 93.23 (92.7 +- 0.9) | 93.81 |
|RTE          |51.26(51.3)| 71.84 (69.1 +- 3.6) | 74.01 |
|MRPC         |61.88(61.9)| 78.53 (74.5 +- 5.3) | - |
|QNLI         |50.81(50.8)| 69.85 (64.5 +- 4.2) | 69.92 |
