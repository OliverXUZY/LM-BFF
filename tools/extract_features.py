import dataclasses
import logging
import os
import sys
sys.path.append("/home/zhuoyan/LM-BFF")
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator

from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping
from src.dataset import input_example_to_tuple


datasets = ['CoLA',
 'trec',
 'sst-5',
 'SST-2',
 'mr',
 'SNLI',
 'cr',
 'QQP',
 'MRPC',
 'MNLI',
 'STS-B',
 'mpqa',
 'QNLI',
 'RTE',
 'subj']

seed_shot = ['16-13', '16-100', '16-21', '16-87', '16-42']

@dataclass
class DataArguments:
    """
    Arguments for dynamic training.
    """
    # ----------------------------------------------------------------------
    # BEGIN zhuoyan CHANGES.
    # ----------------------------------------------------------------------

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )

class myDataset(Dataset):
    def __init__(self, train_datas, tokenizer, data_args):
        super().__init__()
        self.data = train_datas
        self.tokenizer = tokenizer
        self.data_args = data_args
        if data_args.pad_to_max_length:
            self.padding = "max_length"
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        tok = self.tokenizer(*input_example_to_tuple(self.data[index]),padding=self.padding, max_length=self.data_args.max_seq_length, truncation=True)
        return tok

def extract(datasets_name, tokenizer, data_args, model):
    dataset = datasets_name
    train_datas = []
    for seed in seed_shot:
        processor = processors_mapping[dataset.lower()]
        train_datas += processor.get_train_examples(os.path.join("data/k-shot/{}/{}".format(dataset,seed)))
    
    # print(len(train_datas))
    # data0 = train_datas[0]
    # print(input_example_to_tuple(data0))
    # tok = tokenizer(*input_example_to_tuple(data0),padding="max_length", max_length=128, truncation=True)

    ######################################################################################################################################################
    ########################################################################################################################
    ### was trying to forward directly, memory not support to forward entirely, might just better to construct dataset/dataloader
    # input_ids = []
    # attention_mask = []
    # for data in train_datas:
    #     tok = tokenizer(*input_example_to_tuple(data),padding="max_length", max_length=128, truncation=True)
    #     input_ids.append(tok['input_ids'])
    #     attention_mask.append(tok['attention_mask'])
    # batch = {'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_mask)}
    # batch = batch.cuda()
    # for k,v in batch.items():
    #     print("key: {}, shape: {}, v: {}".format(k, v.shape, v))

    # model = RobertaModel.from_pretrained("roberta-large")
    # model = model.cuda()
    # out = model(**batch)
    # print(out['pooler_output'].shape)
    ########################################################################################################################
    ########################################################################################################################


    train_dataset = myDataset(train_datas, tokenizer, data_args)
    loader = DataLoader(train_dataset, batch_size = 20, collate_fn = default_data_collator) 
    batch_rep = []
    for batch in loader:
        # batch = next(iter(loader))
        for k,v in batch.items():
            batch[k] = v.cuda()
        out = model(**batch)
        # print(out[1].shape)
        batch_rep.append(out[1].cpu().data.numpy())
    batch_rep = np.concatenate(batch_rep)
    # print(batch_rep.shape)
    # print(batch_rep.dtype)
    return batch_rep


def main():
    parser = HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-large",
    )
    model = RobertaModel.from_pretrained("roberta-large")
    model = model.cuda()
    for datasets_name in datasets:
        print(datasets_name)

        batch_rep = extract(datasets_name, tokenizer, data_args, model)
        print("name: {} | shape of features: {} | type of numpy array: {}".format(
            datasets_name, batch_rep.shape, batch_rep.dtype))
        
        # np.save("result/features/{}.npy".format(datasets_name), batch_rep)

    
    
        


if __name__ == "__main__":
    main()

