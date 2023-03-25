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
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator

from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping
from src.dataset import input_example_to_tuple, tokenize_multipart_input
from src.metaDataset import template_dict, mapping_dict

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
#  'STS-B',
 'mpqa',
 'QNLI',
 'RTE',
 'subj']

seed_shot = ['16-13', '16-100', '16-21', '16-87', '16-42']

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-large-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

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
    def __init__(self, train_datas, tokenizer, data_args, task_name):
        super().__init__()
        self.data = train_datas
        self.tokenizer = tokenizer
        self.data_args = data_args
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        
        # add extra arguments
        extra_args = {}
        if task_name in ["mnli", 'snli', 'rte']:
            extra_args['max_seq_length'] = 256
        if task_name in ['rte']:
            extra_args['first_sent_limit'] = 240
        if task_name in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa']:
            extra_args['first_sent_limit'] = 110
        if task_name in ['mr', 'subj', 'cr']:
            extra_args['other_sent_limit'] = 50
        if task_name in ['sst-5']:
            extra_args['other_sent_limit'] = 20
        self.extra_args = extra_args

        # pass template and labels                            # take MNLI task for example
        self.template = template_dict[task_name]              # '*cls**sent_0*_It_was*mask*.*sep+*'
        mapping_label_to_word = mapping_dict[task_name]       # "{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        
        processor = processors_mapping[task_name]
        label_list = processor.get_labels()                   # ['contradiction', 'entailment', 'neutral']

        assert mapping_label_to_word is not None
        label_to_word = eval(mapping_label_to_word)
        self.label_to_i = {}

        for idx, key in enumerate(label_to_word):
            # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
            if label_to_word[key][0] not in ['<', '[', '.', ',']:
                # Make sure space+word is in the vocabulary
                assert len(tokenizer.tokenize(' ' + label_to_word[key])) == 1
                label_to_word[key] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + label_to_word[key])[0])
            else:
                label_to_word[key] = tokenizer.convert_tokens_to_ids(label_to_word[key])
            print("Label {} to word {} ({})".format(key, tokenizer._convert_id_to_token(label_to_word[key]), label_to_word[key]))
            self.label_to_i[key] = idx                      # {'contradiction': 0, 'entailment': 1, 'neutral': 2}


        self.label_word_list = [label_to_word[label] for label in label_list]  # [440, 3216, 5359]


    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        template_label = self.template.replace("mask", "label_{}".format(self.label_to_i[self.data[index].label]))
        tok = tokenize_multipart_input(
            input_text_list=input_example_to_tuple(self.data[index]),
            max_length=self.extra_args.get('max_seq_length') or 128,
            tokenizer=self.tokenizer,
            prompt=True,
            template=template_label,
            label_word_list=self.label_word_list,
            first_sent_limit=self.extra_args.get('first_sent_limit') or None,
            other_sent_limit=self.extra_args.get('other_sent_limit') or None,
            # truncate_head=self.args.truncate_head,
        )
        return tok

def extract(datasets_name, tokenizer, data_args, model):
    dataset = datasets_name
    train_datas = []
    for seed in seed_shot:
        processor = processors_mapping[dataset.lower()]
        train_datas += processor.get_train_examples(os.path.join("data/k-shot/{}/{}".format(dataset,seed)))
    
    
    train_dataset = myDataset(train_datas, tokenizer, data_args, dataset.lower())
    loader = DataLoader(train_dataset, batch_size = 5, collate_fn = default_data_collator) 
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
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    if 'BertTokenizer' in type(tokenizer).__name__:
        print("bert!")
        model_fn = BertModel
    else:
        print("roberta!")
        model_fn = RobertaModel
    model = model_fn.from_pretrained(model_args.model_name_or_path)
    model = model.cuda()
    for datasets_name in datasets:
        print(datasets_name)

        batch_rep = extract(datasets_name, tokenizer, data_args, model)
        print("name: {} | shape of features: {} | type of numpy array: {}".format(
            datasets_name, batch_rep.shape, batch_rep.dtype))
        
        np.save("result/features/{}-{}-label_info.npy".format(datasets_name, model_args.model_name_or_path), batch_rep)

if __name__ == "__main__":
    main()

