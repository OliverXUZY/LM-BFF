"""Dataset utils for different data settings for GLUE."""

import os
import copy
import logging
import torch
import numpy as np
import time
from filelock import FileLock
import json
import itertools
import random
import transformers
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, median_mapping
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd

# from torch.utils.data import Dataset

from .dataset import (
    OurInputFeatures,
    input_example_to_string,
    input_example_to_tuple,
    tokenize_multipart_input,
)

logger = logging.getLogger(__name__)

full_task_names = ['cola', 'mrpc', 'qqp', 'snli', 'sts-b', 'cr', 'mr', 'subj',
                       'mnli', 'qnli', 'rte', 'sst-2', 'mpqa', 'sst-5', 'trec']

# full_task_names = ['cola',  'cr', 'mr', 'subj',
#                         'sst-2',  'sst-5']

# full_task_names = ['trec',  'mpqa']

template_dict = {
    'cola': '*cls**sent_0*_This_is*mask*.*sep+*',
    'sst-2': '*cls**sent_0*_It_was*mask*.*sep+*',
    'mrpc': '*cls**sent_0**mask*,*+sentl_1**sep+*',
    'qqp': '*cls**sent_0**mask*,*+sentl_1**sep+*',
    'sts-b': '*cls**sent_0**mask*,*+sentl_1**sep+*',
    'mnli': '*cls**sent-_0*?*mask*,*+sentl_1**sep+*',
    'snli': '*cls**sent-_0*?*mask*,*+sentl_1**sep+*',
    'qnli': '*cls**sent-_0*?*mask*,*+sentl_1**sep+*',
    'rte': '*cls**sent-_0*?*mask*,*+sentl_1**sep+*',
    'mr': '*cls**sent_0*_It_was*mask*.*sep+*',
    'sst-5': '*cls**sent_0*_It_was*mask*.*sep+*',
    'subj': '*cls**sent_0*_This_is*mask*.*sep+*',
    'trec': "*cls**mask*:*+sent_0**sep+*",
    'cr': '*cls**sent_0*_It_was*mask*.*sep+*',
    'mpqa': '*cls**sent_0*_It_was*mask*.*sep+*'
}

mapping_dict = {
    'cola': "{'0':'incorrect','1':'correct'}",
    'sst-2': "{'0':'terrible','1':'great'}",
    'mrpc': "{'0':'No','1':'Yes'}",
    'qqp': "{'0':'No','1':'Yes'}",
    'sts-b': "{'0':'No','1':'Yes'}",
    'mnli': "{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}",
    'snli': "{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}",
    'qnli': "{'not_entailment':'No','entailment':'Yes'}",
    'rte': "{'not_entailment':'No','entailment':'Yes'}",
    'mr': "{0:'terrible',1:'great'}",
    'sst-5': "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}",
    'subj': "{0:'subjective',1:'objective'}",
    'trec': "{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}",
    'cr': "{0:'terrible',1:'great'}",
    'mpqa': "{0:'terrible',1:'great'}"
}

data_dir_dict = {
    'cola': "CoLA",
    'sst-2': "SST-2",
    'mrpc': "MRPC",
    'qqp': "QQP",
    'sts-b': "STS-B",
    'mnli': "MNLI",
    'snli': "SNLI",
    'qnli': "QNLI",
    'rte': "RTE"
}

## The reason is default collator in dataloader will descard task_name when dataset ==> dataloader
task_name_to_id = {
    'cola': 0,
    'sst-2': 1,
    'mrpc': 2,
    'qqp': 3,
    'sts-b': 4,
    'mnli': 5,
    'snli': 6,
    'qnli': 7,
    'rte': 8,
    'mr': 9,
    'sst-5': 10,
    'subj': 11,
    'trec': 12,
    'cr': 13,
    'mpqa': 14,
}

class metaDataset(torch.utils.data.Dataset):
    """meta Few-shot dataset."""
    def __init__(self, args, tokenizer, cache_dir=None, use_demo=False):
        self.args = args
        test_task = args.task_name
        self.task_names = [name for name in full_task_names if name != test_task and name not in ['sts-b']]
        print("zhuoyan: train datasets: ", self.task_names)
        self.tokenizer = tokenizer
        self.datasets = {}
        self.use_demo = use_demo

        self.seed = 0 ## for determinitic random in getitem()

        # Multiple sampling: when using demonstrations, we sample different combinations of demonstrations during 
        # inference and aggregate the results by averaging the logits. The number of different samples is num_sample.
        mode = "train"
        if (mode == "train") or not self.use_demo:
            # We do not do multiple sampling when not using demonstrations or when it's the training mode 
            self.num_sample = 1
        else:
            self.num_sample = args.num_sample

        for name in self.task_names:
            label_list = processors_mapping[name].get_labels()
            self.datasets[name] = {'label_list': label_list}
            self.datasets[name].update({'num_labels': len(label_list)})
            if args.prompt:
                label_to_word = eval(mapping_dict[name])
                for key in label_to_word:
                    # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                    if label_to_word[key][0] not in ['<', '[', '.', ',']:
                        # Make sure space+word is in the vocabulary
                        # print(name)
                        # print(label_to_word[key])
                        assert len(tokenizer.tokenize(' ' + label_to_word[key])) == 1
                        label_to_word[key] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + label_to_word[key])[0])
                    else:
                        label_to_word[key] = tokenizer.convert_tokens_to_ids(label_to_word[key])
                    logger.info("Label {} to word {} ({})".format(key, tokenizer._convert_id_to_token(label_to_word[key]), label_to_word[key]))
                
                if len(label_list) > 1:
                    label_word_list = [label_to_word[label] for label in label_list]
                else:
                    # Regression task
                    # '0' represents low polarity and '1' represents high polarity.
                    label_word_list = [label_to_word[label] for label in ['0', '1']]

            else:
                label_to_word = None
                label_word_list = None
            self.datasets[name].update({'label_to_word': label_to_word})
            self.datasets[name].update({'label_word_list': label_word_list})

            # print("=================================================  Done loading label_list ==========================================")

            processor = processors_mapping[name]

            # Load cache
            direc = data_dir_dict.get(name) or name
            data_dir = "data/k-shot/{}/16-{}".format(direc, args.few_shot_data_seed)
            # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}".format(
                    mode,
                    tokenizer.__class__.__name__,
                    str(args.max_seq_length),
                    name,
                ),
            )

            logger.info(f"Creating/loading examples from dataset file at {data_dir}")

            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not args.overwrite_cache:
                    start = time.time()
                    support_examples, query_examples = torch.load(cached_features_file)
                    logger.info(
                        f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                    )
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")

                    # The support examples are sourced from the training set.
                    support_examples = processor.get_train_examples(data_dir)
                    query_examples = support_examples

                    start = time.time()
                    torch.save([support_examples, query_examples], cached_features_file)
                    # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                    logger.info(
                        "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )
            self.datasets[name].update({'support_examples': support_examples})
            self.datasets[name].update({'query_examples': query_examples})
            
            # print("=================================================  Done loading data ==========================================")
            # Size is expanded by num_sample
        

    def __len__(self):
        return self.args.num_batch # not make a difference
    
    def __getitem__(self, i):
        np.random.seed(self.seed)
        self.seed += 1
        ### random dataset ###
        task_name = random.sample(self.task_names, 1)[0]
        # task_name = 'sst-2'
        # for task_name in self.task_names:

        # if task_name in ["qqp", "mnli", 'snli']:
        #     self.args.num_sample = 4
        # if task_name in ["mnli", 'snli', 'rte']:
        #     self.args.max_seq_length = 256
        # if task_name in ['rte']:
        #     self.args.first_sent_limit = 240
        # if task_name in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa']:
        #     self.args.first_sent_limit = 110
        # if task_name in ['mr', 'subj', 'cr']:
        #     self.args.other_sent_limit = 50
        # if task_name in ['sst-5']:
        #     self.args.other_sent_limit = 20
        
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
       
        dataset = self.datasets[task_name]
        
        # print("select task: ", task_name)
        query_examples = dataset['query_examples']
        ### sample 2 example per getitem, this is the largest batch can go though the GPU, 
        # not necessarily equal to batch size

        query_indices= random.sample([i for i in range(len(query_examples))], 2)
        # query_indices = [0,1]

        # print("select querys: " ,query_indices)
        # The input (query) example
        example1 = query_examples[query_indices[0]]
        # example2 = query_examples[query_indices[1]]
        
        # two query ------------------------
        supports = None
        template = template_dict[task_name]
        # print("Zhuoyan:=======", task_name)


        ex1 = self.convert_fn(
            example=example1,
            supports=supports,
            use_demo=self.use_demo,
            label_list=dataset['label_list'],
            prompt=self.args.prompt,
            template=template,
            label_word_list=['label_word_list'],
            verbose=False,
            task_id = task_name_to_id[task_name]
        )
        # ex2 = self.convert_fn(
        #     example=example2,
        #     supports=supports,
        #     use_demo=self.use_demo,
        #     label_list=dataset['label_list'],
        #     prompt=self.args.prompt,
        #     template=template,
        #     label_word_list=['label_word_list'],
        #     verbose=False,
        # )

        # features = OurInputFeatures(
        #     input_ids = [ex1.input_ids, ex2.input_ids],
        #     attention_mask = [ex1.attention_mask, ex2.attention_mask],
        #     token_type_ids = [ex1.token_type_ids, ex2.token_type_ids],
        #     label = [ex1.label, ex2.label],
        #     mask_pos = [ex1.mask_pos, ex2.mask_pos],
        #     label_word_list = ex1.label_word_list
        #     )
        features = ex1
        
        return features

    def convert_fn(
        self,
        example,
        supports,
        use_demo=False,
        label_list=None,
        prompt=False,
        template=None,
        label_word_list=None,
        verbose=False,
        task_id = None,
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = self.extra_args.get('max_seq_length') or self.args.max_seq_length

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)} # Mapping the label names to label ids
        if len(label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            # Regerssion
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]

        # Prepare other features
        
        # No using demonstrations
        inputs = tokenize_multipart_input(
            input_text_list=input_example_to_tuple(example),
            max_length=max_length,
            tokenizer=self.tokenizer,
            prompt=prompt,
            template=template,
            label_word_list=label_word_list,
            first_sent_limit=self.extra_args.get('first_sent_limit') or None,
            other_sent_limit=self.extra_args.get('other_sent_limit') or None,
            truncate_head=self.args.truncate_head,
        )



        features = OurInputFeatures(**inputs, label=example_label, task_id = task_id)

        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features)
            logger.info("text: %s" % self.tokenizer.decode(features.input_ids))

        return features

class testDataset(torch.utils.data.Dataset):  
    def __init__(self) -> None:
        super().__init__()
        self.seed = 0
    def __len__(self):
        return 10
    def __getitem__(self,index):
        print(self.seed)
        np.random.seed(self.seed)
        self.seed += 1
        a = ["a","b","c","d","e"]
        b = [i for i in range(10)]
        return np.random.choice(a,1)[0], np.random.choice(b,2)








