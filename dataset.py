from __future__ import absolute_import, division, print_function

from argparse import Namespace  # For debugging (dict -> args)
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


import transformers
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          XLMConfig, XLMWithLMHeadModel, XLMTokenizer)

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', lang_id=0):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_name_or_path + '_cached_lm_' + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s",
                        cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            self.examples = []
            


            with open(file_path, encoding="utf-8") as f:
                for seq in tqdm(f):
                    enc = tokenizer.encode(
                        seq, max_length=args.sequence_length)
                    tokens = torch.tensor(enc)

                    padded_tokens = torch.empty(
                        args.sequence_length, dtype=tokens.dtype).fill_(tokenizer.pad_token_id)
                    pad_mask = torch.zeros(args.sequence_length, dtype=torch.bool)
                    padded_tokens[0:tokens.shape[0]] = tokens
                    pad_mask[0:tokens.shape[0]] = 1
                    self.examples.append((padded_tokens, pad_mask, torch.LongTensor([lang_id])))

            logger.info("Saving features into cached file %s",
                        cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

tag2id = None
id2tag = None

def tag_init(mode):
    global tag2id
    global id2tag
    if mode == 'conll':
        tag2id = {'B-LOC': 1,
        'B-MISC': 2,
        'B-ORG': 3,
        'B-PER': 4,
        'I-LOC': 5,
        'I-MISC': 6,
        'I-ORG': 7,
        'I-PER': 8,
        'O': 0,
        'UNDEFINED': -100}
    elif mode == 'wikiann':
        tag2id = {'B-LOC': 1,
        'B-ORG': 2,
        'B-PER': 3,
        'I-LOC': 4,
        'I-ORG': 5,
        'I-PER': 6,
        'O': 0,
        'UNDEFINED': -100}
    else:
        assert(0)
    id2tag = {v:k for k,v in tag2id.items()}

class PANX_corpus():
    def __init__(self, corpus_path, lang, tokenizer):
        """ 
        Input:
            corpus_path: path to a directory 
            where files in the format (**.train/testa/testb) exists

        Example Usage:
            eng_corpus = ConLL_corpus('./data/eng')
            train_tokens, train_tags = eng_corpus.get_training_data()
        """
        assert(len(tag2id) == 8)
        corpus_path = os.path.join(corpus_path, lang)
        self.corpus = {}
        self.datasets = {}
        self.lang = lang
        for f in glob.glob(corpus_path + '/*'):
            
            part = f.split('/')[-1]
            if part not in ('train', 'dev', 'test'): continue
            tokens, tags, td = self.parse_file(f, tokenizer)

            
            self.corpus[part] = (tokens, tags)
            self.datasets[part] = td
            
        

    def parse_file(self, file, tokenizer):
        
        tokens = []
        tags = []
        seq_tokens = []
        seq_tags = []

        bert_tokens = []
        bert_attns = []
        bert_tags = []

        for line in tqdm(open(file, encoding="utf-8")):

            if 'DOCSTART' in line:
                continue

            fields = line.strip().split('\t')

            if not line.strip():
                if seq_tokens and seq_tags:
                    assert(len(seq_tokens) == len(seq_tags))
                    iids = [tokenizer.cls_token_id]
                    attns = [1]
                    labels = [tag2id['UNDEFINED']]

                    for token, tag in zip(seq_tokens, seq_tags):
                        tokenized_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                        if tokenized_token:
                            iids.extend(tokenized_token)
                            labels.extend([tag2id[tag]] + [tag2id['UNDEFINED']] * (len(tokenized_token)-1))
                            attns.extend([1] * len(tokenized_token))
                    
                    iids.append(tokenizer.sep_token_id)
                    attns.append(1)
                    labels.append(tag2id['UNDEFINED'])

                    assert(len(iids) == len(labels))
                    if len(iids) > 128:
                        seq_tokens = []
                        seq_tags = []
                        continue
                        
                    pad_length = 128 - len(iids)
                    iids.extend([tokenizer.pad_token_id] * pad_length)
                    attns.extend([0] * pad_length)
                    labels.extend([tag2id['UNDEFINED']] * pad_length)

                    tokens.append(seq_tokens)
                    tags.append(seq_tags)

                    assert(len(iids) == len(labels))

                    bert_tokens.append(iids)
                    bert_attns.append(attns)
                    bert_tags.append(labels)

                    seq_tokens = []
                    seq_tags = []
            else:
             

                seq_tokens.append(fields[0].split(':', maxsplit=1)[1])
                seq_tags.append(fields[-1])
        td = TensorDataset(torch.LongTensor(bert_tokens), torch.FloatTensor(bert_attns), torch.LongTensor(bert_tags))
        return tokens, tags, td
    
    def get_training_data(self):
        return self.corpus['train'][1], self.datasets['train']
    
    def get_validation_data(self):
        return self.corpus['dev'][1], self.datasets['dev']
    
    def get_test_data(self):
        return self.corpus['test'][1], self.datasets['test']
