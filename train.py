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
from torch.utils.data import (DataLoader, Dataset, SequentialSampler,
 RandomSampler, ConcatDataset, Subset)
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter




import transformers
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          XLMConfig, XLMWithLMHeadModel, XLMTokenizer,
                          XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForMaskedLM)

from tqdm.auto import tqdm

from dataset import TextDataset, tag2id, id2tag
from model import *
from utils import get_lexicon_matching, score_predictions

logger = logging.getLogger(__name__)



def replace_word_translation(args, tokens, tokenizer, matching):

    pad_id = tokenizer.pad_token_id
    for i in range(tokens.shape[0]):
        for j in range(tokens.shape[1]):
            int_token_id = int(tokens[i][j])
            if int_token_id == pad_id: break
            if int_token_id in matching and random.random() < args.translation_replacement_probability:
                tokens[i][j] = random.choice(matching[int_token_id])

def replace_word_translation_labels(args, tokens, labels, tokenizer, matching):

    pad_id = tokenizer.pad_token_id
    for i in range(tokens.shape[0]):
        for j in range(tokens.shape[1]):
            int_token_id = int(tokens[i][j])
            if int_token_id == pad_id: break
            if int_token_id in matching and random.random() < args.translation_replacement_probability:
                labels[i][j] = random.choice(matching[int_token_id])
        

def load_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(args.train_data_file, list):
        datasets = []
        for lang_id, train_data_file in enumerate(args.train_data_file):
            dataset = TextDataset(tokenizer, args, file_path=train_data_file, lang_id=lang_id)
            datasets.append(dataset)
        
        return ConcatDataset(datasets)


            
    else:
        dataset = TextDataset(
            tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file)
    
        return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_mask_token_id(args, tokenizer):
    return tokenizer.mask_token_id

def mask_tokens(tokens, mask, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    mask_token_id = get_mask_token_id(args, tokenizer)
    labels = tokens.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(
        val, already_has_special_tokens=True) for val in tokens.tolist()]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix[special_tokens_mask | ~mask] = 0.0
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # We only compute loss on masked tokens
    # TODO: Figure out why -1 does not work with xlm
    labels[~masked_indices] = -100
    

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    tokens[indices_replaced] = mask_token_id 

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced

    tokens[indices_random] = torch.randint(
        len(tokenizer), (indices_random.sum(),), dtype=tokens.dtype)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return tokens, labels

def mask_tokens_legacy(tokens, mask, tokenizer, args):
    mask_token_id = get_mask_token_id(args, tokenizer)
    labels = tokens.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    mask = mask.to(torch.uint8)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(
        val, already_has_special_tokens=True) for val in tokens.tolist()]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.uint8)
    
    probability_matrix[special_tokens_mask | ~mask] = 0.0
    masked_indices = torch.bernoulli(probability_matrix).to(torch.uint8)
    # We only compute loss on masked tokens
    # TODO: Figure out why -1 does not work with xlm
    labels[~masked_indices] = -100
    

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (torch.bernoulli(torch.full(
        labels.shape, 0.8))).to(torch.uint8) & masked_indices
    tokens[indices_replaced] = mask_token_id 

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (torch.bernoulli(torch.full(
        labels.shape, 0.5))).to(torch.uint8) & masked_indices & ~indices_replaced

    tokens[indices_random] = torch.randint(
        len(tokenizer), (indices_random.sum(),), dtype=tokens.dtype)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return tokens, labels


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, data, models, discriminator, tokenizer):
    torch.autograd.set_detect_anomaly(True)
    model_mlm, model_ner = models
    train_dataset_mlm, ner_corpus = data
    train_dataset_ner = ner_corpus[args.src].datasets['train']
    dev_dataset_src = ner_corpus[args.src].datasets['dev']
    dev_dataset_tgt = ner_corpus[args.tgt].datasets['dev']

    dataset_size = min(len(train_dataset_mlm), len(train_dataset_ner))

    train_dataset_mlm = Subset(train_dataset_mlm, random.sample(range(len(train_dataset_mlm)), dataset_size))
    train_dataset_ner = Subset(train_dataset_ner, random.sample(range(len(train_dataset_ner)), dataset_size))

    train_dataloader_mlm = DataLoader(
        train_dataset_mlm, shuffle=True, batch_size=args.per_gpu_train_batch_size)

    train_dataloader_ner = DataLoader(
        train_dataset_ner, shuffle=True, batch_size=args.per_gpu_train_batch_size)

    tb_path = os.path.join(args.output_dir, 'tb', args.tb_postfix) if args.tb_postfix else os.path.join(args.output_dir, 'tb')
    tb_writer = SummaryWriter(tb_path)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataset_mlm) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataset_mlm) // args.train_batch_size // args.gradient_accumulation_steps * \
             args.num_train_epochs


    optimizer_mlm = AdamW(model_mlm.parameters(),
                      lr=args.g_lr, eps=args.adam_epsilon)

    optimizer_ner = AdamW(model_ner.parameters(),
                      lr=args.ner_lr, eps=args.adam_epsilon)
    
    d_optimizer = AdamW(discriminator.parameters(),
                      lr=args.d_lr)

    d_criterion = nn.CrossEntropyLoss()

    if args.n_gpu > 1:
        model_mlm = torch.nn.DataParallel(model_mlm)
        model_ner = torch.nn.DataParallel(model_ner)
        discriminator = torch.nn.DataParallel(discriminator)

    # Make mask token backward compatible
    if args.legacy:
        _mask_tokens = mask_tokens_legacy
    else:
        _mask_tokens = mask_tokens
    
    matching = get_lexicon_matching(args, tokenizer)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", dataset_size)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0

    d_acc_last_batch = 0.5
    
    train_iterator = tqdm(range(int(args.num_train_epochs)),
                            desc="Epoch")
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch in train_iterator:

        logging.info("   Begin Epoch {}   ".format(epoch))
        logging_numbers = [0.0, 0.0, 0.0, 0.0, 0.0]
        for step, (batch_mlm, batch_ner) in tqdm(enumerate(zip(train_dataloader_mlm, train_dataloader_ner))):
            """ ner stuff """
            model_ner.train()
            optimizer_ner.zero_grad()
            batch_ner = [t.to(args.device) for t in batch_ner]
            tokens, mask, labels = batch_ner
            
            if args.replace_word_translation_ner:
                replace_word_translation(args, tokens, tokenizer, matching)
            
            loss = model_ner(input_ids=tokens, attention_mask=mask, labels=labels)[0]
            loss.backward()
            optimizer_ner.step()
            logging_numbers[4] += loss.item()

            """ mlm stuff """
            model_mlm.train()
            discriminator.train()
            optimizer_mlm.zero_grad()
            tokens, mask, langs = batch_mlm
            try: 
                inputs, labels = _mask_tokens(
                    tokens, mask, tokenizer, args)
            except TypeError:
                _mask_tokens = mask_tokens_legacy
                inputs, labels = _mask_tokens(
                    tokens, mask, tokenizer, args)

            if args.replace_word_translation:
                replace_word_translation(args, inputs, tokenizer, matching)

            if args.do_word_translation_retrieval:
                replace_word_translation_labels(args, inputs, labels, tokenizer, matching)
            tokens = tokens.to(args.device)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            mask = mask.to(args.device)
            
            
            
            # Label smoothing
            langs = langs.to(args.device).view(-1)
            langs_smoothed = langs.float().clone()
            langs_smoothed[langs_smoothed==1.0] = 1.0 - args.smoothing
            langs_smoothed[langs_smoothed==0.0] = args.smoothing
            d_labels = langs_smoothed.clone()
            

            g_labels = 1 - langs_smoothed.clone()
            

            outputs = model_mlm(inputs, attention_mask=mask, masked_lm_labels=labels)
            # Get masked-lm loss & last hidden state
            lm_loss = outputs[0]
            last_layer = outputs[2][-1]
            

            # Train D
            if global_step % args.d_update_steps == 0:
                discriminator.zero_grad()
                d_input = last_layer.detach()
                mask = mask.to(torch.float)
                d_output = discriminator(inputs_embeds=d_input, attention_mask=mask,
                labels=None)[0].view(-1)          
                d_loss = F.binary_cross_entropy_with_logits(d_output, d_labels)       
                if args.n_gpu > 1:
                    d_loss = d_loss.mean()
                d_loss.backward()
                d_optimizer.step()

            # Train G
            model_mlm.zero_grad()

            # Update generator w/ fake labels
            d_output = discriminator(inputs_embeds=last_layer, attention_mask=mask,
             labels=None)[0].view(-1)
            

            g_loss = F.binary_cross_entropy_with_logits(d_output, g_labels)
            
            if args.n_gpu > 1:
                lm_loss = lm_loss.mean()
                g_loss = g_loss.mean()

            loss = lm_loss + args.alpha * g_loss
            loss.backward()
            optimizer_mlm.step()

            global_step += 1

            d_preds = (d_output > 0).to(torch.long)
            d_acc = int((d_preds == langs).sum()) / (langs.shape[0])
            logging_numbers[0] += lm_loss.item()
            logging_numbers[1] += g_loss.item()
            logging_numbers[2] += d_loss.item()
            logging_numbers[3] += d_acc

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                # Only evaluate when single GPU otherwise metrics may not average well
                logging.info('Step: {}'.format(global_step))
                logging.info('MLM Loss: {}'.format(logging_numbers[0]/args.logging_steps))
                logging.info('Generator Loss: {}'.format(logging_numbers[1]/args.logging_steps))
                logging.info('Total Loss: {}'.format((logging_numbers[0] + args.alpha * logging_numbers[1])/args.logging_steps))
                logging.info('Discriminator Loss: {}'.format(logging_numbers[2]/args.logging_steps))
                logging.info('Discriminator Acc: {}'.format(logging_numbers[3]/args.logging_steps))
                logging.info('NER Loss: {}'.format(logging_numbers[4]/args.logging_steps))

                tb_writer.add_scalar('MLM_Loss', logging_numbers[0]/args.logging_steps, global_step)
                tb_writer.add_scalar('Generator_Loss', logging_numbers[1]/args.logging_steps, global_step)
                tb_writer.add_scalar('Total_Loss', logging_numbers[0] + args.alpha * logging_numbers[1], global_step)
                tb_writer.add_scalar('Discriminator_Loss', logging_numbers[2]/args.logging_steps, global_step)
                tb_writer.add_scalar('Discriminator_Acc', logging_numbers[3]/args.logging_steps, global_step)
                tb_writer.add_scalar('NER_Loss', logging_numbers[4]/args.logging_steps, global_step)

                logging_numbers = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            if args.quick_evaluate_steps > 0 and global_step % args.quick_evaluate_steps == 0:
                _,p,r,f = evaluate_ner(args, model_ner, dev_dataset_src)
                logging.info("P/R/F1 on {} dev set: {}, {}, {}".format(args.src, p,r,f))
                tb_writer.add_scalar("Precision_{}_dev".format(args.src), p, global_step)
                tb_writer.add_scalar("Recall_{}_dev".format(args.src), r, global_step)
                tb_writer.add_scalar("F1_{}_dev".format(args.src), f, global_step)

                if dev_dataset_tgt:
                    _,p,r,f = evaluate_ner(args, model_ner, dev_dataset_tgt)
                    logging.info("P/R/F1 on {} dev set: {}, {}, {}".format(args.tgt, p,r,f))
                    tb_writer.add_scalar("Precision_{}_dev".format(args.tgt), p, global_step)
                    tb_writer.add_scalar("Recall_{}_dev".format(args.tgt), r, global_step)
                    tb_writer.add_scalar("F1_{}_dev".format(args.tgt), f, global_step)
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = 'checkpoint'
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Take care of distributed/parallel training
                model_to_save = model_ner.module if hasattr(
                    model_ner, 'module') else model_ner
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(
                    output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

                _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        

        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    
    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate_ner(args, model_ner, ner_dataset):
    model_ner.eval()
    if args.quick_evaluate_ratio < 1.0:
        dataset_size = min(len(ner_dataset), int(10000 * args.quick_evaluate_ratio))
        ner_dataset = Subset(ner_dataset, random.sample(range(len(ner_dataset)), dataset_size))
    logger.info("***** Running evaluation on {} examples *****".format(len(ner_dataset)))
    dataloader = DataLoader(ner_dataset, shuffle=False, batch_size=args.per_gpu_eval_batch_size)
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = [t.to(args.device) for t in batch]
            tokens, mask, labels = batch
            loss, scores = model_ner(input_ids=tokens, attention_mask=mask, labels=labels)[0:2]
            preds = scores.argmax(dim=-1)

            for pred_seq, label_seq in zip(preds, labels):
                preds_list = []
                labels_list = []
                for i in range(len(pred_seq)):
                    if label_seq[i] != -100:
                        preds_list.append(id2tag[int(pred_seq[i])])
                        labels_list.append(id2tag[int(label_seq[i])])
                all_preds.append(preds_list)
                all_labels.append(labels_list)
    
    return score_predictions(all_preds, all_labels)
    
    


            