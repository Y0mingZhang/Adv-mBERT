import logging
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
    AdamW)

from dataset import PANX_corpus, tag_init
tag_init('wikiann')
from dataset import tag2id, id2tag
import os
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import random
import numpy as np
from utils import score_predictions

logging.basicConfig(level = logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", default=427, type=int
    )
    parser.add_argument(
        "--no_cuda", action="store_true"
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=40, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--src", default="en", required=True, type=str, help="src language.")
    parser.add_argument("--tgt", default="de", required=True, type=str, help="tgt language.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The model checkpoint for weights initialization.",
        required=True
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="bert-base-multilingual-cased"
    )
    parser.add_argument(
        "--sequence_length", default=128, type=int, help="Sequence length for language model."
    )
    parser.add_argument(
        "--num_train_epochs", default=4, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument("--ner-lr", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--ner_dir", type=str, required=True)

    args = parser.parse_args()

    

    device = torch.device("cuda" if torch.cuda.is_available()
                            and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    if os.path.isdir(args.model_name_or_path):
        args.output_dir = os.path.join(args.model_name_or_path, 'ner')
    else:
        args.output_dir = os.path.join(args.output_dir, '{}-{}'.format(args.src, args.tgt))
    
    args.tb_dir = os.path.join(args.output_dir, 'tb')
    if not os.path.isdir(args.tb_dir):
        os.makedirs(args.tb_dir)
    
    tb_writer = SummaryWriter(args.tb_dir)

    args.device = device
   
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,
                                 cache_dir=args.cache_dir, use_fast=True)
    model_ner = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, 
                                        cache_dir=args.cache_dir)
    model_ner.to(args.device)

    if args.n_gpu > 1:
        model_ner = torch.nn.DataParallel(model_ner)

    ner_cache_dir = os.path.join(args.cache_dir, 'ner')
    if not os.path.isdir(ner_cache_dir):
        os.makedirs(ner_cache_dir)
    ner_cache = os.path.join(ner_cache_dir, 
     '_'.join([args.model_name_or_path, args.src, args.tgt]) + '.pt')

    if os.path.exists(ner_cache):
        ner_corpus = torch.load(ner_cache)
    else:
        ner_corpus = {
            args.src : PANX_corpus(args.ner_dir, args.src, tokenizer),
            args.tgt : PANX_corpus(args.ner_dir, args.tgt, tokenizer)
        }
        torch.save(ner_corpus, ner_cache)
    
    train(args, ner_corpus, model_ner, tb_writer)

    logging.info("Saving model checkpoint to %s", args.output_dir)
    model_to_save = model_ner.module if hasattr(model_ner, "module") else model_ner
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    best_checkpoint = os.path.join(args.output_dir, "checkpoint-best")                
    model_ner = AutoModelForTokenClassification.from_pretrained(best_checkpoint)
    model_ner.to(args.device)

    if args.n_gpu > 1:
        model_ner = torch.nn.DataParallel(model_ner)

    test_dataset = ner_corpus[args.tgt].datasets['test']
    preds_path = os.path.join(args.output_dir)   
    a,p,r,f = evaluate_ner(args, model_ner, test_dataset, preds_path)

    logging.info("A/P/R/F1 on {} test set: {} {},{}, {}".format(args.tgt,a,p,r,f))
    tb_writer.add_scalar("Accuracy_{}_test".format(args.tgt), a, 0)
    tb_writer.add_scalar("Precision_{}_test".format(args.tgt), p, 0)
    tb_writer.add_scalar("Recall_{}_test".format(args.tgt), r, 0)
    tb_writer.add_scalar("F1_{}_test".format(args.tgt), f, 0)
    tb_writer.close()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def train(args, ner_corpus, model_ner, tb_writer):

    train_dataset = ner_corpus[args.src].datasets['train']
    dev_dataset_src = ner_corpus[args.src].datasets['dev']
    dev_dataset_tgt = ner_corpus[args.tgt].datasets['dev']

    batch_size = args.per_gpu_train_batch_size * args.n_gpu
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model_ner.parameters(),
                      lr=args.ner_lr, eps=args.adam_epsilon)
    
    model_ner.zero_grad()

    best_score = 0.0
    global_step = 0
    for _ in tqdm(range(args.num_train_epochs), desc='Epoch'):
        for step, batch in tqdm(enumerate(train_dataloader), desc='Iteration'):
            model_ner.train()
            batch = tuple(t.to(args.device) for t in batch)
            loss, scores = model_ner(input_ids=batch[0], attention_mask=batch[1], 
                labels=batch[2])[:2]
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model_ner.parameters(), args.max_grad_norm)
                optimizer.step()
                model_ner.zero_grad()
                global_step += 1  
            
                if global_step % args.save_steps == 0:

                    a,p,r,f = evaluate_ner(args, model_ner, dev_dataset_src)
                    logging.info("A/P/R/F1 on {} dev set: {} {}, {}, {}".format(args.src,a,p,r,f))
                    tb_writer.add_scalar("Accuracy_{}_dev".format(args.src), a, global_step)
                    tb_writer.add_scalar("Precision_{}_dev".format(args.src), p, global_step)
                    tb_writer.add_scalar("Recall_{}_dev".format(args.src), r, global_step)
                    tb_writer.add_scalar("F1_{}_dev".format(args.src), f, global_step)

                    if f > best_score:
                        output_dir = os.path.join(args.output_dir, "checkpoint-best")
                        best_checkpoint = output_dir
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model_ner.module if hasattr(model_ner, "module") else model_ner
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logging.info("Saving the best model checkpoint to %s", output_dir)

                    a,p,r,f = evaluate_ner(args, model_ner, dev_dataset_tgt)
                    logging.info("A/P/R/F1 on {} dev set: {} {}, {}, {}".format(args.src,a,p,r,f))
                    tb_writer.add_scalar("Accuracy_{}_dev".format(args.tgt), a, global_step)
                    tb_writer.add_scalar("Precision_{}_dev".format(args.tgt), p, global_step)
                    tb_writer.add_scalar("Recall_{}_dev".format(args.tgt), r, global_step)
                    tb_writer.add_scalar("F1_{}_dev".format(args.tgt), f, global_step)



def evaluate_ner(args, model_ner, ner_dataset, preds_path=None):
    model_ner.eval()
    logging.info("***** Running evaluation on {} examples *****".format(len(ner_dataset)))
    dataloader = DataLoader(ner_dataset, shuffle=False, batch_size=args.per_gpu_eval_batch_size)
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval"):
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
    if preds_path:
        torch.save(all_preds, os.path.join(preds_path, "preds.pt"))
        torch.save(labels_list, os.path.join(preds_path, "labels.pt"))
    return score_predictions(all_preds, all_labels)


    
if __name__ == '__main__':
    main()