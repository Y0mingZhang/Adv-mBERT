import argparse
import logging
import torch

import os



from dataset import PANX_corpus, tag_init
tag_init('wikiann')
from dataset import tag2id

from train import *


from transformers import (AutoModelForTokenClassification,
    AutoConfig, AutoModelWithLMHead, AutoTokenizer,
    AutoModelForSequenceClassification)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=40, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=20, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--src", default="en", required=True, type=str, help="src language.")
    parser.add_argument("--tgt", default="de", required=True, type=str, help="tgt language.")
    parser.add_argument(
        "--model_name_or_path",
        default="xlm-mlm-enfr-1024",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--sequence_length", default=128, type=int, help="Sequence length for language model."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=80, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--ner-lr", default=2e-5, type=float)
    parser.add_argument("--g-lr", default=1e-6, type=float)
    parser.add_argument("--d-lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    ) 
    parser.add_argument(
        "--quick_evaluate_steps", type=int, default=0
    )
    parser.add_argument(
        "--quick_evaluate_ratio", type=float, default=1.0
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
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
    parser.add_argument("--train_data_file", nargs='+', help='Training file(s)')
    parser.add_argument("--seed", type=int, default=427, help="random seed for initialization")
    parser.add_argument("--legacy", action="store_true", help="Legacy code for compatibility with older pytorch versions.")
    parser.add_argument(
        "--lexicon_path",
        default="",
        type=str,
        help="Directory of lexicon"
    )
    parser.add_argument("--replace_word_translation", action="store_false")
    parser.add_argument("--replace_word_translation_ner", action="store_true")
    parser.add_argument('--translation_replacement_probability', type=float, default=0.25, help="probability of replacing a word with its translation")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ner_dir", type=str, required=True)
    parser.add_argument("--tb_postfix", type=str, default="")
    parser.add_argument("--smoothing", type=float, default=0.0)



    args = parser.parse_args()

    if args.smoothing < 0 or args.smoothing > 0.5:
        raise ValueError("Label smoothing must be between 0 and 0.5")

    device = torch.device("cuda" if torch.cuda.is_available()
                            and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()


    args.output_dir = os.path.join(args.output_dir, '{}-{}'.format(args.src, args.tgt))
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   -1, device, args.n_gpu, False, False)

    set_seed(args)
    

 
    


    config = AutoConfig.from_pretrained(args.model_name_or_path,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)
    
    config.output_hidden_states=True
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                 cache_dir=args.cache_dir if args.cache_dir else None,
                                                    use_fast=True)

    
    model_mlm = AutoModelWithLMHead.from_pretrained(args.model_name_or_path, config=config,
                                        cache_dir=args.cache_dir if args.cache_dir  else None)
    config.num_labels = len(tag2id) - 1                                       
    model_ner = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, config=config,
                                        cache_dir=args.cache_dir if args.cache_dir  else None)
    
    model_ner.bert = model_mlm.bert
    model_mlm.to(args.device)
    model_ner.to(args.device)

    bc = BertConfig(hidden_size=model_mlm.config.hidden_size, num_hidden_layers=6, num_attention_heads=6,intermediate_size=768)
    bc.num_labels = 2
    discriminator = AutoModelForSequenceClassification.from_config(bc)

    discriminator.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training


    train_dataset = load_and_cache_examples(
    args, tokenizer, evaluate=False)

   

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


    global_step, tr_loss = train(args, (train_dataset, ner_corpus), 
     (model_mlm, model_ner),
     discriminator, tokenizer)
    logger.info(" global_step = %s, average loss = %s",
                global_step, tr_loss)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model_ner.module if hasattr(model_ner, 'module') else model_ner
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


if __name__ == '__main__':
    main()