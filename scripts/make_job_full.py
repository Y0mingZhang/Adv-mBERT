script="""#!/bin/bash
#SBATCH --job-name mbert
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=gpu
#SBATCH --gpus=v100:1
#SBATCH --time=0-10:00:00
#SBATCH --account=mihalcea1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yimingz@umich.edu

TGT={}
set -e
set -x

cat /scratch/mihalcea_root/mihalcea1/yimingz/WIKI_DATA/en/en.all | head -n10000 \
 > /scratch/mihalcea_root/mihalcea1/yimingz/WIKI_DATA/en/en.10000

cat /scratch/mihalcea_root/mihalcea1/yimingz/WIKI_DATA/$TGT/$TGT.all | head -n10000 \
 > /scratch/mihalcea_root/mihalcea1/yimingz/WIKI_DATA/$TGT/$TGT.10000

python3 /scratch/mihalcea_root/mihalcea1/yimingz/src/Adv-mBERT/main.py \
--per_gpu_train_batch_size 8 \
--src en \
--tgt $TGT \
--model_name_or_path bert-base-multilingual-cased \
--num_train_epochs 10 \
--logging_steps 50 \
--ner-lr 2e-5 \
--g-lr 2e-5 \
--d-lr 5e-5 \
--quick_evaluate_steps 1000 \
--quick_evaluate_ratio 0.1 \
--cache_dir /scratch/mihalcea_root/mihalcea1/yimingz/data/cache_dir \
--output_dir /scratch/mihalcea_root/mihalcea1/yimingz/data/training_output/adv-mbert/full \
--train_data_file /scratch/mihalcea_root/mihalcea1/yimingz/WIKI_DATA/en/en.10000 /scratch/mihalcea_root/mihalcea1/yimingz/WIKI_DATA/$TGT/$TGT.10000 \
--lexicon_path /scratch/mihalcea_root/mihalcea1/yimingz/data/lexicon/en-$TGT.txt \
--ner_dir /scratch/mihalcea_root/mihalcea1/yimingz/data/panx_dataset \
--save_steps 1000 \
--smoothing 0.2 \
--alpha 0.5 \
--d_update_steps 5 \
--translation_replacement_probability 0.5 \
--do_word_translation_retrieval

python3 /scratch/mihalcea_root/mihalcea1/yimingz/src/Adv-mBERT/ner_finetune.py \
--per_gpu_train_batch_size 8 \
--src en \
--tgt $TGT \
--model_name_or_path /scratch/mihalcea_root/mihalcea1/yimingz/data/training_output/adv-mbert/no_trans/en-$TGT/ \
--num_train_epochs 10 \
--ner-lr 2e-5 \
--cache_dir /scratch/mihalcea_root/mihalcea1/yimingz/data/cache_dir \
--output_dir - \
--gradient_accumulation_steps 4 \
--ner_dir /scratch/mihalcea_root/mihalcea1/yimingz/data/panx_dataset \
--save_steps 125
"""
import sys
print(script.format(sys.argv[1]))