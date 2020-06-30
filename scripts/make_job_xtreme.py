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

python3 /scratch/mihalcea_root/mihalcea1/yimingz/src/Adv-mBERT/ner_finetune.py \
--per_gpu_train_batch_size 8 \
--src en \
--tgt $TGT \
--model_name_or_path bert-base-multilingual-cased \
--num_train_epochs 10 \
--ner-lr 2e-5 \
--cache_dir /scratch/mihalcea_root/mihalcea1/yimingz/data/cache_dir \
--output_dir /scratch/mihalcea_root/mihalcea1/yimingz/data/training_output/adv-mbert/xtreme \
--gradient_accumulation_steps 4 \
--ner_dir /scratch/mihalcea_root/mihalcea1/yimingz/data/panx_dataset \
--save_steps 125
"""
import sys
print(script.format(sys.argv[1]))