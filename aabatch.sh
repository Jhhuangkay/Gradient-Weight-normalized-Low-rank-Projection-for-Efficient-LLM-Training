#!/bin/bash
#SBATCH --job-name=LLLora
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=y.shen@uva.nl

source /projects/0/prjs1031/anaconda3/etc/profile.d/conda.sh
conda activate Galore

export HF_HOME=/projects/0/prjs1031/.cache/huggingface
export PIP_CACHE_DIR=/projects/0/prjs1031/.cache/pip

python amlora_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name wnli \
    --enable_galore \
    --lora_all_modules \
    --max_length 512 \
    --seed 1234 \
    --lora_r 8 \
    --galore_scale 4 \
    --per_device_train_batch_size 16 \
    --update_proj_gap 500 \
    --learning_rate 3e-5 \
    --num_train_epochs 30 \
    --output_dir "results_bertglora/ft/berta_base/$task_name"

