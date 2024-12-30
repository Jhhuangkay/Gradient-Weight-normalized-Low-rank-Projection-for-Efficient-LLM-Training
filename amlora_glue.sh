#!/bin/bash
#SBATCH --job-name=LLLora
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --output=./bertglora8/my_job_%j.out
#SBATCH --error=./bertglora8/my_job_%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=y.shen@uva.nl

source /projects/0/prjs1031/anaconda3/etc/profile.d/conda.sh
conda activate Galore

export HF_HOME=/projects/0/prjs1031/.cache/huggingface
export PIP_CACHE_DIR=/projects/0/prjs1031/.cache/pip

task_name=$(echo "$1" | tr -cd 'A-Za-z0-9_-')
echo "Running task: $task_name"
python amlora_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $task_name \
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
    --output_dir "results_bertglora8/$task_name"
