#!/bin/bash
#SBATCH --job-name=LLLora
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=y.shen@uva.nl

source /projects/0/prjs1031/anaconda3/etc/profile.d/conda.sh
conda activate Galore

export HF_HOME=/projects/0/prjs1031/.cache/huggingface
export PIP_CACHE_DIR=/projects/0/prjs1031/.cache/pip

# LLaMA-60M, weight_norm_galore_adamw-Adam, 1 A100, 1 Node
torchrun --standalone --nproc_per_node 1 test_train.py \
    --model_config configs/llama_60m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 256 \
    --update_proj_gap 250 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer weight_norm_galore_adamw 