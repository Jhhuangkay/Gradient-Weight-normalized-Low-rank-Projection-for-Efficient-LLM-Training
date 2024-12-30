source /projects/0/prjs1031/anaconda3/etc/profile.d/conda.sh
conda activate Galore

export HF_HOME=/projects/0/prjs1031/.cache/huggingface
export PIP_CACHE_DIR=/projects/0/prjs1031/.cache/pip

# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node
torchrun --standalone --nproc_per_node 1 test_train.py \
    --model_config configs/llama_60m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 256 \
    --update_proj_gap 200 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 5 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer lowRank
    # --optimizer 
    #weight_norm_galore_adamw 
    # galore_adamw

    # Capture the exit status of the torchrun command
