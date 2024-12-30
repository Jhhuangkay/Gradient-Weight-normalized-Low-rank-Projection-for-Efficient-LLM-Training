
source /projects/0/prjs1031/anaconda3/etc/profile.d/conda.sh
conda activate Galore

export HF_HOME=/projects/0/prjs1031/.cache/huggingface
export PIP_CACHE_DIR=/projects/0/prjs1031/.cache/pip

python amlora_glue.py \
    --model_name_or_path facebook/bart-base  \
    --task_name wnli \
    --enable_galore \
    --lora_all_modules \
    --max_length 512 \
    --seed 1234 \
    --lora_r 512 \
    --galore_scale 4 \
    --per_device_train_batch_size 16 \
    --update_proj_gap 500 \
    --learning_rate 3e-5 \
    --num_train_epochs 30 \
    --output_dir "results_bertglora8/ft/bart_base/wnli1"

