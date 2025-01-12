# GradNormLoRP

This repository contains the **GradNormLoRP** algorithm, proposed in the paper: [Gradient Weight-Normalized Low-Rank Projection for Efficient LLM Training](https://arxiv.org/pdf/2412.19616).

**Gradient Weight-Normalized Low-Rank Projection (GradNormLoRP)** is a novel approach that enhances both parameter and memory efficiency while maintaining comparable performance to full fine-tuning. GradNormLoRP normalizes the weight matrix to improve gradient conditioning, facilitating better convergence during optimization. Additionally, it applies low-rank approximations to the weight and gradient matrices, significantly reducing memory usage during training.

---

## Installation

### Install GradNormLoRP Optimizer
Install the GradNormLoRP optimizer using pip:

```bash
pip install GradNormLoRP
```

### Install Experiment Dependencies
To install the dependencies required for running the experiments, use the following command:

```bash
pip install -r exp_requirements.txt
```

---

## Usage

### Pre-training

`torchrun_main.py` is the main script for training LLaMA models on the C4 dataset with GradNormLoRP. Benchmark scripts for various model sizes are located in the `scripts/benchmark_c4` folder.

For example, to train a 60M parameter model on C4, execute the following command:

```bash
# LLaMA-60M, GradNormLoRP-Adam, 1 A100 GPU, 1 Node
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 250 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer weight_norm_galore_adamw
```

### Fine-tuning

The fine-tuning process involves leveraging PEFT techniques such as LoRA or DoRA. The script `test_fine.py` supports fine-tuning using GradNormLoRP.

An example configuration for fine-tuning:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
    use_dora=True
)
model = get_peft_model(model, lora_config)
```

To run fine-tuning on a specific GLUE task (e.g., MRPC), execute the following:

```bash
task_name="mrpc"
echo "Running task: $task_name"
python test_fine.py \
    --model_name_or_path roberta-base \
    --task_name $task_name \
    --enable_lowrank \
    --lora_all_modules \
    --max_length 512 \
    --seed 1234 \
    --lora_r 4 \
    --galore_scale 4 \
    --per_device_train_batch_size 16 \
    --update_proj_gap 500 \
    --learning_rate 3e-5 \
    --num_train_epochs 30 \
    --output_dir "results_wn/$task_name"
```

---

## Repository Structure

- `configs/`: Configuration files for various model architectures.
- `scripts/benchmark_c4/`: Benchmark scripts for pre-training on C4.
- `test_fine.py`: Main script for fine-tuning on downstream tasks.
- `torchrun_main.py`: Main script for pre-training with GradNormLoRP.

---

## Citation

If you use GradNormLoRP in your research, please cite the paper:

```bibtex
@inproceedings{huang2025gradient,
  title={Gradient Weight-Normalized Low-Rank Projection for Efficient LLM Training},
  author={Huang, Jia-Hong and Shen, Yixian and Zhu, Hongyi and Rudinac, Stevan and Kanoulas, Evangelos},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  year={2025}
}
```

