#!/bin/bash

# Clean up existing installations
pip uninstall -y numpy transformers

# Install requirements with specific versions
pip install --upgrade pip
pip install numpy==1.24.3
pip install transformers==4.37.2
pip install -r requirements.txt

# Find the latest checkpoint directory
latest_checkpoint=$(ls -d ./output/layer_looping_qwen_open_web_text/checkpoint-* 2>/dev/null | sort -V | tail -n 1)

# Set up checkpoint argument
checkpoint_arg=""
if [ ! -z "$latest_checkpoint" ]; then
    echo "Resuming from checkpoint: $latest_checkpoint"
    checkpoint_arg="--resume_from_checkpoint=$latest_checkpoint"
fi

# Run training with accelerate
python3 train.py \
    --model_name_or_path="Qwen/Qwen2.5-0.5B" \
    --n=8 \
    --max_loop_count=10 \
    --per_device_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --learning_rate=5e-5 \
    --num_train_epochs=3 \
    --warmup_ratio=0.03 \
    --sequence_length=512 \
    --use_distillation \
    --kl_weight=1 \
    --lm_weight=1 \
    --output_dir="./output/layer_looping_qwen_open_web_text" \
    --use_wandb \
    $checkpoint_arg