#!/bin/bash
#SBATCH --job-name=download_qwen
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=./results/download_qwen80b.log

# Set LOCAL_DIR to where you want the model weights saved.
LOCAL_DIR=${LOCAL_DIR:-/path/to/Qwen3-Next-80B-A3B-Thinking}

export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download Qwen/Qwen3-Next-80B-A3B-Thinking \
    --local-dir "$LOCAL_DIR"

