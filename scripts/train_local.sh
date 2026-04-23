#!/bin/bash
#
# Local/Debug Training Script for SSPO
#
# Usage:
#   bash scripts/train_local.sh configs/mistral-7b-it/sspo/fb0.01_ch0.1_sspo_mistral-7b-it.yaml
#
# This script is for debugging on a single GPU.
# For full training, use train_sspo.sh with SLURM.

set -euo pipefail

CONFIG_FILE="${1:-}"

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash scripts/train_local.sh <config_file>"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "SSPO Local/Debug Training"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: 1 (local debug)"
echo "=========================================="

# Single GPU training
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=29500 \
    src/src_sspo/train.py \
    --config "$CONFIG_FILE"

echo "Training complete!"
