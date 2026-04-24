#!/bin/bash
#
# Generic Training Script for All Methods (SSPO, DPO, ORPO, SimPO, KTO, SSRM, SPA)
#
# Usage:
#   sbatch scripts/train.sh configs/mistral-7b-it/sspo/fb0.01_ch0.1_sspo_mistral-7b-it.yaml
#   sbatch scripts/train.sh configs/mistral-7b-it/dpo/fb0.01_ch0.1_dpo_mistral-7b-it.yaml
#
# Paper configuration:
#   - GPUs: 8x H100 (80GB each) for cluster, 1x GPU for local
#   - Per device batch: 8
#   - Gradient accumulation: 16
#   - Total batch: 64 (per node)
#   - Training time: ~2 hours per experiment

set -euo pipefail

# SLURM configuration (commented for cluster submission)
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Configuration
CONFIG_FILE="${1:-}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
LOCAL="${LOCAL:-0}"  # Set to 1 for local single-GPU training

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: sbatch scripts/train.sh <config_file> [GPUS] [LOCAL]"
    echo "  config_file: Path to YAML config"
    echo "  GPUS: Number of GPUs (default: 8)"
    echo "  LOCAL: Set to 1 for local training (default: 0)"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Determine method from path
METHOD=$(echo "$CONFIG_FILE" | grep -oP '(dpo|orpo|simpo|kto|ssrm|spa|sspo)' | head -1 | tr '[:lower:]' '[:upper:]')
MODEL_NAME=$(basename "$(dirname "$(dirname "$CONFIG_FILE")")")

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Method: $METHOD"
echo "Model: $MODEL_NAME"
echo "GPUs: $GPUS"
echo "Mode: $([ "$LOCAL" = "1" ] && echo "Local" || echo "Cluster")"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Setup for cluster or local
if [ "$LOCAL" = "1" ]; then
    # Local single-GPU training
    echo "Starting local training (1 GPU)..."

    NNODES=1
    NPROC_PER_NODE=1

    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_port=$MASTER_PORT \
        src/src_sspo/train.py \
        --config "$CONFIG_FILE"
else
    # Cluster multi-GPU training
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        # Running via SLURM
        NNODES=$SLURM_JOB_NUM_NODES
        NPROC_PER_NODE=$GPUS
        MASTER_ADDR=$SLURM_JOB_NODENAME
    else
        # Running manually on cluster (not via sbatch)
        NNODES=1
        NPROC_PER_NODE=$GPUS
        MASTER_ADDR="localhost"
    fi

    echo "Starting cluster training..."
    echo "NNODES=$NNODES, NPROC_PER_NODE=$NPROC_PER_NODE"
    echo "MASTER_ADDR=$MASTER_ADDR"

    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_port=$MASTER_PORT \
        --master_addr=$MASTER_ADDR \
        src/src_sspo/train.py \
        --config "$CONFIG_FILE"
fi

echo "Training complete!"
