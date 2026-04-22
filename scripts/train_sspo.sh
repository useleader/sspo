#!/bin/bash
#
# SSPO Training Script for 8x H100 Cluster
#
# Usage:
#   sbatch scripts/train_sspo.sh configs/mistral-7b-it/sspo/fb0.01_ch0.1_sspo_mistral-7b-it.yaml
#
# Paper configuration (from ADR-0005):
#   - GPUs: 8x H100 (80GB each)
#   - Per device batch: 4
#   - Gradient accumulation: 16
#   - Total batch: 64
#   - Training time: ~2 hours per experiment

set -euo pipefail

#SBATCH --job-name=sspo_train
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

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: sbatch scripts/train_sspo.sh <config_file>"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Setup environment
module purge
module load cuda/12.4  # Adjust based on cluster
module load cudnn/12.4 # Adjust based on cluster

# Create logs directory
mkdir -p logs

# Print configuration
echo "=========================================="
echo "SSPO Training Configuration"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: $GPUS"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "=========================================="

# Calculate number of processes
NNODES=$SLURM_JOB_NUM_NODES
NPROC_PER_NODE=$GPUS

echo "Starting training..."
echo "NNODES=$NNODES, NPROC_PER_NODE=$NPROC_PER_NODE"

# Launch training with torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    --master_addr=$SLURM_JOB_NODENAME \
    src/train.py \
    --config "$CONFIG_FILE"

echo "Training complete!"
