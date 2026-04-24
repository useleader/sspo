#!/bin/bash
#
# Local training script (single GPU debug)
# This is a convenience wrapper that calls train.sh with LOCAL=1
#
# Usage:
#   bash scripts/train_local.sh configs/mistral-7b-it/sspo/fb0.01_ch0.1_sspo_mistral-7b-it.yaml

set -euo pipefail

CONFIG_FILE="${1:-}"

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash scripts/train_local.sh <config_file>"
    exit 1
fi

# Call train.sh with LOCAL=1
LOCAL=1 bash scripts/train.sh "$CONFIG_FILE"
