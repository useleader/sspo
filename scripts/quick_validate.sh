#!/bin/bash
#
# Quick Validation Script - Run representative experiments on 4 GPUs
# Purpose: Verify all methods work correctly before full cluster run
#
# Usage:
#   bash scripts/quick_validate.sh        # Run all 7 methods in parallel (4 GPUs)
#   bash scripts/quick_validate.sh 1     # Run method 1 only (SSPO)
#   bash scripts/quick_validate.sh --status  # Check status of running jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Quick test configs - one per method (using mistral-7b-it, fb=0.01)
declare -A QUICK_CONFIGS
QUICK_CONFIGS[1]="configs/cluster/mistral-7b-it/sspo/mistral-7b-it_sspo_fb0.01_ch0.1.yaml"
QUICK_CONFIGS[2]="configs/cluster/mistral-7b-it/dpo/mistral-7b-it_dpo_fb0.01.yaml"
QUICK_CONFIGS[3]="configs/cluster/mistral-7b-it/simpo/mistral-7b-it_simpo_fb0.01.yaml"
QUICK_CONFIGS[4]="configs/cluster/mistral-7b-it/orpo/mistral-7b-it_orpo_fb0.01.yaml"
QUICK_CONFIGS[5]="configs/cluster/mistral-7b-it/kto/mistral-7b-it_kto.yaml"
QUICK_CONFIGS[6]="configs/cluster/mistral-7b-it/ssrm/mistral-7b-it_ssrm_fb0.01_ch0.1.yaml"
QUICK_CONFIGS[7]="configs/cluster/mistral-7b-it/spa/mistral-7b-it_spa_fb0.01_ch0.1.yaml"

declare -A METHOD_NAMES
METHOD_NAMES[1]="SSPO"
METHOD_NAMES[2]="DPO"
METHOD_NAMES[3]="SimPO"
METHOD_NAMES[4]="ORPO"
METHOD_NAMES[5]="KTO"
METHOD_NAMES[6]="SSRM"
METHOD_NAMES[7]="SPA"

# GPU allocation for 4-GPU machine
# Each method gets 1 GPU, run 4 in parallel, then remaining 3
GPU_ALLOC=(
    "0"  # SSPO - GPU 0
    "1"  # DPO - GPU 1
    "2"  # SimPO - GPU 2
    "3"  # ORPO - GPU 3
    "0"  # KTO - GPU 0 (second wave)
    "1"  # SSRM - GPU 1 (second wave)
    "2"  # SPA - GPU 2 (second wave)
)

show_help() {
    echo "Usage: bash scripts/quick_validate.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  (none)           Run all 7 methods - wave 1 (4 parallel), wave 2 (3 parallel)"
    echo "  1-7              Run specific method only"
    echo "  --status         Check status of running validation jobs"
    echo "  --gpu ID         Run on specific GPU ID (0-3)"
    echo "  --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  bash scripts/quick_validate.sh          # Run all methods"
    echo "  bash scripts/quick_validate.sh 1        # Run SSPO only"
    echo "  bash scripts/quick_validate.sh --status # Check status"
}

run_single() {
    local id=$1
    local config="${QUICK_CONFIGS[$id]}"
    local method="${METHOD_NAMES[$id]}"
    local gpu="${GPU_ALLOC[$id]}"

    if [ ! -f "$config" ]; then
        echo "Error: Config not found: $config"
        return 1
    fi

    echo "=========================================="
    echo "Running $method quick validation"
    echo "Config: $config"
    echo "GPU: $gpu"
    echo "=========================================="

    # Set CUDA visible devices for this job
    export CUDA_VISIBLE_DEVICES=$gpu

    # Run training with reduced steps for quick validation
    LOCAL=1 bash "$SCRIPT_DIR/train.sh" "$config"
}

run_all_wave1() {
    echo "=========================================="
    echo "WAVE 1: Running 4 methods in parallel (GPUs 0-3)"
    echo "Methods: SSPO, DPO, SimPO, ORPO"
    echo "=========================================="

    # Launch first 4 in background
    for id in 1 2 3 4; do
        run_single $id &
        sleep 2  # Stagger starts
    done

    # Wait for all to complete
    wait
    echo "Wave 1 complete!"
}

run_all_wave2() {
    echo "=========================================="
    echo "WAVE 2: Running 3 methods in parallel (GPUs 0-2)"
    echo "Methods: KTO, SSRM, SPA"
    echo "=========================================="

    # Launch remaining 3 in background
    for id in 5 6 7; do
        run_single $id &
        sleep 2  # Stagger starts
    done

    # Wait for all to complete
    wait
    echo "Wave 2 complete!"
}

check_status() {
    echo "=========================================="
    echo "Validation Job Status"
    echo "=========================================="

    # Check for running python processes
    if pgrep -f "train.py" > /dev/null; then
        echo "Training jobs running:"
        ps aux | grep -E "train.py|sspotrain" | grep -v grep | head -10
    else
        echo "No training jobs currently running"
    fi

    # Check logs
    if [ -d "logs" ]; then
        echo ""
        echo "Recent log files:"
        ls -lt logs/*.out 2>/dev/null | head -5
    fi
}

# Main
case "${1:-}" in
    --help|-h)
        show_help
        ;;
    --status)
        check_status
        ;;
    1|2|3|4|5|6|7)
        run_single $1
        ;;
    all)
        run_all_wave1
        run_all_wave2
        echo ""
        echo "=========================================="
        echo "ALL QUICK VALIDATION COMPLETE!"
        echo "=========================================="
        ;;
    "")
        # Default: run all
        run_all_wave1
        run_all_wave2
        echo ""
        echo "=========================================="
        echo "ALL QUICK VALIDATION COMPLETE!"
        echo "=========================================="
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
