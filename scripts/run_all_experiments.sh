#!/bin/bash
#
# Generate all experiment configs and optionally submit SLURM jobs
#
# Supports:
#   - All methods: SSPO, DPO, ORPO, SimPO, KTO, SSRM, SPA
#   - All models: mistral, llama3, qwen2, phi2, meerkat, ultramedical, mistral-business, finance
#   - All ablation experiments: toy, prior, scheduler, all
#
# Usage:
#   bash scripts/run_all_experiments.sh --generate              # Generate all configs
#   bash scripts/run_all_experiments.sh --generate --method sspo  # Generate SSPO only
#   bash scripts/run_all_experiments.sh --ablation toy        # Generate Toy Experiment configs
#   bash scripts/run_all_experiments.sh --ablation all       # Generate all ablation configs
#   bash scripts/run_all_experiments.sh --submit               # Generate and submit all jobs
#   bash scripts/run_all_experiments.sh --local                # Generate and run locally (debug)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train.sh"

cd "$PROJECT_ROOT"

# Default values
METHOD="${METHOD:-all}"
MODEL="${MODEL:-all}"
ABLATION="${ABLATION:-}"
FB_RATIOS="${FB_RATIOS:-0.01 0.05 0.10}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --generate)
            MODE="generate"
            shift
            ;;
        --submit)
            MODE="submit"
            shift
            ;;
        --local)
            MODE="local"
            shift
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --ablation)
            ABLATION="$2"
            shift 2
            ;;
        --fb)
            FB_RATIOS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure TRAIN_SCRIPT is executable
chmod +x "$TRAIN_SCRIPT"

generate_configs() {
    echo "=========================================="
    echo "Generating experiment configurations..."
    echo "=========================================="

    CMD="python3 scripts/generate_model_configs.py --output configs/"

    if [ "$METHOD" != "all" ]; then
        CMD="$CMD --method $METHOD"
    fi

    if [ "$MODEL" != "all" ]; then
        CMD="$CMD --model $MODEL"
    fi

    if [ -n "$ABLATION" ]; then
        CMD="$CMD --ablation $ABLATION"
    fi

    echo "Running: $CMD"
    eval $CMD

    echo ""
    echo "Configs generated in configs/"
    echo ""
    echo "Directory structure:"
    find configs -type f -name "*.yaml" | head -20
    echo "..."
}

submit_jobs() {
    echo "=========================================="
    echo "Submitting SLURM jobs..."
    echo "=========================================="

    # Find all configs (excluding ablation subdirectories)
    if [ -n "$ABLATION" ]; then
        # For ablation experiments, find matching configs
        configs=$(find configs -name "*.yaml" | grep -E "($ABLATION|prior|fixed_gamma|noise)" | sort | head -50)
    else
        # For standard experiments, exclude ablation configs
        configs=$(find configs -name "*.yaml" | grep -v -E "(prior|fixed_gamma|noise)" | sort)
    fi

    job_count=0
    fail_count=0

    for config in $configs; do
        # Skip if file doesn't exist (might have been filtered out)
        if [ ! -f "$config" ]; then
            continue
        fi

        echo "Submitting: $config"
        if sbatch "$TRAIN_SCRIPT" "$config"; then
            job_count=$((job_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi

        # Small delay to avoid overwhelming SLURM
        sleep 0.5
    done

    echo ""
    echo "=========================================="
    echo "Submitted: $job_count jobs"
    if [ $fail_count -gt 0 ]; then
        echo "Failed: $fail_count jobs"
    fi
    echo "=========================================="
}

run_local() {
    echo "=========================================="
    echo "Running experiments locally (single GPU debug)..."
    echo "=========================================="

    # Find first config as example
    if [ -n "$ABLATION" ]; then
        config=$(find configs -name "*.yaml" | grep -E "($ABLATION|prior|fixed_gamma|noise)" | head -1)
    else
        config=$(find configs -name "*.yaml" | grep -v -E "(prior|fixed_gamma|noise)" | head -1)
    fi

    if [ -z "$config" ]; then
        echo "Error: No configs found. Run --generate first."
        exit 1
    fi

    echo "Running example: $config"
    echo ""

    # Set LOCAL=1 for single GPU mode
    LOCAL=1 bash "$TRAIN_SCRIPT" "$config"
}

show_help() {
    echo "Usage: bash scripts/run_all_experiments.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --generate              Generate configs only (default if no mode specified)"
    echo "  --submit                Generate and submit SLURM jobs"
    echo "  --local                 Generate and run locally (debug)"
    echo "  --method METHOD         Training method: all, sspo, dpo, orpo, simpo, kto, ssrm, spa"
    echo "  --model MODEL           Model: all, mistral, llama3, qwen2, phi2, meerkat, ultramedical, mistral-business, finance"
    echo "  --ablation ABLATION    Ablation type: toy, prior, scheduler, all"
    echo "  --fb RATIOS             Labeled ratios (space-separated): 0.01 0.05 0.10"
    echo ""
    echo "Examples:"
    echo "  # Generate all configs"
    echo "  bash scripts/run_all_experiments.sh --generate"
    echo ""
    echo "  # Generate and submit SSPO jobs only"
    echo "  bash scripts/run_all_experiments.sh --method sspo --submit"
    echo ""
    echo "  # Generate Toy Experiment configs"
    echo "  bash scripts/run_all_experiments.sh --ablation toy"
    echo ""
    echo "  # Generate and run all ablation experiments locally"
    echo "  bash scripts/run_all_experiments.sh --ablation all --local"
}

# Main
case "${MODE:-generate}" in
    generate)
        generate_configs
        ;;
    submit)
        generate_configs
        submit_jobs
        ;;
    local)
        generate_configs
        run_local
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown mode: ${MODE:-}"
        show_help
        exit 1
        ;;
esac

echo ""
echo "Done!"
