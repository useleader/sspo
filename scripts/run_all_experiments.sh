#!/bin/bash
#
# Generate all experiment configs and optionally submit SLURM jobs
#
# Usage:
#   bash scripts/run_all_experiments.sh --generate  # Generate configs only
#   bash scripts/run_all_experiments.sh --submit    # Generate and submit
#   bash scripts/run_all_experiments.sh --local    # Generate and run locally

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Parse arguments
MODE="${1:-generate}"

generate_configs() {
    echo "Generating experiment configurations..."
    python3 scripts/generate_model_configs.py --output configs/
    echo "Configs generated in configs/"
}

submit_jobs() {
    echo "Submitting SLURM jobs for all experiments..."
    
    # Find all SSPO configs
    configs=$(find configs -name "*sspo*.yaml" | sort)
    
    job_count=0
    for config in $configs; do
        echo "Submitting: $config"
        sbatch scripts/train_sspo.sh "$config"
        job_count=$((job_count + 1))
        
        # Small delay to avoid overwhelming SLURM
        sleep 1
    done
    
    echo "Submitted $job_count jobs"
}

run_local() {
    echo "Running experiments locally (single GPU debug)..."
    
    # Find first SSPO config as example
    config=$(find configs -name "*sspo*.yaml" | head -1)
    
    if [ -z "$config" ]; then
        echo "Error: No configs found. Run --generate first."
        exit 1
    fi
    
    echo "Running: $config"
    bash scripts/train_local.sh "$config"
}

case "$MODE" in
    --generate)
        generate_configs
        ;;
    --submit)
        generate_configs
        submit_jobs
        ;;
    --local)
        generate_configs
        run_local
        ;;
    *)
        echo "Usage: bash scripts/run_all_experiments.sh [--generate|--submit|--local]"
        echo "  --generate: Generate configs only"
        echo "  --submit:   Generate and submit SLURM jobs"
        echo "  --local:   Generate and run locally (debug)"
        exit 1
        ;;
esac

echo "Done!"
