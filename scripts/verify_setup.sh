#!/bin/bash
#
# Verify SSPO Project Setup
#
# This script checks that all components are properly configured.
# Run this after cloning the repo to verify setup.

set -euo pipefail

echo "=========================================="
echo "SSPO Project Setup Verification"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

check() {
    local name="$1"
    local cmd="$2"
    echo -n "Checking: $name ... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        FAIL=$((FAIL + 1))
    fi
}

check_dir() {
    local name="$1"
    local dir="$2"
    echo -n "Checking: $name ($dir) ... "
    if [ -d "$dir" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        FAIL=$((FAIL + 1))
    fi
}

check_file() {
    local name="$1"
    local file="$2"
    echo -n "Checking: $name ($file) ... "
    if [ -f "$file" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "--- Directory Structure ---"
check_dir "src/" "src/src_sspo"
check_dir "scripts/" "scripts"
check_dir "configs/" "configs"
check_dir "tests/" "tests"
check_dir "docs/adr/" "docs/adr"
check_dir "docs/knowledge/SSPO/" "docs/knowledge/SSPO"

echo ""
echo "--- Core Scripts ---"
check_file "download_data.py" "scripts/download_data.py"
check_file "preprocess_data.py" "scripts/preprocess_data.py"
check_file "generate_model_configs.py" "scripts/generate_model_configs.py"
check_file "analyze_data.py" "scripts/analyze_data.py"
check_file "train_local.sh" "scripts/train_local.sh"
check_file "train_sspo.sh" "scripts/train_sspo.sh"
check_file "run_all_experiments.sh" "scripts/run_all_experiments.sh"

echo ""
echo "--- Eval Scripts ---"
check_file "generate_responses.py" "scripts/eval/generate_responses.py"
check_file "alpaca_eval_evaluator.py" "scripts/eval/alpaca_eval_evaluator.py"
check_file "mtbench_evaluator.py" "scripts/eval/mtbench_evaluator.py"
check_file "aggregate_results.py" "scripts/eval/aggregate_results.py"

echo ""
echo "--- Test Files ---"
check_file "test_download.py" "tests/data/test_download.py"
check_file "test_preprocess.py" "tests/data/test_preprocess.py"
check_file "test_preprocess_comprehensive.py" "tests/data/test_preprocess_comprehensive.py"
check_file "test_model_configs.py" "tests/data/test_model_configs.py"
check_file "test_data_analysis.py" "tests/data/test_data_analysis.py"
check_file "test_generate_responses.py" "tests/eval/test_generate_responses.py"
check_file "test_alpaca_eval_evaluator.py" "tests/eval/test_alpaca_eval_evaluator.py"
check_file "test_mtbench_evaluator.py" "tests/eval/test_mtbench_evaluator.py"
check_file "test_aggregate_results.py" "tests/eval/test_aggregate_results.py"

echo ""
echo "--- Generated Configs ---"
check_dir "configs/mistral-7b-it/" "configs/mistral-7b-it"
check_dir "configs/llama3-8b-it/" "configs/llama3-8b-it"
check_dir "configs/qwen2-7b-it/" "configs/qwen2-7b-it"

echo ""
echo "--- Python Environment ---"
check "Python 3" "python3 --version"
check "pytest in venv" ".venv/bin/python -c 'import pytest'"

echo ""
echo "=========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=========================================="

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Activate environment: source .venv/bin/activate"
    echo "  2. Run tests: python -m pytest tests/ -v"
    echo "  3. Download data: python scripts/download_data.py --dataset all"
    exit 0
else
    echo -e "${RED}Some checks failed. Please review.${NC}"
    exit 1
fi
