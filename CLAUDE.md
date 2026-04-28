# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSPO (Semi-Supervised Preference Optimization) is an ICLR 2026 paper reproduction project. It implements a semi-supervised approach to LLM alignment where only a small fraction of preference data is labeled, and the model learns from both labeled and unlabeled data using pseudo-labeling.

## Environment Setup

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create Python environment
cd /home/yanzm/sspo
uv venv --python 3.10 .venv
source .venv/bin/activate

# Install dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
uv pip install tqdm numpy
cd src && pip install -r requirements.txt
```

**Requirements**: GPU with 6GB+ VRAM (A100 40GB / RTX 4090 24GB recommended), 16GB+ RAM, 50GB+ storage.

**Note**: Current WSL environment has no GPU. `torch.cuda.is_available() = False` is expected. Training runs on 8x H100 cluster.

## Project Structure

```
/home/yanzm/sspo/
├── src/                           # Paper source code (LLaMA-Factory fork)
│   ├── src_sspo/llamafactory/     #   SSPO implementation
│   │   ├── train/dpo/trainer.py   #   ★ SSPO core algorithm
│   │   ├── data/processors/       #   Data processors
│   │   └── hparams/finetuning_args.py  # Hyperparameters
│   ├── preprocessing_data/        #   Paper's preprocessing reference
│   ├── examples/                  #   Paper training examples
│   ├── data/                      #   LLaMA-Factory data + dataset_info.json
│   └── requirements.txt           #   Dependencies
│
├── scripts/                       # Project pipeline scripts (our implementation)
│   ├── download_data.py           #   Download UltraFeedback + UltraChat
│   ├── preprocess_data.py         #   Sample fb%/ch% ratios
│   ├── generate_model_configs.py  #   Generate training YAML configs
│   ├── analyze_data.py           #   Data analysis & statistics
│   ├── train_sspo.sh             #   SLURM training (8x H100)
│   ├── train_local.sh             #   Local debug (1 GPU)
│   ├── run_all_experiments.sh     #   Orchestrator
│   └── eval/                     #   Evaluation package
│       ├── generate_responses.py  #   Generate model responses
│       ├── alpaca_eval_evaluator.py  # LC-Win Rate
│       ├── mtbench_evaluator.py  #   MT-Bench 8-category
│       └── aggregate_results.py   #   Result aggregation
│
├── tests/                        # Test suite (TDD workflow)
│   ├── data/                     #   Data tests
│   │   ├── test_download.py
│   │   ├── test_preprocess.py
│   │   ├── test_model_configs.py
│   │   └── test_data_analysis.py
│   └── eval/                     #   Eval tests
│       └── test_generate_responses.py
│
├── configs/                      # Training YAML configs
│   ├── cluster/                 #   Cluster training configs (by model)
│   │   ├── llama3-8b/
│   │   ├── mistral-7b-it/
│   │   └── qwen2-7b/
│   ├── local/                    #   Local/debug configs
│   │   ├── sft_hybrid/
│   │   ├── ablation_gamma/
│   │   ├── ablation_prior/
│   │   └── ...
│   └── test_*.yaml              #   Test configs
├── data/                        # Raw downloaded data
│   ├── ultrafeedback/           #   UltraFeedback preference dataset
│   └── ultrachat/               #   UltraChat conversations
├── processed/                   # Preprocessed intermediate data
├── saves/                       # Trained model checkpoints (by model & method)
│   ├── llama3-8b/             #   Llama-3-8B saves
│   ├── mistral-7b-it/         #   Mistral-7B saves
│   ├── qwen2-7b/              #   Qwen2-7B saves
│   ├── kto/                   #   KTO method saves
│   └── toy/                   #   Toy experiment saves
├── logs/                        # Training logs
├── plots/                       # Training visualizations (loss curves)
├── results/                     # Evaluation results
│   ├── samples/                 #   Generated responses for evaluation
│   ├── alpaca_eval/             #   AlpacaEval scores
│   └── mtbench/                 #   MT-Bench scores
├── notebooks/                    # Jupyter notebooks for analysis
├── docs/
│   ├── adr/                    #   8 Architecture Decision Records
│   └── knowledge/SSPO/         #   Paper knowledge base (9 files)
└── .gitignore
```

## Data Flow

```
scripts/download_data.py    → HuggingFace → cache/
scripts/preprocess_data.py → Sample fb%/ch% → src/data/dataset_info.json
scripts/generate_model_configs.py → YAML configs
scripts/train_sspo.sh     → torchrun → src/train.py → src/src_sspo/llamafactory/
scripts/eval/              ← Trained model → Evaluation
```

## Common Commands

### Data Pipeline
```bash
# Download data
python scripts/download_data.py --output cache/

# Preprocess data (fb=labeled%, ch=unlabeled%)
python scripts/preprocess_data.py --fb 0.01 --ch 0.10

# Generate training configs
python scripts/generate_model_configs.py --output configs/
```

### Training
```bash
# Generate configs + run locally (debug)
bash scripts/run_all_experiments.sh --local

# Generate configs + submit SLURM jobs
bash scripts/run_all_experiments.sh --submit
```

### Evaluation
```bash
# Generate responses
python scripts/eval/generate_responses.py \
    --model_path saves/mistral-7b-it/sspo/fb0.01_ch0.1/best_model \
    --dataset alpacaeval --output results/responses.json

# Evaluate and aggregate
python scripts/eval/aggregate_results.py --results-dir results/
```

## Key Parameters (from Paper Table 4, ADR-0008)

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 |
| Learning rate | 1e-5 |
| Batch size | 64 (per node) |
| Context length | 1024 |
| γ_0 | 1.0 |
| γ_min | 0.22 |
| γ_decay | 0.001 |
| sspo_prior | 0.5 |
| β (beta) | 2.0 |

## Key Files

| File | Purpose |
|------|---------|
| `src/src_sspo/llamafactory/train/dpo/trainer.py` | SSPO loss implementation |
| `src/src_sspo/llamafactory/hparams/finetuning_args.py` | SSPO hyperparameters |
| `scripts/preprocess_data.py` | Data sampling (fb%/ch%) |
| `scripts/eval/generate_responses.py` | Response generation for benchmarks |

## Models (from Paper)

1. `mistralai/Mistral-7B-Instruct-v0.2`
2. `meta-llama/Meta-Llama-3-8B-Instruct` (requires HF_TOKEN)
3. `Qwen/Qwen2-7B-Instruct`

## Data Ratios (from Paper Table 1)

| Labeled (fb) | Unlabeled (ch) | D_L count | D_U count |
|--------------|----------------|-----------|-----------|
| 1% | 10% | ~611 | 20,000 |
| 5% | 10% | ~3,057 | 20,000 |
| 10% | 10% | ~6,114 | 20,000 |

## Baselines Supported

SSPO, DPO, ORPO, SimPO, KTO, SSRM, SPA — all use similar training infrastructure but with different loss functions in their respective trainers.

## Documentation

- ADRs: `docs/adr/` — 8 architecture decision records
- Knowledge base: `docs/knowledge/SSPO/` — 9 files covering full paper
