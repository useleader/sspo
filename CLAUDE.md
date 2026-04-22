# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSPO (Semi-Supervised Preference Optimization) is an ICLR 2026 paper reproduction project. It implements a semi-supervised approach to LLM alignment where only a small fraction of preference data is labeled, and the model learns from both labeled and unlabeled data using pseudo-labeling.

## Environment Setup

```bash
conda create -n sspo python==3.10.0
conda activate sspo
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
cd /home/yanzm/sspo/src
pip install -r requirements.txt
```

**Requirements**: GPU with 6GB+ VRAM (A100 40GB / RTX 4090 24GB recommended), 16GB+ RAM, 50GB+ storage.

## Common Commands

### Data Preprocessing
```bash
python preprocessing_data/preprocessing_ultrachat.py --fb [feedback_ratio] --ch [chat_ratio]
```
`fb` = labeled feedback ratio, `ch` = unlabeled chat ratio

### Training Configuration
```bash
python examples/train/make_yaml.py --peft lora --method sspo --model_path mistralai/Mistral-7B-Instruct-v0.2
```

### Execute Training
```bash
bash examples/train/train.sh
```

## Architecture

### Core Code Structure
```
src/src_sspo/llamafactory/
├── train/                    # Training implementations
│   ├── dpo/trainer.py       # DPO trainer (contains SSPO implementation)
│   ├── kto/trainer.py       # KTO trainer
│   ├── ppo/trainer.py       # PPO trainer
│   └── rm/trainer.py        # Reward model trainer
├── data/                    # Data processing
│   └── processors/           # Dataset processors (feedback, pairwise, supervised, unsupervised)
├── model/                   # Model loading and patching
│   └── loader.py            # Model initialization
└── hparams/                 # Hyperparameter definitions
```

### SSPO Implementation
The SSPO algorithm is implemented within `src/src_sspo/llamafactory/train/dpo/trainer.py` as a subclass/extension of DPO training. It uses pseudo-labeling on unlabeled preference data to leverage semi-supervised learning.

### Data Flow
1. Raw data → Preprocessing scripts in `preprocessing_data/`
2. Preprocessed data → DataLoaders in `llamafactory/data/`
3. DataLoaders → Trainer.compute_loss() in `llamafactory/train/*/trainer.py`
4. Loss gradients → Model update via PEFT (LoRA default)

## Key Files

| File | Purpose |
|------|---------|
| `src/src_sspo/llamafactory/train/dpo/trainer.py` | DPO/SSPO trainer implementation |
| `src/examples/train/make_yaml.py` | Training config YAML generator |
| `src/preprocessing_data/preprocessing_ultrachat.py` | UltraFeedback dataset preprocessing |
| `src/requirements.txt` | Python dependencies |

## Knowledge Base

Project documentation and research notes are in `docs/` (Obsidian format). Key files:
- `docs/SSPO 论文精读.md` - Paper analysis
- `docs/SSPO 理论分析.md` - Theoretical analysis
- `docs/SSPO 复现规划.md` - Reproduction plan

## Baselines Supported

SSPO, DPO, ORPO, SimPO, KTO, SSRM, SPA - all use similar training infrastructure but with different loss functions in their respective trainers.
