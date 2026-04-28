# SSPO Cluster Experiment Configurations

## Directory Structure

```
configs_cluster/
├── README.md                    # This file
├── data/                        # Processed datasets
│   ├── dataset_info.json        # Dataset registry
│   ├── ultra_paired_fb0.01.json # 1% labeled (for DPO/SimPO/ORPO)
│   ├── ultra_combined_fb0.01_ch0.1.json  # 1% labeled + 10% unlabeled
│   ├── ultra_combined_fb0.05_ch0.1.json  # 5% labeled + 10% unlabeled
│   └── ultra_combined_fb0.10_ch0.1.json  # 10% labeled + 10% unlabeled
├── mistral-7b-it/              # Mistral-7B-Instruct-v0.2
│   ├── dpo/
│   ├── simpo/
│   ├── orpo/
│   ├── sspo/
│   ├── ssrm/
│   ├── spa/
│   └── kto/
├── qwen2-7b/                   # Qwen2-7B-Instruct
│   └── ...
└── llama3-8b/                  # Llama-3-8B-Instruct
    └── ...
```

## Experiments

### 3 Models × 6 Methods × 3 Data Ratios = 54 + 3 KTO = 57 configs

| Method | Datasets Used | Description |
|--------|--------------|-------------|
| DPO | ultra_paired_fb{0.01,0.05,0.10} | Baseline with reference model |
| SimPO | ultra_paired_fb{0.01,0.05,0.10} | SimPO without reference |
| ORPO | ultra_paired_fb{0.01,0.05,0.10} | ORPO without reference |
| SSPO | ultra_combined_fb{0.01,0.05,0.10}_ch0.1 | Semi-supervised |
| SSRM | ultra_combined_fb{0.01,0.05,0.10}_ch0.1 | Reward model pseudo-labeling |
| SPA | ultra_combined_fb{0.01,0.05,0.10}_ch0.1 | Self-preference annotation |
| KTO | kto_mix_en | Demo dataset |

## Hyperparameters (from Paper)

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 |
| Learning rate | 1e-5 |
| Per device batch size | 4 |
| Gradient accumulation steps | 16 |
| Total batch size | 64 (8 GPUs × 4 × 2) |
| Context length | 1024 |
| β (beta) | 0.1 (DPO/ORPO), 2.0 (SimPO) |
| SSPO γ_0 | 1.0 |
| SSPO γ_min | 0.22 |
| SSPO γ_decay | 0.001 |
| SSPO prior | 0.5 |

## Usage

### 1. Setup Environment

```bash
# Activate environment
source .venv/bin/activate

# Set HuggingFace/ModelScope token if needed
export HF_TOKEN=xxx
export MODELSCOPE_TOKEN=xxx
```

### 2. Download Models

```bash
# Download to ./cache directory
USE_MODELSCOPE_HUB=1 python scripts/download_models.py --model mistral --source modelscope --output ./cache
USE_MODELSCOPE_HUB=1 python scripts/download_models.py --model qwen2 --source modelscope --output ./cache
USE_MODELSCOPE_HUB=1 python scripts/download_models.py --model llama3 --source modelscope --output ./cache --token <token>
```

### 3. Update Config Paths

Update `cache_dir` and `model_name_or_path` in configs to point to your model locations.

### 4. Run Training

Single experiment:
```bash
USE_MODELSCOPE_HUB=1 python src/src_sspo/train.py configs_cluster/mistral-7b-it/sspo/mistral-7b-it_sspo_fb0.01_ch0.1.yaml
```

Or use the SLURM script:
```bash
sbatch scripts/train_sspo.sh configs_cluster/mistral-7b-it/sspo/mistral-7b-it_sspo_fb0.01_ch0.1.yaml
```

## Cluster Configuration

- GPUs: 8x H100 (80GB each)
- CPUs per task: 64
- Time limit: 24:00:00
- Partition: compute

## Notes

- All configs use `report_to: none` to disable wandb
- All configs use `bf16: true` for mixed precision training
- `preprocessing_num_workers: 8` for faster data loading
