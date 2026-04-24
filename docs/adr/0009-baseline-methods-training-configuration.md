# ADR-0009: Baseline Methods Training Configuration

**Date**: 2026-04-23
**Status**: proposed
**Deciders**: SSPO Reproduction Team

## Context

The SSPO paper compares against 6 baseline methods (DPO, ORPO, SimPO, KTO, SSRM, SPA) in Table 1 and Table 2. We need to implement reproducible training configurations for all baselines using a unified framework with method-specific config switching.

## Decision

We use a unified training framework in `src/src_sspo/llamafactory/` where all methods share the same infrastructure (data loading, LoRA, logging) but differ in loss functions and hyperparameters. Configs are generated via `scripts/generate_model_configs.py` with method-specific YAML fields.

### Supported Methods and Loss Types

| Method | `pref_loss` | `stage` | Key Hyperparameters |
|--------|-------------|---------|---------------------|
| DPO | `sigmoid` | `dpo` | `pref_beta=0.1` |
| ORPO | `orpo` | `dpo` | `pref_beta=0.1`, `orpo_lambda=0.5` |
| SimPO | `simpo` | `dpo` | `pref_beta=2.0`, `simpo_gamma=2.0` |
| KTO | `kto_pair` | `kto` | `pref_beta=0.1`, `kto_chosen_weight=1.0` |
| SSRM | `ssrm` | `dpo` | `ssrm_prior=0.5` (semi-supervised) |
| SPA | `spa` | `dpo` | `spa_iterations=3` (iterative self-annotation) |
| **SSPO** | `sspo` | `dpo` | `sspo_gamma_0=1.0`, `sspo_gamma_min=0.22`, `sspo_prior=0.5` |

### Config Generation

File: `scripts/generate_model_configs.py`

Each method generates YAML with shared base config + method-specific fields:

```python
# Shared base config (all methods)
base_config = {
    "model_name_or_path": model_path,
    "trust_remote_code": True,
    "do_train": True,
    "finetuning_type": "lora",
    "template": "default",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "lora_rank": 8,
    "lora_target": "all",
    "bf16": True,
    "logging_steps": 20,
    "save_steps": 300,
    "plot_loss": True,
}

# Method-specific (appended based on method argument)
method_configs = {
    "dpo": {"stage": "dpo", "pref_loss": "sigmoid", "pref_beta": 0.1},
    "orpo": {"stage": "dpo", "pref_loss": "orpo", "pref_beta": 0.1, "orpo_lambda": 0.5},
    "simpo": {"stage": "dpo", "pref_loss": "simpo", "pref_beta": 2.0, "simpo_gamma": 2.0},
    "kto": {"stage": "kto", "pref_loss": "kto_pair", "pref_beta": 0.1},
    "ssrm": {"stage": "dpo", "pref_loss": "ssrm", "ssrm_prior": 0.5},
    "spa": {"stage": "dpo", "pref_loss": "spa", "spa_iterations": 3},
    "sspo": {"stage": "dpo", "pref_loss": "sspo", "pref_beta": 2.0, "sspo_gamma_0": 1.0,
             "sspo_gamma_min": 0.22, "sspo_gamma_decay": 0.001, "sspo_prior": 0.5,
             "sspo_base": "simpo", "simpo_gamma": 2.0},
}
```

### Output Directory Structure

```
configs/
├── mistral-7b-it/
│   ├── dpo/fb0.01_ch0.1_dpo_mistral-7b-it.yaml
│   ├── orpo/fb0.01_ch0.1_orpo_mistral-7b-it.yaml
│   ├── simpo/fb0.01_ch0.1_simpo_mistral-7b-it.yaml
│   ├── kto/fb0.01_ch0.1_kto_mistral-7b-it.yaml
│   ├── ssrm/fb0.01_ch0.1_ssrm_mistral-7b-it.yaml
│   ├── spa/fb0.01_ch0.1_spa_mistral-7b-it.yaml
│   └── sspo/fb0.01_ch0.1_sspo_mistral-7b-it.yaml
├── llama3-8b/
│   └── ... (same structure)
└── qwen2-7b/
    └── ... (same structure)
```

### Training Command

All methods use the same torchrun command, differing only in config path:

```bash
python -m torchrun --nproc_per_node=1 src/train.py \
    --config configs/{model}/{method}/fb{fb}_ch{ch}_{method}_{model}.yaml
```

## Alternatives Considered

### Alternative 1: Separate Training Scripts per Method
- **Pros**: Each method is self-contained
- **Cons**: Code duplication, inconsistent behavior, harder to maintain
- **Why not**: We want fair comparison with identical infrastructure

### Alternative 2: Use LLaMA-Factory Built-in Trainers
- **Pros**: Already implemented DPO, ORPO, SimPO, KTO
- **Cons**: Missing SSRM, SPA (our additions); fork anyway for SSPO
- **Why not**: We need SSPO-specific modifications and unified logging

## Consequences

### Positive
- Unified framework ensures fair comparison (same data, same LoRA, same infrastructure)
- Easy to add new baselines by extending `generate_yaml()`
- Consistent logging and checkpointing across all methods
- Easy to reproduce paper Table 1 results

### Negative
- Configuration generation script must be kept in sync with trainer changes
- KTO uses different `stage` which may require different data preprocessing

### Risks
- **Risk**: SSRM and SPA implementations may diverge from paper due to interpretation
- **Mitigation**: Document implementation decisions in ADR-0010; verify against paper pseudocode
