# ADR-0008: Hyperparameter Configuration (Paper-Aligned)

**Date**: 2026-04-22
**Status**: accepted
**Deciders**: SSPO Reproduction Team

## Context

The SSPO paper provides explicit hyperparameter configurations in Section B (Appendix B, page 16-17). For full reproduction, we must use these exact values.

## Decision

We follow paper hyperparameter configurations exactly as specified in Appendix B.

### B.1 Toy Experiment Hyperparameters (page 16)

| Parameter | Value Range | Description |
|-----------|-------------|-------------|
| n_L | {10, 50, 100} | Number of labeled samples |
| n_U | {100, 500, 1000} | Number of unlabeled samples |
| Learning rate | {1e-5, 5e-6, 1e-6} | Grid search |
| Decay rate γ_decay | {0.001, 0.01, 0.1} | For adaptive scheduler |
| Prior P(s=1) | {0.1, 0.3, 0.5, 0.7, 0.9} | Sensitivity analysis |
| Noise ratio | {0%, 10%, 30%, 50%} | Label corruption |

### B.2 Real Experiment Hyperparameters (page 16-17)

**Model Configuration**:
| Parameter | Value |
|-----------|-------|
| Model | Mistral-7B-Instruct-v0.2 |
| LoRA rank | 8 |
| LoRA target | all |
| Precision | bf16 |

**Training Configuration**:
| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-5 |
| Scheduler | cosine |
| Warmup ratio | 0.1 |
| Epochs | 1 |
| Per device batch size | 4 |
| Gradient accumulation steps | 16 |
| Total batch size | 64 (per GPU) or 512 (8 GPUs) |

**SSPO-Specific Parameters (Table 4)**:
| Parameter | Value |
|-----------|-------|
| γ_0 (initial gamma) | 1.0 |
| γ_min (minimum gamma) | 0.22 |
| γ_decay | 0.001 |
| sspo_prior | 0.5 |
| sspo_base | simpo |

**SimPO Configuration**:
| Parameter | Value |
|-----------|-------|
| β (beta) | 2.0 |
| γ (target margin) | 2.0 |

**Baseline Parameters**:
| Method | Key Parameters |
|--------|---------------|
| DPO | β=0.1, label_smoothing=0.0 |
| SimPO | β=2.0, γ=2.0 |
| ORPO | β=0.1, λ=0.5 |
| KTO | β=0.1, z_bias=0.0 |
| CPO | α (conformal penalty) |

### B.3 Data Configuration (page 7)

| Parameter | Value |
|-----------|-------|
| UltraFeedback total | 61,135 pairs |
| UltraChat total | 200,000 conversations |
| Labeled ratios | {1%, 5%, 10%} |
| Unlabeled ratio | 10% |

## Alternative Configurations Tested (page 16)

### C.1 Alternative Model Sizes
| Model | Size | LoRA rank |
|-------|------|-----------|
| Phi-2 | 3B | 8 |
| Mistral-7B | 7B | 8 |
| Llama-3-8B | 8B | 8 |
| Qwen2-7B | 7B | 8 |

### C.2 Alternative Data Ratios
| Labeled % | Unlabeled % | D_L count | D_U count |
|-----------|-------------|-----------|-----------|
| 1% | 10% | ~611 | 20,000 |
| 5% | 10% | ~3,057 | 20,000 |
| 10% | 10% | ~6,114 | 20,000 |

## Implementation Notes (from trainer.py)

```yaml
# SSPO Configuration in YAML
pref_loss: sspo
pref_beta: 2.0
sspo_gamma_0: 1.0
sspo_gamma_min: 0.22
sspo_gamma_decay: 0.001
sspo_prior: 0.5
sspo_base: simpo
simpo_gamma: 2.0
```

## Alternatives Considered

### Alternative 1: Use Default Hyperparameters
- **Pros**: Faster initial setup
- **Cons**: May not reproduce paper results
- **Why not**: Paper provides exact hyperparameters for reproducibility

### Alternative 2: Different β values
- **Pros**: Could find better alignment
- **Cons**: Not aligned with paper
- **Why not**: Paper shows β=2.0 is optimal for SimPO

## Consequences

### Positive
- Reproducible: Exact paper values enable verification
- Fair comparison: Same hyperparams across baselines
- Complete: Covers all methods in paper

### Negative
- Inflexible: Cannot deviate from paper values
- Complexity: Many hyperparameters to track

### Risks
- **Risk**: Hyperparameter mismatch causes divergence
- **Mitigation**: Strict version control on training configs
