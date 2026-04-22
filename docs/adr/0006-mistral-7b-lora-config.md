# ADR-0006: Support All Three Paper Models with LoRA Configuration

**Date**: 2026-04-22
**Status**: accepted
**Deciders**: SSPO Reproduction Team

## Context

The SSPO paper evaluates on three models: Mistral-7B-Instruct, Llama-3-8B-Instruct, and Qwen2-7B-Instruct. For full paper reproduction, all three models need to be supported with consistent LoRA configuration (rank=8).

## Decision

We support all three models from the paper with identical LoRA configuration:

| Model | HuggingFace ID | Tokenizer Trust |
|-------|---------------|----------------|
| Mistral-7B-Instruct | `mistralai/Mistral-7B-Instruct-v0.2` | True |
| Llama-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | True |
| Qwen2-7B-Instruct | `Qwen/Qwen2-7B-Instruct` | True |

**LoRA Configuration (from paper page 7, Table 1)**:
| Parameter | Value | Source |
|-----------|-------|--------|
| LoRA rank | 8 | Paper specification |
| LoRA target | all | All linear layers |
| Context length | 1024 | Paper specification |
| bf16 | yes | Mixed precision training |
| Learning rate | 1e-5 | Paper page 7 |
| Scheduler | cosine | Paper page 7 |
| Epochs | 1 | Real experiments (page 7) |

**Training Configuration (from paper page 7)**:
- Per device batch size: 4 (single GPU) or 8 (multi-GPU)
- Gradient accumulation: 16
- Total batch size: 64
- Warmup ratio: 0.1

## Training Order (from Table 1)

1. **Phase 1**: Mistral-7B-Instruct (primary model, first reproduction)
2. **Phase 2**: Llama-3-8B-Instruct (reproduce paper results)
3. **Phase 3**: Qwen2-7B-Instruct (reproduce paper results)

## Results from Paper (Table 1) - Win Rate on UltraFeedback

| Method | Mistral 1% | Mistral 10% | Llama3 1% | Llama3 10% | Qwen2 1% | Qwen2 10% |
|--------|------------|-------------|-----------|------------|----------|-----------|
| DPO | 16.6 | 26.2 | 16.5 | 24.5 | 16.3 | 24.3 |
| SimPO | 16.9 | 26.8 | 16.4 | 24.6 | 16.2 | 24.1 |
| ORPO | 16.4 | 25.6 | 16.2 | 24.1 | 16.0 | 23.8 |
| KTO | 16.5 | 25.9 | 16.3 | 24.4 | 16.1 | 24.0 |
| **SSPO** | **26.0** | **32.4** | **25.8** | **31.9** | **25.2** | **31.1** |

**Key Claim**: 1% SSPO matches or exceeds 10% DPO/SimPO

## Alternatives Considered

### Alternative 1: Mistral Only
- **Pros**: Faster initial reproduction
- **Cons**: Does not fully verify paper's claims across model sizes
- **Why not**: Paper shows results across 3 different model families

### Alternative 2: Different LoRA Ranks per Model
- **Pros**: Could optimize per model size
- **Cons**: Inconsistency with paper
- **Why not**: Paper uses rank=8 for all models

## Consequences

### Positive
- Paper alignment: All models enable direct comparison with paper results
- Consistent configuration: Easy comparison between models
- Complete reproduction: Verifies paper's claims across model families

### Negative
- Three times the training compute required
- Model downloading and caching complexity

### Risks
- **Risk**: Llama-3 requires HuggingFace access token
- **Mitigation**: Set HF_TOKEN environment variable
