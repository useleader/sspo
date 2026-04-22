# ADR-0001: Use LLaMA-Factory as Base Framework

**Date**: 2026-04-22
**Status**: accepted
**Deciders**: SSPO Reproduction Team

## Context

SSPO (Semi-Supervised Preference Optimization) is an ICLR 2026 paper requiring LLM alignment training. Implementing SSPO from scratch requires handling model loading, LoRA/peft integration, data collation, training loops, checkpointing, and distributed training — all complex components that are error-prone to implement correctly. The external network is unreliable, making it difficult to download and verify large models.

## Decision

We use LLaMA-Factory as the base framework for SSPO implementation, extending it with a custom `CustomDPOTrainer` in `src_sspo/llamafactory/train/dpo/trainer.py`.

## Alternatives Considered

### Alternative 1: Implement Everything from Scratch
- **Pros**: Full control over every component, no framework lock-in
- **Cons**: Massive engineering effort, high risk of bugs, must reimplement DPO/SimPO/KTO baselines
- **Why not**: SSPO reproduction focuses on algorithm validation, not infrastructure

### Alternative 2: Use TRL Library Only (trl.DPOTrainer)
- **Pros**: Lightweight, direct access to DPOTrainer
- **Cons**: Must reimplement data loading, checkpointing, YAML config, CLI, evaluation pipeline
- **Why not**: LLaMA-Factory already provides all these with HuggingFace integration

## Consequences

### Positive
- Fast iteration: leverage existing LoRA, data loading, and training infrastructure
- Baseline compatibility: easy comparison with DPO, SimPO, ORPO, KTO using same framework
- Production-tested: LLaMA-Factory is widely used and maintained

### Negative
- Framework coupling: changes to LLaMA-Factory upstream may require updates
- Code complexity: forked/customized trainer adds indirection

### Risks
- **Risk**: LLaMA-Factory version incompatibility
- **Mitigation**: Pin transformers==4.46.1 and trl==0.9.6 in requirements.txt
