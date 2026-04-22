# ADR-0007: Complete Experimental Topology for SSPO Reproduction

**Date**: 2026-04-22
**Status**: accepted
**Deciders**: SSPO Reproduction Team

## Context

The SSPO paper contains 19 tables forming a complete experimental topology (page 8-9). For full paper reproduction, all experimental modules must be supported. The experiments form a logical闭环: Mathematical Proof → Toy → Real-data → Ablation → Engineering.

## Decision

We reproduce all experimental modules in the paper with the following topology (page 8-9):

### Module 1: Mathematical Analysis (Appendix A.1)
| Section | Content | Purpose |
|---------|---------|---------|
| Theorem 1 | Reward Threshold Theorem | Existence of optimal τ* |
| Theorem 2 | Pseudo-label Quality Bound | ε-bound on noise |
| Theorem 3 | Convergence Analysis | Semi-supervised convergence |

### Module 2: Synthetic Toy Experiments (Figure 2, Table 5)
| Experiment | Variables | Purpose |
|------------|-----------|---------|
| Figure 2 | NLTK word length pairing, n_L ∈ {10, 50, 100} | Visualize decision boundary |
| Table 5 | Noise ratio ∈ {0, 10, 30, 50}%, prior ∈ {0.1, 0.3, 0.5, 0.7, 0.9} | Robustness to noise |

**Toy Experiment Setup (Table 5, page 14)**:
- Synthetic data: NLTK word length pairing (short-short, short-long, long-short, long-long)
- Variables: label flip ratio, n_L (10, 50, 100), prior P(s=1) ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
- Key result: 50% noise still converges, 0.5 prior is safest default

### Module 3: Core Performance Benchmarks (Table 1, Table 2)

**Table 1: Main Results on UltraFeedback (page 8)**:
| Method | Mistral 1% | Mistral 10% | Llama3 1% | Llama3 10% | Qwen2 1% | Qwen2 10% |
|--------|------------|-------------|-----------|------------|----------|-----------|
| DPO | 16.6 | 26.2 | 16.5 | 24.5 | 16.3 | 24.3 |
| SimPO | 16.9 | 26.8 | 16.4 | 24.6 | 16.2 | 24.1 |
| ORPO | 16.4 | 25.6 | 16.2 | 24.1 | 16.0 | 23.8 |
| KTO | 16.5 | 25.9 | 16.3 | 24.4 | 16.1 | 24.0 |
| **SSPO** | **26.0** | **32.4** | **25.8** | **31.9** | **25.2** | **31.1** |

**Table 2: Cross-Domain Results (page 8)**:
| Domain | Dataset | 1% DPO | 1% SSPO | Improvement |
|--------|---------|--------|---------|-------------|
| General | UltraFeedback | 16.6 | 26.0 | +56.6% |
| Medical | UltraMedical | 15.8 | 24.3 | +53.8% |
| Business | DSP Business | 15.2 | 23.8 | +56.6% |

### Module 4: Ablation Studies (Table 3, Table 4, Figure 3)

**Table 3: Prior Sensitivity (page 9)**:
| Prior P(s=1) | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|--------------|-----|-----|-----|-----|-----|
| Win Rate | 24.1 | 25.3 | 26.0 | 25.4 | 24.2 |

**Table 4: Scheduler Ablation (page 9)**:
| Scheduler | Win Rate |
|-----------|----------|
| Fixed γ=0.1 | 23.8 |
| Fixed γ=0.5 | 24.6 |
| Adaptive (γ_0=1.0, γ_min=0.22) | **26.0** |

**Figure 3**: Training curve showing γ_t decay from 1.0 → 0.22

### Module 5: SFT Hybrid Comparison (Table 6, page 15)

| Method | Description | Win Rate |
|--------|-------------|----------|
| DPO + SFT | Blindly add D_U as positive | 21.2 |
| SimPO + SFT | Blindly add D_U as positive | 21.8 |
| **SSPO** | **Threshold-based filtering** | **26.0** |

**Key insight**: SSPO's gain is from discrimination, not data exposure

### Module 6: Engineering Analysis (Table 7, page 15)

| Metric | Value |
|--------|-------|
| KDE overhead | O(M²), negligible |
| Single epoch time | ~45 min (8x H100) |
| Throughput | ~500 samples/sec |

### Module 7: Additional Baselines (Table 8, page 16)

| Method | Configuration |
|--------|---------------|
| CPO | α (conformal penalty) sweep |
| CPO-SimPO | Hybrid approach |
| π_{SFT} | SFT baseline |

## Training Infrastructure

Retain from ADR-0005: 8x H100 cluster for all training.

## Alternatives Considered

### Alternative 1: Skip Toy Experiments
- **Pros**: Faster core benchmark reproduction
- **Cons**: Missing theoretical validation
- **Why not**: Paper's mathematical foundation requires verification

### Alternative 2: Skip Ablation Studies
- **Pros**: Focus on final numbers
- **Cons**: Cannot validate algorithm mechanisms
- **Why not**: Ablation proves SSPO's improvement is structural, not accidental

## Consequences

### Positive
- Complete reproduction: All tables and figures reproducible
- Verifiable claims: Each paper assertion has corresponding experiment
- Algorithm understanding: Ablation reveals why SSPO works

### Negative
- Significant compute: Full suite requires extensive GPU time
- Long timeline: Complete experiments span weeks

### Risks
- **Risk**: Compute budget exceeded
- **Mitigation**: Prioritize core benchmarks (Table 1, Table 2) over toy experiments
