# ADR-0003: UltraFeedback + UltraChat Combined Dataset

**Date**: 2026-04-22
**Status**: accepted
**Deciders**: SSPO Reproduction Team

## Context

SSPO requires two types of data: (1) labeled preference pairs D_L with (prompt, chosen, rejected), and (2) unlabeled SFT data D_U with (prompt, response). UltraFeedback provides 64k preference pairs, UltraChat provides 200k SFT conversations. Paper experiments use 1%, 5%, 10% labeled data ratios with 10% UltraChat usage.

## Decision

We use `preprocessing_data/preprocessing_ultrachat.py` to:
1. Sample fb% of UltraFeedback as labeled data (keeping chosen/rejected)
2. Sample ch% of UltraChat as unlabeled data (keeping messages as unlabeled)
3. Combine into single JSON with fields: instruction, chosen, rejected, unlabeled
4. Register in `data/dataset_info.json` for LLaMA-Factory compatibility

**Paper-matched ratios (from Table 1, Table 2)**:
- Labeled data (UltraFeedback): fb ∈ {0.01, 0.05, 0.10} → 1%, 5%, 10%
- Unlabeled data (UltraChat): ch = 0.10 → 10% of UltraChat

**Dataset sizes (from paper page 7)**:
| Dataset | Total Size | 1% Labeled | 10% Labeled |
|---------|------------|------------|-------------|
| UltraFeedback | 61,135 pairs | ~611 pairs | ~6,114 pairs |
| UltraChat | 200,000 convos | 20,000 unlabeled | 20,000 unlabeled |

**Data compositions for experiments (from paper Table 1)**:
| Experiment | Labeled (fb) | Unlabeled (ch) | Total Samples |
|------------|--------------|----------------|---------------|
| 1% Labeled | 1% | 10% | ~611 D_L + 20,000 D_U |
| 5% Labeled | 5% | 10% | ~3,057 D_L + 20,000 D_U |
| 10% Labeled | 10% | 10% | ~6,114 D_L + 20,000 D_U |

## Cross-Domain Datasets (from Table 2, Table 3)

| Domain | Dataset | Source |
|--------|---------|--------|
| General | UltraFeedback | HuggingFaceH4/ultrafeedback_binarized |
| Medical | UltraMedical | Custom medical preference dataset |
| Business | DSP Business | Custom business preference dataset |

## Alternatives Considered

### Alternative 1: Use Only UltraFeedback with Masked Labels
- **Pros**: Single dataset, simpler preprocessing
- **Cons**: Does not match paper's unlabeled data setup
- **Why not**: Paper specifically uses UltraChat as unlabeled data source

### Alternative 2: Different Unlabeled Data Ratio
- **Pros**: Could reduce compute
- **Cons**: Does not match paper configuration
- **Why not**: Paper uses ch=0.10 (10% of UltraChat)

## Consequences

### Positive
- Matches paper exactly: same datasets, same mixing strategy
- Paper ratios: 1%, 5%, 10% labeled experiments fully supported
- Cross-domain: Medical and Business datasets included

### Negative
- External dependency: requires downloading from HuggingFace
- Preprocessing time: combining datasets takes ~10 minutes

### Risks
- **Risk**: HuggingFace download fails on unreliable network
- **Mitigation**: Implement caching and resume support in download scripts
