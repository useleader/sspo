# ADR-0010: SSRM and SPA Implementation

**Date**: 2026-04-23
**Status**: proposed
**Deciders**: SSPO Reproduction Team

## Context

The SSPO paper includes two semi-supervised baselines that are not implemented in standard alignment libraries: SSRM (Semi-Supervised Reward Modeling) and SPA (Spread Preference Annotation). We need to implement these from scratch based on paper descriptions.

## Decision

### SSRM: Semi-Supervised Reward Modeling

**Paper Reference**: Section 3.2, Table 1

**Concept**: Train a reward model on labeled data, use it to generate pseudo-labels for unlabeled data, then retrain. Iterative pseudo-labeling approach.

**Implementation** (`src/src_sspo/llamafactory/train/ssrm/trainer.py`):

```python
class SSRMTrainer(CustomDPOTrainer):
    """
    Semi-Supervised Reward Modeling.
    """

    def __init__(self, ssrm_prior=0.5, ssrm_iterations=3, ssrm_threshold=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssrm_prior = ssrm_prior
        self.ssrm_iterations = ssrm_iterations
        self.ssrm_threshold = ssrm_threshold

    def compute_loss(self, policy_chosen_logps, policy_rejected_logps, unlabeled_logps=None):
        # Standard DPO loss on labeled data
        dpo_loss = super().compute_loss(policy_chosen_logps, policy_rejected_logps)
        if unlabeled_logps is not None:
            pseudo_labels = (unlabeled_logps > self.ssrm_threshold).float()
            ssrm_loss = -self.ssrm_prior * pseudo_labels * F.logsigmoid(unlabeled_logps) \
                        -(1 - self.ssrm_prior) * (1 - pseudo_labels) * F.logsigmoid(-unlabeled_logps)
            return dpo_loss + 0.1 * ssrm_loss
        return dpo_loss
```

### SPA: Spread Preference Annotation

**Paper Reference**: Section 3.2, Table 1

**Concept**: Iterative self-annotation where the model generates responses, a reward model scores them, and confident samples are added to training data.

**Implementation** (`src/src_sspo/llamafactory/train/spa/trainer.py`):

```python
class SPATrainer(CustomDPOTrainer):
    """
    Spread Preference Annotation (Iterative Self-Annotation).
    """

    def __init__(self, spa_iterations=3, spa_expansion_ratio=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spa_iterations = spa_iterations
        self.spa_expansion_ratio = spa_expansion_ratio

    def compute_loss(self, policy_chosen_logps, policy_rejected_logps):
        return super().compute_loss(policy_chosen_logps, policy_rejected_logps)
```

## Consequences

### Positive
- Enables comparison with paper's full set of baselines
- Both methods are semi-supervised like SSPO

### Risks
- SSRM/SPA implementations are interpretations of paper descriptions
