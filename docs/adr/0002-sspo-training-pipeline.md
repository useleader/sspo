# ADR-0002: Use SimPO-based SSPO Loss with Dynamic Gamma Scheduling

**Date**: 2026-04-22
**Status**: accepted
**Deciders**: SSPO Reproduction Team

## Context

The SSPO paper proposes two variants: DPO-based and SimPO-based. The key difference is whether a reference model is used. DPO-based requires keeping a reference model in memory (2x GPU memory), while SimPO-based does not. The paper uses SimPO-based for efficiency. The algorithm uses adaptive gamma scheduling (γ_t) that decays from γ_0 to γ_min during training.

## Decision

We implement SimPO-based SSPO (no reference model) as the default, controlled by `sspo_base: simpo` in training YAML. The SSPO loss combines:
- Labeled data: SimPO/ORPO loss on (chosen, rejected) pairs
- Unlabeled data: Binary classification loss with adaptive threshold τ*

**SSPO Algorithm (from paper Algorithm 1, page 5)**:
```
Input: Labeled data D_L, unlabeled data D_U, initial policy π_0
Output: Trained policy π_θ

1. for each training step t do
2.   Compute policy logps: log π_θ(y|x) for all samples
3.   if use_ref_model:
4.       Compute reference logps: log π_ref(y|x)
5.       Compute reward: r = β * (logps - ref_logps)
6.   else:
7.       Compute reward: r = β * logps (SimPO-based)
8.   
9.   # Adaptive gamma scheduling
10.  γ_t = max(γ_min, γ_0 * exp(-γ_decay * t))
11.  
12.  # Labeled loss (SimPO on preference pairs)
13:  L_labeled = -log σ(β * (log π(y_w) - log π(y_l)))
14.  
15.  # Unlabeled loss (binary classification with threshold)
16:  τ* = min(r_chosen)  # threshold
17:  L_unlabeled = -[s * log σ(r - τ*) + (1-s) * log σ(τ* - r)]
18.  
19:  # Combined loss with curriculum
20:  L_total = γ_t * L_labeled + (1 - γ_t) * L_unlabeled
21.  
22:  Update θ via gradient descent
23. end for
```

## SSPO Loss Implementation (from trainer.py)

```python
def sspo_loss(policy_chosen_logps, policy_rejected_logps, policy_unlabeled_logps):
    # 1. Normalize rewards with moving average
    normalized_logps = (logps - running_mean) / sqrt(running_var + eps)
    
    # 2. Compute threshold τ* = min(chosen logps)
    threshold = torch.min(policy_chosen_logps)
    
    # 3. Labeled loss (SimPO-based)
    logits = beta * (policy_chosen_logps - policy_rejected_logps)
    pn_loss = -F.logsigmoid(logits)
    
    # 4. Unlabeled loss (binary classification)
    diff = beta * (policy_unlabeled_logps - threshold)
    u_loss = sspo_prior * (-F.logsigmoid(diff)) + (1-sspo_prior) * (-F.logsigmoid(-diff))
    
    # 5. Curriculum: gamma decays from γ_0 to γ_min
    gamma_t = max(gamma_min, gamma_0 * exp(-gamma_decay * t))
    sspo_loss = gamma_t * pn_loss + (1 - gamma_t) * u_loss
    
    return sspo_loss
```

## Key Parameters (from paper Table 4, page 9)

| Parameter | Value | Description |
|-----------|-------|-------------|
| β (beta) | 2.0 | Scaling factor for log ratios |
| γ_0 | 1.0 | Initial gamma (100% labeled loss) |
| γ_min | 0.22 | Minimum gamma (curriculum floor) |
| γ_decay | 0.001 | Exponential decay rate |
| sspo_prior | 0.5 | Prior for unlabeled classification |

**Gamma Schedule**: γ_t = max(0.22, 1.0 * exp(-0.001 * t))
- Early training (t=0): γ ≈ 1.0 → focus on labeled data
- Late training (t→∞): γ → 0.22 → include unlabeled data

## Threshold Selection (Reward Threshold Theorem)

The optimal threshold τ* is set to the minimum reward among chosen responses:
```python
threshold = min(reward_chosen)  # τ* = min_i r(x, y_w^(i))
```

This ensures that responses above the weakest "winning" response are considered positive.

## Alternatives Considered

### Alternative 1: DPO-based SSPO
- **Pros**: Theoretically more stable due to reference model regularization
- **Cons**: 2x memory requirement, slower training
- **Why not**: SimPO-based is more practical for 8x H100 setup

### Alternative 2: Static Gamma (no scheduling)
- **Pros**: Simpler implementation
- **Cons**: Does not match paper's curriculum learning design
- **Why not**: Paper Table 4 shows adaptive decay is critical (+1.4 win rate)

### Alternative 3: Fixed threshold τ*
- **Pros**: No need to compute min(chosen)
- **Cons**: Does not adapt to training dynamics
- **Why not**: Paper uses min(chosen) as implicit threshold

## Consequences

### Positive
- Memory efficient: no reference model needed
- Fast training: same speed as SimPO baseline
- Clear baseline: easy comparison with SimPO
- Curriculum learning: labeled → unlabeled transition is smooth

### Negative
- Not fully faithful to paper's iterative pseudo-labeling
- Threshold τ* is implicit (min of chosen logps) rather than explicitly computed

### Risks
- **Risk**: SimPO-based may underperform on some tasks
- **Mitigation**: Support switching to DPO-based via `sspo_base: dpo` config
