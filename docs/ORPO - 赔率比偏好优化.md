# ORPO - 赔率比偏好优化

> **论文**: "ORPO: Monolithic Preference Optimization without a Reference Model"
> **作者**: Hong et al., 2024
> **arXiv**: 2403.18671

## ORPO的核心创新

**单阶段训练**：DPO需要先做SFT warmup再做偏好学习，ORPO将两个阶段合并为单一的端到端训练。

## 核心方法

### 损失函数

ORPO的损失函数由两部分组成：

$$
L_{ORPO} = L_{SFT} + \lambda \cdot L_{OR}
$$

### 1. SFT损失（标准语言建模）

$$
L_{SFT} = -\mathbb{E}_{(x,y) \sim D}[log\ \pi_\theta(y|x)]
$$

保持语言模型的基本生成能力。

### 2. 赔率比损失（Odds Ratio Loss）

**赔率（Odds）定义**：
$$
odds(y|x) = \frac{P(y|x)}{1 - P(y|x)} = exp(log\ P(y|x))
$$

**赔率比（Odds Ratio）**：
$$
OR(x, y_w, y_l) = \frac{odds(y_w|x)}{odds(y_l|x)} = exp(log\ \pi_\theta(y_w|x) - log\ \pi_\theta(y_l|x))
$$

**OR损失**：
$$
L_{OR} = -\mathbb{E}_{(x, y_w, y_l)}[log\ \sigma(log\ OR(x, y_w, y_l))]
$$

其中 $\sigma$ 是sigmoid函数。

## ORPO vs DPO

| 方面 | DPO | ORPO |
|------|-----|------|
| 训练阶段 | 2阶段（SFT+DPO） | **单阶段** |
| 参考模型 | 需要 | 不需要 |
| 损失函数 | 单一DPO loss | SFT loss + OR loss |
| odds ratio | 隐式使用 | **显式使用** |

## 关键设计

### 为什么单阶段？

**DPO的问题**：
1. 先SFT会让模型偏向某个分布
2. 两阶段训练复杂，需要协调两个阶段的数据
3. 两阶段可能导致suboptimal

**ORPO的解决**：
- 在同一个训练过程中同时优化：
  - 生成质量（SFT loss）
  - 偏好对齐（OR loss）

### 赔率比的直观理解

```
odds(y|x) = P(y|x) / (1 - P(y|x))

odds ratio = odds(y_w|x) / odds(y_l|x)
           = P(y_w|x) / (1-P(y_w|x)) ÷ P(y_l|x) / (1-P(y_l|x))
```

- 当 $P(y_w|x) > P(y_l|x)$，odds ratio > 1
- OR loss 最大化这个比值，即让preferred response的概率相对更高

## 实现要点

```python
def orpo_loss(log_probs_chosen, log_probs_rejected, lambda_or=0.1):
    """
    ORPO损失计算
    log_probs_chosen: [batch] chosen response的log prob
    log_probs_rejected: [batch] rejected response的log prob
    """
    # SFT loss (负对数似然)
    sft_loss = -log_probs_chosen.mean()
    
    # Odds Ratio loss
    odds_chosen = torch.exp(log_probs_chosen)
    odds_rejected = torch.exp(log_probs_rejected)
    odds_ratio = odds_chosen / odds_rejected
    or_loss = -torch.log(torch.sigmoid(torch.log(odds_ratio))).mean()
    
    # 总损失
    return sft_loss + lambda_or * or_loss
```

## ORPO的优势

1. **训练更简单**：单阶段，不需要调整SFT和DPO的相对权重
2. **不需要参考模型**：节省内存和计算
3. **避免两阶段suboptimal**：端到端优化

## 参考文献

- Hong et al., "ORPO: Monolithic Preference Optimization without a Reference Model" (2024)
