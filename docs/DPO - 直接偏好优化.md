# DPO - 直接偏好优化

> **论文**: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" 
> **作者**: Rafael et al., 2023
> **arXiv**: 2305.10425

## DPO的核心思想

DPO通过数学推导，将RLHF中的强化学习问题转化为**直接策略优化问题**，无需显式的奖励模型和PPO训练。

## 核心推导

### 从RLHF到DPO

**RLHF目标（含奖励函数）**:
$$
\max_\pi \mathbb{E}_{x \sim D, y \sim \pi}[r(x, y)] - \beta \cdot D_{KL}(\pi || \pi_{ref})
$$

### 关键洞察：奖励函数的闭式解

对于给定的最优策略 $\pi^*$，可以解析地求出奖励函数：

$$
r(x, y) = \beta \cdot \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + const
$$

这意味着我们**不需要单独训练奖励模型**，可以直接从偏好数据学习策略。

### DPO损失函数

**最终目标**:
$$
L_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim D}[log\ \sigma(\beta \cdot (log\ \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - log\ \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))]
$$

**简化形式**:
$$
L_{DPO} = -\mathbb{E}_{(x, y_w, y_l)}[log\ \sigma(r_\beta(x, y_w) - r_\beta(x, y_l))]
$$

其中:
$$
r_\beta(x, y) = \beta \cdot (log\ \pi_\theta(y|x) - log\ \pi_{ref}(y|x))
$$

## DPO vs RLHF

| 方面 | RLHF | DPO |
|------|------|-----|
| 奖励模型 | 需要单独训练 | 不需要 |
| 训练算法 | PPO | 直接梯度下降 |
| 参考模型 | 仅用于KL约束 | 用于计算log ratio |
| 训练稳定性 | 较难 | 相对稳定 |
| 实现复杂度 | 高 | 低 |

## DPO的工作原理

### 为什么DPO有效？

1. **隐式奖励**: 虽然没有显式奖励函数，但 $\beta \cdot log(\pi_\theta/\pi_{ref})$ 起到奖励作用
2. **对比学习**: 最大化正负样本的log ratio差
3. **KL约束**: 自动通过参考模型实现，避免过度优化

### 隐式奖励的直观理解

```
r_β(x, y) = β · (log π(y|x) - log π_ref(y|x))
         = β · log (π(y|x) / π_ref(y|x))
```

- 当 $\pi(y|x) > \pi_{ref}(y|x)$ → 正奖励
- 当 $\pi(y|x) < \pi_{ref}(y|x)$ → 负奖励

## DPO的局限性

### 1. 需要参考模型

- 训练全程需要维护参考模型 $\pi_{ref}$
- 内存和计算开销大
- 推理时仍需要参考模型（有时）

### 2. 偏向参考模型

- 如果 $\beta$ 设置过大，策略会接近参考模型
- 如果 $\beta$ 设置过小，KL约束失效

### 3. 分布偏移

- 长文本生成时，token-level的log ratio可能累积
- 可能导致生成长度偏差

## 代码实现要点

```python
# DPO Loss核心计算
def dpo_loss(policy_logps, ref_logps, yw_logps, yl_logps, beta=0.1):
    """
    policy_logps: 策略模型对chosen/rejected的log probs
    ref_logps: 参考模型对chosen/rejected的log probs
    """
    # 计算隐式奖励
    chosen_reward = beta * (yw_logps - ref_logps[:, 0])
    rejected_reward = beta * (yl_logps - ref_logps[:, 1])
    
    # Hinge loss
    loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward))
    return loss.mean()
```

## 参考文献

- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
