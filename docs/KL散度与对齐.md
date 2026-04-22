# KL散度与对齐

## KL散度的定义

**KL散度（Kullback-Leibler Divergence）**：衡量两个概率分布 $P$ 和 $Q$ 之间的差异。

$$
D_{KL}(P || Q) = \sum_x P(x) log \frac{P(x)}{Q(x)} = \mathbb{E}_P[log \frac{P(x)}{Q(x)}]
$$

### 性质

1. **非负性**: $D_{KL}(P || Q) \geq 0$
2. **非对称性**: $D_{KL}(P || Q) \neq D_{KL}(Q || P)$
3. **当P=Q时**: $D_{KL}(P || P) = 0$

## 在LLM对齐中的作用

### 控制分布偏移

对齐优化的核心挑战：在提升性能的同时，避免模型偏离原始分布太远。

```
原始分布: π_ref (SFT后的模型)
目标分布: π* (对齐后的模型)

我们希望: π* 接近 π_ref，但性能更好
```

### KL散度惩罚

RLHF和DPO都使用KL散度作为正则项：

**RLHF目标**:
$$
\max_\pi \mathbb{E}_{x,y}[r(x,y)] - \beta \cdot D_{KL}(\pi || \pi_{ref})
$$

**DPO中的隐式KL**:
$$
D_{KL}(\pi_\theta(y|x) || \pi_{ref}(y|x))
$$

## 为什么KL散度重要？

### 1. 防止过度优化

 بدونKL约束，模型可能：
- 过度迎合reward信号
- 产生重复或刻板的输出
- 失去多样性

### 2. 保持有用性

KL散度确保：
- 模型仍然能够生成多样化的响应
- 不会完全偏离原始能力
- 保持一定的"安全性"

### 3. 理论保证

KL散度提供了分布偏移的**可量化度量**。

## KL散度在偏好优化中的形式

### Token-level KL

对于序列生成：

$$
D_{KL}(\pi || \pi_{ref}) = \sum_{t=1}^{T} \pi(y_t|x_{<t}) log \frac{\pi(y_t|x_{<t})}{\pi_{ref}(y_t|x_{<t})}
$$

### Sequence-level KL

对于完整序列：

$$
D_{KL}(\pi(y|x) || \pi_{ref}(y|x)) = log\ \pi(y|x) - log\ \pi_{ref}(y|x)
$$

## 不同方法对KL的处理

### DPO

- **显式使用**：通过 $\beta \cdot log(\pi/\pi_{ref})$ 计算隐式奖励
- **需要参考模型**：全程维护 $\pi_{ref}$
- **KL约束强度**：通过 $\beta$ 参数控制

### SimPO

- **隐式处理**：通过长度归一化的log probability
- **不需要参考模型**
- **理论上仍有KL约束效果**

### ORPO

- **SFT loss保持**：确保生成分布接近参考
- **不需要显式KL项**

### KTO

- **无参考模型**
- **通过对称损失隐式控制**

## KL散度与训练稳定性

### 过大β的问题

$$
L = L_{reward} + \beta \cdot D_{KL}
$$

- 如果 $\beta$ 太大 → 模型太保守，不敢偏离参考
- 如果 $\beta$ 太小 → 可能过度优化

### 实际选择

| 任务 | β范围 |
|------|-------|
| DPO | 0.1 - 0.3 |
| SimPO | 1.0 - 2.0 (implicit) |
| SSPO | 待确定 |

## KL散度的计算

### 实战代码

```python
def compute_kl_divergence(log_probs, ref_log_probs):
    """
    计算token-level KL散度
    log_probs: [batch, seq_len]
    ref_log_probs: [batch, seq_len]
    """
    kl = torch.exp(log_probs) * (log_probs - ref_log_probs)
    return kl.sum(dim=-1).mean()

# 或者使用transformers内置
from transformers import TrainerCallback

class KLCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            kl = compute_kl_divergence(model_logps, ref_logps)
            print(f"KL divergence: {kl:.4f}")
```

## 参考

- KL散度原始论文: Kullback & Leibler (1951)
- On Information Sufficiency (Cover & Thomas, 2006)
