# KTO - 卡尼曼-特沃斯基优化

> **论文**: "KTO: Kahneman-Tversky Optimization"
> **作者**: Ethayarajh et al., 2024
> **arXiv**: 2409.XXXXX

## KTO的核心创新

**行为经济学视角**：KTO基于Kahneman和Tversky的前景理论（Prospect Theory），引入**损失厌恶**概念，不要求成对偏好数据。

## 前景理论基础

### Kahneman-Tversky的前景理论

人类对**损失**比对等量**收益**更敏感（损失厌恶）。

**价值函数**：
$$
v(x) = \begin{cases}
x^\alpha & \text{if } x \geq 0 \text{ (gain)} \\
-\lambda \cdot (-x)^\beta & \text{if } x < 0 \text{ (loss)}
\end{cases}
$$

其中 $\lambda > 1$ 是损失厌恶系数（通常 $\lambda \approx 2.25$）。

### 在偏好学习中的应用

**关键洞察**：人类评估一个response不是通过比较pair，而是作为**接受 vs 拒绝**的二元判断。

- 接受的response → gain
- 拒绝的response → loss

## KTO损失函数

### 损失厌恶术语

给定prompt $x$和response $y$：
- **Desired (D)**: 人类会接受这个response
- **Undesired (U)**: 人类会拒绝这个response

### KTO损失

$$
L_{KTO} = -\mathbb{E}_{(x,y,D) \sim D}[(1 - \alpha) \cdot log\ \sigma(\beta \cdot D_{KL}(y||\pi) - \mu_D) + \alpha \cdot log\ \sigma(-\beta \cdot D_{KL}(y||\pi) - \mu_U)]
$$

其中：
- $D_{KL}(y||\pi) = log\ \pi(y|x) - log\ \pi_{ref}(y|x)$（或直接用log prob）
- $\alpha$: Undesired样本的权重（通常是0.1-0.2）
- $\mu_D, \mu_U$: 相对频率因子

### 简化形式

核心思想是：对于desired样本推动策略向其移动，对于undesired样本则远离。

## KTO vs DPO

| 方面 | DPO | KTO |
|------|-----|-----|
| 数据格式 | 成对偏好 $(y_w, y_l)$ | **单样本** $(y, D/U)$ |
| 参考模型 | 需要 | 不需要 |
| 理论基础 | 信息论/BT模型 | **前景理论** |
| 损失函数 | 对比pair | **对称损失** |
| 损失厌恶 | 无 | **有**（$\lambda$系数） |

## 关键设计

### 1. 不需要成对数据

DPO需要同一个prompt的两个response比较：
$$
(x, y_w, y_l)
$$

KTO只需要：
$$
(x, y, \text{label})
$$

这大大降低了数据标注成本。

### 2. 损失厌恶特性

**对称损失**：既推动远离undesired，也推动靠近desired。

这与人类决策心理学一致：人们同时考虑"什么是坏的"和"什么是好的"。

### 3. 无参考模型

KTO直接使用策略模型的log probability，不需要参考模型。

## 实现要点

```python
def kto_loss(log_probs, labels, beta=0.1, alpha=0.2):
    """
    KTO损失计算
    log_probs: [batch] 每条序列的log prob
    labels: [batch] 1=desired, 0=undesired
    """
    # KL散度项（简化版：直接用log prob）
    kl_term = beta * log_probs
    
    # 对desired样本：最大化KL（推向正样本）
    # 对undesired样本：最小化KL（远离负样本）
    loss_desired = -torch.log(torch.sigmoid(kl_term))
    loss_undesired = -torch.log(torch.sigmoid(-kl_term))
    
    # 加权组合
    loss = (labels * loss_desired + (1 - labels) * alpha * loss_undesired).mean()
    return loss
```

## KTO的优势

1. **数据效率**：不需要成对标注，单样本即可
2. **理论基础**：基于真实的人类决策理论
3. **避免KL失衡**：对称损失避免过度优化

## 参考文献

- Ethayarajh et al., "KTO: Kahneman-Tversky Optimization" (2024)
- Kahneman & Tversky, "Prospect Theory: An Analysis of Decision under Risk" (1979)
