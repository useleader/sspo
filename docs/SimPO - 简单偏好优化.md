# SimPO - 简单偏好优化

> **论文**: "SimPO: Simple Preference Optimization with a Reference-Free Reward"
> **作者**: Ji et al., 2024
> **arXiv**: 2405.14734

## SimPO的核心创新

**去除参考模型**：DPO需要参考模型计算log ratio，SimPO直接使用策略模型的log probability，不需要参考模型。

## 核心方法

### 隐式奖励定义

SimPO定义了一个**不依赖参考模型**的隐式奖励：

$$
r(x, y) = \frac{\beta}{|y|} \cdot \sum_{i=1}^{|y|} log\ \pi_\theta(y_i|x_{<i})
$$

其中 $\beta$ 是温度参数，$|y|$ 是序列长度（用于长度归一化）。

### 偏好概率

使用Bradley-Terry模型：

$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$

### SimPO损失函数

$$
L_{SimPO} = -\mathbb{E}_{(x, y_w, y_l)}[log\ \sigma(r(x, y_w) - r(x, y_l) - \gamma)]
$$

其中 $\gamma > 0$ 是一个**目标奖励margin**参数。

## SimPO vs DPO

| 方面 | DPO | SimPO |
|------|-----|-------|
| 参考模型 | 需要 | **不需要** |
| 奖励定义 | $\beta \cdot log(\pi/\pi_{ref})$ | 直接用 $\beta \cdot log\ \pi$ |
| 长度处理 | 无归一化 | **长度归一化** |
| 目标margin | 无 | 有（$\gamma$参数） |

## 关键设计

### 1. 长度归一化

DPO的reward没有考虑序列长度，导致模型可能偏向生成短文本（因为短文本的log probability和更大）。

SimPO通过**除以序列长度**来解决这个问题。

### 2. 目标Margin

$\gamma$ 参数强制正负样本之间有一个最小的reward差距。

**直觉**：有margin的要求让模型更明确地学习区分正负样本。

## 为什么SimPO有效？

### 理论分析

DPO中，隐式奖励是：
$$
r_{DPO}(x, y) = \beta \cdot (log\ \pi_\theta(y|x) - log\ \pi_{ref}(y|x))
$$

SimPO中：
$$
r_{SimPO}(x, y) = \beta \cdot log\ \pi_\theta(y|x) / |y|
$$

**关键区别**：
- DPO使用**log ratio**（相对于参考）
- SimPO直接使用**log probability**，但做了长度归一化

### 隐式对齐

虽然SimPO没有参考模型，但通过**长度归一化的log probability**，隐式地实现了与参考模型类似的对齐效果。

## 实验结果

论文显示SimPO在多个基准上与DPO相当或更好，同时：
- 训练速度更快（无需参考模型前向传播）
- 推理时不需要参考模型
- 对超参数（$\beta$, $\gamma$）更鲁棒

## 实现要点

```python
def simpo_reward(log_probs, beta=2.0):
    """计算SimPO隐式奖励（长度归一化）"""
    # log_probs: [batch, seq_len] 每个token的log prob
    # 返回: [batch] 每条序列的奖励
    return beta * log_probs.mean(dim=-1)  # 长度归一化

def simpo_loss(chosen_rewards, rejected_rewards, gamma=1.0):
    """SimPO损失函数"""
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards - gamma)).mean()
```

## 参考文献

- Ji et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward" (2024)
