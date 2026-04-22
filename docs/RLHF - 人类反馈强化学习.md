# RLHF - 人类反馈强化学习

> **论文**: "Training language models to follow instructions with human feedback" (Ouyang et al., 2022) - InstructGPT

## RLHF三阶段

```
Pretrained LLM → SFT → Reward Modeling → PPO Fine-tuning → Aligned LLM
```

### Stage 1: Supervised Fine-Tuning (SFT)

使用人工标注的demonstration数据进行监督学习。

$$
L_{SFT} = -\mathbb{E}_{(x,y) \sim D}[log\ \pi_\theta(y|x)]
$$

**目的**: 初始化策略模型，使其具备基本对话能力

### Stage 2: Reward Modeling

训练一个奖励模型 $r_\phi(x, y)$ 来预测人类偏好。

**训练数据**: 人类对同一prompt下不同response的偏好排序

**Bradley-Terry模型**: 偏好概率可以表示为：

$$
P(y_w \succ y_l | x) = \frac{exp(r(x, y_w))}{exp(r(x, y_w)) + exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))
$$

**损失函数**:

$$
L_R = -\mathbb{E}_{(x, y_w, y_l) \sim D}[log\ \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]
$$

### Stage 3: RL Fine-tuning with PPO

使用强化学习（PPO算法）优化策略模型。

**目标函数**:

$$
L_{PPO} = -\mathbb{E}_{x \sim D, y \sim \pi_\theta}[r_\phi(x, y)] + \beta \cdot D_{KL}(\pi_\theta || \pi_{SFT})
$$

其中:
- $r_\phi(x, y)$: 奖励模型给出的奖励值
- $\pi_{SFT}$: SFT阶段的策略（作为参考）
- $\beta$: KL惩罚系数，控制偏离程度
- $D_{KL}$: KL散度，限制策略不要偏离参考太远

**PPO关键更新**:

$$
L^{CLIP} = -\mathbb{E}[min(r_t(\theta) \cdot A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t)]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ 是概率比值。

## RLHF的问题

### 1. 训练复杂性

- 需要同时训练多个模型（Reward Model + Policy + Reference）
- PPO的超参数调优困难
- 训练不稳定

### 2. 人类标注成本

- 需要大量配对偏好标注
- 标注质量受标注者主观影响
- 扩展性差

### 3. 分布控制困难

- 过度优化导致奖励黑客（reward hacking）
- KL散度难以精确控制

## 为什么需要替代方法？

```
问题: PPO训练复杂 + 人工标注成本高
解决方向: 简化训练流程 + 减少标注需求
```

## 后续发展

| 方向 | 代表方法 | 核心思想 |
|------|----------|----------|
| 简化RL | DPO, SimPO | 去除PPO，直接优化 |
| 减少标注 | RLAIF, SSPO | AI反馈/半监督 |
| 理论改进 | ORPO, KTO | 新的优化目标 |

## 参考

- Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
