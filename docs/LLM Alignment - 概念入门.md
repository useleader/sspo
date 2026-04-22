# LLM Alignment - 概念入门

## 什么是LLM Alignment（对齐）？

LLM Alignment 是让大语言模型(LLM)的输出与人类**意图和价值观**保持一致的过程。

### 为什么需要对弈？

| 能力 | 对齐前 | 对齐后 |
|------|--------|--------|
| 遵循指令 | 可能偏离 | 精准执行 |
| 安全性 | 可能输出有害内容 | 拒绝有害请求 |
| 有用性 | 回答可能无用 | 提供有帮助的回答 |
| 诚实性 | 可能编造信息 | 尽可能准确 |

## 对齐训练流程（标准RLHF）

```
Pretrained LLM → SFT(监督微调) → Reward Model → RL(PPO) → Aligned LLM
```

### 1. SFT (Supervised Fine-Tuning)

使用人类标注的问答对进行监督学习。

```
损失函数: L_SFT = -E_{(x,y)∼D} [log π(y|x)]
```

### 2. Reward Modeling

训练一个奖励模型来预测人类偏好。

$$
r_\theta(x, y) = \text{Reward Model}(x, y)
$$

### 3. RL Fine-tuning (PPO)

使用强化学习优化策略模型。

$$
L_{PPO} = -E_{x\sim D, y\sim\pi}[r_\theta(x,y)] - \beta \cdot D_{KL}(\pi || \pi_{SFT})
$$

## 偏好学习的核心问题

### 传统方法的局限

1. **需要大量人工标注** - 配对偏好数据标注成本高
2. **PPO训练复杂** - 需要平衡reward和KL散度
3. **参考模型开销** - DPO需要持续维护参考模型

### 核心挑战

- **数据效率**: 如何用更少的人类反馈学习？
- **分布漂移**: 如何避免过度优化导致分布偏移？
- **伪标签质量**: 如何确保自动生成的标签可靠？

## 对齐方法演进

| 方法 | 年份 | 核心创新 | 参考模型 |
|------|------|----------|----------|
| RLHF/PPO | 2022 | 强化学习框架 | 否 |
| DPO | 2023 | 直接优化策略 | 是 |
| SimPO | 2024 | 去除参考模型 | 否 |
| ORPO | 2024 | 单阶段训练 | 否 |
| KTO | 2024 | 行为经济学 | 否 |
| **SSPO** | **2026** | **半监督学习** | **否** |

## 参考文献

- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022)
- Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback" (2022)
