# SSPO 论文精读

> **论文**: Semi-Supervised Preference Optimization with Limited Feedback
> **作者**: Seonggyun Lee, Sungjun Lim, Seojin Park, Soeun Cheon, Kyungwoo Song (Yonsei University & KAIST)
> **发表**: ICLR 2026
> **arXiv**: 2511.00040v3
> **GitHub**: https://github.com/MLAI-Yonsei/SSPO

## 核心问题

**现有方法的痛点**：DPO、SimPO等方法虽然简化了RLHF，但仍然需要大量**成对偏好标注数据**。

- 标注成本：每条偏好数据需要人类比较两个response
- 数据需求：通常需要数万到数百万条
- 扩展困难：随着模型变大，标注需求增加

## SSPO的核心思想

**半监督学习**：同时利用
1. **少量标注数据**：成对偏好 $(x, y_w, y_l)$
2. **大量无标注数据**：SFT数据池，无偏好标签

## 问题定义

### 标准偏好优化

给定少量成对偏好数据：
$$
D_L = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N
$$

学习策略 $\pi$ 最大化偏好似然。

### SSPO的设置

额外给定大量无配对数据：
$$
D_U = \{x^{(j)}, y^{(j)}\}_{j=1}^M, \quad M >> N
$$

这些数据**没有偏好标签**，但可能包含隐式偏好信号。

## 核心方法

### 1. 概率二分类视角

SSPO将偏好学习重新定义为**二分类问题**：

$$
P_\theta(y \succ x) = \sigma(f_\theta(x, y))
$$

其中 $f_\theta$ 是策略模型定义的奖励函数，$\sigma$ 是sigmoid。

### 2. 最优奖励阈值

**关键定理（Reward Threshold Theorem）**：

存在一个最优阈值 $\tau^*$，使得：
$$
P(y_w \succ y_l | x) \approx \mathbb{1}[f(x, y_w) - f(x, y_l) > \tau^*]
$$

直观理解：
- 当两个response的reward差超过阈值时，偏好是确定的
- 阈值附近的样本是"模糊"的，需要更谨慎处理

### 3. 伪标签生成

**核心创新**：利用阈值对无配对数据进行伪标签。

对于无配对数据 $(x, y) \in D_U$：
1. 计算reward: $r = f(x, y)$
2. 计算pseudo-label: $\hat{y} = \mathbb{1}[r > \tau^*]$

**关键洞察**：即使是无配对的SFT数据，也存在winning和losing responses。SSPO证明可以可靠地识别它们。

### 4. 联合优化

最终的优化目标结合：
- **标注数据**：标准偏好损失
- **伪标签数据**：二分类交叉熵损失

$$
L_{SSPO} = L_{labeled} + \lambda \cdot L_{pseudo}
$$

## 算法流程

```
输入: 少量标注数据 D_L, 大量无标注数据 D_U, 初始策略 π_0

1. Warm-up: 用 D_L 训练基础偏好模型
2. For each iteration:
   a. 计算无标注数据的reward: r(x,y) for (x,y) ∈ D_U
   b. 确定阈值 τ (基于理论或验证集)
   c. 生成伪标签: ŷ = 1[r(x,y) > τ]
   d. 用 D_L ∪ {(x,y,ŷ)} 联合训练策略
   e. 更新策略模型
3. 输出: 对齐后的策略 π*
```

## 关键结果

### 惊人的数据效率

> **1% UltraFeedback + SSPO ≈ 10% UltraFeedback + DPO**

在Mistral-7B-Instruct上：
- SSPO使用1%标注数据 超越 使用10%数据的DPO/SimPO
- 在多个基准测试上验证有效

### 理论保证

论文提供了：
1. Reward threshold的存在性证明
2. 伪标签质量的理论边界
3. 半监督学习的收敛性分析

## 实验设置

### 数据集

- **UltraFeedback**: 主流偏好数据集，包含64k条prompt，每条4个response
- 使用不同采样比例：1%, 5%, 10%

### 模型

- Mistral-7B-Instruct
- Llama-3-8B-Instruct
- Qwen2-7B-Instruct

### 基准方法

- DPO
- SimPO
- ORPO
- KTO
- CPO

## 主要贡献总结

1. **问题定义**：首个半监督偏好优化问题
2. **理论分析**：证明最优reward threshold的存在
3. **方法创新**：伪标签技术蒸馏隐式偏好
4. **实验验证**：1%数据超越10%数据的baseline

## 开放问题

1. 阈值选择的最优策略
2. 伪标签噪声的影响分析
3. 在更多模态上的应用
