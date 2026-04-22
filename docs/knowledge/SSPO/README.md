# SSPO 论文知识库

> **论文**: Semi-Supervised Preference Optimization with Limited Feedback
> **会议**: ICLR 2026
> **GitHub**: https://github.com/MLAI-Yonsei/SSPO
> **状态**: 精读完成 ✅

## 快速导航

### 核心内容
- [[01_论文概述]] - 论文基本信息、研究动机、核心贡献
- [[02_预备知识]] - ⭐ **详细背景知识**，偏好学习完整指南
- [[03_问题定义]] - 问题形式化、符号说明
- [[04_方法_SSPO]] - SSPO算法核心（Theorem 1、KDE阈值估计、自适应调度）
- [[05_理论分析]] - 最优阈值存在性证明、收敛性分析
- [[06_实验设置]] - 数据集、基线、评估基准
- [[07_实验结果]] - Toy实验、主实验、消融研究
- [[08_相关工作]] - 与DPO、KTO、ORPO等方法对比
- [[09_附录]] - 定理证明、算法伪代码、超参数配置

## 论文核心发现

### 🔑 关键发现
1. **惊人数据效率**: 仅用1% UltraFeedback数据训练的SSPO，超越使用10%数据的DPO/KTO/SimPO等基线
2. **理论保证**: 证明了存在最优reward threshold δ*可以高概率分离winning/losing responses
3. **KDE阈值估计**: 使用Kernel Density Estimation从paired data学习最优阈值
4. **自适应调度**: curriculum learning动态平衡paired/unpaired数据权重

### 📊 实验结果摘要

| 配置 | Phi-2 (2.7B) | Mistral (7B) | Llama3 (8B) |
|------|-------------|--------------|-------------|
| SSPO 1% | **LC=7.2%** | **LC=26.7%** | **LC=14.8%** |
| Best Baseline 1% | 4.3% (SSRM) | 18.2% (SPA) | 14.9% (SSRM) |
| Best Baseline 10% | 4.9% (SPA) | 19.1% (SPA) | 16.7% (KTO) |

**SSPO用1%数据超越了所有基线用10%数据的结果！**

## 推荐阅读顺序

### 新手入门（强烈建议按顺序阅读）

1. **[[02_预备知识]]** - 这是你了解偏好学习领域的最佳起点！
   - 从Alignment Problem讲起，解释为什么需要偏好学习
   - 详细讲解RLHF三阶段Pipeline（SFT → Reward Model → PPO）
   - 深入讲解Bradley-Terry模型（所有偏好学习的基础）
   - 逐一讲解DPO、KTO、ORPO、SimPO等方法的动机、数学推导、优缺点
   - 最后介绍半监督方法（SSRM、SPA）作为SSP0的铺垫

2. **[[03_问题定义]]** - 在理解预备知识后，学习SSP0的问题形式化

3. **[[04_方法_SSPO]]** - 核心算法，包含Theorem 1的直观解释

### 有经验者快速参考

1. **[[01_论文概述]]** → **[[04_方法_SSPO]]** → **[[07_实验结果]]**

## 方法流程图

```
                    Theorem 1: 最优阈值存在性
                           │
                           ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Paired D_L   │───▶│  Reward Model   │───▶│ KDE Threshold    │
│ (labeled)    │    │  r_θ from D_L   │    │ δ̂ = argmin R(δ) │
└──────────────┘    └─────────────────┘    └────────┬─────────┘
                                                    │
┌──────────────┐    ┌─────────────────┐             │
│ Unpaired D_U │───▶│  Pseudo-label   │◀────────────┘
│ (SFT data)    │    │  if r > δ̂ → ŝ=1 │
└──────────────┘    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Adaptive Training          │
              │  L = γ'·R_D_L + (1-γ')·R_D_U │
              │  γ' = max(γ_min, γ_0·e^{-λτ}) │
              └──────────────────────────────┘
                             │
                             ▼
                      Aligned Model
```

## 核心公式

### 偏好分类器 (Bradley-Terry)
$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

### 最优阈值存在性 (Theorem 1)
$$\mathbb{P}\left(\max_i r_l^{(i)} \leq \delta^* \leq \min_j r_w^{(j)}\right) \geq 1 - \alpha$$

### 训练目标
$$\mathcal{L} = \gamma' \cdot R_{D_L} + (1-\gamma') \cdot R_{D_U}, \quad \gamma' = \max\{\gamma_{min}, \gamma_0 e^{-\lambda\tau}\}$$

## 源码位置

解压后的源码位于 `/tmp/` 目录：
- `iclr2026_conference_CameraReady.tex` - 完整论文LaTeX源码
- `sspo_figure1.png` - 方法概览图
- `sspo_figure2.png` - 自适应调度可视化
- `sspo_figure3.png` - Reward分布演化图

## 相关概念

- [[concepts/preference-optimization]] - 偏好优化
- [[concepts/semi-supervised-learning]] - 半监督学习
- [[concepts/bayes-optimal-classification]] - Bayes最优分类
- [[concepts/kernel-density-estimation]] - 核密度估计
- [[concepts/curriculum-learning]] - 课程学习
