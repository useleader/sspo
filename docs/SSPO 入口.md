# SSPO: Semi-Supervised Preference Optimization

> **论文**: [Semi-Supervised Preference Optimization with Limited Feedback](https://arxiv.org/abs/2511.00040)
> **作者**: Seonggyun Lee, Sungjun Lim, Seojin Park, Soeun Cheon, Kyungwoo Song (Yonsei University & KAIST)
> **发表**: ICLR 2026
> **GitHub**: https://github.com/MLAI-Yonsei/SSPO

## 核心问题

现有偏好优化方法（DPO、SimPO等）依赖大量人工标注的**配对反馈数据**，标注成本高昂。

## 核心创新

SSPO将偏好学习重新定义为**概率二分类问题**，证明存在最优奖励阈值用于伪标签生成，从而从大量无配对SFT数据中蒸馏隐式偏好。

## 关键结果

> SSPO在仅使用**1% UltraFeedback**数据时，性能超越使用**10% UltraFeedback**训练的强baseline

## 知识库结构

```dataview
TABLE file.ctime as Created
FROM ""
WHERE contains(file.folder, "LLM Alignment/SSPO")
SORT file.ctime ASC
```

---

## 学习路径

### 阶段1: 基础概念
1. [[LLM Alignment - 概念入门]] - LLM对齐的基本问题
2. [[RLHF - 人类反馈强化学习]] - 传统RLHF方法
3. [[DPO - 直接偏好优化]] - DPO核心原理

### 阶段2: 偏好优化方法演进
4. [[SimPO - 简单偏好优化]] - 去除参考模型
5. [[ORPO - 赔率比偏好优化]] - 单阶段训练
6. [[KTO - 卡尼曼-特沃斯基优化]] - 行为经济学视角

### 阶段3: SSPO核心
7. [[SSPO 论文精读]] - 论文核心内容
8. [[SSPO 理论分析]] - 奖励阈值与伪标签
9. [[SSPO 与其他方法对比]]

### 阶段4: 复现与实践
10. [[SSPO 复现规划]] - 系统复现思路
11. [[SSPO 代码实现分析]] - 核心代码解读

---

## 相关概念

- [[Bradley-Terry 模型]] - 偏好概率建模
- [[伪标签 Pseudo-labeling]] - 半监督学习核心
- [[KL散度与对齐]] - 分布控制

## 外部资源

- [SSPO GitHub](https://github.com/MLAI-Yonsei/SSPO)
- [arXiv论文](https://arxiv.org/abs/2511.00040)
- [UltraFeedback数据集](https://arxiv.org/abs/2310.01377)
