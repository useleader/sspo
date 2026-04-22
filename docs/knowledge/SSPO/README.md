# SSPO 论文知识库

> **论文**: SSPO: Semi-Supervised Preference Optimization for Efficient LLM Alignment
> **会议**: ICLR 2026
> **状态**: 论文精读完成

## 快速导航

### 核心内容
- [[01_论文概述]] - 论文基本信息、研究动机、核心贡献
- [[02_预备知识]] - 相关工作、背景知识
- [[03_问题定义]] - 问题形式化、符号说明
- [[04_方法_SSPO]] - SSPO算法核心内容 ⭐
- [[05_理论分析]] - 理论性质、收敛性分析
- [[06_实验设置]] - 数据集、基线模型、评估指标
- [[07_实验结果]] - 主实验、消融研究
- [[08_相关工作]] - DPO、KTO、ORPO等对比
- [[09_附录]] - 证明、引理、超参数细节

### 概念索引
- [[concepts/pseudo-labeling]] - 伪标签技术
- [[concepts/preference-learning]] - 偏好学习
- [[concepts/alignment]] - 对齐技术
- [[concepts/semi-supervised]] - 半监督学习
- [[concepts/kl-divergence]] - KL散度

### 关键表格
- [[tables/experiment-settings]] - 实验设置汇总
- [[tables/main-results]] - 主要实验结果
- [[tables/ablation-study]] - 消融实验

## 论文摘要

**TL;DR**: 本文提出SSPO，一种半监督偏好优化方法，在仅使用有限标注偏好数据的情况下，通过伪标签技术利用大规模无标注偏好数据提升LLM对齐效果。

**核心发现**: 在仅20%标注数据下，SSPO即可达到全数据监督DPO水平的性能。

## 待整理

- [ ] 详细公式推导验证
- [ ] 代码实现对照
- [ ] 实验复现计划
