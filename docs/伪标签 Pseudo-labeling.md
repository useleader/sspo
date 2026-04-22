# 伪标签 Pseudo-labeling

## 概念定义

**伪标签（Pseudo-labeling）**：使用模型为无标注数据自动生成标签，然后用这些"伪"标签进行训练。

## 核心思想

```
有标注数据 (N)          无标注数据 (M >> N)
     │                        │
     ▼                        ▼
  标准监督学习 ──────► 训练模型 ──────► 预测伪标签
                                            │
                                            ▼
                                      伪标签数据
                                            │
                                            ▼
              ◄────── 联合训练 ◄────────────┘
```

## 伪标签流程

### 1. Train on Labeled Data
使用标注数据训练初始模型 $f_\theta$

### 2. Generate Pseudo-labels
对无标注数据生成标签：
$$
\hat{y} = argmax_y\ f_\theta(x, y)
$$

### 3. Retrain with Combined Data
使用 $(x, \hat{y})$ 联合训练

## 为什么伪标签有效？

### 1. 数据增强效应

无标注数据提供了更广泛的输入分布覆盖

### 2. 自我训练（Self-training）

模型对自己生成的标签进行"确认"，强化已学到的决策边界

### 3. 隐式信息蒸馏

高质量模型生成的伪标签包含了对数据分布的隐式知识

## 伪标签的挑战

### 1. 确认偏差（Confirmation Bias）

模型生成的错误标签会被模型自己强化，导致错误累积。

### 2. 噪声标签

低置信度的伪标签可能误导训练。

### 3. 分布偏移

伪标签可能偏向模型已见过的分布。

## 应对策略

### 置信度阈值

```python
def pseudo_label_with_threshold(model, unlabeled_data, threshold=0.9):
    pseudo_labels = []
    for x in unlabeled_data:
        probs = model.predict_proba(x)
        if max(probs) > threshold:
            pseudo_labels.append(argmax(probs))
        else:
            pseudo_labels.append(None)  # 丢弃低置信度样本
    return pseudo_labels
```

### 迭代细化

```
迭代1: 用高置信度伪标签训练
迭代2: 重新预测，升高阈值
迭代3: ...
```

### 课程学习

从高置信度到低置信度，逐步引入更多伪标签。

## 在SSPO中的应用

### SSPO的特殊设置

- **有标注数据**: $(x, y_w, y_l)$ 人类标注的偏好对
- **无标注数据**: $(x, y)$ SFT数据，只有prompt和response

### SSPO的伪标签策略

1. **Reward计算**: $r = f_\theta(x, y)$
2. **阈值判断**: $\hat{y} = \mathbb{1}[r > \tau^*]$
3. **标签含义**: $\hat{y}=1$ 表示这是一个winning response

### 关键创新

不是预测偏好对，而是识别**绝对质量**：哪个response是winning（可以被接受）。

## SSPO的伪标签优势

### 与传统伪标签对比

| 方面 | 传统伪标签 | SSPO伪标签 |
|------|------------|------------|
| 标签来源 | 模型预测 | Reward阈值 |
| 标签类型 | 类别标签 | 二值标签(win/lose) |
| 质量控制 | 置信度阈值 | 最优阈值定理 |
| 理论基础 | 启发式 | 理论证明 |

### SSPO的理论保证

论文证明了存在最优阈值 $\tau^*$ 使得：
1. 伪标签错误率可控
2. 半监督学习收敛
3. 性能单调提升

## 参考文献

- Lee et al., "Pseudo-labeling: A simple method for enhancing semi-supervised learning" (various)
- SSPO论文 Section 3.2: Pseudo-labeling Analysis
