# Bradley-Terry 模型

## 历史背景

Bradley和Terry在1952年提出用于分析**成对比较**的统计模型。

## 基本形式

### 问题定义

给定两个对象 $A$ 和 $B$，观测到偏好 $A \succ B$。

### 模型假设

每个对象有一个**潜在价值**（latent quality）$\theta_A$, $\theta_B$。

偏好概率定义为：

$$
P(A \succ B) = \frac{\theta_A}{\theta_A + \theta_B}
$$

### 指数形式

通常用指数形式参数化：

$$
P(A \succ B) = \frac{exp(\theta_A)}{exp(\theta_A) + exp(\theta_B)} = \sigma(\theta_A - \theta_B)
$$

其中 $\sigma$ 是sigmoid函数。

## 在偏好学习中的应用

### 语言模型偏好

给定prompt $x$和两个response $y_w$（preferred）和 $y_l$（dispreferred）。

定义reward函数 $r_\theta(x, y)$，则：

$$
P(y_w \succ y_l | x; \theta) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))
$$

### 损失函数推导

**负对数似然**：

$$
L = -\mathbb{E}_{(x, y_w, y_l)}[log\ P(y_w \succ y_l | x)] = -\mathbb{E}[log\ \sigma(r_w - r_l)]
$$

## 与其他模型的关系

### Luce-Shepard模型

Bradley-Terry模型是Luce-Shepard模型的特例。

### Plackett-Luce模型

用于多选项比较的推广。

## 在LLM对齐中的应用

### DPO的BT基础

DPO隐式使用Bradley-Terry模型：

$$
P(y_w \succ y_l) = \sigma(\beta \cdot log\frac{\pi(y_w)}{\pi_{ref}(y_w)} - \beta \cdot log\frac{\pi(y_l)}{\pi_{ref}(y_l)})
$$

### SimPO的BT基础

SimPO同样使用BT模型定义偏好概率：

$$
P(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))
$$

## 估计方法

### 最大似然估计

给定成对比较数据 $\{(A_i \succ B_i)\}$，似然函数：

$$
L(\theta) = \prod_i \sigma(\theta_{A_i} - \theta_{B_i})
$$

### 优化

通过对数似然最大化：

$$
\hat{\theta} = argmax_\theta \sum_i log\ \sigma(\theta_{A_i} - \theta_{B_i})
$$

## 局限性

### 1. 传递性假设

模型假设如果 $A \succ B$ 且 $B \succ C$，则 $A \succ C$。这在实际中可能不成立。

### 2. 只有二元比较

不能直接处理多选项比较（需要推广到Plackett-Luce）。

### 3. 独立假设

假设每对比较是独立的。

## 参考

- Bradley, R.A. and Terry, M.E., "Rank Analysis of Incomplete Block Designs" (1952)
- textbooks on choice models and discrete choice theory
