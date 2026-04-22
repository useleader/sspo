# 04_方法_SSPO

## 4.1 核心思想

SSPO将偏好学习重构为**概率分类任务**，通过学习一个reward threshold来为unpaired data分配伪标签。

### 与传统方法的关键区别
- 传统方法只能利用paired data $D_L$
- SSPO同时利用$D_L$（监督）和$D_U$（伪标签）

## 4.2 最优Reward Threshold的存在性（Theorem 1）

### 定理内容
在sub-Gaussian假设下：
- losing rewards: $\{r_\theta(x^{(i)}, y_l^{(i)})\}_{i=1}^{n_L}$ ~ $\mathcal{N}(\mu_l, \sigma_l^2)$
- winning rewards: $\{r_\theta(x^{(j)}, y_w^{(j)})\}_{j=1}^{n_L}$ ~ $\mathcal{N}(\mu_w, \sigma_w^2)$，其中 $\mu_w > \mu_l$

则存在最优threshold $\delta^* = \mu_l + t_1 = \mu_w - t_2$，使得：
$$\mathbb{P}\left(\max_i r_\theta(x^{(i)}, y_l^{(i)}) \leq \delta^* \leq \min_j r_\theta(x^{(j)}, y_w^{(j)})\right) \geq 1 - \alpha$$

### 直观理解
存在一个阈值可以高概率地完美分离winning和losing responses。

## 4.3 实际阈值估计：Kernel Density Estimation

由于$\mu_l, \mu_w$未知，使用KDE估计reward分布：

$$\hat{p}_w(r) = \frac{1}{n_L \cdot h}\sum_{j=1}^{n_L}\mathcal{K}\left(\frac{r - r_\theta(x^{(j)}, y_w^{(j)})}{h}\right)$$

$$\hat{p}_l(r) = \frac{1}{n_L \cdot h}\sum_{i=1}^{n_L}\mathcal{K}\left(\frac{r - r_\theta(x^{(i)}, y_l^{(i)})}{h}\right)$$

其中$\mathcal{K}(u) = \frac{1}{\sqrt{2\pi}}\exp(-\frac{1}{2}u^2)$是高斯核，$h$是bandwidth。

### 最小化Bayes Risk估计阈值
$$\hat{\delta} = \arg\min_{\delta} \hat{R}(\delta)$$

其中：
$$\hat{R}(\delta) = \mathbb{P}(s=1) \cdot \int_{-\infty}^{\delta}\hat{p}_w(r)dr + \mathbb{P}(s=0) \cdot \int_{\delta}^{\infty}\hat{p}_l(r)dr$$

## 4.4 伪标签生成

对每个unpaired sample $(x_u, y_u)$：
$$\tilde{s} = \mathbb{I}\{r_\theta(x_u, y_u) > \hat{\delta}\}$$

- $\tilde{s}=1$: $y_u$被标记为pseudo-winning
- $\tilde{s}=0$: $y_u$被标记为pseudo-losing

## 4.5 自适应调度训练

### 损失函数
$$\mathcal{L}(f_\theta) = \gamma' \cdot R_{D_L}(f_\theta) + (1-\gamma') \cdot R_{D_U}(f_\theta)$$

### 自适应系数
$$\gamma' = \max\left\{\gamma_{min}, \gamma_0 \cdot \exp(-\lambda\tau)\right\}$$

**训练初期** ($\tau=0$): $\gamma' \approx 1$，主要学习paired data
**训练后期**: $\gamma'$衰减，逐步引入pseudo-labeled unpaired data

### 设计动机
- 早期：reward function尚未学好，需要paired data的可靠监督
- 后期：reward分离明显，可以利用大量unpaired data

## 4.6 算法流程

```
输入: Paired data D_L, Unpaired data D_U
初始化: θ (from SFT model), γ' = 1

for τ = 1 to T do:
    # 1. 用D_L训练reward function
    r_θ ← train_reward(D_L)

    # 2. 用KDE估计最优threshold
    δ̂ ← KDE_threshold(r_θ, D_L)

    # 3. 为D_U生成伪标签
    for (x_u, y_u) in D_U:
        if r_θ(x_u, y_u) > δ̂:
            ŝ = 1  # pseudo-winning
        else:
            ŝ = 0  # pseudo-losing

    # 4. 联合优化
    γ' = max(γ_min, γ_0·exp(-λτ))
    L = γ'·R_D_L + (1-γ')·R_D_U
    θ ← θ - η∇L
end for
```

## 4.7 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| $\lambda$ | - | 衰减率 |
| $\gamma_0$ | 1 | 初始系数 |
| $\gamma_{min}$ | $n_L/(n_L+n_U)$ | 最小系数 |
| Prior $\mathbb{P}_{D_U}(s=1)$ | 0.5 | unpaired数据中winning