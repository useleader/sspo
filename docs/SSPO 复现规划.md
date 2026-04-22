# SSPO 复现规划

> 创建时间: 2026-04-14
> 目标: 系统复现SSPO论文

## 论文信息

- **标题**: Semi-Supervised Preference Optimization with Limited Feedback
- **作者**: Seonggyun Lee et al. (Yonsei University & KAIST)
- **会议**: ICLR 2026
- **arXiv**: 2511.00040v3
- **GitHub**: https://github.com/MLAI-Yonsei/SSPO

---

## 一、复现思路总览

### SSPO核心流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        SSPO 训练流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │  Labeled D_L │         │ Unlabeled D_U│                      │
│  │ (pair prefs) │         │  (SFT data)  │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
│         │                         │                              │
│         ▼                         ▼                              │
│  ┌──────────────────────────────────────┐                       │
│  │     1. Warm-up: 用D_L训练偏好模型      │                       │
│  └──────────────────┬───────────────────┘                       │
│                     │                                            │
│                     ▼                                            │
│  ┌──────────────────────────────────────┐                       │
│  │     2. 计算无标注数据的reward          │                       │
│  │        r(x,y) = f_θ(x,y)             │                       │
│  └──────────────────┬───────────────────┘                       │
│                     │                                            │
│                     ▼                                            │
│  ┌──────────────────────────────────────┐                       │
│  │     3. 确定最优阈值 τ*                │                       │
│  │        (Reward Threshold Theorem)      │                       │
│  └──────────────────┬───────────────────┘                       │
│                     │                                            │
│                     ▼                                            │
│  ┌──────────────────────────────────────┐                       │
│  │     4. 生成伪标签                     │                       │
│  │        ŷ = 1[r(x,y) > τ*]           │                       │
│  └──────────────────┬───────────────────┘                       │
│                     │                                            │
│                     ▼                                            │
│  ┌──────────────────────────────────────┐                       │
│  │     5. 联合优化                       │                       │
│  │        L_total = L_labeled + λ·L_pseudo │                    │
│  └──────────────────────────────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 与其他方法的对比

| 阶段 | DPO | SimPO | SSPO |
|------|-----|-------|------|
| 数据输入 | 配对偏好 | 配对偏好 | 配对偏好 + 无标注SFT |
| 参考模型 | 需要 | 不需要 | 不需要 |
| 伪标签 | ✗ | ✗ | ✓ |
| 训练阶段 | SFT + DPO | 单阶段 | 单阶段 |

---

## 二、复现步骤详细规划

### 阶段1: 环境准备与基线复现

**目标**: 搭建开发环境，理解代码结构

#### 1.1 环境配置

```bash
# 创建conda环境
conda create -n sspo python=3.10
conda activate sspo

# 安装PyTorch (根据你的CUDA版本)
pip install torch torchvision torchaudio

# 安装transformers
pip install transformers accelerate

# 安装数据集处理
pip install datasets trl peft

# 安装其他依赖
pip install numpy pandas tqdm matplotlib
```

#### 1.2 代码获取与结构分析

```bash
git clone https://github.com/MLAI-Yonsei/SSPO.git
cd SSPO
ls -la
```

**预期目录结构**:
```
SSPO/
├── src/
│   ├── sspo.py          # 核心SSPO实现
│   ├── trainer.py        # 训练器
│   └── utils.py         # 工具函数
├── scripts/
│   ├── train_sspo.py    # 训练脚本
│   └── evaluate.py      # 评估脚本
├── configs/              # 配置文件
├── data/                # 数据处理
└── README.md
```

#### 1.3 依赖分析

```python
# 检查 requirements.txt 或 pyproject.toml
# 核心依赖:
# - torch
# - transformers
# - trl (TRL库提供DPO等实现)
# - peft (LoRA支持)
# - datasets
```

#### 1.4 基线方法复现

**先复现DPO基线**，确保训练流程正确：

```python
# DPO基线代码框架
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
```

**验证指标**:
- Training loss下降
- 生成的response质量改善

---

### 阶段2: 数据准备

**目标**: 正确加载和预处理数据

#### 2.1 数据集选择

论文使用的数据集：
- **UltraFeedback**: 64k prompts, 每条4个response
- 也测试了其他数据集

#### 2.2 数据格式

**标注数据 (D_L)**:
```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "Paris is the capital of France."
}
```

**无标注数据 (D_U)**:
```json
{
  "prompt": "Explain quantum entanglement.",
  "response": "Quantum entanglement is a phenomenon where..."
}
```

#### 2.3 数据划分

```
UltraFeedback (64k):
├── 1% 标注数据: ~640 条
├── 5% 标注数据: ~3,200 条  
└── 10% 标注数据: ~6,400 条

无标注数据 (全部SFT数据)
```

#### 2.4 Reward模型准备

```python
# 训练reward model用于计算伪标签
reward_model = RewardModel(config)
```

---

### 阶段3: 核心算法实现

**目标**: 实现SSPO核心逻辑

#### 3.1 Reward计算

```python
def compute_reward(prompt, response, model, tokenizer):
    """
    计算 (prompt, response) 的reward
    这是SSPO伪标签生成的关键
    """
    inputs = tokenizer(prompt, response, return_tensors='pt')
    # 使用模型计算reward (具体实现依赖论文细节)
    with torch.no_grad():
        outputs = model(**inputs)
    reward = ...  # 根据论文定义
    return reward
```

#### 3.2 最优阈值计算

```python
def find_optimal_threshold(rewards, labels=None):
    """
    实现Reward Threshold Theorem
    找到最优阈值 τ* 用于伪标签生成
    """
    # 方法1: 基于验证集选择
    # 方法2: 理论估算 τ* ≈ -log(ε) for small ε
    # 方法3: 网格搜索最优阈值
    pass
```

#### 3.3 伪标签生成

```python
def generate_pseudo_labels(data, reward_fn, threshold):
    """
    为无标注数据生成伪标签
    """
    pseudo_labels = []
    for item in data:
        r = reward_fn(item['prompt'], item['response'])
        label = 1 if r > threshold else 0
        pseudo_labels.append(label)
    return pseudo_labels
```

#### 3.4 SSPO损失函数

```python
def sspo_loss(labeled_logits, pseudo_logits, lambda_pseudo=1.0):
    """
    SSPO总损失
    L_total = L_labeled + λ * L_pseudo
    """
    # Labeled data: 标准偏好损失
    L_labeled = -torch.log(torch.sigmoid(labeled_logits))
    
    # Pseudo-labeled data: 二分类交叉熵
    L_pseudo = F.binary_cross_entropy_with_logits(
        pseudo_logits, 
        pseudo_labels.float()
    )
    
    return L_labeled + lambda_pseudo * L_pseudo
```

#### 3.5 训练循环

```python
def sspo_train_loop(model, labeled_data, unlabeled_data, config):
    """
    SSPO主训练循环
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    for epoch in range(config.num_epochs):
        # 1. 用标注数据计算损失
        labeled_loss = compute_labeled_loss(model, labeled_data)
        
        # 2. 用当前模型计算无标注数据的reward
        rewards = compute_rewards(model, unlabeled_data)
        
        # 3. 生成伪标签
        pseudo_labels = (rewards > config.threshold).float()
        
        # 4. 用伪标签数据计算损失
        pseudo_loss = compute_pseudo_loss(model, unlabeled_data, pseudo_labels)
        
        # 5. 联合优化
        total_loss = labeled_loss + config.lambda_pseudo * pseudo_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

### 阶段4: 训练配置与调优

#### 4.1 关键超参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| learning_rate | 学习率 | 1e-6 到 5e-6 |
| batch_size | 批大小 | 8-16 |
| beta/alpha | KL散度系数 | 0.1 |
| threshold (τ*) | 伪标签阈值 | 需要搜索 |
| lambda_pseudo | 伪标签损失权重 | 0.1-1.0 |
| num_epochs | 训练轮数 | 3-10 |

#### 4.2 阈值选择策略

```python
# 方案1: 固定阈值
threshold = 0.0

# 方案2: 百分位阈值
threshold = np.percentile(all_rewards, 70)  # top 70% 作为winning

# 方案3: 验证集搜索
best_threshold = find_best_threshold_on_val(val_data)
```

#### 4.3 伪标签权重

$$
\lambda = \frac{N_{labeled}}{N_{labeled} + N_{unlabeled}}
$$

---

### 阶段5: 评估与验证

#### 5.1 评估指标

| 指标 | 说明 | 测量方式 |
|------|------|----------|
| Win Rate | 胜率 | 人类评估/AutoEval |
| GPT-4 Score | GPT-4评分 | 使用GPT-4作为裁判 |
| Length | 长度控制 | 响应平均长度 |
| Diversity | 多样性 | n-gram统计 |

#### 5.2 评估数据集

- **MT-Bench**: 多轮对话
- **AlpacaEval**: 指令遵循
- **HH-RLHF**: helpful/harmless评估

#### 5.3 消融实验

```
1. 标注数据比例: 1% vs 5% vs 10%
2. 阈值选择: 不同τ的影响
3. 伪标签权重: 不同λ的影响
4. 无标注数据量: 不同M的影响
```

---

### 阶段6: 代码调试与优化

#### 6.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| Loss不下降 | 学习率过大/过小 | 调整lr或使用warmup |
| 伪标签全1或全0 | 阈值不合理 | 检查reward分布，调整阈值 |
| 训练不稳定 | batch size太小 | 增大batch或减小学习率 |
| 内存不足 | 模型太大 | 使用LoRA或减小模型 |

#### 6.2 调试建议

```python
# 1. 先在小数据上验证
small_labeled = labeled_data[:100]
small_unlabeled = unlabeled_data[:1000]

# 2. 打印中间变量
print(f"Rewards: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
print(f"Pseudo labels: {pseudo_labels.sum()}/{len(pseudo_labels)}")

# 3. 梯度检查
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
```

---

## 三、复现检查清单

### 环境搭建
- [ ] Python环境创建
- [ ] 依赖安装
- [ ] Git clone代码

### 数据处理
- [ ] 数据集下载
- [ ] 数据格式转换
- [ ] 数据划分脚本

### 基线验证
- [ ] DPO基线运行
- [ ] SimPO基线运行
- [ ] 确认基线性能

### SSPO实现
- [ ] Reward计算模块
- [ ] 阈值选择模块
- [ ] 伪标签生成模块
- [ ] 联合损失函数
- [ ] 完整训练循环

### 实验验证
- [ ] 1%数据实验
- [ ] 5%数据实验
- [ ] 10%数据实验
- [ ] 对比基线方法

---

## 四、预期结果

### 论文报告的性能

| 方法 | 数据比例 | 预期性能 |
|------|----------|----------|
| DPO | 10% | 100% (baseline) |
| SSPO | 1% | ~95-100% |
| SSPO | 5% | >100% |
| SSPO | 10% | >>100% |

### 复现目标

- 主要结果复现: 1% SSPO ≈ 10% DPO
- 趋势验证: 数据量增加性能提升
- 阈值敏感性: 找到合适的τ

---

## 五、参考资料

1. [SSPO GitHub](https://github.com/MLAI-Yonsei/SSPO)
2. [TRL Library](https://github.com/huggingface/trl)
3. [UltraFeedback Dataset](https://arxiv.org/abs/2310.01377)
4. [DPO Paper](https://arxiv.org/abs/2305.10425)
