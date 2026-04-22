# SSPO 代码实现分析

> 基于GitHub: https://github.com/MLAI-Yonsei/SSPO

## 代码结构

```
SSPO/
├── src/
│   ├── sspo.py          # 核心SSPO类
│   ├── reward.py        # Reward模型
│   ├── trainer.py       # 训练器
│   └── utils.py         # 工具函数
├── scripts/
│   ├── train_sspo.py    # 训练入口
│   └── evaluate.py      # 评估
├── configs/             # 配置文件
└── requirements.txt
```

## 核心类分析

### SSPO Trainer

```python
class SSPOTrainer:
    def __init__(
        self,
        model,
        ref_model=None,
        beta=0.1,
        threshold=0.0,
        lambda_pseudo=1.0,
        ...
    ):
        self.model = model
        self.ref_model = ref_model
        self.threshold = threshold
        self.lambda_pseudo = lambda_pseudo
    
    def compute_loss(self, batch):
        # 1. 计算标注数据的损失
        labeled_loss = self.compute_labeled_loss(batch['labeled'])
        
        # 2. 计算无标注数据的伪标签和损失
        pseudo_loss = self.compute_pseudo_loss(batch['unlabeled'])
        
        # 3. 联合损失
        return labeled_loss + self.lambda_pseudo * pseudo_loss
```

### Reward计算

```python
def compute_reward(prompt, response, model):
    """
    计算给定(prompt, response)的reward
    可能使用:
    - 单独的reward模型
    - 策略模型与参考模型的log prob差
    - 某种隐式奖励函数
    """
    inputs = tokenize(prompt, response)
    outputs = model(**inputs)
    reward = ...  # 根据论文实现
    return reward
```

### 伪标签生成

```python
def generate_pseudo_labels(unlabeled_data, reward_fn, threshold):
    """
    根据阈值生成伪标签
    """
    pseudo_labels = []
    for item in unlabeled_data:
        r = reward_fn(item['prompt'], item['response'])
        label = 1 if r > threshold else 0  # winning/losing
        pseudo_labels.append(label)
    return pseudo_labels
```

## 训练流程

```python
def train_sspo():
    # 1. 加载数据和模型
    model = load_model()
    tokenizer = load_tokenizer()
    labeled_data, unlabeled_data = load_data()
    
    # 2. 初始化训练器
    trainer = SSPOTrainer(
        model=model,
        threshold=0.0,
        lambda_pseudo=0.1,
        ...
    )
    
    # 3. 训练循环
    for epoch in range(num_epochs):
        for batch in dataloader:
            loss = trainer.compute_loss(batch)
            loss.backward()
            optimizer.step()
    
    # 4. 保存模型
    save_model(model)
```

## 关键超参数

| 参数 | 说明 | 参考值 |
|------|------|--------|
| threshold | 伪标签阈值 | 0.0 或搜索 |
| lambda_pseudo | 伪标签损失权重 | 0.1-1.0 |
| beta | KL系数 | 0.1 |
| learning_rate | 学习率 | 1e-6 |
| batch_size | 批大小 | 8-16 |

## 潜在实现细节

### Reward Model的选择

论文可能使用:
1. 单独的reward模型（如InstructGPT的reward network）
2. 隐式reward（从策略模型推导）

### 阈值选择

```python
# 方案1: 固定阈值
threshold = 0.0

# 方案2: 百分位
threshold = np.percentile(all_rewards, percentile=70)

# 方案3: 验证集搜索
best_acc = 0
for tau in np.linspace(-1, 1, 100):
    pseudo_labels = (rewards > tau).astype(float)
    # 在验证集上评估
    acc = evaluate(val_data, pseudo_labels)
    if acc > best_acc:
        best_acc = acc
        best_threshold = tau
```

## 与官方实现对照

> ⚠️ **注意**: 实际代码需要等clone官方仓库后确认细节

```
git clone https://github.com/MLAI-Yonsei/SSPO.git
cd SSPO
```

### 需要的依赖

```bash
torch>=2.0
transformers>=4.30
trl>=0.7.0
peft>=0.4.0
datasets>=2.14
accelerate
```

## 调试建议

1. **先跑通官方demo**: 按照README的示例运行
2. **检查loss曲线**: 确认损失正常下降
3. **验证伪标签分布**: 确认不是全1或全0
4. **小数据集验证**: 用小数据快速迭代

## 参考

- [SSPO GitHub](https://github.com/MLAI-Yonsei/SSPO)
- [TRL Library](https://github.com/huggingface/trl)
