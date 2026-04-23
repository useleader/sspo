# SSPO - Semi-Supervised Preference Optimization

> ICLR 2026 论文复现项目 | [官方GitHub](https://github.com/MLAI-Yonsei/SSPO)

---

## 项目状态

| 模块 | 状态 | 说明 |
|------|------|------|
| 知识库 | ✅ | 10个ADRs + 9个知识文件 |
| 数据下载 | ✅ | UltraFeedback + UltraChat |
| 数据预处理 | ✅ | Combined dataset 生成 |
| 模型配置 | ✅ | 27个训练配置 |
| 测试套件 | ✅ | 73 tests passing |
| 训练脚本 | ✅ | SLURM + Local |

---

## 快速开始

### 1. 环境配置

```bash
# 安装 uv (Python包管理器)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建Python环境
cd /home/yanzm/sspo
uv venv --python 3.10 .venv
source .venv/bin/activate

# 安装依赖
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
uv pip install tqdm numpy requests datasets
cd src && pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载数据集 (需要 ~1.5GB 存储)
python scripts/download_data.py --dataset all --output data/

# 预处理 (fb=标注比例, ch=无标注比例)
python scripts/preprocess_data.py --fb 0.01 --ch 0.1 --output processed/
```

### 3. 模型下载 (ModelScope)

```bash
# 安装 ModelScope SDK (已包含在依赖中)
uv pip install modelscope

# 设置 Token (可选，部分模型需要)
export MODELSCOPE_TOKEN=your_token_here

# 下载模型 (以 Mistral 为例，约 14GB)
python scripts/download_models.py --model mistral --output models/

# 或下载所有模型
python scripts/download_models.py --model all --output models/
```

### 4. 生成训练配置

```bash
# 生成所有实验配置
python scripts/generate_model_configs.py --output configs/

# 查看生成结果
ls configs/mistral-7b-it/sspo/
```

### 5. 本地调试训练

```bash
# 单GPU本地调试
bash scripts/run_all_experiments.sh --local

# 或直接运行
bash scripts/train_local.sh configs/mistral-7b-it/sspo/fb0.01_ch0.1_sspo_mistral-7b-it.yaml
```

### 6. 提交SLURM任务 (8x H100集群)

```bash
# 生成配置并提交所有实验
bash scripts/run_all_experiments.sh --submit
```

---

## 项目结构

```
/home/yanzm/sspo/
├── src/                              # LLaMA-Factory fork (SSPO实现)
│   └── src_sspo/llamafactory/
│       ├── train/dpo/trainer.py     # SSPO核心算法
│       ├── data/processors/         # 数据处理器
│       └── hparams/finetuning_args.py
│
├── scripts/                          # 训练Pipeline
│   ├── download_data.py             # 数据下载
│   ├── preprocess_data.py           # 数据预处理
│   ├── generate_model_configs.py    # 配置生成
│   ├── analyze_data.py             # 数据分析
│   ├── train_sspo.sh               # SLURM训练
│   ├── train_local.sh              # 本地训练
│   ├── run_all_experiments.sh      # 实验编排
│   └── eval/                       # 评估模块
│       ├── generate_responses.py
│       ├── alpaca_eval_evaluator.py
│       ├── mtbench_evaluator.py
│       └── aggregate_results.py
│
├── tests/                           # 测试套件 (73 tests)
│   ├── data/                       # 数据测试
│   └── eval/                       # 评估测试
│
├── configs/                         # 生成的训练配置 (27个)
│   ├── mistral-7b-it/sspo/
│   ├── llama3-8b-it/sspo/
│   └── qwen2-7b-it/sspo/
│
├── data/                            # 原始数据集
│   ├── ultrafeedback/
│   └── ultrachat/
│
├── processed/                       # 预处理后的数据
│   └── ultra_combined_fb*.json
│
├── saves/                          # 模型权重 (LoRA)
├── logs/                           # 训练日志
├── plots/                          # Loss曲线
├── results/                        # 评估结果
│   ├── samples/
│   ├── alpaca_eval/
│   └── mtbench/
│
├── docs/
│   ├── adr/                        # 架构决策记录 (10个)
│   └── knowledge/SSPO/            # 论文知识库 (9个文件)
│
└── CLAUDE.md                       # Claude Code 指南
```

---

## 实验配置

| 模型 | 方法 | 标注比例 (fb) | 无标注比例 (ch) |
|------|------|---------------|-----------------|
| Mistral-7B-Instruct | SSPO | 1%, 5%, 10% | 10% |
| Llama-3-8B-Instruct | DPO | 1%, 5%, 10% | 10% |
| Qwen2-7B-Instruct | SimPO | 1%, 5%, 10% | 10% |

**关键超参数 (来自论文):**

| 参数 | 值 |
|------|-----|
| LoRA rank | 8 |
| Learning rate | 1e-5 |
| Batch size | 64 (per node) |
| Context length | 1024 |
| β (beta) | 2.0 |
| SSPO γ_0 | 1.0 |
| SSPO γ_min | 0.22 |

---

## 测试

```bash
# 运行所有测试
source .venv/bin/activate
python -m pytest tests/ -v

# 只运行数据测试
python -m pytest tests/data/ -v

# 只运行评估测试
python -m pytest tests/eval/ -v
```

**测试覆盖:**
- 边缘情况 (ratio = 0.0, 1.0, 极小值)
- 数据质量 (缺失字段，空值处理)
- 数据完整性 (labeled/unlabeled 分离)
- 规模测试 (100k样本)
- 评估脚本签名和导入

---

## 评估

```bash
# 生成模型响应
python scripts/eval/generate_responses.py \
    --model_path saves/mistral-7b-it/sspo/fb0.01_ch0.1/best_model \
    --dataset alpacaeval \
    --output results/samples/mistral_sspo.json

# 评估 AlpacaEval (LC-Win Rate)
python scripts/eval/alpaca_eval_evaluator.py \
    --model-outputs results/samples/mistral_sspo.json \
    --output-dir results/alpaca_eval/

# 评估 MT-Bench
python scripts/eval/mtbench_evaluator.py \
    --model-outputs results/samples/mistral_sspo.json \
    --output-dir results/mtbench/

# 聚合结果
python scripts/eval/aggregate_results.py \
    --results-dir results/ \
    --output results/aggregated.json
```

---

## 参考资源

- [SSPO GitHub](https://github.com/MLAI-Yonsei/SSPO)
- [arXiv论文](https://arxiv.org/abs/2511.00040)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
