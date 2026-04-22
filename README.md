# SSPO - Semi-Supervised Preference Optimization

> ICLR 2026 论文复现项目 | [官方GitHub](https://github.com/MLAI-Yonsei/SSPO)

---

## 项目状态

| 项目 | 状态 | 说明 |
|------|------|------|
| 知识库 | ✅ 完成 | 19个笔记在 `docs/` |
| 代码仓库 | ✅ 已克隆 | `/home/yanzm/sspo/src/` |
| 环境配置 | ⚠️ 待配置 | 需要GPU环境 |
| 数据准备 | ⏳ 待执行 | 需要先配置环境 |
| 模型训练 | ⏳ 待执行 | 需要先配置环境 |

---

## 环境要求

**当前环境**: CPU only (7.6GB RAM)

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA 6GB+ | A100 40GB / RTX 4090 24GB |
| 内存 | 16GB | 32GB+ |
| 存储 | 50GB | 100GB+ SSD |

**详细指南**: [[SSPO 环境搭建指南]]

---

## 快速开始

### 已有内容

```
/home/yanzm/sspo/
├── docs/                          # 知识库 (Obsidian)
│   ├── SSPO 入口.md
│   ├── SSPO 论文精读.md
│   ├── SSPO 理论分析.md
│   ├── SSPO 复现规划.md
│   ├── SSPO 环境搭建指南.md       ← 新增
│   ├── SSPO 代码实现分析.md
│   ├── SSPO 与其他方法对比.md
│   ├── 半监督学习概念.md
│   ├── 伪标签 Pseudo-labeling.md
│   ├── Bradley-Terry 模型.md
│   ├── KL散度与对齐.md
│   ├── LLM Alignment - 概念入门.md
│   ├── RLHF - 人类反馈强化学习.md
│   ├── DPO - 直接偏好优化.md
│   ├── SimPO - 简单偏好优化.md
│   ├── ORPO - 赔率比偏好优化.md
│   ├── KTO - 卡尼曼特沃斯基优化.md
│   ├── Obsidian插件安装指南.md
│   └── Git多端同步工作流.md
│
├── src/                           # 官方代码 (已克隆)
│   ├── src_sspo/                  # SSPO 核心实现
│   ├── examples/                  # 训练示例
│   ├── preprocessing_data/        # 数据预处理
│   └── data/                      # 演示数据
│
├── course_pdfs/                   # 课程PPT
├── configs/                       # 配置文件 (待填充)
├── data/                          # 数据集 (待填充)
└── README.md
```

### 官方仓库结构

```
src/
├── src_sspo/                      # SSPO核心代码
│   └── llamafactory/              # 基于LLaMA-Factory
│       ├── train/                # 训练器 (dpo, kto, ppo, sft, rm)
│       │   └── dpo/              # DPO trainer (含SSPO)
│       ├── data/                 # 数据处理
│       └── model/                # 模型加载
│
├── examples/
│   ├── train/                    # 训练脚本
│   │   ├── make_yaml.py         # 生成配置
│   │   └── train.sh             # 训练入口
│   └── SSRM/                    # SSRM基线
│
└── preprocessing_data/           # 数据预处理
    ├── preprocessing_ultrachat.py
    ├── preprocessing_medical.py
    └── preprocessing_business.py
```

---

## 复现步骤

### Phase 1: 环境搭建 (当前)

```bash
# 1. 配置 GPU 环境 (云GPU或本地)
#    参考: docs/SSPO 环境搭建指南.md

# 2. 创建 conda 环境
conda create -n sspo python==3.10.0
conda activate sspo

# 3. 安装 PyTorch (CUDA版本)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装依赖
cd /home/yanzm/sspo/src
pip install -r requirements.txt

# 5. 验证
python -c "import torch; print(torch.cuda.is_available())"
```

### Phase 2: 数据准备

```bash
# 预处理数据 (fb=标注比例, ch=无标注比例)
python preprocessing_data/preprocessing_ultrachat.py --fb 0.1 --ch 0.1
```

### Phase 3: 配置生成

```bash
# 生成训练配置
python examples/train/make_yaml.py \
    --peft lora \
    --method sspo \
    --model_path mistralai/Mistral-7B-Instruct-v0.2
```

### Phase 4: 开始训练

```bash
# 编辑 train.sh 添加训练命令
nano examples/train/train.sh

# 执行训练
bash examples/train/train.sh
```

---

## 核心文件

| 文件 | 说明 |
|------|------|
| `src/src_sspo/llamafactory/train/dpo/trainer.py` | DPO trainer (含SSPO实现) |
| `src/examples/train/make_yaml.py` | 配置生成脚本 |
| `src/preprocessing_data/preprocessing_ultrachat.py` | UltraFeedback预处理 |
| `docs/SSPO 复现规划.md` | 详细复现计划 |

---

## 复现进度

- [x] Phase 0: 知识库建立
- [ ] Phase 1: 环境搭建 ← **当前**
- [ ] Phase 2: 数据准备
- [ ] Phase 3: 算法实现
- [ ] Phase 4: 训练配置
- [ ] Phase 5: 实验与评估

---

## 学习路径

```
1. 概念入门
   docs/LLM Alignment - 概念入门.md
   docs/Bradley-Terry 模型.md
   docs/KL散度与对齐.md

2. 方法演进
   docs/RLHF - 人类反馈强化学习.md
   docs/DPO - 直接偏好优化.md
   docs/SimPO - 简单偏好优化.md  (可选)
   docs/ORPO - 赔率比偏好优化.md  (可选)
   docs/KTO - 卡尼曼特沃斯基优化.md  (可选)

3. SSPO核心
   docs/伪标签 Pseudo-labeling.md
   docs/半监督学习概念.md
   docs/SSPO 论文精读.md
   docs/SSPO 理论分析.md

4. 动手实践 ← 你在这里
   docs/SSPO 环境搭建指南.md
   docs/SSPO 复现规划.md
```

---

## 云GPU推荐

如果没有本地GPU，可以使用:

| 提供商 | 特点 | 链接 |
|--------|------|------|
| **Modal** | 按秒计费，代码即部署 | modal.com |
| **Vast.ai** | 性价比高 | vast.ai |
| **RunPod** | 实例多样 | runpod.io |

参考: `docs/SSPO 环境搭建指南.md` 的 Modal 章节

---

## 参考资源

- [SSPO GitHub](https://github.com/MLAI-Yonsei/SSPO)
- [arXiv论文](https://arxiv.org/abs/2511.00040)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Alignment Handbook](https://github.com/huggingface/alignment-handbook)
