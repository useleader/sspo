# SSPO 环境搭建指南

## 当前环境

| 项目 | 状态 |
|------|------|
| Python | ✅ Python 3.12.3 |
| Conda | ❌ 未安装 |
| GPU | ❌ 无GPU (仅CPU) |
| 内存 | 7.6GB |
| 磁盘 | 949GB 可用 |

**⚠️ 当前环境限制**: 没有GPU，无法直接运行模型训练。需要配置GPU环境。

---

## 硬件要求

### 最低要求 (LoRA/QLoRA)
| 组件 | 要求 |
|------|------|
| GPU | NVIDIA GPU (≥6GB VRAM for 7B model) |
| 内存 | 16GB+ RAM |
| 存储 | 50GB+ |

### 推荐配置
| 组件 | 要求 |
|------|------|
| GPU | A100 40GB / A6000 48GB / RTX 4090 24GB |
| 内存 | 32GB+ RAM |
| 存储 | 100GB+ SSD |

---

## 环境搭建选项

### 选项 1: 云GPU (推荐用于初期学习)

#### Modal (推荐)
```bash
# 使用 Modal 租借 GPU
modal setup
modal run --gpu A100 train.py
```

#### RunPod / Vast.ai
- 按小时计费
- 选择有优惠的GPU实例

### 选项 2: 本地 conda 环境

```bash
# 1. 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. 创建环境
conda create -n sspo python=3.10
conda activate sspo

# 3. 安装 PyTorch (GPU版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 安装 llamafactory-cli
pip install llamafactory
```

### 选项 3: WSL + Windows GPU

```bash
# 在 WSL 中确保 CUDA 可用
nvidia-smi  # 应该能看到 Windows GPU
```

---

## 安装步骤 (当有GPU后)

### 1. 克隆仓库
```bash
cd /home/yanzm/sspo/src
git clone https://github.com/MLAI-Yonsei/SSPO.git .
```

### 2. 创建conda环境
```bash
conda create -n sspo python==3.10.0
conda activate sspo
```

### 3. 安装依赖
```bash
# 先安装 PyTorch (根据你的 CUDA 版本选择)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 4. 验证安装
```bash
python -c "import torch; print(torch.cuda.is_available())"
# 应该输出 True
```

---

## 数据准备

### 预处理 UltraFeedback 数据

```bash
cd /home/yanzm/sspo/src

# fb = labeled data 保留比例, ch = unlabeled data 保留比例
python preprocessing_data/preprocessing_ultrachat.py --fb 0.1 --ch 0.1
```

这会生成:
- `data/ultra_combined_fb0.1_ch0.1.json`

### 查看数据格式

```bash
head -100 data/ultra_combined_fb0.1_ch0.1.json
```

---

## 配置文件生成

```bash
cd /home/yanzm/sspo/src

# 生成 YAML 配置文件
python examples/train/make_yaml.py \
    --peft lora \
    --method sspo \
    --model_path mistralai/Mistral-7B-Instruct-v0.2
```

输出示例:
```
llamafactory-cli train ./examples/train/mistral-7b-it/sspo/fb0.1_ch0.1/fb0.1_ch0.1_...
```

---

## 训练命令

### 编辑 train.sh

```bash
# 编辑 examples/train/train.sh
nano examples/train/train.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES="0,1"
export WANDB_PROJECT="sspo-reproduction"
export WANDB_NAME="sspo-7b-lora"

# 添加训练命令 (从 make_yaml.py 输出复制)
llamafactory-cli train YOUR_YAML_FILE_PATH
```

### 开始训练

```bash
cd /home/yanzm/sspo/src
bash examples/train/train.sh
```

---

## 验证训练是否正常

观察日志，确保:
1. Loss 正常下降
2. 没有 NaN/Inf
3. GPU 内存使用正常

```bash
# 监控 GPU
watch -n 1 nvidia-smi

# 查看训练日志
tail -f logs/training.log
```

---

## 常见问题

### Q: `conda: command not found`
**A**: 需要先安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Q: `torch.cuda.is_available()` 返回 False
**A**: 
1. 检查 NVIDIA 驱动: `nvidia-smi`
2. 检查 PyTorch CUDA 版本匹配: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Q: GPU 内存不足 (OOM)
**A**:
1. 使用 QLoRA (4-bit量化): `--peft q-lora`
2. 减小 batch_size
3. 使用更小的模型 (如 phi-2)

### Q: 训练太慢
**A**:
1. 使用多卡: `CUDA_VISIBLE_DEVICES="0,1,2,3"`
2. 启用 gradient checkpointing
3. 使用更小的 cutoff_len

---

## 下一步

环境准备好后，参考:
- [[SSPO 复现规划]] - 详细复现步骤
- [[SSPO 代码实现分析]] - 代码架构
