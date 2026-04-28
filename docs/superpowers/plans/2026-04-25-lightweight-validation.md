# Lightweight Validation Run - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate all toy experiment code runs correctly on A6000 GPU before full-scale experiments. Cover paper Table 6/7 full scope.

**Architecture:** Use existing `toy_experiment` configs in `configs/local/toy_experiment/`. Run each method+noise+prior combination to validate pipeline integrity.

**Tech Stack:** PyTorch, LLaMA-Factory fork, single A6000 48GB GPU, toy datasets

---

## Scope: Paper Table 6 Coverage

**实验完整列表 (对照论文 Table 6/7):**

| Method | n_L | Noise | Priors | Configs |
|--------|-----|-------|--------|---------|
| DPO | 10/50/100 | 0/10/30/50% | - | 12 |
| ORPO | 10/50/100 | 0/10/30/50% | - | 12 |
| SimPO | 10/50/100 | 0/10/30/50% | - | 12 |
| SSPO | 10/50/100 | 0% | 0.1/0.3/0.5/0.7/0.9 | 15 |
| SSPO | 10/50/100 | 10/30/50% | 0.5 | 9 |
| **Total** | | | | **60** |

**验证策略:** 运行代表性样本来覆盖所有方法、噪声级别和prior值

---

## Task 1: DPO Validation (nl10, nl50, nl100)

**Configs:**
- `configs/local/toy_experiment/toy_dpo_nl10.yaml`
- `configs/local/toy_experiment/toy_dpo_nl50.yaml`
- `configs/local/toy_experiment/toy_dpo_nl100.yaml`

- [ ] **Run DPO nl10**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_dpo_nl10.yaml
```

- [ ] **Run DPO nl50**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_dpo_nl50.yaml
```

- [ ] **Run DPO nl100**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_dpo_nl100.yaml
```

---

## Task 2: ORPO Validation (nl10, nl50, nl100)

**Configs:**
- `configs/local/toy_experiment/toy_orpo_nl10.yaml`
- `configs/local/toy_experiment/toy_orpo_nl50.yaml`
- `configs/local/toy_experiment/toy_orpo_nl100.yaml`

- [ ] **Run ORPO nl10**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_orpo_nl10.yaml
```

- [ ] **Run ORPO nl50**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_orpo_nl50.yaml
```

- [ ] **Run ORPO nl100**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_orpo_nl100.yaml
```

---

## Task 3: SimPO Validation (nl10, nl50, nl100)

**Configs:**
- `configs/local/toy_experiment/toy_simpo_nl10.yaml`
- `configs/local/toy_experiment/toy_simpo_nl50.yaml`
- `configs/local/toy_experiment/toy_simpo_nl100.yaml`

- [ ] **Run SimPO nl10**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_simpo_nl10.yaml
```

- [ ] **Run SimPO nl50**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_simpo_nl50.yaml
```

- [ ] **Run SimPO nl100**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_simpo_nl100.yaml
```

---

## Task 4: SSPO Validation (nl10, nl50, nl100 + Priors + Noise)

**Configs - No Noise:**
- `configs/local/toy_experiment/toy_sspo_nl10.yaml` (prior=0.5 default)
- `configs/local/toy_experiment/toy_sspo_nl10_prior1.yaml` (prior=0.1)
- `configs/local/toy_experiment/toy_sspo_nl10_prior3.yaml` (prior=0.3)
- `configs/local/toy_experiment/toy_sspo_nl10_prior7.yaml` (prior=0.7)
- `configs/local/toy_experiment/toy_sspo_nl10_prior9.yaml` (prior=0.9)
- `configs/local/toy_experiment/toy_sspo_nl50.yaml`
- `configs/local/toy_experiment/toy_sspo_nl100.yaml`

**Configs - With Noise:**
- `configs/local/toy_experiment/toy_sspo_nl10_noise50.yaml`
- `configs/local/toy_experiment/toy_sspo_nl50_noise50.yaml`
- `configs/local/toy_experiment/toy_sspo_nl100_noise50.yaml`

- [ ] **Run SSPO nl10 (prior=0.5, no noise)**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_sspo_nl10.yaml
```

- [ ] **Run SSPO nl10 prior sensitivity (priors 0.1, 0.3, 0.7, 0.9)**
```bash
for p in 1 3 7 9; do
  LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_sspo_nl10_prior${p}.yaml
done
```

- [ ] **Run SSPO nl10, nl50, nl100 with 50% noise**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_sspo_nl10_noise50.yaml
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_sspo_nl50_noise50.yaml
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_sspo_nl100_noise50.yaml
```

---

## Verification Criteria

每个训练应满足:
1. 模型加载 `./cache/mistralai/Mistral-7B-Instruct-v0.2`
2. 数据加载 `src/data/toy*.json`
3. 完成至少5步训练无错误
4. 保存checkpoint到output_dir

**成功标志:**
- 日志中出现loss值
- `output_dir/` 包含 `adapter_model.bin` (LoRA权重)
- A6000 48GB处理nl10无CUDA OOM

---

## Execution Handoff

**计划保存于 `docs/superpowers/plans/2026-04-25-lightweight-validation.md`**

**建议执行方式:** 手动顺序执行，便于监控错误

**快速验证命令 (一个方法):**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_dpo_nl10.yaml
```
