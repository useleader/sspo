# Complete Experiment Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate all SSPO paper experiments can run on A6000 GPU using lightweight configs, then verify full experiment pipeline.

**Architecture:** Use existing toy_experiment configs for local validation, then run cluster configs with reduced settings.

**Tech Stack:** PyTorch, LLaMA-Factory fork, A6000 48GB GPU, toy + real datasets

---

## Status Review

**已完成:**
- Toy configs: 87个 (DPO/ORPO/SimPO/SSPO, nl10/50/100, noise 0/10/30/50%, priors)
- SFT Hybrid: 6个 (DPO+SFT, SimPO+SFT for 3 models)
- Real configs: 29个 cluster configs for mistral-7b-it
- 100% data config exists: `configs/cluster/mistral-7b-it/sspo/mistral-7b-it_sspo_fb1.0_ch0.1.yaml`

**缺失:**
- Figure 2 loss contribution logging (commented out in trainer.py:593-594)
- Figure 3 reward distribution logging

---

## Task 1: Verify SFT Hybrid Implementation

**检查SFT Hybrid是否真正实现:**

- [ ] **Step 1: 检查trainer.py中pref_ftx的处理**

```bash
grep -n "pref_ftx" /home/yzm/sspo/src/src_sspo/llamafactory/train/dpo/trainer.py
```

Expected: 找到 `self.ftx_gamma = finetuning_args.pref_ftx` 和相关逻辑

- [ ] **Step 2: 确认SFT hybrid configs配置正确**

Run: `cat configs/local/sft_hybrid/mistral-7b-it_dpo_sft_hybrid.yaml`
Expected: `pref_ftx: 1.0` 字段存在

---

## Task 2: Update generate_model_configs.py for SFT Hybrid

**Files:**
- Modify: `scripts/generate_model_configs.py`

- [ ] **Step 1: 添加SFT hybrid配置生成**

在 `generate_model_configs.py` 中添加 `--type sft_hybrid` 选项:

```python
def generate_sft_hybrid_configs(output_dir: Path, models: List[str]) -> List[Path]:
    """Generate DPO+SFT and SimPO+SFT configs (Paper Table 10)."""
    generated = []
    sft_dir = output_dir / "sft_hybrid"
    sft_dir.mkdir(parents=True, exist_ok=True)

    for model_key in models:
        model = MODELS[model_key]
        for method, pref_loss in [("dpo", "sigmoid"), ("simpo", "simpo")]:
            config = REAL_TEMPLATE.copy()
            config["model_name_or_path"] = model.hf_path
            config["trust_remote_code"] = model.trust_remote_code
            config["cache_dir"] = model.cache_dir
            config["dataset"] = "ultra_combined_fb0.01_ch0.1"
            config["template"] = "mistral"
            config["cutoff_len"] = 1024
            config["num_train_epochs"] = 1
            config["per_device_train_batch_size"] = 4
            config["gradient_accumulation_steps"] = 16
            config["learning_rate"] = 1e-5
            config["pref_loss"] = pref_loss
            config["pref_beta"] = 0.1 if method == "dpo" else 2.0
            config["pref_ftx"] = 1.0
            config["ref_model"] = model.hf_path
            config["output_dir"] = f"./saves/{model.name}/{method}_sft_hybrid"

            filename = f"{model.name}_{method}_sft_hybrid.yaml"
            filepath = sft_dir / filename

            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Generated: {filepath}")
            generated.append(filepath)

    return generated
```

- [ ] **Step 2: 更新main()处理sft_hybrid类型**

```python
if args.type in ["sft_hybrid", "all"]:
    print("\n=== Generating SFT Hybrid Configs ===")
    all_generated += generate_sft_hybrid_configs(output_dir, models)
```

---

## Task 3: Generate All Configs

- [ ] **Step 1: 生成所有配置文件**

```bash
cd /home/yzm/sspo
python scripts/generate_model_configs.py --type all --output configs/
```

Expected output:
```
=== Generating Real-Data Configs ===
=== Generating Toy Experiment Configs ===
=== Generating Large Paired Data Configs ===
=== Generating SFT Hybrid Configs ===
Total configs generated: N
```

- [ ] **Step 2: 验证生成的配置文件数量**

```bash
echo "Toy configs: $(ls configs/local/toy_experiment/*.yaml | wc -l)"
echo "Cluster configs: $(find configs/cluster -name '*.yaml' | wc -l)"
echo "SFT Hybrid configs: $(ls configs/local/sft_hybrid/*.yaml | wc -l)"
```

Expected: Toy ~87+, Cluster 29+, SFT Hybrid 6+

---

## Task 4: Lightweight Toy Experiment Validation

**在A6000上运行轻量toy实验验证代码可用性:**

- [ ] **Step 1: 测试DPO nl10**

```bash
cd /home/yzm/sspo
LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_dpo_nl10.yaml 2>&1 | head -50
```

Expected: 加载模型，运行训练步骤，loss下降

- [ ] **Step 2: 测试SSPO nl10**

```bash
cd /home/yzm/sspo
LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_sspo_nl10.yaml 2>&1 | head -50
```

Expected: 同上，SSPO loss应该比DPO高一些

- [ ] **Step 3: 测试ORPO nl10**

```bash
cd /home/yzm/sspo
LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_orpo_nl10.yaml 2>&1 | head -50
```

- [ ] **Step 4: 测试SimPO nl10**

```bash
cd /home/yzm/sspo
LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_simpo_nl10.yaml 2>&1 | head -50
```

- [ ] **Step 5: 测试SSPO with noise nl10**

```bash
cd /home/yzm/sspo
LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_sspo_nl10_noise50.yaml 2>&1 | head -50
```

- [ ] **Step 6: 测试SSPO prior sensitivity nl10**

```bash
cd /home/yzm/sspo
for p in 1 3 7 9; do
  LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_sspo_nl10_prior${p}.yaml 2>&1 | head -30
done
```

---

## Task 5: Verify Checkpoint Saving

- [ ] **Step 1: 检查DPO checkpoint**

```bash
ls -la /home/yzm/sspo/saves/toy/dpo/nl10/ 2>/dev/null | head -10
```

Expected: `adapter_model.bin` 和 `adapter_config.json` 存在

- [ ] **Step 2: 检查SSPO checkpoint**

```bash
ls -la /home/yzm/sspo/saves/toy/sspo/nl10/ 2>/dev/null | head -10
```

---

## Task 6: SFT Hybrid Validation (Optional - if time permits)

- [ ] **Step 1: 测试DPO+SFT hybrid**

```bash
cd /home/yzm/sspo
LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/sft_hybrid/mistral-7b-it_dpo_sft_hybrid.yaml 2>&1 | head -50
```

---

## Verification Checklist

每个实验通过的标准:
- [ ] 模型加载成功
- [ ] 数据加载成功
- [ ] 训练至少5步无错误
- [ ] Loss下降
- [ ] Checkpoint保存成功

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-25-complete-validation.md`.**

**Two execution options:**

**1. Subagent-Driven** - Dispatch subagent per task for parallel execution

**2. Inline Execution** - Run each validation manually in sequence

**Recommended approach:** Option 2 (Inline) since you want to monitor GPU memory and errors interactively on A6000.

**First command:**
```bash
cd /home/yzm/sspo && LOCAL=1 GPUS=1 bash scripts/train.sh configs/local/toy_experiment/toy_dpo_nl10.yaml
```
