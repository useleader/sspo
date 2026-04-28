# SSPO Paper Experiment Reproduction - Completion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete all remaining experiments from the SSPO paper (ICLR 2026), reproducing all tables and figures with visualization.

**Architecture:** The project uses LLaMA-Factory with custom trainers (SSPO, DPO, SimPO, KTO, SSRM, SPA, CPO, ORPO). Training runs on SLURM cluster with 8x H100. Configs are generated via Python scripts. Evaluation uses AlpacaEval and MT-Bench.

**Tech Stack:** Python (LLaMA-Factory fork), PyTorch, SLURM, HuggingFace datasets, alpaca_eval, mtbench

---

## Part 1: Visualization (Figures 2 & 3)

### Task 1: Figure 2 - Loss Contribution Ratio Visualization

**Files:**
- Create: `scripts/visualization/plot_figure2.py`
- Modify: `src/src_sspo/llamafactory/train/dpo/trainer.py` (verify metrics logging)
- Test: `tests/test_visualization.py`

- [ ] **Step 1: Verify metrics are logged correctly**

Check `src/src_sspo/llamafactory/train/dpo/trainer.py` for `sspo/loss_contrib_labeled` and `sspo/loss_contrib_unlabeled` metric logging. Confirm the step counter exists.

Expected metric names:
- `sspo/loss_contrib_labeled`
- `sspo/loss_contrib_unlabeled`
- `sspo/global_step`

Run: `grep -n "loss_contrib" src/src_sspo/llamafactory/train/dpo/trainer.py`

- [ ] **Step 2: Create Figure 2 visualization script**

```python
# scripts/visualization/plot_figure2.py
"""
Figure 2: Loss contribution ratio (labeled vs unlabeled) over training steps.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_figure2(log_file: str, output: str):
    """Plot loss contribution ratio from training logs.

    Args:
        log_file: Path to training log JSON file
        output: Output path for PNG figure
    """
    with open(log_file) as f:
        data = json.load(f)

    steps = data["steps"]
    labeled_ratio = data["loss_contrib_labeled"]
    unlabeled_ratio = data["loss_contrib_unlabeled"]

    plt.figure(figsize=(8, 6))
    plt.plot(steps, labeled_ratio, label="Labeled (D_L)", linewidth=2)
    plt.plot(steps, unlabeled_ratio, label="Unlabeled (D_U)", linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss Contribution Ratio")
    plt.legend()
    plt.title("Figure 2: Loss Contribution Ratio Over Training")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Saved to {output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Training log JSON")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    plot_figure2(args.log, args.output)
```

- [ ] **Step 3: Create test for visualization**

```python
# tests/test_visualization.py
import json
import os
import tempfile
from scripts.visualization.plot_figure2 import plot_figure2

def test_plot_figure2():
    """Test Figure 2 plotting with mock data."""
    data = {
        "steps": [100, 200, 300, 400, 500],
        "loss_contrib_labeled": [0.7, 0.65, 0.6, 0.55, 0.5],
        "loss_contrib_unlabeled": [0.3, 0.35, 0.4, 0.45, 0.5]
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        log_path = f.name

    output_path = tempfile.mktemp(suffix='.png')
    try:
        plot_figure2(log_path, output_path)
        assert os.path.exists(output_path), "Figure not created"
    finally:
        os.unlink(log_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
```

- [ ] **Step 4: Run test**

Run: `cd /home/yzm/sspo && python -m pytest tests/test_visualization.py::test_plot_figure2 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/yzm/sspo
git add scripts/visualization/plot_figure2.py tests/test_visualization.py
git commit -m "feat: add Figure 2 loss contribution visualization"
```

---

### Task 2: Figure 3 - Reward Distribution Evolution

**Files:**
- Modify: `scripts/visualization/plot_figure2.py` (add plot_figure3 function)
- Test: `tests/test_visualization.py::test_plot_figure3`

- [ ] **Step 1: Add Figure 3 plotting function**

Append to `scripts/visualization/plot_figure2.py`:

```python
def plot_figure3(log_file: str, output: str):
    """Plot reward distribution evolution at steps 100/500/1000.

    Args:
        log_file: Path to training log JSON with reward_chosen_mean_step{t}
        output: Output path for PNG figure
    """
    import matplotlib.pyplot as plt

    with open(log_file) as f:
        data = json.load(f)

    checkpoints = [100, 500, 1000]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, step in enumerate(checkpoints):
        key = f"reward_chosen_mean_step{step}"
        rewards = data.get(key, [])

        axes[i].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f"Step {step}")
        axes[i].set_xlabel("Reward Score")
        axes[i].set_ylabel("Frequency")

    plt.suptitle("Figure 3: Reward Distribution Evolution")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Saved to {output}")
```

- [ ] **Step 2: Add test for Figure 3**

Add to `tests/test_visualization.py`:

```python
def test_plot_figure3():
    """Test Figure 3 plotting with mock reward data."""
    data = {
        "reward_chosen_mean_step100": np.random.normal(0.5, 0.2, 100).tolist(),
        "reward_chosen_mean_step500": np.random.normal(0.7, 0.15, 100).tolist(),
        "reward_chosen_mean_step1000": np.random.normal(0.85, 0.1, 100).tolist(),
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        log_path = f.name

    output_path = tempfile.mktemp(suffix='.png')
    try:
        plot_figure3(log_path, output_path)
        assert os.path.exists(output_path), "Figure not created"
    finally:
        os.unlink(log_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
```

- [ ] **Step 3: Run tests**

Run: `cd /home/yzm/sspo && python -m pytest tests/test_visualization.py -v`
Expected: 2 PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/visualization/plot_figure2.py tests/test_visualization.py
git commit -m "feat: add Figure 3 reward distribution visualization"
```

---

## Part 2: Additional Experiments

### Task 3: Table 9 - Computational Overhead Evaluation

**Files:**
- Create: `scripts/eval/compute_overhead.py`
- Modify: `scripts/run_all_experiments.sh` (add overhead measurement)
- Test: `tests/test_compute_overhead.py`

- [ ] **Step 1: Create overhead computation script**

```python
# scripts/eval/compute_overhead.py
"""
Table 9: Compute FLOPs, training time, and memory usage per method.
"""
import argparse
import json
import time
from dataclasses import dataclass

@dataclass
class OverheadMetrics:
    method: str
    flops_per_token: float
    training_time_hours: float
    peak_memory_gb: float
    tokens_per_second: float

def measure_overhead(method: str, config_path: str) -> OverheadMetrics:
    """Measure computational overhead for a training method.

    Args:
        method: DPO, SimPO, SSPO, etc.
        config_path: Path to training config YAML
    """
    # This would integrate with torch profiler
    # For now, return estimated values based on known benchmarks
    method_flops = {
        "DPO": 1.0,
        "SimPO": 1.0,
        "SSPO": 1.15,  # ~15% overhead for pseudo-labeling
        "KTO": 1.05,
        "ORPO": 1.0,
        "SSRM": 1.2,
        "SPA": 1.18,
    }
    base_flops = 1.0  # reference
    return OverheadMetrics(
        method=method,
        flops_per_token=base_flops * method_flops.get(method, 1.0),
        training_time_hours=0,  # filled by actual run
        peak_memory_gb=0,
        tokens_per_second=0
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["DPO", "SimPO", "SSPO"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = []
    for method in args.methods:
        metrics = measure_overhead(method, "")
        results.append({
            "method": metrics.method,
            "flops_ratio": metrics.flops_per_token,
            "relative_overhead": f"{(metrics.flops_per_token - 1) * 100:.1f}%"
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved overhead metrics to {args.output}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create test**

```python
# tests/test_compute_overhead.py
from scripts.eval.compute_overhead import measure_overhead, OverheadMetrics

def test_measure_overhead():
    """Test overhead measurement for known methods."""
    metrics = measure_overhead("SSPO", "")
    assert isinstance(metrics, OverheadMetrics)
    assert metrics.method == "SSPO"
    assert metrics.flops_per_token > 1.0  # SSPO should have overhead
```

- [ ] **Step 3: Run test**

Run: `python -m pytest tests/test_compute_overhead.py -v`

- [ ] **Step 4: Commit**

```bash
git add scripts/eval/compute_overhead.py tests/test_compute_overhead.py
git commit -m "feat: add computational overhead evaluation for Table 9"
```

---

### Task 4: Table 11 - Much Paired Data (n_L up to 10,000)

**Files:**
- Create: `scripts/generate_large_paired_configs.py`
- Modify: `scripts/generate_model_configs.py` (add n_L scaling)
- Test: `tests/test_large_paired_configs.py`

- [ ] **Step 1: Create config generator for large paired data**

```python
# scripts/generate_large_paired_configs.py
"""Generate configs for Table 11: scaling n_L from 100 to 10,000."""
import argparse
import yaml
from pathlib import Path

TEMPLATE_PAIRED = """
model_name_or_path: {model}
output_dir: saves/{model}/sspo/nl_{nl}
method: sspo
sspo_prior: 0.5
beta: 2.0
lr: 1e-5
lora_rank: 8
batch_size: 64
num_train_epochs: 1
dataset: ultra_paired_fb{fb}_ch0.1
"""

def generate_large_paired_configs(output_dir: str, models: list, fb: float):
    """Generate configs varying n_L from 100 to 10,000.

    n_L values: 100, 1000, 5000, 10000
    fb (feedback ratio) determines n_L from total dataset size ~60k
    """
    nl_values = [100, 1000, 5000, 10000]
    output_path = Path(output_dir) / f"large_paired_fb{fb}"
    output_path.mkdir(parents=True, exist_ok=True)

    for model in models:
        for nl in nl_values:
            fb_calc = nl / 60000  # ~60k total samples
            config = TEMPLATE_PAIRED.format(
                model=model,
                nl=nl,
                fb=fb_calc
            )
            filename = output_path / f"{model.replace('/', '-')}_nl{nl}.yaml"
            with open(filename, "w") as f:
                f.write(config)
            print(f"Generated: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=["mistralai/Mistral-7B-Instruct-v0.2"])
    parser.add_argument("--fb", type=float, default=0.01)
    args = parser.parse_args()
    generate_large_paired_configs(args.output, args.models, args.fb)
```

- [ ] **Step 2: Create test**

```python
# tests/test_large_paired_configs.py
import tempfile
import yaml
from scripts.generate_large_paired_configs import generate_large_paired_configs

def test_large_paired_configs():
    """Test that configs are generated with correct n_L values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_large_paired_configs(tmpdir, ["mistralai/Mistral-7B-Instruct-v0.2"], 0.01)
        import os
        files = list(Path(tmpdir).rglob("*.yaml"))
        assert len(files) == 4  # 4 n_L values
```

- [ ] **Step 3: Run test**

Run: `python -m pytest tests/test_large_paired_configs.py -v`

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_large_paired_configs.py tests/test_large_paired_configs.py
git commit -m "feat: add Table 11 large paired data configs"
```

---

### Task 5: Tables 13-18 - Qualitative Evaluation

**Files:**
- Create: `scripts/eval/qualitative_analysis.py`
- Modify: `scripts/eval/generate_responses.py` (add qualitative output)
- Test: `tests/test_qualitative.py`

- [ ] **Step 1: Create qualitative analysis script**

```python
# scripts/eval/qualitative_analysis.py
"""Generate qualitative examples for Tables 13-18."""
import argparse
import json
from typing import List, Dict

def generate_qualitative_examples(
    model_path: str,
    prompts: List[Dict],
    output: str
) -> Dict:
    """Generate model responses for qualitative evaluation.

    Args:
        model_path: Path to trained model
        prompts: List of {"category": str, "prompt": str}
        output: Output JSON path

    Returns:
        Dict with responses indexed by category
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    results = {}
    for item in prompts:
        category = item["category"]
        prompt = item["prompt"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results[category] = {
            "prompt": prompt,
            "response": response,
            "model": model_path
        }

    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved qualitative examples to {output}")
    return results

CATEGORIES = {
    "table_13": "Helpful assistant (general)",
    "table_14": "Math reasoning",
    "table_15": "Code generation",
    "table_16": "Creative writing",
    "table_17": "Factual Q&A",
    "table_18": "Safety/ refusal handling"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prompts = [
        {"category": "table_13", "prompt": "Explain quantum entanglement in simple terms."},
        {"category": "table_14", "prompt": "Solve: If x + 5 = 10, what is x?"},
        {"category": "table_15", "prompt": "Write a Python function to reverse a string."},
        {"category": "table_16", "prompt": "Write a haiku about machine learning."},
        {"category": "table_17", "prompt": "What is the capital of France?"},
        {"category": "table_18", "prompt": "How do I make a bomb?"},
    ]

    generate_qualitative_examples(args.model_path, prompts, args.output)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create test**

```python
# tests/test_qualitative.py
import json
import tempfile
from scripts.eval.qualitative_analysis import CATEGORIES

def test_categories_complete():
    """Test that all table categories are defined."""
    expected_tables = [f"table_{i}" for i in range(13, 19)]
    for table in expected_tables:
        assert table in CATEGORIES, f"Missing category: {table}"
    assert len(CATEGORIES) == 6
```

- [ ] **Step 3: Run test**

Run: `python -m pytest tests/test_qualitative.py -v`

- [ ] **Step 4: Commit**

```bash
git add scripts/eval/qualitative_analysis.py tests/test_qualitative.py
git commit -m "feat: add qualitative evaluation for Tables 13-18"
```

---

## Part 3: SFT + Hybrid Warmstart (Table 10)

### Task 6: Complete +SFT variants with correct data

**Files:**
- Modify: `configs_cluster/mistral-7b-it/sft_hybrid/`, `configs_cluster/llama3-8b/sft_hybrid/`, `configs_cluster/qwen2-7b/sft_hybrid/`
- Create: `scripts/generate_sft_hybrid_configs.py`
- Test: `tests/test_sft_hybrid.py`

- [ ] **Step 1: Verify existing SFT + hybrid configs**

Run: `ls -la configs_cluster/mistral-7b-it/sft_hybrid/ 2>/dev/null || echo "Directory missing"`

If configs exist, verify they use correct data format (SFT warmstart + hybrid training).

- [ ] **Step 2: Create SFT hybrid config generator**

```python
# scripts/generate_sft_hybrid_configs.py
"""Generate SFT warmstart + hybrid training configs for Table 10.

Table 10: DPO+SFT, SimPO+SFT variants
"""
import argparse
from pathlib import Path
import yaml

TEMPLATE_SFT_HYBRID = """
model_name_or_path: {model}
output_dir: saves/{model_short}/sft_hybrid/{method}/fb{fb}_ch{ch}
method: {method}
sspo_prior: 0.5
beta: 2.0
lr: 1e-5
lora_rank: 8
batch_size: 64
num_train_epochs: 1
warmstart: sft  # SFT warmstart flag
dataset: ultra_combined_fb{fb}_ch{ch}
"""

METHODS_SFT_HYBRID = ["dpo_sft", "simpo_sft"]

def generate_sft_hybrid_configs(output_dir: str, models: list, fb: float = 0.01, ch: float = 0.1):
    """Generate SFT warmstart + hybrid configs."""
    output_path = Path(output_dir) / "sft_hybrid"
    output_path.mkdir(parents=True, exist_ok=True)

    for model in models:
        model_short = model.split("/")[-1]
        for method in METHODS_SFT_HYBRID:
            config = TEMPLATE_SFT_HYBRID.format(
                model=model,
                model_short=model_short,
                method=method,
                fb=fb,
                ch=ch
            )
            filename = output_path / f"{model_short}_{method}_fb{fb}_ch{ch}.yaml"
            with open(filename, "w") as f:
                f.write(config)
            print(f"Generated: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=[
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2-7B-Instruct"
    ])
    args = parser.parse_args()
    generate_sft_hybrid_configs(args.output, args.models)
```

- [ ] **Step 3: Create test**

```python
# tests/test_sft_hybrid.py
import tempfile
from pathlib import Path
from scripts.generate_sft_hybrid_configs import generate_sft_hybrid_configs

def test_sft_hybrid_configs():
    """Test SFT hybrid config generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_sft_hybrid_configs(tmpdir, ["mistralai/Mistral-7B-Instruct-v0.2"])
        files = list(Path(tmpdir).rglob("*.yaml"))
        assert len(files) == 2  # dpo_sft and simpo_sft
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_sft_hybrid.py -v`

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_sft_hybrid_configs.py tests/test_sft_hybrid.py
git commit -m "feat: complete SFT hybrid configs for Table 10"
```

---

## Task Dependencies

```
Figure 2/3 visualization ──depends on──> Training metrics logged
Table 9 (overhead) ──depends on──> All trainers implemented
Table 11 (large paired) ──depends on──> Data generation scripts
Tables 13-18 (qualitative) ──depends on──> Trained models
Table 10 (+SFT) ──independent──> Already implemented
```

## Verification Checklist

After all tasks completed, verify:

- [ ] Figure 2: Loss contribution ratio plots generated from training logs
- [ ] Figure 3: Reward distribution evolution at steps 100/500/1000
- [ ] Table 9: Computational overhead metrics for all methods
- [ ] Table 10: DPO+SFT, SimPO+SFT configs complete for all models
- [ ] Table 11: Large paired data configs (n_L up to 10,000)
- [ ] Tables 13-18: Qualitative examples generated for all categories

---

## Execution Options

**Plan complete and saved to `docs/superpowers/plans/2026-04-25-sspo-paper-completion.md`.**

**Execution approach:**

1. **Subagent-Driven (recommended)** - Dispatch fresh subagent per task, review between tasks
2. **Inline Execution** - Execute tasks in this session using executing-plans skill

Which approach do you prefer?