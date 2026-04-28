#!/usr/bin/env python3
"""Generate experiment configs for all methods and data ratios."""

import os
from pathlib import Path

# Base config template
BASE_CONFIG = """{model_header}stage: dpo
do_train: true
finetuning_type: lora
template: {template}
dataset: {dataset}
dataset_dir: ./data
cutoff_len: 1024
max_samples: null
overwrite_cache: false
preprocessing_num_workers: 8
num_train_epochs: 1
per_device_train_batch_size: 4
gradient_accumulation_steps: 16
learning_rate: 1.0e-5
lr_scheduler_type: cosine
warmup_ratio: 0.1
max_grad_norm: 1.0
lora_rank: 8
lora_target: all
val_size: 0.0
per_device_eval_batch_size: 2
eval_strategy: 'no'
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true
bf16: true
cache_dir: ./cache
output_dir: ./saves/{model_short}/{method}/{output_suffix}
pref_loss: {pref_loss}
pref_beta: {pref_beta}
sspo_gamma_0: 1.0
sspo_gamma_min: 0.22
sspo_gamma_decay: 0.001
sspo_prior: 0.5
report_to: none
{ref_model}
"""

# Models
MODELS = {
    "mistral-7b-it": {
        "model_path": "./cache/mistralai/Mistral-7B-Instruct-v0.2",
        "template": "mistral",
        "pref_beta": "0.1",
        "short": "mistral-7b-it",
    },
    "qwen2-7b": {
        "model_path": "./cache/qwen2",
        "template": "qwen",
        "pref_beta": "0.1", 
        "short": "qwen2-7b",
    },
    "llama3-8b": {
        "model_path": "./cache/llama3/LLM-Research/Meta-Llama-3-8B-Instruct",
        "template": "llama3",
        "pref_beta": "0.1",
        "short": "llama3-8b",
    },
}

# Methods and their configurations
METHODS = {
    "dpo": {
        "pref_loss": "sigmoid",
        "pref_beta": "0.1",
        "datasets": ["ultra_paired_fb0.01", "ultra_paired_fb0.05", "ultra_paired_fb0.10"],
        "ch": "paired",
        "needs_ref": True,
    },
    "simpo": {
        "pref_loss": "simpo",
        "pref_beta": "2.0",
        "datasets": ["ultra_paired_fb0.01", "ultra_paired_fb0.05", "ultra_paired_fb0.10"],
        "ch": "paired",
        "needs_ref": False,
    },
    "orpo": {
        "pref_loss": "orpo",
        "pref_beta": "0.1",
        "datasets": ["ultra_paired_fb0.01", "ultra_paired_fb0.05", "ultra_paired_fb0.10"],
        "ch": "paired",
        "needs_ref": False,
    },
    "sspo": {
        "pref_loss": "sspo",
        "pref_beta": "0.1",
        "datasets": ["ultra_combined_fb0.01_ch0.1", "ultra_combined_fb0.05_ch0.1", "ultra_combined_fb0.10_ch0.1"],
        "ch": "0.1",
        "needs_ref": False,
    },
    "ssrm": {
        "pref_loss": "ssrm",
        "pref_beta": "0.1",
        "datasets": ["ultra_combined_fb0.01_ch0.1", "ultra_combined_fb0.05_ch0.1", "ultra_combined_fb0.10_ch0.1"],
        "ch": "0.1",
        "needs_ref": False,
    },
    "spa": {
        "pref_loss": "spa",
        "pref_beta": "0.1",
        "datasets": ["ultra_combined_fb0.01_ch0.1", "ultra_combined_fb0.05_ch0.1", "ultra_combined_fb0.10_ch0.1"],
        "ch": "0.1",
        "needs_ref": False,
    },
}

KTO_METHOD = {
    "kto": {
        "stage": "kto",
        "datasets": ["kto_mix_en"],
        "needs_ref": True,
    }
}

def get_dataset_name(method, ch, fb_str):
    """Get dataset name based on method and ratios."""
    if ch == "paired":
        return fb_str
    else:
        return f"ultra_combined_{fb_str}_ch{ch}"

def extract_fb(dataset):
    """Extract fb value from dataset name."""
    if "fb0.01" in dataset:
        return "0.01"
    elif "fb0.05" in dataset:
        return "0.05"
    elif "fb0.10" in dataset:
        return "0.10"
    return "0.01"

def main():
    output_dir = Path("/home/yzm/sspo/configs_cluster")
    
    for model_name, model_info in MODELS.items():
        for method_name, method_info in METHODS.items():
            for dataset in method_info["datasets"]:
                fb = extract_fb(dataset)
                ch = method_info["ch"]
                
                # Generate output filename
                if ch == "paired":
                    output_file = f"{model_name}_{method_name}_fb{fb}.yaml"
                    dataset_name = f"ultra_paired_fb{fb}"
                else:
                    output_file = f"{model_name}_{method_name}_fb{fb}_ch{ch}.yaml"
                    dataset_name = f"ultra_combined_fb{fb}_ch{ch}"
                
                # Build config
                model_header = f"model_name_or_path: {model_info['model_path']}\ntrust_remote_code: true\n"

                ref_model = f"ref_model: {model_info['model_path']}" if method_info["needs_ref"] else ""

                # Determine output_suffix format
                if ch == "paired":
                    output_suffix = f"fb{fb}"
                else:
                    output_suffix = f"fb{fb}_ch{ch}"

                config = BASE_CONFIG.format(
                    model_header=model_header,
                    template=model_info["template"],
                    dataset=dataset_name,
                    model_short=model_info["short"],
                    method=method_name,
                    output_suffix=output_suffix,
                    pref_loss=method_info["pref_loss"],
                    pref_beta=method_info["pref_beta"],
                    ref_model=ref_model,
                )
                
                # Write config
                method_dir = output_dir / model_name / method_name
                method_dir.mkdir(parents=True, exist_ok=True)
                output_path = method_dir / output_file
                with open(output_path, "w") as f:
                    f.write(config)
                print(f"Generated: {output_path}")
        
        # KTO configs (separate - uses different stage and dataset)
        kto_info = KTO_METHOD["kto"]
        for dataset in kto_info["datasets"]:
            output_file = f"{model_name}_kto.yaml"
            
            model_header = f"model_name_or_path: {model_info['model_path']}\ntrust_remote_code: true\nstage: kto\n"

            ref_model = f"ref_model: {model_info['model_path']}" if kto_info["needs_ref"] else ""

            config = BASE_CONFIG.format(
                model_header=model_header,
                template=model_info["template"],
                dataset=dataset,
                model_short=model_info["short"],
                method="kto",
                output_suffix="demo",
                pref_loss="sigmoid",  # not used for kto
                pref_beta=model_info["pref_beta"],
                ref_model=ref_model,
            )
            
            method_dir = output_dir / model_name / "kto"
            method_dir.mkdir(parents=True, exist_ok=True)
            output_path = method_dir / output_file
            with open(output_path, "w") as f:
                f.write(config)
            print(f"Generated: {output_path}")

    print("\nDone! Configs saved to /home/yzm/sspo/configs_cluster/")

if __name__ == "__main__":
    main()
