#!/usr/bin/env python3
"""
Unified config generator for SSPO paper experiments.

Usage:
    # Real-data configs (cluster)
    python scripts/generate_model_configs.py --type real --method sspo --output configs/

    # Toy experiment configs
    python scripts/generate_model_configs.py --type toy --output configs/

    # Large paired data configs (Table 11)
    python scripts/generate_model_configs.py --type large_paired --output configs/

    # All types
    python scripts/generate_model_configs.py --type all --output configs/
"""

import argparse
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


@dataclass
class ModelConfig:
    name: str
    hf_path: str
    trust_remote_code: bool = True
    cache_dir: str = "./cache"


MODELS = {
    "mistral": ModelConfig(
        name="mistral-7b-it",
        hf_path="mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True,
        cache_dir="./cache/mistral-7b-it",
    ),
    "llama3": ModelConfig(
        name="llama3-8b-it",
        hf_path="meta-llama/Meta-Llama-3-8B-Instruct",
        trust_remote_code=True,
        cache_dir="./cache/llama3-8b-it",
    ),
    "qwen2": ModelConfig(
        name="qwen2-7b-it",
        hf_path="Qwen/Qwen2-7B-Instruct",
        trust_remote_code=True,
        cache_dir="./cache/qwen2-7b-it",
    ),
}

# Paper Table 7 hyperparams for toy experiments
TOY_METHOD_PARAMS = {
    "dpo": {"lr": 1e-3, "pref_loss": "sigmoid", "pref_beta": 0.1},
    "orpo": {"lr": 1e-5, "pref_loss": "orpo", "pref_beta": 0.1},
    "simpo": {"lr": 1e-3, "pref_loss": "simpo", "pref_beta": 2.0},
}

SSPO_TOY_PARAMS = {
    (10, 0.1): {"lr": 1e-3, "decay": 0.01},
    (10, 0.3): {"lr": 5e-3, "decay": 0.001},
    (10, 0.5): {"lr": 2e-2, "decay": 0.03},
    (10, 0.7): {"lr": 1e-2, "decay": 0.01},
    (10, 0.9): {"lr": 5e-3, "decay": 0.001},
    (50, 0.1): {"lr": 3e-3, "decay": 0.001},
    (50, 0.3): {"lr": 1e-3, "decay": 0.03},
    (50, 0.5): {"lr": 1e-3, "decay": 0.001},
    (50, 0.7): {"lr": 1e-3, "decay": 0.001},
    (50, 0.9): {"lr": 1e-3, "decay": 0.03},
    (100, 0.1): {"lr": 1e-3, "decay": 0.03},
    (100, 0.3): {"lr": 5e-3, "decay": 0.05},
    (100, 0.5): {"lr": 5e-3, "decay": 0.005},
    (100, 0.7): {"lr": 1e-3, "decay": 0.001},
    (100, 0.9): {"lr": 5e-4, "decay": 0.001},
}

REAL_TEMPLATE = {
    "stage": "dpo",
    "do_train": True,
    "finetuning_type": "lora",
    "template": "default",
    "cutoff_len": 2048,
    "max_samples": 10000000,
    "overwrite_cache": True,
    "preprocessing_num_workers": 12,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5e-7,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "lora_rank": 8,
    "lora_target": "all",
    "val_size": 0.0,
    "per_device_eval_batch_size": 4,
    "eval_strategy": "no",
    "logging_steps": 20,
    "save_steps": 300,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "bf16": True,
}

TOY_TEMPLATE = {
    "stage": "dpo",
    "do_train": True,
    "finetuning_type": "lora",
    "template": "default",
    "cutoff_len": 512,
    "max_samples": 10,
    "overwrite_cache": False,
    "preprocessing_num_workers": 4,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "lora_rank": 8,
    "lora_target": "all",
    "val_size": 0.0,
    "per_device_eval_batch_size": 2,
    "eval_strategy": "no",
    "logging_steps": 20,
    "save_steps": 200,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "bf16": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SSPO paper experiment configs")
    parser.add_argument("--type", type=str, default="real",
                        choices=["real", "toy", "large_paired", "sft_hybrid", "all"],
                        help="Type of configs to generate")
    parser.add_argument("--output", type=str, default="configs",
                        help="Output directory")
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "sspo", "dpo", "orpo", "simpo", "kto", "ssrm", "spa"],
                        help="Training method")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "mistral", "llama3", "qwen2"],
                        help="Model")
    parser.add_argument("--fb", type=float, nargs="+", default=[0.01, 0.05, 0.10],
                        help="Labeled data ratios")
    return parser.parse_args()


def get_toy_dataset(method: str, nl: int, noise: int) -> str:
    if method in ["dpo", "orpo", "simpo"]:
        suffix = f"_noise{noise}" if noise > 0 else ""
        return f"toy_paired_nl{nl}{suffix}"
    else:
        suffix = f"_noise{noise}" if noise > 0 else ""
        return f"toy_nl{nl}{suffix}"


def generate_toy_configs(output_dir: Path) -> List[Path]:
    """Generate toy experiment configs (Paper Table 6/7)."""
    generated = []
    noise_levels = [0, 10, 30, 50]
    nl_values = [10, 50, 100]
    priors = [0.1, 0.3, 0.5, 0.7, 0.9]
    toy_dir = output_dir / "toy_experiment"
    toy_dir.mkdir(parents=True, exist_ok=True)

    for method in ["dpo", "orpo", "simpo"]:
        params = TOY_METHOD_PARAMS[method]
        for nl in nl_values:
            for noise in noise_levels:
                config = TOY_TEMPLATE.copy()
                config["model_name_or_path"] = "./cache/mistralai/Mistral-7B-Instruct-v0.2"
                config["trust_remote_code"] = True
                config["dataset"] = get_toy_dataset(method, nl, noise)
                config["dataset_dir"] = "src/data"
                config["learning_rate"] = params["lr"]
                config["pref_loss"] = params["pref_loss"]
                config["pref_beta"] = params["pref_beta"]
                config["cache_dir"] = "./cache"

                noise_str = f"_noise{noise}" if noise > 0 else ""
                config["output_dir"] = f"./saves/toy/{method}/nl{nl}{noise_str}"

                filename = f"toy_{method}_nl{nl}{noise_str}.yaml"
                filepath = toy_dir / filename
                if not filepath.exists():
                    with open(filepath, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    print(f"Created: {filename}")
                    generated.append(filepath)
                else:
                    print(f"Exists: {filename}")

    # SSPO: noise 0 with all priors
    for nl in nl_values:
        for prior in priors:
            config = TOY_TEMPLATE.copy()
            config["model_name_or_path"] = "./cache/mistralai/Mistral-7B-Instruct-v0.2"
            config["trust_remote_code"] = True
            config["dataset"] = get_toy_dataset("sspo", nl, 0)
            config["dataset_dir"] = "src/data"
            config["cache_dir"] = "./cache"

            key = (nl, prior)
            sp = SSPO_TOY_PARAMS.get(key, SSPO_TOY_PARAMS[(nl, 0.5)])
            config["learning_rate"] = sp["lr"]
            config["pref_loss"] = "sspo"
            config["pref_beta"] = 10.0
            config["sspo_gamma_0"] = 1.0
            config["sspo_gamma_min"] = 0.22
            config["sspo_gamma_decay"] = sp["decay"]
            config["sspo_prior"] = prior

            if prior != 0.5:
                config["output_dir"] = f"./saves/toy/sspo/nl{nl}_prior0{int(prior*10)}"
            else:
                config["output_dir"] = f"./saves/toy/sspo/nl{nl}"

            filename = f"toy_sspo_nl{nl}_prior{int(prior*10)}.yaml"
            filepath = toy_dir / filename
            if not filepath.exists():
                with open(filepath, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"Created: {filename}")
                generated.append(filepath)
            else:
                print(f"Exists: {filename}")

    # SSPO: noise 10/30/50 with prior=0.5
    for nl in nl_values:
        for noise in [10, 30, 50]:
            config = TOY_TEMPLATE.copy()
            config["model_name_or_path"] = "./cache/mistralai/Mistral-7B-Instruct-v0.2"
            config["trust_remote_code"] = True
            config["dataset"] = get_toy_dataset("sspo", nl, noise)
            config["dataset_dir"] = "src/data"
            config["cache_dir"] = "./cache"

            key = (nl, 0.5)
            sp = SSPO_TOY_PARAMS.get(key, SSPO_TOY_PARAMS[(nl, 0.5)])
            config["learning_rate"] = sp["lr"]
            config["pref_loss"] = "sspo"
            config["pref_beta"] = 10.0
            config["sspo_gamma_0"] = 1.0
            config["sspo_gamma_min"] = 0.22
            config["sspo_gamma_decay"] = sp["decay"]
            config["sspo_prior"] = 0.5
            config["output_dir"] = f"./saves/toy/sspo/nl{nl}_noise{noise}"

            filename = f"toy_sspo_nl{nl}_noise{noise}.yaml"
            filepath = toy_dir / filename
            if not filepath.exists():
                with open(filepath, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"Created: {filename}")
                generated.append(filepath)
            else:
                print(f"Exists: {filename}")

    return generated


def generate_real_configs(output_dir: Path, methods: List[str], models: List[str], fb_ratios: List[float]) -> List[Path]:
    """Generate real-data configs for cluster (Paper Table 2/3/4)."""
    generated = []
    for model_key in models:
        model = MODELS[model_key]
        for method in methods:
            for fb in fb_ratios:
                config = REAL_TEMPLATE.copy()
                config["model_name_or_path"] = model.hf_path
                config["trust_remote_code"] = model.trust_remote_code
                config["cache_dir"] = model.cache_dir
                config["dataset"] = f"ultra_combined_fb{fb}_ch0.1"
                config["output_dir"] = f"./saves_{model.name}/{method}/fb{fb}_ch0.1"

                if method == "sspo":
                    config["pref_loss"] = "sspo"
                    config["pref_beta"] = 2.0
                    config["simpo_gamma"] = 2.0
                    config["sspo_gamma_0"] = 1.0
                    config["sspo_gamma_min"] = 0.22
                    config["sspo_gamma_decay"] = 0.001
                    config["sspo_prior"] = 0.5
                    config["sspo_base"] = "simpo"
                elif method == "dpo":
                    config["pref_loss"] = "sigmoid"
                    config["pref_beta"] = 0.1
                elif method == "orpo":
                    config["pref_loss"] = "orpo"
                    config["pref_beta"] = 0.1
                elif method == "simpo":
                    config["pref_loss"] = "simpo"
                    config["pref_beta"] = 2.0
                    config["simpo_gamma"] = 2.0
                elif method == "kto":
                    config["stage"] = "kto"
                    config["pref_loss"] = "kto_pair"
                    config["pref_beta"] = 0.1
                elif method == "ssrm":
                    config["pref_loss"] = "ssrm"
                    config["ssrm_prior"] = 0.5
                    config["ssrm_iterations"] = 3
                    config["ssrm_threshold"] = 0.9
                elif method == "spa":
                    config["pref_loss"] = "spa"
                    config["spa_iterations"] = 3
                    config["spa_expansion_ratio"] = 0.1

                method_dir = output_dir / model.name / method
                method_dir.mkdir(parents=True, exist_ok=True)
                filename = f"fb{fb}_ch0.1_{method}_{model.name}.yaml"
                filepath = method_dir / filename

                with open(filepath, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"Generated: {filepath}")
                generated.append(filepath)

    return generated


def generate_large_paired_configs(output_dir: Path, models: List[str]) -> List[Path]:
    """Generate large paired data configs (Paper Table 11)."""
    generated = []
    nl_values = [100, 1000, 5000, 10000]
    total = 60000

    for nl in nl_values:
        fb = nl / total
        for model_key in models:
            model = MODELS[model_key]
            config = REAL_TEMPLATE.copy()
            config["model_name_or_path"] = model.hf_path
            config["trust_remote_code"] = model.trust_remote_code
            config["cache_dir"] = model.cache_dir
            config["dataset"] = f"ultra_paired_fb{fb:.4f}"
            config["output_dir"] = f"./saves_{model.name}/large_paired/nl{nl}"

            large_dir = output_dir / model.name / "large_paired"
            large_dir.mkdir(parents=True, exist_ok=True)
            filename = f"nl{nl}_large_paired_{model.name}.yaml"
            filepath = large_dir / filename

            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Generated: {filepath}")
            generated.append(filepath)

    return generated


def generate_sft_hybrid_configs(output_dir: Path, models: List[str]) -> List[Path]:
    """Generate SFT Hybrid configs for DPO+SFT and SimPo+SFT (Paper Section 4.3)."""
    generated = []
    methods = ["dpo", "simpo"]
    sft_dir = output_dir / "local" / "sft_hybrid"
    sft_dir.mkdir(parents=True, exist_ok=True)

    for model_key in models:
        model = MODELS[model_key]
        for method in methods:
            config = REAL_TEMPLATE.copy()
            config["model_name_or_path"] = model.hf_path
            config["trust_remote_code"] = model.trust_remote_code
            config["cache_dir"] = model.cache_dir
            config["template"] = "mistral"
            config["dataset"] = "ultra_combined_fb0.01_ch0.1"
            config["cutoff_len"] = 1024
            config["num_train_epochs"] = 1
            config["per_device_train_batch_size"] = 4
            config["gradient_accumulation_steps"] = 16
            config["learning_rate"] = 1e-5
            config["pref_ftx"] = 1.0
            config["ref_model"] = model.hf_path
            config["output_dir"] = f"./saves/{model.name}/{method}_sft_hybrid"

            if method == "dpo":
                config["pref_loss"] = "sigmoid"
                config["pref_beta"] = 0.1
            elif method == "simpo":
                config["pref_loss"] = "simpo"
                config["pref_beta"] = 2.0
                config["simpo_gamma"] = 2.0

            filename = f"{model.name}_{method}_sft_hybrid.yaml"
            filepath = sft_dir / filename
            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Generated: {filepath}")
            generated.append(filepath)

    return generated


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models = list(MODELS.keys())
    else:
        models = [args.model]

    if args.method == "all":
        methods = ["sspo", "dpo", "orpo", "simpo", "kto", "ssrm", "spa"]
    else:
        methods = [args.method]

    all_generated = []

    if args.type in ["real", "all"]:
        print("\n=== Generating Real-Data Configs ===")
        all_generated += generate_real_configs(output_dir, methods, models, args.fb)

    if args.type in ["toy", "all"]:
        print("\n=== Generating Toy Experiment Configs ===")
        all_generated += generate_toy_configs(output_dir)

    if args.type in ["large_paired", "all"]:
        print("\n=== Generating Large Paired Data Configs ===")
        all_generated += generate_large_paired_configs(output_dir, models)

    if args.type in ["sft_hybrid", "all"]:
        print("\n=== Generating SFT Hybrid Configs ===")
        all_generated += generate_sft_hybrid_configs(output_dir, models)

    print(f"\n=== Total configs generated: {len(all_generated)} ===")


if __name__ == "__main__":
    main()
