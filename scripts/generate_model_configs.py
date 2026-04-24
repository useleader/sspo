#!/usr/bin/env python3
"""
Generate training configuration YAML files for SSPO experiments.

This script generates YAML configs for:
1. Eight models: Mistral-7B, Llama-3-8B, Qwen2-7B, Phi-2, Meerkat-7B, UltraMedical, Mistral-Business, Finance-LLaMA
2. Three labeled ratios: 1%, 5%, 10%
3. Seven methods: SSPO, DPO, ORPO, SimPO, KTO, SSRM, SPA
4. Ablation experiments: Toy Experiment, Prior Sensitivity, Adaptive Scheduling

Paper configuration (from ADR-0006, ADR-0008, ADR-0010):
    - Models: Mistral-7B, Llama-3-8B, Qwen2-7B + domain models
    - LoRA rank: 8, target: all
    - Learning rate: 5e-7
    - Epochs: 3
    - Batch size: 64 (per node)
    - Context length: 2048

Model list:
    General: mistral, llama3, qwen2, phi2
    Medical: meerkat, ultramedical
    Business: mistral-business, finance

Ablation experiments (from paper section 7.3):
    --ablation toy: Toy Experiment (n_L=10,50,100, noise=0%,50%)
    --ablation prior: Prior Sensitivity (prior=0.1,0.3,0.5,0.7,0.9)
    --ablation scheduler: Adaptive Scheduling (fixed_gamma=0.1,0.5 vs adaptive)
    --ablation all: All ablation experiments

Usage:
    # Standard configs
    python scripts/generate_model_configs.py --output configs/
    python scripts/generate_model_configs.py --model mistral --method sspo --output configs/

    # Ablation experiments
    python scripts/generate_model_configs.py --ablation toy --output configs/
    python scripts/generate_model_configs.py --ablation prior --output configs/
    python scripts/generate_model_configs.py --ablation scheduler --output configs/
    python scripts/generate_model_configs.py --ablation all --output configs/

Note:
    Llama-3 models require HF_TOKEN environment variable:
    export HF_TOKEN=your_huggingface_token
"""

import argparse
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    hf_path: str
    trust_remote_code: bool = True
    cache_dir: str = "./cache"


@dataclass
class TrainingConfig:
    """Training hyperparameters from paper."""
    # Model
    stage: str = "dpo"  # dpo for SSPO/DPO/SimPO
    finetuning_type: str = "lora"
    template: str = "default"
    
    # Data
    cutoff_len: int = 2048
    max_samples: int = 10000000
    overwrite_cache: bool = True
    preprocessing_num_workers: int = 12
    
    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-7
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # LoRA
    lora_rank: int = 8
    lora_target: str = "all"
    
    # Evaluation
    val_size: float = 0.0  # No validation for faster training
    per_device_eval_batch_size: int = 4
    eval_strategy: str = "no"
    
    # Logging
    logging_steps: int = 20
    save_steps: int = 300
    plot_loss: bool = True
    overwrite_output_dir: bool = True
    
    # Precision
    bf16: bool = True
    
    # SSPO parameters
    pref_loss: str = "sspo"
    pref_beta: float = 2.0
    simpo_gamma: float = 2.0
    sspo_gamma_0: float = 1.0
    sspo_gamma_min: float = 0.22
    sspo_gamma_decay: float = 0.001
    sspo_prior: float = 0.5
    sspo_base: str = "simpo"


MODELS = {
    # General domain models
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
    "phi2": ModelConfig(
        name="phi-2",
        hf_path="microsoft/Phi-2",
        trust_remote_code=True,
        cache_dir="./cache/phi-2",
    ),
    # Medical domain models
    "meerkat": ModelConfig(
        name="meerkat-7b-v1.0",
        hf_path="CognitiveLLVM/Meerkat-7B-v1.0",
        trust_remote_code=True,
        cache_dir="./cache/meerkat-7b",
    ),
    "ultramedical": ModelConfig(
        name="llama3-8b-ultramedical",
        hf_path="zelrn/llama3-8b-ultramedical",
        trust_remote_code=True,
        cache_dir="./cache/llama3-8b-ultramedical",
    ),
    # Business domain models
    "mistral-business": ModelConfig(
        name="mistral-7b-business",
        hf_path="sadavart/finrl-chatbot-mistral-7b",
        trust_remote_code=True,
        cache_dir="./cache/mistral-7b-business",
    ),
    "finance": ModelConfig(
        name="finance-llama-8b",
        hf_path="Derivative/Finance-LLaMA-8B",
        trust_remote_code=True,
        cache_dir="./cache/finance-llama-8b",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training configuration YAML files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs",
        help="Output directory for YAML files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=[
            "all",
            "mistral", "llama3", "qwen2", "phi2",
            "meerkat", "ultramedical",
            "mistral-business", "finance",
        ],
        help="Which model to generate configs for",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "sspo", "dpo", "orpo", "simpo", "kto", "ssrm", "spa"],
        help="Which training method",
    )
    parser.add_argument(
        "--fb",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10],
        help="Labeled data ratios (fb)",
    )
    # Ablation experiment arguments
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=["toy", "prior", "scheduler", "all", None],
        help="Run ablation studies: toy (Toy Experiment), prior (Prior Sensitivity), scheduler (Adaptive Scheduling)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Noise ratio for Toy Experiment (0.0 or 0.5)",
    )
    parser.add_argument(
        "--prior",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="SSPO prior values for Prior Sensitivity ablation",
    )
    parser.add_argument(
        "--no-adaptive-scheduler",
        action="store_true",
        help="Disable adaptive scheduler (use fixed gamma instead)",
    )
    parser.add_argument(
        "--fixed-gamma",
        type=float,
        default=0.1,
        help="Fixed gamma value when adaptive scheduler is disabled",
    )
    return parser.parse_args()


def get_cache_dir(model_name: str) -> str:
    """Get cache directory for model."""
    return MODELS[model_name].cache_dir


def generate_yaml(
    model_key: str,
    method: str,
    fb_ratio: float,
    ch_ratio: float = 0.10,
    output_dir: Path = None,
    # Ablation parameters
    noise: float = 0.0,
    prior: float = None,
    use_adaptive_scheduler: bool = True,
    fixed_gamma: float = 0.1,
) -> Path:
    """
    Generate a YAML configuration file.

    Args:
        model_key: Model key (mistral, llama3, qwen2)
        method: Training method (sspo, dpo, simpo)
        fb_ratio: Labeled data ratio
        ch_ratio: Unlabeled data ratio
        output_dir: Output directory
        noise: Noise ratio for Toy Experiment (0.0 or 0.5)
        prior: SSPO prior value for Prior Sensitivity ablation
        use_adaptive_scheduler: Use adaptive scheduler or fixed gamma
        fixed_gamma: Fixed gamma when adaptive scheduler is disabled

    Returns:
        Path to generated YAML file
    """
    model = MODELS[model_key]
    base_config = TrainingConfig()

    # Dataset name
    dataset_name = f"ultra_combined_fb{fb_ratio}_ch{ch_ratio}"

    # Build output directory structure
    if output_dir is None:
        output_dir = Path("configs")

    model_dir = output_dir / model.name / method
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build config dict
    config = {
        # Model
        "model_name_or_path": model.hf_path,
        "trust_remote_code": model.trust_remote_code,

        # Stage
        "stage": "dpo",  # All use dpo trainer
        "do_train": True,
        "finetuning_type": "lora",
        "template": base_config.template,

        # Data
        "dataset": dataset_name,
        "cutoff_len": base_config.cutoff_len,
        "max_samples": base_config.max_samples,
        "overwrite_cache": base_config.overwrite_cache,
        "preprocessing_num_workers": base_config.preprocessing_num_workers,

        # Training
        "num_train_epochs": base_config.num_train_epochs,
        "per_device_train_batch_size": base_config.per_device_train_batch_size,
        "gradient_accumulation_steps": base_config.gradient_accumulation_steps,
        "learning_rate": base_config.learning_rate,
        "lr_scheduler_type": base_config.lr_scheduler_type,
        "warmup_ratio": base_config.warmup_ratio,
        "max_grad_norm": base_config.max_grad_norm,

        # LoRA
        "lora_rank": base_config.lora_rank,
        "lora_target": base_config.lora_target,

        # Evaluation
        "val_size": base_config.val_size,
        "per_device_eval_batch_size": base_config.per_device_eval_batch_size,
        "eval_strategy": base_config.eval_strategy,

        # Logging
        "logging_steps": base_config.logging_steps,
        "save_steps": base_config.save_steps,
        "plot_loss": base_config.plot_loss,
        "overwrite_output_dir": base_config.overwrite_output_dir,

        # Precision
        "bf16": base_config.bf16,

        # Cache
        "cache_dir": get_cache_dir(model_key),

        # Output
        "output_dir": f"./saves_{model.name}/{method}/fb{fb_ratio}_ch{ch_ratio}",
    }

    # Build filename suffix for ablation
    ablation_suffix = ""

    # Method-specific parameters
    if method == "sspo":
        sspo_config = {
            "pref_loss": "sspo",
            "pref_beta": base_config.pref_beta,
            "simpo_gamma": base_config.simpo_gamma,
            "sspo_base": base_config.sspo_base,
        }

        # Prior sensitivity ablation
        if prior is not None:
            sspo_config["sspo_prior"] = prior
            ablation_suffix += f"_prior{prior}"
        else:
            sspo_config["sspo_prior"] = base_config.sspo_prior

        # Adaptive scheduler ablation
        if use_adaptive_scheduler:
            sspo_config["sspo_gamma_0"] = base_config.sspo_gamma_0
            sspo_config["sspo_gamma_min"] = base_config.sspo_gamma_min
            sspo_config["sspo_gamma_decay"] = base_config.sspo_gamma_decay
        else:
            # Fixed gamma mode
            sspo_config["sspo_gamma_0"] = fixed_gamma
            sspo_config["sspo_gamma_min"] = fixed_gamma
            sspo_config["sspo_gamma_decay"] = 0.0  # No decay
            ablation_suffix += f"_fixed_gamma{fixed_gamma}"

        config.update(sspo_config)
    elif method == "dpo":
        config.update({
            "pref_loss": "sigmoid",
            "pref_beta": 0.1,  # DPO typically uses lower beta
        })
    elif method == "orpo":
        config.update({
            "pref_loss": "orpo",
            "pref_beta": 0.1,
        })
    elif method == "simpo":
        config.update({
            "pref_loss": "simpo",
            "pref_beta": base_config.pref_beta,
            "simpo_gamma": base_config.simpo_gamma,
        })
    elif method == "kto":
        config.update({
            "stage": "kto",
            "pref_loss": "kto_pair",
            "pref_beta": 0.1,
        })
    elif method == "ssrm":
        config.update({
            "pref_loss": "ssrm",
            "ssrm_prior": 0.5,
            "ssrm_iterations": 3,
            "ssrm_threshold": 0.9,
        })
    elif method == "spa":
        config.update({
            "pref_loss": "spa",
            "spa_iterations": 3,
            "spa_expansion_ratio": 0.1,
        })

    # Toy experiment noise suffix
    if noise > 0:
        ablation_suffix += f"_noise{noise}"

    # Generate filename
    filename = f"fb{fb_ratio}_ch{ch_ratio}_{method}_{model.name}{ablation_suffix}.yaml"
    filepath = model_dir / filename

    # Write YAML
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return filepath


def generate_toy_experiment_configs(
    model_keys: List[str],
    output_dir: Path,
    noise_levels: List[float] = [0.0, 0.5],
    n_l_values: List[int] = [10, 50, 100],
) -> List[Path]:
    """
    Generate Toy Experiment configurations.

    From paper section 7.1:
    - Tests with n_L = 10, 50, 100 samples
    - Noise settings: 0% and 50%
    - Compare DPO, ORPO, SimPO, SSPO

    Args:
        model_keys: List of model keys to use
        output_dir: Output directory
        noise_levels: Noise ratios to test
        n_l_values: Number of labeled samples to test

    Returns:
        List of generated config paths
    """
    generated = []
    methods = ["dpo", "orpo", "simpo", "sspo"]

    # Map n_L to fb_ratio (approximate, assuming ~60k total samples)
    # n_L=10 -> ~0.017%, n_L=50 -> ~0.083%, n_L=100 -> ~0.17%
    # But for toy experiment we use fixed small datasets
    fb_ratios = {
        10: 0.00017,   # ~10 samples from 60k
        50: 0.00083,   # ~50 samples
        100: 0.00167,  # ~100 samples
    }

    for noise in noise_levels:
        for n_l in n_l_values:
            fb = fb_ratios.get(n_l, 0.01)
            for model_key in model_keys:
                for method in methods:
                    filepath = generate_yaml(
                        model_key=model_key,
                        method=method,
                        fb_ratio=fb,
                        ch_ratio=0.10,
                        output_dir=output_dir,
                        noise=noise,
                    )
                    generated.append(filepath)
                    print(f"Generated: {filepath}")

    return generated


def generate_prior_sensitivity_configs(
    model_keys: List[str],
    prior_values: List[float],
    output_dir: Path,
    fb_ratio: float = 0.10,
) -> List[Path]:
    """
    Generate Prior Sensitivity ablation configurations.

    From paper section 7.3:
    - Tests prior = 0.1, 0.3, 0.5, 0.7, 0.9
    - Uses 10% paired data

    Args:
        model_keys: List of model keys to use
        prior_values: List of prior values to test
        output_dir: Output directory
        fb_ratio: Labeled data ratio (default 10%)

    Returns:
        List of generated config paths
    """
    generated = []

    for prior in prior_values:
        for model_key in model_keys:
            filepath = generate_yaml(
                model_key=model_key,
                method="sspo",
                fb_ratio=fb_ratio,
                ch_ratio=0.10,
                output_dir=output_dir,
                prior=prior,
            )
            generated.append(filepath)
            print(f"Generated: {filepath}")

    return generated


def generate_scheduler_ablation_configs(
    model_keys: List[str],
    output_dir: Path,
    fb_ratios: List[float] = [0.01, 0.10],
    fixed_gamma_values: List[float] = [0.1, 0.5],
) -> List[Path]:
    """
    Generate Adaptive Scheduling ablation configurations.

    From paper section 7.3:
    - Compare adaptive scheduler vs fixed gamma
    - Fixed gamma values: 0.1, 0.5

    Args:
        model_keys: List of model keys to use
        output_dir: Output directory
        fb_ratios: Labeled data ratios to test
        fixed_gamma_values: Fixed gamma values to compare

    Returns:
        List of generated config paths
    """
    generated = []

    for fb_ratio in fb_ratios:
        # Adaptive scheduler (baseline)
        for model_key in model_keys:
            filepath = generate_yaml(
                model_key=model_key,
                method="sspo",
                fb_ratio=fb_ratio,
                ch_ratio=0.10,
                output_dir=output_dir,
                use_adaptive_scheduler=True,
            )
            generated.append(filepath)
            print(f"Generated: {filepath}")

        # Fixed gamma variants
        for fixed_gamma in fixed_gamma_values:
            for model_key in model_keys:
                filepath = generate_yaml(
                    model_key=model_key,
                    method="sspo",
                    fb_ratio=fb_ratio,
                    ch_ratio=0.10,
                    output_dir=output_dir,
                    use_adaptive_scheduler=False,
                    fixed_gamma=fixed_gamma,
                )
                generated.append(filepath)
                print(f"Generated: {filepath}")

    return generated


def print_summary(
    generated: List[Path],
    methods: List[str],
    models: List[str],
    ratios: List[float],
) -> None:
    """Print generation summary."""
    print("=" * 60)
    print("Config Generation Summary")
    print("=" * 60)
    print(f"  Models: {', '.join(models)}")
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Ratios: {ratios}")
    print(f"  Total configs generated: {len(generated)}")
    print("=" * 60)
    
    for path in generated:
        print(f"  {path}")


def main():
    args = parse_args()

    # Determine models to generate
    if args.model == "all":
        model_keys = list(MODELS.keys())
    else:
        model_keys = [args.model]

    # Determine methods
    if args.method == "all":
        methods = ["sspo", "dpo", "orpo", "simpo", "kto", "ssrm", "spa"]
    else:
        methods = [args.method]

    # Generate configs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # Handle ablation experiments
    if args.ablation == "toy":
        print("Generating Toy Experiment configurations...")
        generated = generate_toy_experiment_configs(
            model_keys=model_keys,
            output_dir=output_dir,
            noise_levels=[0.0, 0.5],
            n_l_values=[10, 50, 100],
        )
    elif args.ablation == "prior":
        print("Generating Prior Sensitivity ablation configurations...")
        generated = generate_prior_sensitivity_configs(
            model_keys=model_keys,
            prior_values=args.prior,
            output_dir=output_dir,
            fb_ratio=0.10,
        )
    elif args.ablation == "scheduler":
        print("Generating Adaptive Scheduling ablation configurations...")
        generated = generate_scheduler_ablation_configs(
            model_keys=model_keys,
            output_dir=output_dir,
            fb_ratios=[0.01, 0.10],
            fixed_gamma_values=[0.1, 0.5],
        )
    elif args.ablation == "all":
        print("Generating ALL ablation configurations...")
        generated = generate_toy_experiment_configs(
            model_keys=model_keys,
            output_dir=output_dir,
            noise_levels=[0.0, 0.5],
            n_l_values=[10, 50, 100],
        )
        generated += generate_prior_sensitivity_configs(
            model_keys=model_keys,
            prior_values=args.prior,
            output_dir=output_dir,
            fb_ratio=0.10,
        )
        generated += generate_scheduler_ablation_configs(
            model_keys=model_keys,
            output_dir=output_dir,
            fb_ratios=[0.01, 0.10],
            fixed_gamma_values=[0.1, 0.5],
        )
    else:
        # Standard config generation
        for model_key in model_keys:
            for method in methods:
                for fb_ratio in args.fb:
                    filepath = generate_yaml(
                        model_key=model_key,
                        method=method,
                        fb_ratio=fb_ratio,
                        ch_ratio=0.10,
                        output_dir=output_dir,
                        noise=args.noise,
                    )
                    generated.append(filepath)
                    print(f"Generated: {filepath}")

    print_summary(generated, methods, model_keys, args.fb)


if __name__ == "__main__":
    main()
