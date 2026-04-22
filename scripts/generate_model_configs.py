#!/usr/bin/env python3
"""
Generate training configuration YAML files for SSPO experiments.

This script generates YAML configs for:
1. Three models: Mistral-7B, Llama-3-8B, Qwen2-7B
2. Three labeled ratios: 1%, 5%, 10%
3. Multiple methods: SSPO (main), DPO, SimPO (baselines)

Paper configuration (from ADR-0006, ADR-0008):
    - Model: Mistral-7B-Instruct-v0.2, Llama-3-8B-Instruct, Qwen2-7B-Instruct
    - LoRA rank: 8, target: all
    - Learning rate: 1e-5
    - Epochs: 1
    - Batch size: 64 (per node)
    - Context length: 1024

Usage:
    python scripts/generate_model_configs.py --output configs/
    python scripts/generate_model_configs.py --model mistral --output configs/
"""

import argparse
import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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
    cutoff_len: int = 1024
    max_samples: int = 10000000
    overwrite_cache: bool = True
    preprocessing_num_workers: int = 12
    
    # Training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
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
        choices=["all", "mistral", "llama3", "qwen2"],
        help="Which model to generate configs for",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "sspo", "dpo", "simpo"],
        help="Which training method",
    )
    parser.add_argument(
        "--fb",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10],
        help="Labeled data ratios (fb)",
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
) -> Path:
    """
    Generate a YAML configuration file.
    
    Args:
        model_key: Model key (mistral, llama3, qwen2)
        method: Training method (sspo, dpo, simpo)
        fb_ratio: Labeled data ratio
        ch_ratio: Unlabeled data ratio
        output_dir: Output directory
    
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
    
    # Method-specific parameters
    if method == "sspo":
        config.update({
            "pref_loss": "sspo",
            "pref_beta": base_config.pref_beta,
            "simpo_gamma": base_config.simpo_gamma,
            "sspo_gamma_0": base_config.sspo_gamma_0,
            "sspo_gamma_min": base_config.sspo_gamma_min,
            "sspo_gamma_decay": base_config.sspo_gamma_decay,
            "sspo_prior": base_config.sspo_prior,
            "sspo_base": base_config.sspo_base,
        })
    elif method == "dpo":
        config.update({
            "pref_loss": "sigmoid",
            "pref_beta": 0.1,  # DPO typically uses lower beta
        })
    elif method == "simpo":
        config.update({
            "pref_loss": "simpo",
            "pref_beta": base_config.pref_beta,
            "simpo_gamma": base_config.simpo_gamma,
        })
    
    # Generate filename
    filename = f"fb{fb_ratio}_ch{ch_ratio}_{method}_{model.name}.yaml"
    filepath = model_dir / filename
    
    # Write YAML
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return filepath


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
        methods = ["sspo", "dpo", "simpo"]
    else:
        methods = [args.method]
    
    # Generate configs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated = []
    
    for model_key in model_keys:
        for method in methods:
            for fb_ratio in args.fb:
                filepath = generate_yaml(
                    model_key=model_key,
                    method=method,
                    fb_ratio=fb_ratio,
                    ch_ratio=0.10,
                    output_dir=output_dir,
                )
                generated.append(filepath)
                print(f"Generated: {filepath}")
    
    print_summary(generated, methods, model_keys, args.fb)


if __name__ == "__main__":
    main()
