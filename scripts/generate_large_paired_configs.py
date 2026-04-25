#!/usr/bin/env python3
"""
Generate large paired data configuration YAML files for Table 11 experiments.

Table 11 from the paper studies the effect of n_L (labeled sample size) on
performance with large paired data. This script generates configs for:
- n_L values: 100, 1000, 5000, 10000
- fb calculation: nl / 60000 (total dataset ~60k)

Usage:
    python scripts/generate_large_paired_configs.py --output configs_table11/
    python scripts/generate_large_paired_configs.py --models mistral --fb 0.0017 --output configs/
"""

import argparse
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    hf_path: str
    trust_remote_code: bool = True
    cache_dir: str = "./cache"


# Model configurations (same as generate_model_configs.py)
MODELS: Dict[str, ModelConfig] = {
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


# n_L values for Table 11 large paired data experiments
N_L_VALUES = [100, 1000, 5000, 10000]

# Total dataset size (approximate)
TOTAL_DATASET_SIZE = 60000


# TEMPLATE_PAIRED: Template for large paired data experiments
TEMPLATE_PAIRED = {
    # Model
    "stage": "dpo",
    "do_train": True,
    "finetuning_type": "lora",
    "template": "default",

    # Data
    "cutoff_len": 2048,
    "max_samples": 10000000,
    "overwrite_cache": True,
    "preprocessing_num_workers": 12,

    # Training
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5e-7,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,

    # LoRA
    "lora_rank": 8,
    "lora_target": "all",

    # Evaluation
    "val_size": 0.0,
    "per_device_eval_batch_size": 4,
    "eval_strategy": "no",

    # Logging
    "logging_steps": 20,
    "save_steps": 300,
    "plot_loss": True,
    "overwrite_output_dir": True,

    # Precision
    "bf16": True,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate large paired data configuration YAML files for Table 11"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs_table11",
        help="Output directory for YAML files",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["mistral", "llama3", "qwen2"],
        help="Which models to generate configs for",
    )
    parser.add_argument(
        "--fb",
        type=float,
        default=None,
        help="Labeled data ratio (fb). If not specified, calculated from n_L / 60000",
    )
    parser.add_argument(
        "--nl",
        type=int,
        nargs="+",
        default=N_L_VALUES,
        help=f"n_L values to generate configs for (default: {N_L_VALUES})",
    )
    return parser.parse_args()


def calculate_fb(nl: int, total: int = TOTAL_DATASET_SIZE) -> float:
    """
    Calculate fb ratio from n_L.

    Args:
        nl: Number of labeled samples
        total: Total dataset size (default 60000)

    Returns:
        fb ratio as float
    """
    return nl / total


def generate_large_paired_configs(
    output_dir: Path,
    models: List[str],
    fb: float = None,
    nl_values: List[int] = N_L_VALUES,
) -> List[Path]:
    """
    Generate large paired data configuration files.

    Args:
        output_dir: Output directory for YAML files
        models: List of model keys to generate configs for
        fb: Fixed fb ratio (if None, calculated from n_L / 60000)
        nl_values: List of n_L values to generate configs for

    Returns:
        List of generated config file paths
    """
    generated = []

    for nl in nl_values:
        # Calculate fb from n_L if not provided
        fb_ratio = fb if fb is not None else calculate_fb(nl)

        for model_key in models:
            model = MODELS[model_key]

            # Build output directory structure
            model_dir = output_dir / model.name / "large_paired"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Build config dict from template
            config = TEMPLATE_PAIRED.copy()

            # Update model-specific fields
            config["model_name_or_path"] = model.hf_path
            config["trust_remote_code"] = model.trust_remote_code
            config["cache_dir"] = model.cache_dir

            # Dataset name
            dataset_name = f"ultra_paired_fb{fb_ratio:.4f}"
            config["dataset"] = dataset_name

            # Output directory
            config["output_dir"] = f"./saves_{model.name}/large_paired/nl{nl}"

            # Generate filename
            filename = f"nl{nl}_large_paired_{model.name}.yaml"
            filepath = model_dir / filename

            # Write YAML
            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            generated.append(filepath)
            print(f"Generated: {filepath}")

    return generated


def main():
    args = parse_args()

    # Validate models
    for model in args.models:
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(MODELS.keys())}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate configs
    print("=" * 60)
    print("Table 11 Large Paired Data Config Generation")
    print("=" * 60)
    print(f"  n_L values: {args.nl}")
    print(f"  Models: {', '.join(args.models)}")
    if args.fb is not None:
        print(f"  Fixed fb ratio: {args.fb}")
    else:
        print(f"  fb calculated as n_L / {TOTAL_DATASET_SIZE}")
    print("=" * 60)

    generated = generate_large_paired_configs(
        output_dir=output_dir,
        models=args.models,
        fb=args.fb,
        nl_values=args.nl,
    )

    print(f"\nTotal configs generated: {len(generated)}")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
