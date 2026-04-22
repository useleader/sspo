#!/usr/bin/env python3
"""
Preprocess UltraFeedback and UltraChat into combined dataset.

This script:
1. Loads UltraFeedback (labeled) and UltraChat (unlabeled)
2. Applies sampling ratios (fb for labeled, ch for unlabeled)
3. Combines into single JSON with fields: instruction, chosen, rejected, unlabeled
4. Registers in dataset_info.json for LLaMA-Factory

Usage:
    python scripts/preprocess_data.py --fb 0.01 --ch 0.1 --output data/
    python scripts/preprocess_data.py --fb 0.05 --ch 0.1 --output data/
    python scripts/preprocess_data.py --fb 0.10 --ch 0.1 --output data/

Paper configuration (from ADR-0003):
    - fb (labeled ratio): 0.01, 0.05, 0.10
    - ch (unlabeled ratio): 0.10
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess UltraFeedback and UltraChat for SSPO"
    )
    parser.add_argument(
        "--fb",
        type=float,
        default=0.1,
        help="Ratio of UltraFeedback to keep as labeled data (0.0-1.0)",
    )
    parser.add_argument(
        "--ch",
        type=float,
        default=0.1,
        help="Ratio of UltraChat to keep as unlabeled data (0.0-1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Custom dataset name suffix",
    )
    return parser.parse_args()


def load_datasets() -> DatasetDict:
    """Load UltraFeedback and UltraChat from HuggingFace."""
    logger.info("Loading datasets from HuggingFace...")
    
    ultrafeedback = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        splits=["train_prefs", "test_prefs"],
    )
    
    ultrachat = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=["train_sft", "test_sft", "train_gen", "test_gen"],
    )
    
    return ultrafeedback, ultrachat


def keep_partial_data(dataset, keep_ratio: float) -> List:
    """
    Randomly keep only keep_ratio portion of the dataset.
    
    Args:
        dataset: Original dataset
        keep_ratio: Ratio of data to keep (0.0-1.0)
    
    Returns:
        List of selected samples
    """
    total_samples = len(dataset)
    num_to_keep = int(total_samples * keep_ratio)
    
    indices = random.sample(range(total_samples), num_to_keep)
    
    return [dataset[i] for i in indices]


def create_combined_dataset(
    ultrafeedback: List,
    ultrachat: List,
) -> List[dict]:
    """
    Combine UltraFeedback (labeled) and UltraChat (unlabeled).
    
    Args:
        ultrafeedback: List of UltraFeedback samples
        ultrachat: List of UltraChat samples
    
    Returns:
        Combined dataset with fields: instruction, chosen, rejected, unlabeled
    """
    combined = []
    
    # Add labeled data from UltraFeedback
    logger.info(f"Processing {len(ultrafeedback)} UltraFeedback samples...")
    for sample in tqdm(ultrafeedback, desc="UltraFeedback"):
        combined.append({
            "instruction": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
            "unlabeled": "",  # Empty for labeled data
        })
    
    # Add unlabeled data from UltraChat
    logger.info(f"Processing {len(ultrachat)} UltraChat samples...")
    for sample in tqdm(ultrachat, desc="UltraChat"):
        # UltraChat has messages structure, flatten to single response
        messages = sample.get("messages", [])
        if messages and len(messages) > 0:
            # Get assistant's last message
            response = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    response = msg.get("content", "")
                    break
            
            combined.append({
                "instruction": sample.get("prompt", ""),
                "chosen": "",
                "rejected": "",
                "unlabeled": response,
            })
    
    # Shuffle combined dataset
    random.shuffle(combined)
    
    return combined


def save_combined_dataset(
    dataset: List[dict],
    output_dir: Path,
    fb_ratio: float,
    ch_ratio: float,
    dataset_name: Optional[str] = None,
) -> Path:
    """
    Save combined dataset to JSON file.
    
    Args:
        dataset: Combined dataset
        output_dir: Output directory
        fb_ratio: Labeled data ratio
        ch_ratio: Unlabeled data ratio
        dataset_name: Custom dataset name
    
    Returns:
        Path to saved JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name is None:
        dataset_name = f"ultra_combined_fb{fb_ratio}_ch{ch_ratio}"
    
    json_file = output_dir / f"{dataset_name}.json"
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved combined dataset to {json_file}")
    
    return json_file


def update_dataset_info(
    output_dir: Path,
    fb_ratio: float,
    ch_ratio: float,
    dataset_name: Optional[str] = None,
) -> None:
    """
    Update dataset_info.json for LLaMA-Factory compatibility.
    
    Args:
        output_dir: Directory containing dataset_info.json
        fb_ratio: Labeled data ratio
        ch_ratio: Unlabeled data ratio
        dataset_name: Custom dataset name
    """
    if dataset_name is None:
        dataset_name = f"ultra_combined_fb{fb_ratio}_ch{ch_ratio}"
    
    dataset_info_path = output_dir / "dataset_info.json"
    
    # Load existing or create new
    if dataset_info_path.exists():
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}
    
    # Add new dataset entry
    json_file = f"./{dataset_name}.json"
    dataset_info[dataset_name] = {
        "file_name": json_file,
        "ranking": True,
        "columns": {
            "prompt": "instruction",
            "chosen": "chosen",
            "rejected": "rejected",
            "unlabeled": "unlabeled",
        },
    }
    
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Updated dataset_info.json with {dataset_name}")


def print_summary(
    fb_ratio: float,
    ch_ratio: float,
    ultrafeedback_size: int,
    ultrachat_size: int,
    combined_size: int,
) -> None:
    """Print preprocessing summary."""
    logger.info("=" * 60)
    logger.info("Preprocessing Summary")
    logger.info("=" * 60)
    logger.info(f"  Labeled ratio (fb): {fb_ratio}")
    logger.info(f"  Unlabeled ratio (ch): {ch_ratio}")
    logger.info(f"  UltraFeedback samples: {ultrafeedback_size}")
    logger.info(f"  UltraChat samples: {ultrachat_size}")
    logger.info(f"  Combined dataset size: {combined_size}")
    logger.info("=" * 60)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    logger.info("Starting data preprocessing...")
    logger.info(f"  Labeled ratio (fb): {args.fb}")
    logger.info(f"  Unlabeled ratio (ch): {args.ch}")
    logger.info(f"  Output directory: {args.output}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    ultrafeedback_raw, ultrachat_raw = load_datasets()
    
    # Sample according to ratios
    ultrafeedback = keep_partial_data(
        ultrafeedback_raw["train_prefs"],
        keep_ratio=args.fb
    )
    
    ultrachat = keep_partial_data(
        ultrachat_raw[0],  # train_sft split
        keep_ratio=args.ch
    )
    
    logger.info(f"Kept {len(ultrafeedback)} UltraFeedback samples ({args.fb * 100:.1f}%)")
    logger.info(f"Kept {len(ultrachat)} UltraChat samples ({args.ch * 100:.1f}%)")
    
    # Create combined dataset
    combined = create_combined_dataset(ultrafeedback, ultrachat)
    
    # Save
    dataset_name = f"ultra_combined_fb{args.fb}_ch{args.ch}"
    if args.dataset_name:
        dataset_name = args.dataset_name
    
    json_file = save_combined_dataset(
        combined,
        output_dir,
        args.fb,
        args.ch,
        dataset_name,
    )
    
    update_dataset_info(output_dir, args.fb, args.ch, dataset_name)
    
    print_summary(
        args.fb,
        args.ch,
        len(ultrafeedback),
        len(ultrachat),
        len(combined),
    )
    
    logger.info(f"✓ Preprocessing complete: {json_file}")


if __name__ == "__main__":
    main()
