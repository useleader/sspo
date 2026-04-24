#!/usr/bin/env python3
"""
Preprocess UltraFeedback and UltraChat into combined dataset.

This script:
1. Loads UltraFeedback (labeled) and UltraChat (unlabeled) from local files
2. Applies sampling ratios (fb for labeled, ch for unlabeled)
3. Combines into single JSON with fields: instruction, chosen, rejected, unlabeled
4. Registers in dataset_info.json for LLaMA-Factory
5. Logs all stages to logs/ directory

Usage:
    python scripts/preprocess_data.py --fb 0.01 --ch 0.1 --output processed/
    python scripts/preprocess_data.py --fb 0.05 --ch 0.1 --output processed/
    python scripts/preprocess_data.py --fb 0.10 --ch 0.1 --output processed/

Paper configuration (from ADR-0003):
    - fb (labeled ratio): 0.01, 0.05, 0.10
    - ch (unlabeled ratio): 0.10
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

# Add scripts to path for logging module
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_logging import (
    Colors,
    StepTracker,
    get_logger,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
)


# ANSI color shortcuts
GREEN = Colors.GREEN
RED = Colors.RED
YELLOW = Colors.YELLOW
CYAN = Colors.CYAN
BOLD = Colors.BOLD
RESET = Colors.RESET


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    print_info(f"Random seed set to {seed}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess UltraFeedback and UltraChat for SSPO"
    )
    parser.add_argument(
        "--fb",
        type=float,
        default=0.1,
        help="Ratio of paired data to keep as labeled (0.0-1.0)",
    )
    parser.add_argument(
        "--ch",
        type=float,
        default=0.1,
        help="Ratio of unpaired data to keep as unlabeled (0.0-1.0)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="general",
        choices=["general", "medical", "business"],
        help="Domain for preprocessing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing downloaded data",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for log files",
    )
    return parser.parse_args()


def load_jsonl(filepath: Path) -> List[dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# Domain configurations
DOMAIN_CONFIGS = {
    "general": {
        "paired_name": "ultrafeedback",
        "unpaired_name": "ultrachat",
        "paired_split": "train",
        "unpaired_split": "train_sft",
        "dataset_prefix": "ultra",
    },
    "medical": {
        "paired_name": "ultramedical_preference",
        "unpaired_name": "ultramedical",
        "paired_split": "train",
        "unpaired_split": "train_sft",
        "dataset_prefix": "medical",
    },
    "business": {
        "paired_name": "dsp_business",
        "unpaired_name": "business_book",
        "paired_split": "train",
        "unpaired_split": "train",
        "dataset_prefix": "biz",
    },
}


def get_dataset_paths(data_dir: Path, domain: str) -> tuple:
    """Get dataset paths for specified domain."""
    config = DOMAIN_CONFIGS[domain]
    paired_path = data_dir / config["paired_name"] / f"{config['paired_split']}.json"
    unpaired_path = data_dir / config["unpaired_name"] / f"{config['unpaired_split']}.json"
    return paired_path, unpaired_path, config["dataset_prefix"]


def keep_partial_data(dataset, keep_ratio: float) -> List:
    """Randomly keep only keep_ratio portion of the dataset."""
    total_samples = len(dataset)
    num_to_keep = int(total_samples * keep_ratio)
    indices = random.sample(range(total_samples), num_to_keep)
    return [dataset[i] for i in indices]


def create_combined_dataset(
    paired: List,
    unpaired: List,
    logger=None,
) -> tuple:
    """Combine paired (labeled) and unpaired data into single dataset."""
    combined = []

    # Add labeled data from paired dataset
    label_count = 0
    logger.info(f"Processing {len(paired)} paired samples...") if logger else None
    for sample in tqdm(paired, desc="Paired", unit="samples"):
        combined.append({
            "instruction": sample["instruction"],
            "chosen": sample["chosen_response"],
            "rejected": sample["rejected_response"],
            "unlabeled": "",
        })
        label_count += 1

    # Add unlabeled data from unpaired dataset
    logger.info(f"Processing {len(unpaired)} unpaired samples...") if logger else None
    unlabel_count = 0
    skipped_no_assistant = 0
    for sample in tqdm(unpaired, desc="Unpaired", unit="samples"):
        messages = sample.get("messages", [])
        if not messages:
            skipped_no_assistant += 1
            continue

        response = ""
        instruction = ""
        for msg in messages:
            if msg.get("role") == "user" and not instruction:
                instruction = msg.get("content", "")
            if msg.get("role") == "assistant":
                response = msg.get("content", "")
                break

        if not response:
            skipped_no_assistant += 1
            continue

        combined.append({
            "instruction": instruction,
            "chosen": "",
            "rejected": "",
            "unlabeled": response,
        })
        unlabel_count += 1

    # Shuffle
    logger.info("Shuffling combined dataset...") if logger else None
    random.shuffle(combined)

    return combined, label_count, unlabel_count, skipped_no_assistant


def save_combined_dataset(
    dataset: List[dict],
    output_dir: Path,
    fb_ratio: float,
    ch_ratio: float,
    dataset_name: Optional[str] = None,
) -> tuple:
    """Save combined dataset to JSON file. Returns (path, size_mb)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name is None:
        dataset_name = f"ultra_combined_fb{fb_ratio}_ch{ch_ratio}"

    json_file = output_dir / f"{dataset_name}.json"

    print_info(f"Saving {len(dataset)} samples to {json_file}...")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    size_mb = json_file.stat().st_size / (1024 * 1024)
    print_success(f"Saved: {json_file} ({size_mb:.1f} MB)")

    return json_file, size_mb


def update_dataset_info(
    output_dir: Path,
    fb_ratio: float,
    ch_ratio: float,
    dataset_name: Optional[str] = None,
) -> None:
    """Update dataset_info.json for LLaMA-Factory compatibility."""
    if dataset_name is None:
        dataset_name = f"ultra_combined_fb{fb_ratio}_ch{ch_ratio}"

    dataset_info_path = output_dir / "dataset_info.json"

    if dataset_info_path.exists():
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

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

    print_success(f"Updated dataset_info.json")


def main():
    args = parse_args()
    output_dir = Path(args.output)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"preprocess_{timestamp}.log"
    logger = get_logger("preprocess", log_file)

    # Setup step tracker (5 steps)
    tracker = StepTracker("preprocess", total_steps=5)

    print_header("SSPO Data Preprocessor")
    print(f"{CYAN}Labeled ratio (fb):{RESET} {args.fb}")
    print(f"{CYAN}Unlabeled ratio (ch):{RESET} {args.ch}")
    print(f"{CYAN}Domain:{RESET} {args.domain}")
    print(f"{CYAN}Data directory:{RESET} {args.data_dir}")
    print(f"{CYAN}Output directory:{RESET} {output_dir}")
    print(f"{CYAN}Log file:{RESET} {log_file}")
    print()

    overall_start = time.time()

    # Step 1: Set seed
    tracker.step("Setting random seed", step_num=1)
    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")

    # Step 2: Load datasets
    tracker.step("Loading datasets", step_num=2)
    data_dir = Path(args.data_dir)

    paired_path, unpaired_path, prefix = get_dataset_paths(data_dir, args.domain)
    config = DOMAIN_CONFIGS[args.domain]

    print_info(f"Loading paired data from {paired_path}...")
    paired_raw = load_jsonl(paired_path)
    print_info(f"Loaded {len(paired_raw)} samples from {config['paired_name']}")

    print_info(f"Loading unpaired data from {unpaired_path}...")
    unpaired_raw = load_jsonl(unpaired_path)
    print_info(f"Loaded {len(unpaired_raw)} samples from {config['unpaired_name']}")

    # Step 3: Sample datasets
    tracker.step("Sampling datasets", step_num=3)
    paired = keep_partial_data(paired_raw, keep_ratio=args.fb)
    unpaired = keep_partial_data(unpaired_raw, keep_ratio=args.ch)

    print_info(f"Sampled {len(paired)} paired samples ({args.fb * 100:.1f}%)")
    print_info(f"Sampled {len(unpaired)} unpaired samples ({args.ch * 100:.1f}%)")

    # Step 4: Create combined dataset
    tracker.step("Creating combined dataset", step_num=4)
    combined, label_count, unlabel_count, skipped = create_combined_dataset(
        paired, unpaired, logger
    )

    if skipped > 0:
        print_warning(f"Skipped {skipped} samples with no assistant message")

    # Step 5: Save
    tracker.step("Saving combined dataset", step_num=5)
    dataset_name = f"{prefix}_combined_fb{args.fb}_ch{args.ch}"

    json_file, size_mb = save_combined_dataset(
        combined,
        output_dir,
        args.fb,
        args.ch,
        dataset_name,
    )

    update_dataset_info(output_dir, args.fb, args.ch, dataset_name)

    # Summary
    elapsed = time.time() - overall_start
    print_header("Preprocessing Summary")
    print(f"{CYAN}Domain:{RESET} {args.domain}")
    print(f"{CYAN}Labeled samples:{RESET} {label_count}")
    print(f"{CYAN}Unlabeled samples:{RESET} {unlabel_count}")
    print(f"{CYAN}Total samples:{RESET} {len(combined)}")
    print(f"{CYAN}Skipped:{RESET} {skipped}")
    print(f"{CYAN}Output file:{RESET} {json_file} ({size_mb:.1f} MB)")
    print()
    print_success(f"Complete ({elapsed:.1f}s)")
    tracker.complete()


if __name__ == "__main__":
    main()
