#!/usr/bin/env python3
"""
Download datasets from HuggingFace for SSPO experiments.

Datasets:
- General: UltraFeedback (paired), UltraChat (unpaired)
- Medical: UltraMedical-Preference (paired), UltraMedical (unpaired)
- Business: DSP-Business (paired), Business-Book (unpaired)

Usage:
    python scripts/download_data.py --dataset all --output data/
    python scripts/download_data.py --dataset ultrafeedback --output data/
    python scripts/download_data.py --dataset ultramedical_preference --output data/
    python scripts/download_data.py --dataset dsp_business --output data/
"""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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
)


# ANSI color shortcuts
GREEN = Colors.GREEN
RED = Colors.RED
YELLOW = Colors.YELLOW
CYAN = Colors.CYAN
BOLD = Colors.BOLD
RESET = Colors.RESET


@dataclass
class DatasetConfig:
    name: str
    hf_path: str
    splits: list[str]
    expected_size_mb: Optional[int] = None


DATASETS = {
    # General domain datasets
    "ultrafeedback": DatasetConfig(
        name="UltraFeedback",
        hf_path="argilla/ultrafeedback-binarized-preferences",
        splits=["train"],
        expected_size_mb=500,
    ),
    "ultrachat": DatasetConfig(
        name="UltraChat",
        hf_path="HuggingFaceH4/ultrachat_200k",
        splits=["train_sft", "test_sft", "train_gen", "test_gen"],
        expected_size_mb=1000,
    ),
    # Medical domain datasets
    "ultramedical_preference": DatasetConfig(
        name="UltraMedical-Preference",
        hf_path="medmcqa/ultramedical-preference",
        splits=["train", "validation"],
        expected_size_mb=100,
    ),
    "ultramedical": DatasetConfig(
        name="UltraMedical",
        hf_path="medmcqa/ultramedical",
        splits=["train_sft", "test_sft"],
        expected_size_mb=200,
    ),
    # Business domain datasets
    "dsp_business": DatasetConfig(
        name="DSP-Business",
        hf_path="lawrence/NLP-DSP",
        splits=["train", "validation"],
        expected_size_mb=10,
    ),
    "business_book": DatasetConfig(
        name="Business-Book",
        hf_path="tasksource/book_summaries",
        splits=["train", "validation"],
        expected_size_mb=150,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download UltraFeedback and UltraChat datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=[
            "all",
            "ultrafeedback",
            "ultrachat",
            "ultramedical_preference",
            "ultramedical",
            "dsp_business",
            "business_book",
        ],
        help="Which dataset to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for log files",
    )
    return parser.parse_args()


def verify_dataset(dataset_path: Path, expected_files: list[str]) -> bool:
    """Verify dataset files exist and have content."""
    missing = []
    empty = []

    for fname in expected_files:
        fpath = dataset_path / fname
        if not fpath.exists():
            missing.append(fname)
        elif fpath.stat().st_size == 0:
            empty.append(fname)

    if missing:
        print_warning(f"  Missing files: {missing}")
    if empty:
        print_warning(f"  Empty files: {empty}")

    return len(missing) == 0 and len(empty) == 0


def main():
    args = parse_args()
    output_dir = Path(args.output)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"download_{timestamp}.log"
    logger = get_logger("download", log_file)

    # Determine datasets to download
    datasets_to_download = []
    if args.dataset == "all":
        datasets_to_download = list(DATASETS.keys())
    else:
        datasets_to_download = [args.dataset]

    # Setup step tracker
    tracker = StepTracker("download", total_steps=len(datasets_to_download))

    print_header("SSPO Data Downloader")
    print(f"{CYAN}Output directory:{RESET} {output_dir}")
    print(f"{CYAN}Log file:{RESET} {log_file}")
    print(f"{CYAN}Datasets:{RESET} {', '.join(datasets_to_download)}")
    print()

    # Check for HuggingFace libraries
    tracker.step("Checking dependencies")
    try:
        from huggingface_hub import snapshot_download
        from datasets import load_dataset
        logger.info("HuggingFace libraries available")
        print_success("Dependencies OK")
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print(f"Install with: uv pip install datasets huggingface_hub")
        sys.exit(1)

    results = {}
    overall_start = time.time()

    for i, dataset_key in enumerate(datasets_to_download, 1):
        tracker.step(f"Downloading {DATASETS[dataset_key].name}", step_num=i)
        config = DATASETS[dataset_key]

        dataset_dir = output_dir / config.name.lower()
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  {BOLD}Source:{RESET} {config.hf_path}")
        print(f"  {BOLD}Splits:{RESET} {', '.join(config.splits)}")

        # Check if already downloaded
        if not args.force and verify_dataset(dataset_dir, config.splits):
            print_success(f"{config.name} already downloaded, skipping")
            logger.info(f"Skipped {config.name}: already exists")
            results[dataset_key] = True
            continue

        logger.info(f"Starting download: {config.name}")
        print(f"  {CYAN}Downloading...{RESET}")

        try:
            for split in config.splits:
                print(f"\n  {BOLD}[{split}]{RESET} ", end="", flush=True)

                ds = load_dataset(
                    config.hf_path,
                    split=split,
                    cache_dir=args.cache_dir,
                )

                # Save to local JSONL
                output_file = dataset_dir / f"{split}.json"
                ds.to_json(output_file)

                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"{GREEN}✓{RESET} {size_mb:.1f} MB")

                logger.info(f"Downloaded {split}: {output_file} ({size_mb:.1f} MB)")

            print_success(f"Downloaded {config.name}")
            logger.info(f"Complete: {config.name}")
            results[dataset_key] = True

        except Exception as e:
            print_error(f"Failed: {e}")
            logger.error(f"Failed {config.name}: {e}")
            results[dataset_key] = False

    # Summary
    print_header("Download Summary")
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    elapsed = time.time() - overall_start

    for dataset_key, success in results.items():
        status = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
        print(f"  {status} {DATASETS[dataset_key].name}")

    print()
    if success_count == total_count:
        print_success(f"All {total_count} datasets downloaded ({elapsed:.1f}s)")
    else:
        print_error(f"{success_count}/{total_count} downloaded ({elapsed:.1f}s)")

    logger.info(f"Summary: {success_count}/{total_count} in {elapsed:.1f}s")

    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
