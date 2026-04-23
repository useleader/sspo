#!/usr/bin/env python3
"""
Download UltraFeedback and UltraChat datasets from HuggingFace.

This script handles:
- Dataset downloading from HuggingFace
- Retry logic for unreliable networks
- Progress tracking
- Verification of downloaded data

Usage:
    python scripts/download_data.py --dataset all --output data/
    python scripts/download_data.py --dataset ultrafeedback --output data/
    python scripts/download_data.py --dataset ultrachat --output data/
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


@dataclass
class DatasetConfig:
    name: str
    hf_path: str
    splits: list[str]
    expected_size_mb: Optional[int] = None


DATASETS = {
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
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download UltraFeedback and UltraChat datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "ultrafeedback", "ultrachat"],
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
        "--verify",
        action="store_true",
        default=True,
        help="Verify dataset integrity after download",
    )
    return parser.parse_args()


def download_with_retry(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download file with exponential backoff retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            
            with open(output_path, "wb") as f, tqdm(
                desc=output_path.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
            
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                print(f"  Attempt {attempt + 1} failed: {e}")
                print(f"  Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                print(f"  Failed after {MAX_RETRIES} attempts")
                return False
    return False


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
        print(f"  Missing files: {missing}")
    if empty:
        print(f"  Empty files: {empty}")
    
    return len(missing) == 0 and len(empty) == 0


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_to_download = []
    if args.dataset == "all":
        datasets_to_download = list(DATASETS.keys())
    else:
        datasets_to_download = [args.dataset]
    
    print_section("SSPO Data Downloader")
    print(f"Output directory: {output_dir}")
    print(f"Datasets: {', '.join(datasets_to_download)}")
    
    # Check for HuggingFace CLI
    try:
        from huggingface_hub import snapshot_download
        hf_available = True
    except ImportError:
        print("Warning: huggingface_hub not installed, using fallback download")
        hf_available = False
    
    results = {}
    
    for dataset_key in datasets_to_download:
        config = DATASETS[dataset_key]
        print_section(f"Downloading {config.name}")
        
        dataset_dir = output_dir / config.name.lower()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if not args.force and verify_dataset(dataset_dir, config.splits):
            print(f"✓ {config.name} already downloaded, skipping")
            results[dataset_key] = True
            continue
        
        print(f"Downloading from: {config.hf_path}")
        print(f"Splits: {', '.join(config.splits)}")
        
        if hf_available:
            try:
                cache_dir = args.cache_dir or "~/.cache/huggingface"
                print(f"Using HuggingFace cache: {cache_dir}")

                # Use datasets library to load and save
                from datasets import load_dataset

                for split in config.splits:
                    print(f"  Loading {split} split...")
                    ds = load_dataset(
                        config.hf_path,
                        split=split,
                        cache_dir=cache_dir,
                    )
                    # Save to local directory
                    output_file = dataset_dir / f"{split}.json"
                    ds.to_json(output_file)
                    print(f"    Saved: {output_file}")

                print(f"✓ Downloaded {config.name} to {dataset_dir}")
                results[dataset_key] = True
                
            except Exception as e:
                print(f"✗ Failed to download {config.name}: {e}")
                results[dataset_key] = False
        else:
            print("Error: huggingface_hub required for download")
            print("Install with: pip install huggingface_hub")
            results[dataset_key] = False
    
    # Summary
    print_section("Download Summary")
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for dataset_key, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {DATASETS[dataset_key].name}")
    
    print(f"\n{success_count}/{total_count} datasets downloaded successfully")
    
    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
