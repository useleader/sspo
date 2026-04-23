#!/usr/bin/env python3
"""
Tests for download_data.py

These tests verify:
1. Dataset configuration is correct
2. Dataset directories are properly structured
3. Data format matches expected schema
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.download_data import DATASETS, DatasetConfig


def test_dataset_configs():
    """Verify all expected datasets are configured."""
    assert "ultrafeedback" in DATASETS
    assert "ultrachat" in DATASETS

    # UltraFeedback has train split only
    uf = DATASETS["ultrafeedback"]
    assert "train" in uf.splits

    # UltraChat should have SFT and generation splits
    uc = DATASETS["ultrachat"]
    assert "train_sft" in uc.splits
    assert "test_sft" in uc.splits


def test_ultrafeedback_structure():
    """Test UltraFeedback data structure if downloaded."""
    data_dir = Path("data/ultrafeedback")

    if not data_dir.exists():
        print("SKIP: UltraFeedback not yet downloaded")
        return

    # Check expected files exist
    expected_splits = DATASETS["ultrafeedback"].splits

    # Load sample and verify structure
    for split in expected_splits:
        split_file = data_dir / f"{split}.json"
        if not split_file.exists():
            print(f"SKIP: {split_file} not found")
            continue

        # File is JSONL format (one JSON per line)
        samples = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        if not samples:
            print(f"SKIP: Empty dataset {split_file.name}")
            continue

        sample = samples[0]

        # Verify structure based on paper
        # UltraFeedback has: source, instruction, models, completions
        assert "instruction" in sample, \
            f"Missing instruction in {split}"

        # Check completions structure if present
        if "completions" in sample:
            assert isinstance(sample["completions"], list), \
                "completions should be a list"


def test_ultrachat_structure():
    """Test UltraChat data structure if downloaded."""
    data_dir = Path("data/ultrachat")

    if not data_dir.exists():
        print("SKIP: UltraChat not yet downloaded")
        return

    # Check expected files exist
    expected_splits = DATASETS["ultrachat"].splits

    for split in expected_splits:
        split_file = data_dir / f"{split}.json"
        if not split_file.exists():
            print(f"SKIP: {split_file} not found")
            continue

        # File is JSONL format (one JSON per line)
        samples = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        if not samples:
            print(f"SKIP: Empty dataset {split_file.name}")
            continue

        sample = samples[0]

        # UltraChat has messages structure
        assert "messages" in sample or "prompt" in sample, \
            f"Missing messages/prompt in {split}"


def test_preprocessing_compatibility():
    """Test that downloaded data is compatible with preprocessing script."""
    # This test verifies the data formats expected by preprocessing_ultrachat.py
    
    data_dir = Path("data")
    
    # Check if combined dataset exists
    combined_file = data_dir / "ultra_combined_fb0.01_ch0.1.json"
    if not combined_file.exists():
        print("SKIP: Combined dataset not yet generated")
        return
    
    with open(combined_file) as f:
        data = json.load(f)
    
    # Combined dataset should have these fields
    sample = data[0] if data else {}
    assert "instruction" in sample, "Missing instruction field"
    
    # Fields used by preprocessing_ultrachat.py
    # - instruction: prompt
    # - chosen: preferred response
    # - rejected: dispreferred response
    # - unlabeled: response for unlabeled data
    
    # Check that at least some samples have the expected structure
    has_labeled = any(
        sample.get("chosen") and sample.get("rejected") 
        for sample in data[:100]
    )
    has_unlabeled = any(
        sample.get("unlabeled") and not sample.get("chosen")
        for sample in data[:100]
    )
    
    assert has_labeled or has_unlabeled, \
        "Dataset should have labeled or unlabeled samples"


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("  Data Download Tests")
    print("=" * 60)
    
    tests = [
        ("Dataset Configs", test_dataset_configs),
        ("UltraFeedback Structure", test_ultrafeedback_structure),
        ("UltraChat Structure", test_ultrachat_structure),
        ("Preprocessing Compatibility", test_preprocessing_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            test_fn()
            print(f"  ✓ PASS")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
