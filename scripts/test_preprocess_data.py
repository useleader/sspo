#!/usr/bin/env python3
"""
Tests for preprocess_data.py

These tests verify:
1. Data sampling ratios are correct
2. Combined dataset has correct structure
3. Data formats match LLaMA-Factory expectations
"""

import json
import sys
from pathlib import Path


def test_combined_dataset_structure():
    """Test that combined dataset has correct fields."""
    data_dir = Path("data")
    
    # Find combined dataset files
    combined_files = list(data_dir.glob("ultra_combined_fb*.json"))
    
    if not combined_files:
        print("SKIP: No combined dataset found")
        return
    
    for fpath in combined_files:
        with open(fpath) as f:
            data = json.load(f)
        
        if not data:
            print(f"SKIP: Empty dataset {fpath.name}")
            continue
        
        sample = data[0]
        
        # Check required fields
        assert "instruction" in sample, f"Missing instruction in {fpath.name}"
        assert "chosen" in sample, f"Missing chosen in {fpath.name}"
        assert "rejected" in sample, f"Missing rejected in {fpath.name}"
        assert "unlabeled" in sample, f"Missing unlabeled in {fpath.name}"
        
        # Verify field types
        assert isinstance(sample["instruction"], str), "instruction should be string"
        assert isinstance(sample["chosen"], str), "chosen should be string"
        assert isinstance(sample["rejected"], str), "rejected should be string"
        assert isinstance(sample["unlabeled"], str), "unlabeled should be string"
        
        # For labeled data, chosen and rejected should be non-empty
        # For unlabeled data, unlabeled should be non-empty
        is_labeled = bool(sample["chosen"] and sample["rejected"])
        is_unlabeled = bool(sample["unlabeled"] and not sample["chosen"])
        
        assert is_labeled or is_unlabeled, \
            f"Sample should be either labeled or unlabeled: {sample}"


def test_ratio_calculation():
    """Test that data ratios are calculated correctly."""
    from scripts.preprocess_data import keep_partial_data
    
    # Create mock dataset
    mock_data = [{"id": i} for i in range(100)]
    
    # Test 10% ratio
    kept = keep_partial_data(mock_data, keep_ratio=0.1)
    assert len(kept) == 10, f"Expected 10 samples, got {len(kept)}"
    
    # Test 1% ratio
    kept = keep_partial_data(mock_data, keep_ratio=0.01)
    assert len(kept) == 1, f"Expected 1 sample, got {len(kept)}"
    
    # Test 5% ratio
    kept = keep_partial_data(mock_data, keep_ratio=0.05)
    assert len(kept) == 5, f"Expected 5 samples, got {len(kept)}"


def test_dataset_info_json():
    """Test that dataset_info.json is properly formatted."""
    data_dir = Path("data")
    info_file = data_dir / "dataset_info.json"
    
    if not info_file.exists():
        print("SKIP: dataset_info.json not found")
        return
    
    with open(info_file) as f:
        info = json.load(f)
    
    # Check that entries have required fields
    for name, entry in info.items():
        assert "file_name" in entry, f"Missing file_name in {name}"
        assert "columns" in entry, f"Missing columns in {name}"
        
        columns = entry["columns"]
        assert "prompt" in columns, f"Missing prompt column in {name}"
        assert columns["prompt"] == "instruction", \
            f"prompt should map to 'instruction', got {columns['prompt']}"


def test_paper_configuration():
    """Test that paper configurations are supported."""
    # From ADR-0003, paper uses these ratios
    paper_configs = [
        {"fb": 0.01, "ch": 0.10},  # 1% labeled
        {"fb": 0.05, "ch": 0.10},  # 5% labeled
        {"fb": 0.10, "ch": 0.10},  # 10% labeled
    ]
    
    data_dir = Path("data")
    
    for config in paper_configs:
        fb = config["fb"]
        ch = config["ch"]
        expected_name = f"ultra_combined_fb{fb}_ch{ch}.json"
        
        # Check if file exists (may not be downloaded yet)
        if not (data_dir / expected_name).exists():
            print(f"SKIP: {expected_name} not yet generated")
            continue
        
        print(f"Found: {expected_name}")


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("  Data Preprocessing Tests")
    print("=" * 60)
    
    tests = [
        ("Combined Dataset Structure", test_combined_dataset_structure),
        ("Ratio Calculation", test_ratio_calculation),
        ("Dataset Info JSON", test_dataset_info_json),
        ("Paper Configuration", test_paper_configuration),
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
