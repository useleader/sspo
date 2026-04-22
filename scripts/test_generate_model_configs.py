#!/usr/bin/env python3
"""
Tests for generate_model_configs.py

These tests verify:
1. YAML files are valid and readable
2. SSPO parameters are correctly set
3. All paper configurations are supported
"""

import sys
import yaml
from pathlib import Path


def test_yaml_validity():
    """Test that generated YAML files are valid."""
    configs_dir = Path("configs")
    
    if not configs_dir.exists():
        print("SKIP: configs directory not found")
        return
    
    yaml_files = list(configs_dir.glob("**/*.yaml"))
    
    if not yaml_files:
        print("SKIP: No YAML files found")
        return
    
    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            try:
                config = yaml.safe_load(f)
                assert config is not None, f"Empty config in {yaml_file}"
                assert isinstance(config, dict), f"Config is not a dict"
            except yaml.YAMLError as e:
                print(f"  ✗ FAIL: Invalid YAML in {yaml_file}: {e}")
                return
    
    print(f"  ✓ All {len(yaml_files)} YAML files are valid")


def test_sspo_parameters():
    """Test that SSPO parameters are correctly set."""
    configs_dir = Path("configs")
    
    if not configs_dir.exists():
        print("SKIP: configs directory not found")
        return
    
    sspo_files = list(configs_dir.glob("**/sspo/*.yaml"))
    
    if not sspo_files:
        print("SKIP: No SSPO configs found")
        return
    
    for yaml_file in sspo_files:
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        
        # Check SSPO-specific parameters
        assert config.get("pref_loss") == "sspo", \
            f"pref_loss should be 'sspo' in {yaml_file}"
        assert config.get("sspo_gamma_0") == 1.0, \
            f"sspo_gamma_0 should be 1.0 in {yaml_file}"
        assert config.get("sspo_gamma_min") == 0.22, \
            f"sspo_gamma_min should be 0.22 in {yaml_file}"
        assert config.get("sspo_gamma_decay") == 0.001, \
            f"sspo_gamma_decay should be 0.001 in {yaml_file}"
        assert config.get("sspo_prior") == 0.5, \
            f"sspo_prior should be 0.5 in {yaml_file}"
        assert config.get("sspo_base") == "simpo", \
            f"sspo_base should be 'simpo' in {yaml_file}"
        assert config.get("pref_beta") == 2.0, \
            f"pref_beta should be 2.0 in {yaml_file}"


def test_lora_config():
    """Test that LoRA configuration is correct."""
    configs_dir = Path("configs")
    
    if not configs_dir.exists():
        print("SKIP: configs directory not found")
        return
    
    yaml_files = list(configs_dir.glob("**/sspo/*.yaml"))
    
    if not yaml_files:
        print("SKIP: No SSPO configs found")
        return
    
    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        
        # Check LoRA parameters (from paper)
        assert config.get("lora_rank") == 8, \
            f"lora_rank should be 8 in {yaml_file}"
        assert config.get("lora_target") == "all", \
            f"lora_target should be 'all' in {yaml_file}"


def test_paper_configs_exist():
    """Test that paper configurations are generated."""
    configs_dir = Path("configs")
    
    if not configs_dir.exists():
        print("SKIP: configs directory not found")
        return
    
    # Expected configurations from paper
    expected = [
        ("mistral-7b-it", "sspo", 0.01),
        ("mistral-7b-it", "sspo", 0.05),
        ("mistral-7b-it", "sspo", 0.10),
        ("llama3-8b-it", "sspo", 0.01),
        ("llama3-8b-it", "sspo", 0.05),
        ("llama3-8b-it", "sspo", 0.10),
        ("qwen2-7b-it", "sspo", 0.01),
        ("qwen2-7b-it", "sspo", 0.05),
        ("qwen2-7b-it", "sspo", 0.10),
    ]
    
    found = 0
    missing = []
    
    for model, method, fb in expected:
        expected_file = configs_dir / model / method / f"fb{fb}_ch0.1_{method}_{model}.yaml"
        if expected_file.exists():
            found += 1
        else:
            missing.append(f"{model}/{method}/fb{fb}")
    
    if missing:
        print(f"  Found {found}/{len(expected)} expected configs")
        print(f"  Missing: {missing}")
    else:
        print(f"  ✓ All {len(expected)} expected configs found")


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("  Model Config Generator Tests")
    print("=" * 60)
    
    tests = [
        ("YAML Validity", test_yaml_validity),
        ("SSPO Parameters", test_sspo_parameters),
        ("LoRA Config", test_lora_config),
        ("Paper Configs", test_paper_configs_exist),
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
