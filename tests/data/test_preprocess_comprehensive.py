#!/usr/bin/env python3
"""
Comprehensive Tests for Preprocess Data - Business Logic Validation.

These tests cover POTENTIAL BUSINESS PROBLEMS that simple structure tests miss:
1. Edge cases in ratio calculation (0.0, 1.0, very small values)
2. Missing assistant messages in UltraChat
3. Empty rejected_response in UltraFeedback
4. JSON encoding issues
5. dataset_info.json overwrite protection
6. Empty dataset after sampling
7. Memory efficiency with large datasets
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.preprocess_data import (
    keep_partial_data,
    create_combined_dataset,
    save_combined_dataset,
    update_dataset_info,
    load_jsonl,
)


# ============================================================================
# Edge Case Tests - Ratio Calculation
# ============================================================================

class TestRatioCalculationEdgeCases:
    """Test edge cases in ratio calculation."""

    def test_ratio_zero_keeps_nothing(self):
        """Ratio 0.0 should keep 0 samples."""
        mock_data = [{"id": i} for i in range(100)]
        result = keep_partial_data(mock_data, keep_ratio=0.0)
        assert len(result) == 0, f"Ratio 0.0 should keep 0 samples, got {len(result)}"

    def test_ratio_one_keeps_all(self):
        """Ratio 1.0 should keep all samples."""
        mock_data = [{"id": i} for i in range(100)]
        result = keep_partial_data(mock_data, keep_ratio=1.0)
        assert len(result) == 100, f"Ratio 1.0 should keep 100 samples, got {len(result)}"

    def test_ratio_very_small(self):
        """Very small ratio (0.001) should work correctly."""
        mock_data = [{"id": i} for i in range(100000)]
        result = keep_partial_data(mock_data, keep_ratio=0.001)
        expected = 100  # 0.001 * 100000 = 100
        assert len(result) == expected, f"Expected {expected}, got {len(result)}"

    def test_ratio_rounds_down(self):
        """Fractional samples should round down."""
        mock_data = [{"id": i} for i in range(33)]
        result = keep_partial_data(mock_data, keep_ratio=0.1)
        expected = 3  # 0.1 * 33 = 3.3, should round to 3
        assert len(result) == expected, f"Expected {expected}, got {len(result)}"


# ============================================================================
# Data Quality Tests - Missing Fields
# ============================================================================

class TestDataQualityMissingFields:
    """Test handling of missing or malformed fields."""

    def test_ultrachat_missing_assistant_message(self):
        """UltraChat sample with no assistant message should be skipped."""
        ultrachat = [{
            "instruction": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "Hi again"}
            ]
        }]
        ultrafeedback = []

        combined, *_ = create_combined_dataset(ultrafeedback, ultrachat)

        assert len(combined) == 0, "Should skip samples with no assistant message"

    def test_ultrachat_empty_messages(self):
        """UltraChat with empty messages list should be skipped."""
        ultrachat = [{"instruction": "test", "messages": []}]
        ultrafeedback = []

        combined, *_ = create_combined_dataset(ultrafeedback, ultrachat)

        assert len(combined) == 0, "Should skip samples with empty messages"

    def test_ultrachat_missing_instruction(self):
        """UltraChat with no user message should use empty string."""
        ultrachat = [{
            "messages": [
                {"role": "assistant", "content": "Hi there"}
            ]
        }]
        ultrafeedback = []

        combined, *_ = create_combined_dataset(ultrafeedback, ultrachat)

        assert len(combined) == 1
        assert combined[0]["instruction"] == ""

    def test_ultrafeedback_empty_rejected(self):
        """UltraFeedback with empty rejected_response should still be included."""
        ultrafeedback = [{
            "instruction": "What is 2+2?",
            "chosen_response": "4",
            "rejected_response": "",
            "chosen_avg_rating": 5.0,
            "rejected_avg_rating": 0.0
        }]
        ultrachat = []

        combined, *_ = create_combined_dataset(ultrafeedback, ultrachat)

        assert len(combined) == 1
        assert combined[0]["chosen"] == "4"
        assert combined[0]["rejected"] == ""


# ============================================================================
# Data Integrity Tests
# ============================================================================

class TestDataIntegrity:
    """Test data integrity after processing."""

    def test_labeled_data_has_no_unlabeled(self):
        """Labeled data should have empty unlabeled field."""
        ultrafeedback = [
            {"instruction": "Q1", "chosen_response": "A1", "rejected_response": "A2"},
            {"instruction": "Q2", "chosen_response": "A3", "rejected_response": "A4"},
        ]
        ultrachat = []

        combined, *_ = create_combined_dataset(ultrafeedback, ultrachat)

        for sample in combined:
            assert sample["unlabeled"] == "", "Labeled data should have empty unlabeled"
            assert sample["chosen"] != "", "Labeled data should have non-empty chosen"

    def test_unlabeled_data_has_empty_chosen_rejected(self):
        """Unlabeled data should have empty chosen and rejected."""
        ultrachat = [{
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        }]
        ultrafeedback = []

        combined, *_ = create_combined_dataset(ultrafeedback, ultrachat)

        assert len(combined) == 1
        assert combined[0]["chosen"] == ""
        assert combined[0]["rejected"] == ""
        assert combined[0]["unlabeled"] != ""

    def test_combined_data_is_shuffled(self):
        """Combined data should be shuffled (not all labeled first)."""
        ultrafeedback = [
            {"instruction": f"Q{i}", "chosen_response": f"A{i}", "rejected_response": f"B{i}"}
            for i in range(100)
        ]
        ultrachat = [
            {
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"R{i}"}
                ]
            }
            for i in range(100)
        ]

        combined, *_ = create_combined_dataset(ultrafeedback, ultrachat)

        has_labeled_early = any(s["chosen"] != "" for s in combined[:50])
        has_unlabeled_early = any(s["unlabeled"] != "" for s in combined[:50])

        assert has_labeled_early and has_unlabeled_early, \
            "Data should be shuffled, but appears to be in original order"

    def test_all_samples_have_instruction(self):
        """Every sample should have an instruction field."""
        ultrafeedback = [
            {"instruction": "Q1", "chosen_response": "A1", "rejected_response": "A2"},
        ]
        ultrachat = [
            {"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]}
        ]

        combined, *_ = create_combined_dataset(ultrafeedback, ultrachat)

        for i, sample in enumerate(combined):
            assert "instruction" in sample, f"Sample {i} missing instruction field"


# ============================================================================
# Save/Load Tests
# ============================================================================

class TestSaveLoad:
    """Test save and load functionality."""

    def test_json_save_and_load(self, tmp_path):
        """Test that saved JSONL can be loaded correctly."""
        data = [
            {"instruction": "Q1", "chosen": "A1", "rejected": "A2", "unlabeled": ""},
            {"instruction": "Q2", "chosen": "", "rejected": "", "unlabeled": "R2"},
        ]

        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        loaded = load_jsonl(jsonl_file)

        assert len(loaded) == 2
        assert loaded[0]["instruction"] == "Q1"
        assert loaded[1]["unlabeled"] == "R2"

    def test_unicode_in_data(self, tmp_path):
        """Test handling of unicode characters."""
        data = [
            {"instruction": "你好世界", "chosen": "Hello", "rejected": "Hi", "unlabeled": ""},
        ]

        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        loaded = load_jsonl(jsonl_file)

        assert loaded[0]["instruction"] == "你好世界"

    def test_dataset_info_update(self, tmp_path):
        """Test that dataset_info.json is updated correctly."""
        info_file = tmp_path / "dataset_info.json"
        with open(info_file, "w") as f:
            json.dump({}, f)

        update_dataset_info(tmp_path, fb_ratio=0.01, ch_ratio=0.1)

        with open(info_file) as f:
            info = json.load(f)

        assert "ultra_combined_fb0.01_ch0.1" in info
        entry = info["ultra_combined_fb0.01_ch0.1"]
        assert entry["file_name"] == "./ultra_combined_fb0.01_ch0.1.json"
        assert entry["ranking"] is True
        assert entry["columns"]["prompt"] == "instruction"

    def test_dataset_info_preserves_existing(self, tmp_path):
        """Test that updating dataset_info preserves existing entries."""
        existing_info = {
            "existing_dataset": {
                "file_name": "./existing.json",
                "ranking": True,
                "columns": {"prompt": "text"}
            }
        }
        info_file = tmp_path / "dataset_info.json"
        with open(info_file, "w") as f:
            json.dump(existing_info, f)

        update_dataset_info(tmp_path, fb_ratio=0.05, ch_ratio=0.1)

        with open(info_file) as f:
            info = json.load(f)

        assert "existing_dataset" in info
        assert "ultra_combined_fb0.05_ch0.1" in info


# ============================================================================
# Scale Tests
# ============================================================================

class TestScale:
    """Test handling of large datasets."""

    def test_large_dataset_sampling(self):
        """Test sampling from large dataset."""
        mock_data = [{"id": i} for i in range(100000)]

        result = keep_partial_data(mock_data, keep_ratio=0.1)

        assert len(result) == 10000, f"Expected 10000, got {len(result)}"
        ids = [r["id"] for r in result]
        assert len(set(ids)) == len(ids), "Sampled IDs should be unique"

    def test_large_combined_dataset(self):
        """Test combining large datasets."""
        ultrafeedback = [
            {"instruction": f"Q{i}", "chosen_response": f"A{i}", "rejected_response": f"B{i}"}
            for i in range(50000)
        ]
        ultrachat = [
            {
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"R{i}"}
                ]
            }
            for i in range(50000)
        ]

        combined, *_ = create_combined_dataset(ultrafeedback, ultrachat)

        assert len(combined) == 100000, f"Expected 100000, got {len(combined)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
