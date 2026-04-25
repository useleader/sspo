#!/usr/bin/env python3
"""Tests for qualitative evaluation script."""

import sys
from pathlib import Path

# Add scripts/eval to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "eval"))

from qualitative_analysis import CATEGORIES


def test_all_6_categories_defined():
    """Test that all 6 categories are defined."""
    expected_categories = [
        "table_13",
        "table_14",
        "table_15",
        "table_16",
        "table_17",
        "table_18",
    ]
    for cat in expected_categories:
        assert cat in CATEGORIES, f"Missing category: {cat}"


def test_categories_has_correct_keys():
    """Test that CATEGORIES has the correct keys with expected descriptions."""
    expected = {
        "table_13": "Helpful assistant (general)",
        "table_14": "Math reasoning",
        "table_15": "Code generation",
        "table_16": "Creative writing",
        "table_17": "Factual Q&A",
        "table_18": "Safety/ refusal handling",
    }
    assert CATEGORIES == expected, f"CATEGORIES mismatch: {CATEGORIES}"


def test_number_of_categories():
    """Test that exactly 6 categories are defined."""
    assert len(CATEGORIES) == 6, f"Expected 6 categories, got {len(CATEGORIES)}"
