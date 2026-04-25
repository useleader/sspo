#!/usr/bin/env python3
"""Tests for compute_overhead.py - Table 9 computational overhead evaluation."""

import pytest
from scripts.eval.compute_overhead import (
    OverheadMetrics,
    measure_overhead,
    FLOPS_RATIOS,
)


class TestOverheadMetrics:
    """Test suite for OverheadMetrics dataclass."""

    def test_measure_overhead_returns_overhead_metrics(self):
        """Test that measure_overhead returns OverheadMetrics instance."""
        result = measure_overhead("dpo")
        assert isinstance(result, OverheadMetrics)
        assert result.method == "dpo"
        assert result.flops_per_token == 1.0
        assert result.training_time_hours == 1.0
        assert result.peak_memory_gb == 40.0
        assert result.tokens_per_second == 45000.0

    def test_measure_overhead_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            measure_overhead("unknown_method")

    def test_sspo_has_flops_per_token_greater_than_one(self):
        """Test that SSPO has flops_per_token > 1.0."""
        result = measure_overhead("sspo")
        assert result.flops_per_token > 1.0, (
            f"SSPO flops_per_token should be > 1.0, got {result.flops_per_token}"
        )

    def test_sspo_flops_ratio_is_115(self):
        """Test that SSPO FLOPs ratio matches expected value of 1.15."""
        result = measure_overhead("sspo")
        assert result.flops_per_token == 1.15

    def test_dpo_baseline_flops_is_one(self):
        """Test that DPO baseline FLOPs ratio is exactly 1.0."""
        result = measure_overhead("dpo")
        assert result.flops_per_token == 1.0

    def test_all_methods_have_valid_flops(self):
        """Test that all defined methods return valid FLOPs ratios."""
        for method in FLOPS_RATIOS:
            result = measure_overhead(method)
            assert result.flops_per_token == FLOPS_RATIOS[method]
            assert result.method == method.lower()

    def test_methods_case_insensitive(self):
        """Test that method names are case insensitive."""
        result_lower = measure_overhead("dpo")
        result_upper = measure_overhead("DPO")
        result_mixed = measure_overhead("DpO")
        assert result_lower.flops_per_token == result_upper.flops_per_token
        assert result_upper.flops_per_token == result_mixed.flops_per_token

    def test_training_time_scales_with_flops(self):
        """Test that training time scales linearly with FLOPs ratio."""
        dpo_result = measure_overhead("dpo")
        sspo_result = measure_overhead("sspo")
        # SSPO has 1.15x FLOPs, so training time should be 1.15x
        assert sspo_result.training_time_hours == pytest.approx(
            dpo_result.training_time_hours * 1.15, rel=1e-4
        )

    def test_memory_overhead_for_higher_flops(self):
        """Test that memory increases for higher FLOPs methods."""
        dpo_result = measure_overhead("dpo")
        sspo_result = measure_overhead("sspo")
        # Memory should increase for SSPO due to aux tensors
        assert sspo_result.peak_memory_gb > dpo_result.peak_memory_gb

    def test_tokens_per_second_inversely_scaled(self):
        """Test that tokens_per_second decreases as FLOPs increase."""
        dpo_result = measure_overhead("dpo")
        sspo_result = measure_overhead("sspo")
        # Higher FLOPs means lower throughput
        assert sspo_result.tokens_per_second < dpo_result.tokens_per_second


class TestFLOPSRatios:
    """Test that FLOPs ratios match the specification."""

    def test_flops_ratios_match_specification(self):
        """Verify all FLOPs ratios match the specified values."""
        expected = {
            "dpo": 1.0,
            "simpo": 1.0,
            "sspo": 1.15,
            "kto": 1.05,
            "orpo": 1.0,
            "ssrm": 1.2,
            "spa": 1.18,
        }
        assert FLOPS_RATIOS == expected