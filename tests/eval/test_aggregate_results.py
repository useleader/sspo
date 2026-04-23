#!/usr/bin/env python3
"""Tests for results aggregation."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.eval.aggregate_results import (
    aggregate_results,
    generate_comparison_table,
)


class TestAggregateResults:
    """Tests for results aggregation."""

    def test_aggregate_results_from_directory(self):
        """Test aggregating results from a directory of JSON files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir)

            alpaca_result = {
                "method": "sspo",
                "model": "mistral",
                "lc_win_rate": 32.4,
            }
            mtbench_result = {
                "method": "sspo",
                "model": "mistral",
                "average_score": 7.2,
            }

            with open(results_path / "mistral_sspo_alpaca.json", "w") as f:
                json.dump(alpaca_result, f)
            with open(results_path / "mistral_sspo_mtbench.json", "w") as f:
                json.dump(mtbench_result, f)

            result = aggregate_results(tmp_dir)

            assert "alpaca_eval" in result
            assert "mtbench" in result
            assert "mistral_sspo" in result["alpaca_eval"]
            assert result["alpaca_eval"]["mistral_sspo"] == 32.4
            assert "mistral_sspo" in result["mtbench"]
            assert result["mtbench"]["mistral_sspo"] == 7.2

    def test_aggregate_results_with_nested_directories(self):
        """Test aggregating results from nested directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir) / "models" / "mistral"
            results_path.mkdir(parents=True)

            result_data = {
                "method": "dpo",
                "model": "mistral",
                "lc_win_rate": 26.2,
            }
            with open(results_path / "result.json", "w") as f:
                json.dump(result_data, f)

            result = aggregate_results(tmp_dir)

            assert "mistral_dpo" in result["alpaca_eval"]
            assert result["alpaca_eval"]["mistral_dpo"] == 26.2

    def test_aggregate_results_skips_invalid_json(self):
        """Test that invalid JSON files are skipped with warning."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir)

            with open(results_path / "valid.json", "w") as f:
                json.dump({"method": "dpo", "model": "mistral", "lc_win_rate": 26.2}, f)

            with open(results_path / "invalid.json", "w") as f:
                f.write("{ invalid json }")

            result = aggregate_results(tmp_dir)

            assert "mistral_dpo" in result["alpaca_eval"]

    def test_aggregate_results_handles_missing_method_field(self):
        """Test handling of result files missing method field."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir)

            result_data = {
                "model": "mistral",
                "lc_win_rate": 26.2,
            }
            with open(results_path / "no_method.json", "w") as f:
                json.dump(result_data, f)

            result = aggregate_results(tmp_dir)

            assert "mistral_unknown" in result["alpaca_eval"]

    def test_aggregate_results_saves_to_output_path(self):
        """Test saving aggregated results to file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir)
            output_path = results_path / "aggregated.json"

            with open(results_path / "test.json", "w") as f:
                json.dump({"method": "sspo", "lc_win_rate": 32.4}, f)

            result = aggregate_results(tmp_dir, output_path=str(output_path))

            assert output_path.exists()
            with open(output_path) as f:
                saved = json.load(f)
            assert "alpaca_eval" in saved


class TestGenerateComparisonTable:
    """Tests for comparison table generation."""

    def test_generate_alpaca_eval_table(self):
        """Test generating AlpacaEval comparison table."""
        aggregated = {
            "alpaca_eval": {
                "mistral_sft": 12.1,
                "mistral_dpo": 26.2,
                "mistral_simpo": 26.8,
                "mistral_orpo": 25.6,
                "mistral_kto": 25.9,
                "mistral_sspo": 32.4,
            },
            "mtbench": {},
        }

        table = generate_comparison_table(aggregated, "alpaca_eval")

        assert "| Method |" in table
        assert "| mistral |" in table
        assert "32.4" in table
        assert "SSPO" in table

    def test_generate_mtbench_table(self):
        """Test generating MT-Bench comparison table."""
        aggregated = {
            "alpaca_eval": {},
            "mtbench": {
                "reasoning_sspo": 6.5,
                "math_sspo": 5.8,
                "coding_sspo": 7.2,
                "writing_sspo": 7.8,
            },
        }

        table = generate_comparison_table(aggregated, "mtbench")

        assert "| Category |" in table
        assert "Reasoning" in table
        assert "Math" in table

    def test_generate_table_with_missing_data(self):
        """Test generating table when some data is missing."""
        aggregated = {
            "alpaca_eval": {
                "mistral_dpo": 26.2,
            },
            "mtbench": {},
        }

        table = generate_comparison_table(aggregated, "alpaca_eval")

        assert "N/A" in table
        assert "26.2" in table

    def test_generate_table_no_data(self):
        """Test generating table when data is empty (shows N/A for all)."""
        aggregated = {"alpaca_eval": {}, "mtbench": {}}

        table = generate_comparison_table(aggregated, "alpaca_eval")

        # Empty data still generates table with N/A values
        assert "| Method |" in table
        assert "N/A" in table
        assert "SSPO" in table

    def test_generate_table_unknown_metric(self):
        """Test generating table with unknown metric name."""
        aggregated = {"alpaca_eval": {}, "mtbench": {}}

        table = generate_comparison_table(aggregated, "unknown_metric")

        assert table == "No data available"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
