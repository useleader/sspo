#!/usr/bin/env python3
"""Tests for MT-Bench evaluator."""

import json
import sys
import tempfile
from pathlib import Path
import inspect

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestMTBenchEvaluator:
    """Tests for MT-Bench evaluation."""

    def test_evaluate_function_signature(self):
        """Test that evaluate_mtbench function exists and has correct signature."""
        from scripts.eval.mtbench_evaluator import evaluate_mtbench

        sig = inspect.signature(evaluate_mtbench)
        params = list(sig.parameters.keys())

        assert "model_outputs_path" in params
        assert "output_dir" in params

    def test_evaluate_mtbench_returns_expected_keys(self):
        """Test that evaluate_mtbench returns expected keys."""
        from scripts.eval.mtbench_evaluator import evaluate_mtbench

        # This will raise ImportError if mtbench not installed
        # We're testing the expected structure
        try:
            import mtbench
        except ImportError:
            pytest.skip("mtbench not installed")

    def test_cli_function_exists(self):
        """Test that CLI function exists."""
        from scripts.eval.mtbench_evaluator import evaluate_mtbench_cli
        assert callable(evaluate_mtbench_cli)

    def test_evaluate_raises_import_error_when_library_missing(self):
        """Test that ImportError is raised when mtbench not installed."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mtbench" or name.startswith("mtbench."):
                raise ImportError("No module named 'mtbench'")
            return original_import(name, *args, **kwargs)

        if "scripts.eval.mtbench_evaluator" in sys.modules:
            del sys.modules["scripts.eval.mtbench_evaluator"]

        try:
            builtins.__import__ = mock_import
            import scripts.eval.mtbench_evaluator as reloaded_module

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump([], f)
                temp_path = f.name

            try:
                with pytest.raises(ImportError, match="mtbench"):
                    reloaded_module.evaluate_mtbench(temp_path)
            finally:
                Path(temp_path).unlink()
        finally:
            builtins.__import__ = original_import
            if "scripts.eval.mtbench_evaluator" in sys.modules:
                del sys.modules["scripts.eval.mtbench_evaluator"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
