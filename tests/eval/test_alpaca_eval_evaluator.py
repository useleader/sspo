#!/usr/bin/env python3
"""Tests for AlpacaEval evaluator."""

import json
import sys
import tempfile
from pathlib import Path
import inspect

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestAlpacaEvalEvaluator:
    """Tests for AlpacaEval evaluation."""

    def test_evaluate_function_signature(self):
        """Test that evaluate_alpacaeval function exists and has correct signature."""
        from scripts.eval.alpaca_eval_evaluator import evaluate_alpacaeval

        sig = inspect.signature(evaluate_alpacaeval)
        params = list(sig.parameters.keys())

        assert "model_outputs_path" in params
        assert "output_dir" in params
        assert "reference_output" in params

    def test_cli_function_exists(self):
        """Test that CLI function exists."""
        from scripts.eval.alpaca_eval_evaluator import evaluate_alpacaeval_cli
        assert callable(evaluate_alpacaeval_cli)

    def test_evaluate_raises_import_error_when_library_missing(self):
        """Test that ImportError is raised when alpaca_eval not installed."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "alpaca_eval" or name.startswith("alpaca_eval."):
                raise ImportError("No module named 'alpaca_eval'")
            return original_import(name, *args, **kwargs)

        if "scripts.eval.alpaca_eval_evaluator" in sys.modules:
            del sys.modules["scripts.eval.alpaca_eval_evaluator"]

        try:
            builtins.__import__ = mock_import
            import scripts.eval.alpaca_eval_evaluator as reloaded_module

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump([], f)
                temp_path = f.name

            try:
                with pytest.raises(ImportError, match="alpaca_eval"):
                    reloaded_module.evaluate_alpacaeval(temp_path)
            finally:
                Path(temp_path).unlink()
        finally:
            builtins.__import__ = original_import
            if "scripts.eval.alpaca_eval_evaluator" in sys.modules:
                del sys.modules["scripts.eval.alpaca_eval_evaluator"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
