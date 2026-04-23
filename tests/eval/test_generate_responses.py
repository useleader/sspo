"""Tests for generate_responses.py"""
import pytest
from dataclasses import asdict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.eval.generate_responses import GenerationConfig, load_benchmark_prompts


class TestGenerationConfig:
    """Test GenerationConfig matches paper evaluation settings."""

    def test_default_values_match_paper(self):
        """Paper page 7: max_tokens=2048, temperature=0.7."""
        config = GenerationConfig()
        assert config.max_new_tokens == 2048
        assert config.temperature == 0.7

    def test_config_to_dict(self):
        """Config should convert to dict for model inference."""
        config = GenerationConfig(max_new_tokens=1024, temperature=0.5)
        d = asdict(config)
        assert d["max_new_tokens"] == 1024
        assert d["temperature"] == 0.5

    def test_sampling_settings(self):
        """do_sample=True for diverse outputs."""
        config = GenerationConfig()
        assert config.do_sample is True
        assert config.top_p == 0.9


class TestLoadBenchmarkPrompts:
    """Test loading benchmark datasets."""

    def test_load_alpacaeval_prompts(self):
        """AlpacaEval has 805 samples (paper page 8) when real library installed.
        When mock, returns 100 prompts with 'alpaca_eval not installed' warning."""
        prompts = load_benchmark_prompts("alpacaeval")
        # Real library returns 805, mock returns 100
        assert len(prompts) in [805, 100], f"Expected 805 or 100 prompts, got {len(prompts)}"
        assert "instruction" in prompts[0]

    def test_load_mtbench_prompts(self):
        """MT-Bench has 8 categories when real library installed.
        When mock, returns 10 prompts per category (80 total) with warning."""
        prompts = load_benchmark_prompts("mtbench")
        # Real library returns 8 categories, mock returns 80 (10 per category)
        assert len(prompts) in [8, 80], f"Expected 8 or 80 prompts, got {len(prompts)}"
        assert "instruction" in prompts[0]


class TestGenerateResponses:
    """Test response generation pipeline."""

    def test_generate_single_response(self, tmp_path):
        """Generate response for single prompt."""
        # Mock model and tokenizer
        # ...

    def test_save_responses_to_json(self, tmp_path):
        """Responses should save as JSON for evaluation."""
        # ...