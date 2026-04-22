"""Tests for generate_responses.py"""
import pytest
from dataclasses import asdict
from generation_config import GenerationConfig


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
        """AlpacaEval has 805 samples (paper page 8)."""
        prompts = load_benchmark_prompts("alpacaeval")
        assert len(prompts) == 805

    def test_load_mtbench_prompts(self):
        """MT-Bench has 8 categories."""
        prompts = load_benchmark_prompts("mtbench")
        assert len(prompts) == 8  # 8 categories


class TestGenerateResponses:
    """Test response generation pipeline."""

    def test_generate_single_response(self, tmp_path):
        """Generate response for single prompt."""
        # Mock model and tokenizer
        # ...

    def test_save_responses_to_json(self, tmp_path):
        """Responses should save as JSON for evaluation."""
        # ...