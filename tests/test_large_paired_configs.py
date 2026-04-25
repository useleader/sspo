"""
Tests for large paired data config generation (Table 11).

Tests that generate_large_paired_configs.py produces correct configs for:
- 4 n_L values: 100, 1000, 5000, 10000
- fb calculation: nl / 60000
- YAML output format
"""

import sys
import tempfile
import yaml
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.generate_large_paired_configs import (
    generate_large_paired_configs,
    N_L_VALUES,
    TOTAL_DATASET_SIZE,
    calculate_fb,
)


class TestLargePairedConfigGeneration:
    """Test large paired config generation."""

    @staticmethod
    def test_configs_generated_for_four_nl_values():
        """Configs should be generated for all 4 n_L values."""
        assert len(N_L_VALUES) == 4
        assert 100 in N_L_VALUES
        assert 1000 in N_L_VALUES
        assert 5000 in N_L_VALUES
        assert 10000 in N_L_VALUES

    @staticmethod
    def test_fb_calculation():
        """fb should be calculated as n_L / 60000."""
        import pytest
        assert calculate_fb(100) == pytest.approx(100 / 60000)
        assert calculate_fb(1000) == pytest.approx(1000 / 60000)
        assert calculate_fb(5000) == pytest.approx(5000 / 60000)
        assert calculate_fb(10000) == pytest.approx(10000 / 60000)

    @staticmethod
    def test_generate_configs_for_single_model():
        """Configs should be generated for a single model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generated = generate_large_paired_configs(
                output_dir=output_dir,
                models=["mistral"],
                nl_values=[100, 1000, 5000, 10000],
            )

            # Should generate 4 configs (one per n_L value)
            assert len(generated) == 4

    @staticmethod
    def test_generate_configs_for_multiple_models():
        """Configs should be generated for multiple models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generated = generate_large_paired_configs(
                output_dir=output_dir,
                models=["mistral", "llama3", "qwen2"],
                nl_values=[100, 1000, 5000, 10000],
            )

            # Should generate 12 configs (4 n_L values * 3 models)
            assert len(generated) == 12


class TestLargePairedConfigOutput:
    """Test large paired config output files."""

    @staticmethod
    def test_output_files_have_yaml_extension():
        """All output files should have .yaml extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generated = generate_large_paired_configs(
                output_dir=output_dir,
                models=["mistral"],
                nl_values=[100, 1000],
            )

            for path in generated:
                assert path.suffix == ".yaml"

    @staticmethod
    def test_config_contains_required_fields():
        """Generated config should contain all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generated = generate_large_paired_configs(
                output_dir=output_dir,
                models=["mistral"],
                nl_values=[1000],
            )

            assert len(generated) == 1
            path = generated[0]

            with open(path) as f:
                config = yaml.safe_load(f)

            # Check required fields from TEMPLATE_PAIRED
            assert config["stage"] == "dpo"
            assert config["finetuning_type"] == "lora"
            assert config["lora_rank"] == 8
            assert config["learning_rate"] == 5e-7
            assert config["num_train_epochs"] == 3
            assert config["bf16"] is True

    @staticmethod
    def test_config_output_directory_structure():
        """Configs should be in correct directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generated = generate_large_paired_configs(
                output_dir=output_dir,
                models=["mistral"],
                nl_values=[1000],
            )

            path = generated[0]
            # Should be at: {output}/mistral-7b-it/large_paired/nl1000_large_paired_mistral-7b-it.yaml
            assert path.parent.name == "large_paired"
            assert path.parent.parent.name == "mistral-7b-it"

    @staticmethod
    def test_config_filename_contains_nl():
        """Filename should contain n_L value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            for nl in [100, 1000, 5000, 10000]:
                generated = generate_large_paired_configs(
                    output_dir=output_dir,
                    models=["mistral"],
                    nl_values=[nl],
                )
                path = generated[0]
                assert f"nl{nl}" in path.name


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
