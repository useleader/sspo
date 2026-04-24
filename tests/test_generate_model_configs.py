"""
Tests for model config generation.

Tests that generate_model_configs.py produces correct configs for all methods:
- DPO, ORPO, SimPO, KTO, SSRM, SPA, SSPO
"""

import sys
import tempfile
import yaml
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.generate_model_configs import generate_yaml, MODELS, TrainingConfig


class TestMethodConfigs:
    """Test that each method generates correct config fields."""

    @staticmethod
    def test_sspo_config():
        """SSPO should have sspo-specific fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_yaml("mistral", "sspo", 0.01, 0.10, Path(tmpdir))
            with open(path) as f:
                config = yaml.safe_load(f)

            assert config["pref_loss"] == "sspo"
            assert config["pref_beta"] == 2.0
            assert config["sspo_gamma_0"] == 1.0
            assert config["sspo_gamma_min"] == 0.22
            assert config["sspo_prior"] == 0.5
            assert config["sspo_base"] == "simpo"

    @staticmethod
    def test_dpo_config():
        """DPO should have sigmoid loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_yaml("mistral", "dpo", 0.01, 0.10, Path(tmpdir))
            with open(path) as f:
                config = yaml.safe_load(f)

            assert config["pref_loss"] == "sigmoid"
            assert config["pref_beta"] == 0.1
            assert config["stage"] == "dpo"

    @staticmethod
    def test_orpo_config():
        """ORPO should have orpo loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_yaml("mistral", "orpo", 0.01, 0.10, Path(tmpdir))
            with open(path) as f:
                config = yaml.safe_load(f)

            assert config["pref_loss"] == "orpo"
            assert config["pref_beta"] == 0.1
            assert config["stage"] == "dpo"

    @staticmethod
    def test_simpo_config():
        """SimPO should have simpo loss and gamma."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_yaml("mistral", "simpo", 0.01, 0.10, Path(tmpdir))
            with open(path) as f:
                config = yaml.safe_load(f)

            assert config["pref_loss"] == "simpo"
            assert config["pref_beta"] == 2.0
            assert config["simpo_gamma"] == 2.0
            assert config["stage"] == "dpo"

    @staticmethod
    def test_kto_config():
        """KTO should use kto stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_yaml("mistral", "kto", 0.01, 0.10, Path(tmpdir))
            with open(path) as f:
                config = yaml.safe_load(f)

            assert config["pref_loss"] == "kto_pair"
            assert config["pref_beta"] == 0.1
            assert config["stage"] == "kto"

    @staticmethod
    def test_ssrm_config():
        """SSRM should have ssrm-specific fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_yaml("mistral", "ssrm", 0.01, 0.10, Path(tmpdir))
            with open(path) as f:
                config = yaml.safe_load(f)

            assert config["pref_loss"] == "ssrm"
            assert "ssrm_prior" in config
            assert config["stage"] == "dpo"

    @staticmethod
    def test_spa_config():
        """SPA should have spa-specific fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_yaml("mistral", "spa", 0.01, 0.10, Path(tmpdir))
            with open(path) as f:
                config = yaml.safe_load(f)

            assert config["pref_loss"] == "spa"
            assert "spa_iterations" in config
            assert config["stage"] == "dpo"


class TestSharedConfigFields:
    """Test that all methods share common fields."""

    @staticmethod
    def test_all_methods_have_lora():
        """All methods should have LoRA config."""
        methods = ["sspo", "dpo", "orpo", "simpo", "kto", "ssrm", "spa"]
        with tempfile.TemporaryDirectory() as tmpdir:
            for method in methods:
                path = generate_yaml("mistral", method, 0.01, 0.10, Path(tmpdir))
                with open(path) as f:
                    config = yaml.safe_load(f)

                assert config["finetuning_type"] == "lora"
                assert config["lora_rank"] == 8
                assert config["lora_target"] == "all"

    @staticmethod
    def test_all_methods_have_common_training_params():
        """All methods should share training hyperparameters."""
        methods = ["sspo", "dpo", "orpo", "simpo", "kto", "ssrm", "spa"]
        with tempfile.TemporaryDirectory() as tmpdir:
            for method in methods:
                path = generate_yaml("mistral", method, 0.01, 0.10, Path(tmpdir))
                with open(path) as f:
                    config = yaml.safe_load(f)

                assert config["learning_rate"] == 5e-7
                assert config["num_train_epochs"] == 3
                assert config["bf16"] is True


class TestOutputStructure:
    """Test output directory structure."""

    @staticmethod
    def test_output_directory_structure():
        """Configs should be in correct directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_yaml("mistral", "dpo", 0.01, 0.10, Path(tmpdir))

            # Should be at: {output}/mistral-7b-it/dpo/fb0.01_ch0.1_dpo_mistral-7b-it.yaml
            assert path.parent.name == "dpo"
            assert path.parent.parent.name == "mistral-7b-it"
            assert "fb0.01" in path.name
            assert "dpo" in path.name


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
