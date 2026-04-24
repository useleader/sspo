"""
Tests for SSRM and SPA trainers.

These tests verify that SSRM (Semi-Supervised Reward Modeling)
and SPA (Spread Preference Annotation) trainers are properly implemented.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestSSRMTrainerExists:
    """Test that SSRM trainer exists and can be imported."""

    @staticmethod
    def test_ssrm_trainer_import():
        """SSRM trainer should be importable."""
        from src_sspo.llamafactory.train.ssrm.trainer import SSRMTrainer
        assert SSRMTrainer is not None

    @staticmethod
    def test_ssrm_trainer_has_ssrm_loss():
        """SSRM trainer should have ssrm-specific loss."""
        from src_sspo.llamafactory.train.ssrm.trainer import SSRMTrainer
        import inspect

        # Check that compute_loss method exists
        assert hasattr(SSRMTrainer, 'compute_loss')

        # Check __init__ parameters
        sig = inspect.signature(SSRMTrainer.__init__)
        params = list(sig.parameters.keys())
        assert 'ssrm_prior' in params
        assert 'ssrm_iterations' in params
        assert 'ssrm_threshold' in params


class TestSPATrainerExists:
    """Test that SPA trainer exists and can be imported."""

    @staticmethod
    def test_spa_trainer_import():
        """SPA trainer should be importable."""
        from src_sspo.llamafactory.train.spa.trainer import SPATrainer
        assert SPATrainer is not None

    @staticmethod
    def test_spa_trainer_has_spa_params():
        """SPA trainer should have spa-specific parameters."""
        from src_sspo.llamafactory.train.spa.trainer import SPATrainer
        import inspect

        # Check __init__ parameters
        sig = inspect.signature(SPATrainer.__init__)
        params = list(sig.parameters.keys())
        assert 'spa_iterations' in params
        assert 'spa_expansion_ratio' in params


class TestTrainerIntegration:
    """Test that trainers work with the existing framework."""

    @staticmethod
    def test_custom_dpo_trainer_exists():
        """CustomDPOTrainer (base class) should exist."""
        from src_sspo.llamafactory.train.dpo.trainer import CustomDPOTrainer
        assert CustomDPOTrainer is not None

    @staticmethod
    def test_ssrm_inherits_from_custom_dpo():
        """SSRM should inherit from CustomDPOTrainer."""
        from src_sspo.llamafactory.train.ssrm.trainer import SSRMTrainer
        from src_sspo.llamafactory.train.dpo.trainer import CustomDPOTrainer
        assert issubclass(SSRMTrainer, CustomDPOTrainer)

    @staticmethod
    def test_spa_inherits_from_custom_dpo():
        """SPA should inherit from CustomDPOTrainer."""
        from src_sspo.llamafactory.train.spa.trainer import SPATrainer
        from src_sspo.llamafactory.train.dpo.trainer import CustomDPOTrainer
        assert issubclass(SPATrainer, CustomDPOTrainer)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
