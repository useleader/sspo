"""
Spread Preference Annotation (SPA) Trainer.

This trainer implements the SPA method from the SSPO paper:
- Iterative self-annotation where the model generates responses
- Reward model scores (prompt, response) pairs
- High-scoring pairs are added to training data

Reference: SSPO Paper Section 3.2, Table 1
"""

from typing import TYPE_CHECKING, Optional, Union

import torch

from ..dpo.trainer import CustomDPOTrainer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from ...hparams import FinetuningArguments


class SPATrainer(CustomDPOTrainer):
    """
    Spread Preference Annotation Trainer.

    This trainer extends CustomDPOTrainer with SPA-specific functionality:
    - Iterative self-annotation process
    - High-scoring generated responses added to training
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        spa_iterations: int = 3,
        spa_expansion_ratio: float = 0.1,
        **kwargs,
    ):
        self.spa_iterations = spa_iterations
        self.spa_expansion_ratio = spa_expansion_ratio

        super().__init__(
            model=model,
            ref_model=ref_model,
            finetuning_args=finetuning_args,
            processor=processor,
            **kwargs,
        )

    def spa_score_responses(
        self,
        generated_logps: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Scores generated responses for SPA.

        Higher scores indicate better quality responses that should be
        added to the training set.

        Args:
            generated_logps: Log probabilities of generated responses

        Returns:
            Response quality scores
        """
        # Use log probabilities as quality scores
        # Higher logps = higher quality response
        return generated_logps

    def spa_select_high_quality(
        self,
        scores: "torch.Tensor",
        expansion_ratio: float = None,
    ) -> "torch.Tensor":
        """
        Selects high-quality samples based on scores.

        Args:
            scores: Quality scores for samples
            expansion_ratio: Fraction of samples to select (default: spa_expansion_ratio)

        Returns:
            Boolean mask indicating selected samples
        """
        if expansion_ratio is None:
            expansion_ratio = self.spa_expansion_ratio

        # Select top-k samples by score
        k = int(len(scores) * expansion_ratio)
        if k == 0:
            k = 1

        _, top_indices = torch.topk(scores, k=min(k, len(scores)))

        # Create boolean mask
        selected = torch.zeros_like(scores, dtype=torch.bool)
        selected[top_indices] = True

        return selected

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Compute SPA loss.

        For SPA, the main loss is the standard DPO/SimPO loss.
        The iterative self-annotation is handled externally by the training loop.
        """
        # Standard DPO loss - SPA uses DPO as base
        return super().compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch
        )
