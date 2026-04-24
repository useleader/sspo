"""
Semi-Supervised Reward Modeling (SSRM) Trainer.

This trainer implements the SSRM method from the SSPO paper:
- Train reward model on labeled data
- Use it to generate pseudo-labels for unlabeled data
- Retrain with combined labeled + pseudo-labeled data

Reference: SSPO Paper Section 3.2, Table 1
"""

from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn.functional as F

from ..dpo.trainer import CustomDPOTrainer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from ...hparams import FinetuningArguments


class SSRMTrainer(CustomDPOTrainer):
    """
    Semi-Supervised Reward Modeling Trainer.

    This trainer extends CustomDPOTrainer with SSRM-specific functionality:
    - Uses pseudo-labeling on unlabeled data
    - Implements iterative refinement rounds
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        ssrm_prior: float = 0.5,
        ssrm_iterations: int = 3,
        ssrm_threshold: float = 0.9,
        **kwargs,
    ):
        self.ssrm_prior = ssrm_prior
        self.ssrm_iterations = ssrm_iterations
        self.ssrm_threshold = ssrm_threshold

        super().__init__(
            model=model,
            ref_model=ref_model,
            finetuning_args=finetuning_args,
            processor=processor,
            **kwargs,
        )

    def ssrm_pseudo_label_loss(
        self,
        policy_unlabeled_logps: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Computes pseudo-label loss for SSRM.

        Uses the policy model to score unlabeled samples and computes
        a binary cross-entropy loss based on the confidence threshold.

        Args:
            policy_unlabeled_logps: Log probabilities from policy for unlabeled samples

        Returns:
            Pseudo-label loss tensor
        """
        # Compute confidence scores (using sigmoid on logps as proxy)
        confidence = torch.sigmoid(policy_unlabeled_logps)

        # Assign pseudo-labels based on threshold
        pseudo_labels = (confidence > self.ssrm_threshold).float()

        # Binary cross-entropy loss
        # For high confidence positive samples: -log(sigmoid(logps))
        # For high confidence negative samples: -log(1 - sigmoid(logps))
        ssrm_loss = -(
            self.ssrm_prior * pseudo_labels * F.logsigmoid(policy_unlabeled_logps)
            + (1 - self.ssrm_prior) * (1 - pseudo_labels) * F.logsigmoid(-policy_unlabeled_logps)
        )

        return ssrm_loss.mean()

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Compute SSRM loss combining DPO loss on labeled data and pseudo-label loss on unlabeled data.
        """
        # Get standard DPO loss on labeled data
        dpo_loss = super().compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch
        )

        # Check if we have unlabeled data in inputs
        if "unlabeled_logps" in inputs and inputs["unlabeled_logps"] is not None:
            unlabeled_logps = inputs["unlabeled_logps"]
            ssrm_loss = self.ssrm_pseudo_label_loss(unlabeled_logps)

            # Weighted combination: DPO loss + alpha * SSRM loss
            alpha = 0.1  # Weight for SSRM loss
            total_loss = dpo_loss + alpha * ssrm_loss
            return total_loss

        return dpo_loss
