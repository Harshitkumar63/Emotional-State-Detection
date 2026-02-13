"""
Attention-Based Multimodal Fusion Network
==========================================
The core fusion module: a PyTorch network that fuses embeddings from text,
voice, and face modalities into a burnout risk prediction.

Architecture:
  1. Per-modality projectors map each 768-d embedding to a shared hidden space.
  2. A self-attention mechanism learns PER-SAMPLE importance weights.
  3. The weighted sum of projected embeddings is classified into 3 burnout
     risk classes (Low / Moderate / High).

Why attention-based fusion over simpler approaches?
---------------------------------------------------
| Approach          | Problem                                           |
|-------------------|---------------------------------------------------|
| Averaging         | Treats all modalities equally, even noisy ones    |
| Concatenation     | Grows linearly with modalities; no learned weight |
| Fixed weighting   | Same weights for every input — no adaptivity      |
| **Attention**     | Learns per-sample importance; graceful w/ missing |

Handling missing modalities:
  When a modality is absent, its attention score is masked to -inf before
  softmax, so it receives zero weight.  This is mathematically equivalent
  to "the model only looks at what's available" — no imputation or
  zero-padding tricks needed.

The model is small (~260K parameters) and trains in seconds on CPU with
synthetic data.  For production, it should be fine-tuned on real aligned
multimodal datasets like MELD or CMU-MOSEI.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# The three modalities in fixed order (used for consistent indexing)
MODALITY_ORDER = ("text", "voice", "face")


class AttentionFusionNetwork(nn.Module):
    """Attention-based fusion of multimodal embeddings for burnout risk.

    Parameters
    ----------
    embedding_dim : int
        Dimension of each modality's embedding (768 for all our models).
    hidden_dim : int
        Shared projection dimension (smaller = lighter model).
    num_classes : int
        Number of output classes (3 = Low / Moderate / High Risk).
    dropout : float
        Dropout rate for regularisation.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # --- Per-modality projectors ---
        # Each projects from embedding_dim to a shared hidden_dim space.
        # Why separate projectors?  Each modality's embedding space has
        # different characteristics (language vs. speech vs. visual).
        # Separate projectors let each learn its own optimal mapping.
        self.projectors = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for mod in MODALITY_ORDER
        })

        # --- Cross-modal attention scorer ---
        # Learns a scalar importance score for each modality's projected
        # embedding.  Using Tanh activation followed by a linear layer
        # is the standard Bahdanau-style attention mechanism.
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # --- Classification head ---
        # Takes the attention-weighted fused embedding and predicts risk.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        return_weights: bool = True,
    ) -> dict:
        """Forward pass: fuse modality embeddings and predict burnout risk.

        Parameters
        ----------
        embeddings : dict
            Keys from {"text", "voice", "face"}, values are (B, 768) tensors.
            Missing modalities should be omitted from the dict (not zeroed).
        return_weights : bool
            If True, return attention weights for explainability.

        Returns
        -------
        dict with keys:
            logits             - (B, 3) raw logits
            probabilities      - (B, 3) softmax probabilities
            predicted_class    - (B,) argmax class index
            confidence         - (B,) max probability
            modality_weights   - {modality: (B,) weight} — attention weights
        """
        # Determine batch size and device from any available embedding
        sample_emb = next(iter(embeddings.values()))
        B = sample_emb.shape[0]
        device = sample_emb.device

        projected = []
        mask_vals = []

        for mod in MODALITY_ORDER:
            if mod in embeddings and embeddings[mod] is not None:
                proj = self.projectors[mod](embeddings[mod].to(device))
                projected.append(proj)
                mask_vals.append(1.0)
            else:
                # Placeholder zero vector — will be masked out by attention
                projected.append(torch.zeros(B, self.hidden_dim, device=device))
                mask_vals.append(0.0)

        # Stack: (B, 3, hidden_dim)
        stacked = torch.stack(projected, dim=1)

        # Modality mask: (B, 3)
        mask = torch.tensor([mask_vals], device=device).expand(B, -1)

        # --- Compute attention weights ---
        scores = self.attention(stacked).squeeze(-1)  # (B, 3)

        # Mask unavailable modalities to -inf so they get 0 weight after softmax
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Handle edge case: if all modalities are missing (shouldn't happen)
        all_masked = (mask.sum(dim=-1, keepdim=True) == 0)
        scores = scores.masked_fill(all_masked.expand_as(scores), 0.0)

        weights = F.softmax(scores, dim=-1)  # (B, 3)

        # Replace any NaN from softmax (when all -inf) with uniform
        weights = weights.masked_fill(torch.isnan(weights), 1.0 / len(MODALITY_ORDER))

        # --- Weighted sum (attention-fused representation) ---
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (B, hidden_dim)

        # --- Classify ---
        logits = self.classifier(fused)  # (B, num_classes)
        probs = F.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)

        result = {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": predicted,
            "confidence": confidence,
        }

        if return_weights:
            result["modality_weights"] = {
                mod: weights[:, i] for i, mod in enumerate(MODALITY_ORDER)
            }

        return result


def load_fusion_model(config: dict) -> AttentionFusionNetwork:
    """Factory function: create and optionally load a checkpoint.

    Parameters
    ----------
    config : dict
        Full config loaded from config.yaml.

    Returns
    -------
    AttentionFusionNetwork ready for inference (eval mode).
    """
    from pathlib import Path

    fusion_cfg = config.get("fusion", {})
    model = AttentionFusionNetwork(
        embedding_dim=768,
        hidden_dim=fusion_cfg.get("hidden_dim", 256),
        num_classes=fusion_cfg.get("num_classes", 3),
        dropout=fusion_cfg.get("dropout", 0.3),
    )

    # Try to load trained weights
    checkpoint_path = fusion_cfg.get("checkpoint", "checkpoints/fusion_model.pt")
    if Path(checkpoint_path).exists():
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info("Fusion model checkpoint loaded: %s", checkpoint_path)
        except Exception as e:
            logger.warning("Could not load fusion checkpoint (%s). Using random weights.", e)
    else:
        logger.info(
            "No fusion checkpoint found at '%s'. "
            "Run 'python scripts/train_fusion.py' to train. Using random weights.",
            checkpoint_path
        )

    model.eval()
    return model


# Module-level logger
from src.utils.helpers import setup_logging
logger = setup_logging()
