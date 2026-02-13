"""
Temporal Burnout Predictor — GRU-Based Time-Series Risk Forecasting
=====================================================================
Predicts burnout risk TRENDS from a sequence of past assessments.

Why a temporal model?
  A single assessment is a snapshot.  Burnout is a *process* that develops
  over days/weeks.  A GRU model looks at the trajectory:

  - Is stress consistently rising?
  - Is energy declining over time?
  - Are negative emotions becoming more frequent?

  These patterns are more predictive of actual burnout than any single
  assessment, no matter how accurate.

Architecture:
  Input:  (seq_len, 10) — sequence of 10-d feature vectors from past assessments
  GRU:   (10 → 64) — captures temporal dynamics
  FC:    (64 → 3)  — predicts risk trend: Improving / Stable / Worsening

Why GRU over LSTM?
  - Fewer parameters (faster training, less overfitting with small data)
  - Comparable performance on short sequences (< 50 steps)
  - Simpler implementation (2 gates vs 3)

Limitation:
  This model is trained on synthetic temporal data.  Real-world
  validation requires longitudinal burnout studies.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils.helpers import setup_logging

logger = setup_logging()

# Risk trend labels
TREND_LABELS = {0: "Improving", 1: "Stable", 2: "Worsening"}
TREND_COLORS = {"Improving": "#27ae60", "Stable": "#f39c12", "Worsening": "#e74c3c"}

# Feature dimension per timestep
FEATURE_DIM = 10


class TemporalBurnoutPredictor(nn.Module):
    """GRU-based model that predicts burnout risk trend from assessment history.

    Parameters
    ----------
    input_dim : int
        Feature dimension per timestep (10: energy, stress, work, risk, 6 emotions).
    hidden_dim : int
        GRU hidden state dimension.
    num_classes : int
        Number of trend classes (3: Improving / Stable / Worsening).
    num_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout between GRU layers (only applied if num_layers > 1).
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        num_classes: int = 3,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input normalisation — stabilises GRU training
        self.input_norm = nn.LayerNorm(input_dim)

        # GRU: processes the sequence of assessment feature vectors
        # Why GRU? Fewer parameters than LSTM, performs equally well
        # on short sequences, and is less prone to overfitting.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification head: GRU's final hidden state -> trend prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, input_dim).

        Returns
        -------
        dict with logits, probabilities, predicted_trend, confidence.
        """
        # Normalise input features
        x = self.input_norm(x)

        # GRU processes the full sequence
        gru_out, hidden = self.gru(x)
        # Use the final hidden state of the last layer
        final_hidden = hidden[-1]  # (batch, hidden_dim)

        # Classify
        logits = self.classifier(final_hidden)  # (batch, 3)
        probs = F.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)

        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_trend": predicted,  # 0=Improving, 1=Stable, 2=Worsening
            "confidence": confidence,
        }

    def predict_trend(self, feature_sequence: list[list[float]]) -> dict:
        """Predict burnout trend from a sequence of feature vectors.

        Parameters
        ----------
        feature_sequence : list[list[float]]
            List of 10-d feature vectors, oldest first.
            Minimum 3 entries required.

        Returns
        -------
        dict with trend_label, trend_confidence, trend_probabilities.
        """
        if len(feature_sequence) < 3:
            return {
                "trend_label": "Stable",
                "trend_confidence": 0.0,
                "trend_probabilities": {
                    "Improving": 0.33, "Stable": 0.34, "Worsening": 0.33
                },
                "note": "Insufficient history (need >= 3 assessments)",
            }

        # Convert to tensor: (1, seq_len, 10)
        x = torch.FloatTensor([feature_sequence])

        self.eval()
        with torch.no_grad():
            output = self(x)

        pred_idx = output["predicted_trend"].item()
        conf = output["confidence"].item()
        probs = output["probabilities"][0]

        return {
            "trend_label": TREND_LABELS.get(pred_idx, "Unknown"),
            "trend_confidence": round(conf, 4),
            "trend_probabilities": {
                TREND_LABELS[i]: round(probs[i].item(), 4)
                for i in range(len(TREND_LABELS))
            },
        }


# ======================================================================
# Model loading
# ======================================================================

def load_temporal_model(
    checkpoint_path: str = "checkpoints/temporal_model.pt",
) -> TemporalBurnoutPredictor:
    """Load a trained temporal model, or return untrained if no checkpoint."""
    model = TemporalBurnoutPredictor()

    if Path(checkpoint_path).exists():
        try:
            state_dict = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict)
            logger.info("Temporal model loaded: %s", checkpoint_path)
        except Exception as e:
            logger.warning("Could not load temporal checkpoint: %s", e)
    else:
        logger.info(
            "No temporal checkpoint at '%s'. "
            "Run 'python scripts/train_temporal.py' to train.",
            checkpoint_path,
        )

    model.eval()
    return model


# ======================================================================
# Synthetic data generation for training
# ======================================================================

def generate_temporal_data(
    n_sequences: int = 1000,
    seq_length: int = 10,
    seed: int = 42,
) -> tuple:
    """Generate synthetic temporal assessment sequences for training.

    Creates three types of trajectories:
      0 = Improving: stress decreasing, energy increasing over time
      1 = Stable:    metrics fluctuate around a constant baseline
      2 = Worsening: stress increasing, energy decreasing over time

    Each timestep has 10 features matching SessionStore.get_feature_sequence().
    """
    rng = np.random.RandomState(seed)
    n_per_class = n_sequences // 3

    all_sequences = []
    all_labels = []

    for label in range(3):
        for _ in range(n_per_class):
            seq = []
            # Base levels
            energy_base = rng.uniform(0.3, 0.7)
            stress_base = rng.uniform(0.3, 0.7)
            work_base = rng.uniform(0.3, 0.7)

            for t in range(seq_length):
                progress = t / max(seq_length - 1, 1)
                noise = rng.randn() * 0.05

                if label == 0:  # Improving
                    energy = energy_base + progress * 0.3 + noise
                    stress = stress_base - progress * 0.3 + noise
                    work = work_base + progress * 0.2 + noise
                elif label == 1:  # Stable
                    energy = energy_base + noise
                    stress = stress_base + noise
                    work = work_base + noise
                else:  # Worsening
                    energy = energy_base - progress * 0.3 + noise
                    stress = stress_base + progress * 0.3 + noise
                    work = work_base - progress * 0.2 + noise

                # Clamp to [0, 1]
                energy = max(0.0, min(1.0, energy))
                stress = max(0.0, min(1.0, stress))
                work = max(0.0, min(1.0, work))

                # Risk encoding based on stress
                risk_enc = 0.0 if stress < 0.4 else 0.5 if stress < 0.7 else 1.0

                # Emotion scores (correlated with stress/energy)
                joy = max(0, energy * 0.5 - stress * 0.3 + rng.randn() * 0.1)
                sadness = max(0, stress * 0.4 - energy * 0.2 + rng.randn() * 0.1)
                anger = max(0, stress * 0.3 + rng.randn() * 0.08)
                fear = max(0, stress * 0.3 - energy * 0.1 + rng.randn() * 0.08)
                neutral = max(0, 0.3 + rng.randn() * 0.1)
                disgust = max(0, stress * 0.15 + rng.randn() * 0.05)

                vec = [energy, stress, work, risk_enc,
                       anger, disgust, fear, joy, neutral, sadness]
                seq.append(vec)

            all_sequences.append(seq)
            all_labels.append(label)

    # Convert and shuffle
    sequences = torch.FloatTensor(all_sequences)
    labels = torch.LongTensor(all_labels)
    perm = torch.randperm(len(labels))

    return sequences[perm], labels[perm]
