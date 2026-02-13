"""
Confidence Calibration & Uncertainty Estimation
=================================================
Raw softmax probabilities from neural networks are typically
**overconfident** — a model might output 95% confidence even when
it's essentially guessing.

This module adds two layers of honest uncertainty:

1. **Temperature Scaling** — divides logits by a learned temperature T
   before softmax.  Higher T → softer (less confident) probabilities.
   This is the standard post-hoc calibration method (Guo et al., 2017).

2. **Monte Carlo Dropout** — runs multiple forward passes with dropout
   enabled, then measures prediction variance.  High variance →
   the model is uncertain.  Low variance → confident.

Why this matters for burnout detection:
  A "High Risk" prediction with 40% confidence should be treated very
  differently from one with 90% confidence.  Calibrated confidence
  lets users and practitioners know when to trust the system.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemperatureScaler:
    """Post-hoc temperature scaling for confidence calibration.

    Usage:
      scaler = TemperatureScaler(temperature=1.5)
      calibrated_probs = scaler.calibrate(logits)
    """

    def __init__(self, temperature: float = 1.5):
        """
        Parameters
        ----------
        temperature : float
            Scaling factor.  T > 1 makes predictions less confident,
            T < 1 makes them more confident.  T = 1 is no-op.
            Default 1.5 is a reasonable starting point for models
            trained on synthetic data (which tend to be overconfident).
        """
        self.temperature = temperature

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to raw logits.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model output (B, num_classes).

        Returns
        -------
        torch.Tensor : calibrated probabilities (B, num_classes).
        """
        return F.softmax(logits / self.temperature, dim=-1)


class MCDropoutEstimator:
    """Monte Carlo Dropout for uncertainty estimation.

    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty.  This is a practical approximation
    to Bayesian inference (Gal & Ghahramani, 2016).

    Usage:
      estimator = MCDropoutEstimator(model, n_samples=10)
      result = estimator.estimate(embeddings)
      # result["uncertainty"] is 0-1 (higher = less certain)
    """

    def __init__(self, model: nn.Module, n_samples: int = 10):
        self.model = model
        self.n_samples = n_samples

    def estimate(self, embeddings: dict[str, torch.Tensor]) -> dict:
        """Run MC Dropout and compute uncertainty.

        Parameters
        ----------
        embeddings : dict
            Same format as AttentionFusionNetwork.forward() input.

        Returns
        -------
        dict with:
            mean_probabilities - averaged predictions across MC samples
            uncertainty        - scalar 0-1 (prediction variance)
            predicted_class    - most likely class from mean probs
            confidence         - calibrated confidence score
        """
        # Enable dropout for stochastic forward passes
        self.model.train()

        all_probs = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(embeddings, return_weights=False)
                all_probs.append(output["probabilities"])

        # Restore eval mode
        self.model.eval()

        # Stack: (n_samples, B, num_classes)
        stacked = torch.stack(all_probs, dim=0)

        # Mean prediction
        mean_probs = stacked.mean(dim=0)  # (B, num_classes)

        # Uncertainty: variance of predictions across MC samples
        # Higher variance = more uncertainty
        variance = stacked.var(dim=0).mean(dim=-1)  # (B,)

        # Normalise uncertainty to [0, 1] using a sigmoid-like transform
        # Max theoretical variance for 3-class is ~0.22
        uncertainty = torch.clamp(variance / 0.15, 0.0, 1.0)

        confidence, predicted = mean_probs.max(dim=-1)

        return {
            "mean_probabilities": mean_probs,
            "uncertainty": uncertainty,
            "predicted_class": predicted,
            "confidence": confidence,
        }


def get_calibrated_prediction(
    model: nn.Module,
    embeddings: dict[str, torch.Tensor],
    temperature: float = 1.5,
    mc_samples: int = 10,
) -> dict:
    """Convenience function: get a fully calibrated prediction.

    Combines temperature scaling with MC Dropout uncertainty.

    Returns
    -------
    dict with:
        predicted_class, calibrated_confidence, uncertainty,
        calibrated_probabilities, is_reliable
    """
    # Standard forward pass
    model.eval()
    with torch.no_grad():
        output = model(embeddings)

    # Temperature-scaled probabilities
    scaler = TemperatureScaler(temperature=temperature)
    cal_probs = scaler.calibrate(output["logits"])
    cal_confidence, cal_predicted = cal_probs.max(dim=-1)

    # MC Dropout uncertainty
    mc = MCDropoutEstimator(model, n_samples=mc_samples)
    mc_result = mc.estimate(embeddings)

    uncertainty = mc_result["uncertainty"]

    # Determine reliability: confident AND low uncertainty
    is_reliable = (cal_confidence > 0.5) & (uncertainty < 0.5)

    return {
        "predicted_class": cal_predicted,
        "calibrated_confidence": cal_confidence,
        "calibrated_probabilities": cal_probs,
        "uncertainty": uncertainty,
        "is_reliable": is_reliable,
        "raw_confidence": output["confidence"],
        "modality_weights": output.get("modality_weights", {}),
    }
