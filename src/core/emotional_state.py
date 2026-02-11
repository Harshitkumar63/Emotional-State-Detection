"""
Emotional State Data Model (v2)
================================
Holds the **complete** result of an emotional state + burnout risk assessment.

v2 additions:
  - burnout_risk / burnout_confidence / burnout_label (from fusion model)
  - modality_contributions (attention weights from fusion)
  - per-modality embeddings (optional, for inspection)

Keeping the data model separate from logic (Single Responsibility)
makes it easy to serialise to JSON, log, or pass between components.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class EmotionalState:
    """Structured representation of emotional state + burnout risk."""

    # --- Primary emotion detected across all provided modalities ---
    primary_emotion: str                        # e.g. "sadness", "joy"
    emotion_scores: dict[str, float]            # {emotion: 0-1 score}

    # --- Higher-level dimensions (derived from emotions) ---
    energy_level: str                           # "Exhausted" ... "Energetic"
    energy_score: float                         # 0-1 continuous

    stress_level: str                           # "Calm" ... "High Stress"
    stress_score: float                         # 0-1 continuous

    work_inclination: str                       # "Needs Rest" ... "Motivated"
    work_score: float                           # 0-1 continuous

    # --- Burnout Risk (NEW in v2: from attention fusion model) ---
    burnout_risk: str = "N/A"                   # "Low Risk" / "Moderate Risk" / "High Risk"
    burnout_confidence: float = 0.0             # 0-1 confidence in the prediction
    burnout_probabilities: dict[str, float] = field(default_factory=dict)
    # {"Low Risk": 0.6, "Moderate Risk": 0.3, "High Risk": 0.1}

    # --- Modality contributions (from attention weights) ---
    modality_contributions: dict[str, float] = field(default_factory=dict)
    # {"text": 0.45, "voice": 0.35, "face": 0.20} — sums to 1.0

    # --- Contextual details ---
    mental_summary: str = ""                    # human-readable paragraph
    signals: list[dict] = field(default_factory=list)
    modalities_used: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # --- Optional raw per-modality results for deep inspection ---
    raw_text_result: Optional[dict] = None
    raw_voice_result: Optional[dict] = None
    raw_face_result: Optional[dict] = None

    # --- Voice model info (which model was used) ---
    voice_model_used: str = "acoustic"          # "acoustic" or "deep"

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON export, API responses)."""
        d = asdict(self)
        # Remove raw results from export — they're for internal use
        d.pop("raw_text_result", None)
        d.pop("raw_voice_result", None)
        d.pop("raw_face_result", None)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
