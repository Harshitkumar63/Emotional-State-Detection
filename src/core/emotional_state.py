"""
Emotional State Data Model
===========================
A plain dataclass that holds the **complete** result of an emotional
state assessment.  Every field is populated by the StateEngine; the
Streamlit app and CLI just read from this object.

Keeping the data model separate from logic (Single Responsibility)
makes it easy to serialise to JSON, log, or pass between components.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class EmotionalState:
    """Structured representation of a person's current emotional state."""

    # --- Primary emotion detected across all provided modalities ---
    primary_emotion: str                        # e.g. "sadness", "joy"
    emotion_scores: dict[str, float]            # {emotion: 0-1 score}

    # --- Higher-level dimensions (derived from emotions) ---
    energy_level: str                           # "Exhausted" … "Energetic"
    energy_score: float                         # 0-1 continuous

    stress_level: str                           # "Calm" … "High Stress"
    stress_score: float                         # 0-1 continuous

    work_inclination: str                       # "Needs Rest" … "Motivated"
    work_score: float                           # 0-1 continuous

    # --- Contextual details ---
    mental_summary: str                         # human-readable paragraph
    signals: list[dict] = field(default_factory=list)
    # Each signal: {"source": "text", "observation": "...", "suggests": "..."}

    modalities_used: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # --- Optional raw per-modality results for deep inspection ---
    raw_text_result: Optional[dict] = None
    raw_voice_result: Optional[dict] = None
    raw_face_result: Optional[dict] = None

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
