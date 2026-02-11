"""
Core Module — State Assessment & Burnout Risk Engine
=====================================================
  - EmotionalState:  data model holding assessment results
  - StateEngine:     emotion-based dimensional mapping (v1 logic)
  - BurnoutEngine:   full pipeline with attention fusion (v2 upgrade)
  - Explainer:       human-readable explanations for predictions
"""

from src.core.emotional_state import EmotionalState
from src.core.state_engine import StateEngine
from src.core.burnout_engine import BurnoutEngine
from src.core.explainer import Explainer

__all__ = ["EmotionalState", "StateEngine", "BurnoutEngine", "Explainer"]
