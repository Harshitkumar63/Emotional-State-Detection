"""Core logic — emotional state model, assessment engine, explainability."""

from src.core.emotional_state import EmotionalState
from src.core.state_engine import StateEngine
from src.core.explainer import Explainer

__all__ = ["EmotionalState", "StateEngine", "Explainer"]
