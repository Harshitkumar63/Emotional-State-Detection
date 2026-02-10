"""
Text Analyzer — Real Emotion Classification from Written Text
==============================================================
Uses ``j-hartmann/emotion-english-distilroberta-base``, a DistilRoBERTa
model **fine-tuned on 6 emotion datasets** (including GoEmotions,
ISEAR, and others).  It predicts seven emotion classes:

    anger · disgust · fear · joy · neutral · sadness · surprise

Why this model (instead of raw DistilBERT embeddings)?
  The previous implementation extracted [CLS] embeddings and fed them
  into a randomly-initialised classifier — which produces random output.
  This model was actually *trained* to classify emotions, so its
  predictions are meaningful out of the box.  No custom training needed.

Design decisions:
  • We use the HuggingFace ``pipeline`` API for reliability; it handles
    tokenisation, batching, and softmax internally.
  • ``top_k=None`` returns scores for ALL emotions (not just the winner),
    which the StateEngine needs for nuanced assessment.
  • The analyser also extracts simple lexical signals (emotional keyword
    matches) for the explainability module, so users understand *why*
    the model reached its conclusion.
"""

from __future__ import annotations

import re
from typing import Optional

from transformers import pipeline as hf_pipeline

from src.utils.helpers import load_config, setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# Lexical signal keywords — used for explainability, NOT for prediction.
# The model handles prediction; these give users interpretable evidence.
# ---------------------------------------------------------------------------
_SIGNAL_KEYWORDS = {
    "exhaustion": [
        "exhausted", "drained", "tired", "fatigued", "burned out",
        "burnt out", "no energy", "wiped out", "can't sleep", "insomnia",
    ],
    "stress": [
        "stressed", "overwhelmed", "pressure", "deadline", "anxious",
        "anxiety", "panic", "nervous", "tense", "can't relax",
    ],
    "sadness": [
        "sad", "hopeless", "depressed", "lonely", "empty", "crying",
        "worthless", "pointless", "lost", "numb", "grief",
    ],
    "anger": [
        "angry", "frustrated", "furious", "irritated", "annoyed",
        "resentful", "unfair", "hate", "rage",
    ],
    "joy": [
        "happy", "grateful", "excited", "motivated", "energised",
        "energized", "proud", "fulfilled", "love", "great", "amazing",
    ],
    "disengagement": [
        "don't care", "detached", "disconnected", "apathetic", "bored",
        "unmotivated", "going through the motions", "autopilot",
    ],
}


class TextAnalyzer:
    """Classify emotions in free-form text using a pretrained transformer.

    Returns a dict with emotion probabilities, dominant emotion, and
    lexical signals extracted from the input for explainability.
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        model_name = config["text"]["model_name"]
        self.max_length = config["text"]["max_length"]

        logger.info("Loading text emotion model: %s", model_name)
        self.classifier = hf_pipeline(
            "text-classification",
            model=model_name,
            top_k=None,          # return all 7 emotion scores
            truncation=True,
            max_length=self.max_length,
        )
        logger.info("Text analyzer ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> dict:
        """Analyse text and return emotion scores + explainability signals.

        Parameters
        ----------
        text : str
            Free-form text — journal entry, chat message, feedback, etc.

        Returns
        -------
        dict with keys:
            emotions        – {emotion: score} for all 7 classes
            dominant_emotion – highest-scoring emotion label
            dominant_score   – its probability
            signals          – list of {source, observation, suggests}
        """
        text = text.strip()
        if not text:
            raise ValueError("Text input is empty.")

        # --- Model prediction (the REAL emotion classification) --------
        raw = self.classifier(text)[0]  # list of {label, score}
        emotions = {item["label"]: round(item["score"], 4) for item in raw}
        dominant = max(raw, key=lambda x: x["score"])

        # --- Lexical signals (for explainability only) -----------------
        signals = self._extract_signals(text)

        return {
            "emotions": emotions,
            "dominant_emotion": dominant["label"],
            "dominant_score": round(dominant["score"], 4),
            "signals": signals,
        }

    # ------------------------------------------------------------------
    # Explainability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_signals(text: str) -> list[dict]:
        """Scan text for emotional keyword clusters.

        This is NOT used for prediction — the transformer does that.
        It gives the explainer concrete phrases to cite when telling
        the user *why* the system reached its conclusion.
        """
        text_lower = text.lower()
        signals = []

        for category, keywords in _SIGNAL_KEYWORDS.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                signals.append({
                    "source": "text",
                    "observation": f"Contains phrases: {', '.join(found[:4])}",
                    "suggests": category.replace("_", " "),
                })

        return signals
