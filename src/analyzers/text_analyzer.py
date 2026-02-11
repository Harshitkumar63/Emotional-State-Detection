"""
Text Analyzer — Real Emotion Classification + Embedding Extraction
===================================================================
Uses ``j-hartmann/emotion-english-distilroberta-base``, a DistilRoBERTa
model **fine-tuned on 6 emotion datasets** (GoEmotions, ISEAR, etc.).
Predicts seven emotion classes:

    anger . disgust . fear . joy . neutral . sadness . surprise

v2 additions:
  - ``extract_embedding()`` returns the 768-d [CLS] token for the
    attention-based fusion model.  The embedding is extracted from the
    SAME model (no extra weights loaded) by accessing the pipeline's
    internal model and requesting ``output_hidden_states=True``.

Why this approach?
  We keep the HuggingFace pipeline for ``analyze()`` (clean, reliable)
  AND access the raw model for embeddings.  The model weights are shared,
  so there's zero overhead.  This pattern — classification + embedding
  from a single model — is standard in production multimodal systems.
"""

from __future__ import annotations

import re
from typing import Optional

import torch
from transformers import pipeline as hf_pipeline

from src.utils.helpers import load_config, setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# Lexical signal keywords — used for explainability, NOT for prediction.
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
    """Classify emotions in free-form text AND extract dense embeddings.

    Public API:
      - analyze(text)           -> emotion scores + explainability signals
      - extract_embedding(text) -> 768-d [CLS] embedding for fusion model
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        model_name = config["text"]["model_name"]
        self.max_length = config["text"]["max_length"]
        self.embedding_dim = config["text"].get("embedding_dim", 768)

        logger.info("Loading text emotion model: %s", model_name)
        self.classifier = hf_pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            truncation=True,
            max_length=self.max_length,
        )

        # Expose the underlying model and tokenizer for embedding extraction.
        # This does NOT load separate weights — it references the same objects
        # that the pipeline already holds in memory.
        self._model = self.classifier.model
        self._tokenizer = self.classifier.tokenizer
        logger.info("Text analyzer ready (classification + embedding).")

    # ------------------------------------------------------------------
    # Public API: Emotion Classification
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
            emotions        - {emotion: score} for all 7 classes
            dominant_emotion - highest-scoring emotion label
            dominant_score   - its probability
            signals          - list of {source, observation, suggests}
        """
        text = text.strip()
        if not text:
            raise ValueError("Text input is empty.")

        raw = self.classifier(text)[0]
        emotions = {item["label"]: round(item["score"], 4) for item in raw}
        dominant = max(raw, key=lambda x: x["score"])
        signals = self._extract_signals(text)

        return {
            "emotions": emotions,
            "dominant_emotion": dominant["label"],
            "dominant_score": round(dominant["score"], 4),
            "signals": signals,
        }

    # ------------------------------------------------------------------
    # Public API: Embedding Extraction (NEW in v2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_embedding(self, text: str) -> torch.Tensor:
        """Extract the [CLS] token embedding from the emotion model.

        Returns a (1, 768) tensor suitable for the fusion network.

        Why the [CLS] token?
          In transformer models fine-tuned for classification, the [CLS]
          token is trained to aggregate sentence-level meaning.  For our
          emotion-tuned DistilRoBERTa, this embedding encodes the emotional
          content of the text — exactly what the fusion model needs.
        """
        text = text.strip()
        if not text:
            raise ValueError("Text input is empty.")

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # Move to the same device as the model (CPU in our case)
        device = self._model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # output_hidden_states=True gives us all layer activations
        outputs = self._model(**inputs, output_hidden_states=True)

        # Last hidden state, [CLS] token (position 0)
        cls_embedding = outputs.hidden_states[-1][:, 0, :]  # (1, 768)
        return cls_embedding.cpu()

    # ------------------------------------------------------------------
    # Explainability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_signals(text: str) -> list[dict]:
        """Scan text for emotional keyword clusters.

        NOT used for prediction — the transformer does that.
        Gives the explainer concrete phrases to cite.
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
