"""
Face Analyzer — Facial Emotion Recognition + Embedding Extraction
==================================================================
Uses a Vision Transformer (ViT) fine-tuned on FER-2013 to classify
facial expressions into seven emotions:

    angry . disgust . fear . happy . neutral . sad . surprise

v2 additions:
  - ``extract_embedding()`` returns the 768-d [CLS] token for fusion.
  - Uses the pipeline's internal model (zero extra memory).
  - Better fallback handling with clear model status.

Design decisions:
  - Pipeline API for classification (reliable, handles preprocessing).
  - Raw model access for embedding extraction via ``output_hidden_states``.
  - Normalises labels to shared vocabulary so StateEngine can combine them.
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import torch
from PIL import Image

from src.utils.helpers import load_config, setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# Label normalisation: FER model labels → shared emotion vocabulary
# ---------------------------------------------------------------------------
_LABEL_MAP = {
    "happy":    "joy",
    "sad":      "sadness",
    "angry":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "surprise": "surprise",
    "neutral":  "neutral",
}


class FaceAnalyzer:
    """Classify facial emotions AND extract dense embeddings for fusion.

    Public API:
      - analyze(image)           -> emotion scores + signals
      - extract_embedding(image) -> 768-d [CLS] embedding for fusion model
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        model_name = config["face"]["model_name"]
        self.embedding_dim = config["face"].get("embedding_dim", 768)
        self._model_ready = False

        try:
            from transformers import pipeline as hf_pipeline

            logger.info("Loading face emotion model: %s", model_name)
            self.classifier = hf_pipeline(
                "image-classification",
                model=model_name,
                top_k=None,
            )
            self._model_ready = True

            # Expose internals for embedding extraction (shared weights)
            self._model = self.classifier.model
            self._image_processor = self.classifier.image_processor
            logger.info("Face analyzer ready (classification + embedding).")
        except Exception as exc:
            logger.warning(
                "Face model could not be loaded (%s). "
                "Face analysis will use fallback mode.", exc
            )

    # ------------------------------------------------------------------
    # Public API: Emotion Classification
    # ------------------------------------------------------------------

    def analyze(self, image_input: Union[str, Image.Image]) -> dict:
        """Classify facial emotions in the given image.

        Parameters
        ----------
        image_input : str or PIL.Image.Image
            File path or already-opened PIL image.

        Returns
        -------
        dict with keys:
            emotions         - {emotion: score} normalised to shared vocab
            dominant_emotion - top emotion label
            dominant_score   - its probability
            signals          - explainability evidence
            model_used       - which model produced the result
        """
        image = self._load_image(image_input)

        if self._model_ready:
            return self._analyze_with_model(image)
        else:
            return self._fallback_analysis(image)

    # ------------------------------------------------------------------
    # Public API: Embedding Extraction (NEW in v2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_embedding(self, image_input: Union[str, Image.Image]) -> torch.Tensor:
        """Extract the [CLS] token embedding from the ViT model.

        Returns a (1, 768) tensor suitable for the fusion network.

        Why the ViT [CLS] token?
          ViT (Vision Transformer) adds a learnable [CLS] token that
          attends to all image patches.  After fine-tuning on FER-2013,
          this token encodes the facial emotion content — exactly what
          the fusion model needs for cross-modal attention.
        """
        if not self._model_ready:
            # Return zero embedding when model unavailable (fusion handles this)
            logger.warning("Face model not ready; returning zero embedding.")
            return torch.zeros(1, self.embedding_dim)

        image = self._load_image(image_input)
        inputs = self._image_processor(image, return_tensors="pt")

        device = self._model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self._model(**inputs, output_hidden_states=True)

        # ViT [CLS] token is at position 0 of the last hidden state
        cls_embedding = outputs.hidden_states[-1][:, 0, :]  # (1, 768)
        return cls_embedding.cpu()

    # ------------------------------------------------------------------
    # Model-based analysis
    # ------------------------------------------------------------------

    def _analyze_with_model(self, image: Image.Image) -> dict:
        """Run the ViT classifier and normalise labels."""
        raw = self.classifier(image)

        emotions = {}
        for item in raw:
            norm_label = _LABEL_MAP.get(item["label"].lower(), item["label"].lower())
            emotions[norm_label] = round(item["score"], 4)

        dominant_label = max(emotions, key=emotions.get)
        dominant_score = emotions[dominant_label]
        signals = self._build_signals(dominant_label, dominant_score)

        return {
            "emotions": emotions,
            "dominant_emotion": dominant_label,
            "dominant_score": round(dominant_score, 4),
            "signals": signals,
            "model_used": "vit-face-expression",
        }

    # ------------------------------------------------------------------
    # Fallback when model isn't available
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_analysis(image: Image.Image) -> dict:
        """Basic heuristic when the ViT model is unavailable."""
        import numpy as np

        arr = np.array(image.convert("RGB")).astype(float)
        brightness = arr.mean() / 255.0

        if brightness < 0.3:
            note = "Image appears dark - this may affect analysis accuracy"
        else:
            note = "Image brightness is normal"

        return {
            "emotions": {"neutral": 0.6, "sadness": 0.2, "joy": 0.2},
            "dominant_emotion": "neutral",
            "dominant_score": 0.6,
            "signals": [{
                "source": "face",
                "observation": f"Fallback mode (model unavailable). {note}",
                "suggests": "Unable to perform detailed facial analysis",
            }],
            "model_used": "fallback",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image_input: Union[str, Image.Image]) -> Image.Image:
        """Accept file path or PIL Image; return RGB PIL Image."""
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(
                f"Expected file path or PIL Image, got {type(image_input)}"
            )

    @staticmethod
    def _build_signals(dominant: str, score: float) -> list[dict]:
        """Generate explainability evidence from face prediction."""
        confidence_word = (
            "strongly" if score > 0.6 else "moderately" if score > 0.35 else "slightly"
        )

        emotion_descriptions = {
            "joy":      "happiness or contentment",
            "sadness":  "sadness or low mood",
            "anger":    "frustration or irritation",
            "fear":     "anxiety or worry",
            "surprise": "surprise or alertness",
            "disgust":  "displeasure or aversion",
            "neutral":  "a calm, neutral expression",
        }

        desc = emotion_descriptions.get(dominant, dominant)

        return [{
            "source": "face",
            "observation": (
                f"Facial expression {confidence_word} suggests {desc} "
                f"({score:.0%} confidence)"
            ),
            "suggests": dominant,
        }]
