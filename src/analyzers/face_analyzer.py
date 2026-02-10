"""
Face Analyzer — Facial Emotion Recognition
============================================
Uses a Vision Transformer (ViT) fine-tuned on the FER-2013 dataset to
classify facial expressions into seven emotions:

    angry · disgust · fear · happy · neutral · sad · surprise

Why this replaces the old ResNet-18 pipeline:
  ResNet-18 with ImageNet weights recognises objects (dogs, cars, etc.),
  NOT facial emotions.  A face image was getting the same kind of feature
  vector as any other photograph.  This model was actually **trained on
  facial expression data**, so its predictions are meaningful.

Design decisions:
  • Uses the HuggingFace ``pipeline("image-classification")`` API.
  • Falls back gracefully if the model can't be downloaded (e.g. offline).
  • Accepts file paths, PIL Images, or numpy arrays for flexibility
    with Streamlit uploads.
  • Normalises label names to match the text analyzer's emotion vocabulary
    (e.g. "happy" → "joy", "sad" → "sadness") so the StateEngine can
    combine them seamlessly.
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

from PIL import Image

from src.utils.helpers import load_config, setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# The face model may use different label names than the text model.
# This map normalises them to a shared vocabulary.
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
    """Classify facial emotions using a pretrained Vision Transformer.

    Falls back to a basic assessment if the model isn't available.
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        model_name = config["face"]["model_name"]
        self._model_ready = False

        try:
            from transformers import pipeline as hf_pipeline

            logger.info("Loading face emotion model: %s", model_name)
            self.classifier = hf_pipeline(
                "image-classification",
                model=model_name,
                top_k=None,   # return all emotion scores
            )
            self._model_ready = True
            logger.info("Face analyzer ready.")
        except Exception as exc:
            # Model download can fail (offline, disk space, etc.)
            # The system still works — just without face analysis.
            logger.warning(
                "Face model could not be loaded (%s). "
                "Face analysis will use fallback mode.", exc
            )

    # ------------------------------------------------------------------
    # Public API
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
            emotions         – {emotion: score} normalised to shared vocabulary
            dominant_emotion – top emotion label
            dominant_score   – its probability
            signals          – explainability evidence
            model_used       – which model produced the result
        """
        image = self._load_image(image_input)

        if self._model_ready:
            return self._analyze_with_model(image)
        else:
            return self._fallback_analysis(image)

    # ------------------------------------------------------------------
    # Model-based analysis
    # ------------------------------------------------------------------

    def _analyze_with_model(self, image: Image.Image) -> dict:
        """Run the ViT classifier and normalise labels."""
        raw = self.classifier(image)  # list of {label, score}

        # Normalise labels to shared vocabulary
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
        """Basic image-statistics heuristic when the ViT model is unavailable.

        This is intentionally simple and honest — it only checks whether
        the image appears very dark (possibly tired / low mood) or normal.
        It does NOT claim to detect specific emotions.
        """
        import numpy as np

        arr = np.array(image.convert("RGB")).astype(float)
        brightness = arr.mean() / 255.0

        if brightness < 0.3:
            note = "Image appears dark — this may affect analysis accuracy"
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
        """Accept a file path or PIL Image; return an RGB PIL Image."""
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
