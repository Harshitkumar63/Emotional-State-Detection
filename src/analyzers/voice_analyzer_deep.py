"""
Voice Analyzer (Deep) — Speech Emotion Recognition via Wav2Vec2
================================================================
Uses ``superb/wav2vec2-base-superb-er``, a Wav2Vec2-base model
fine-tuned on the IEMOCAP dataset for 4-class emotion recognition:

    angry (ang) . happy (hap) . neutral (neu) . sad (sad)

Why deep speech models outperform handcrafted features:
------------------------------------------------------
1. **Learned representations**: Wav2Vec2 pre-trains on 960h of unlabelled
   speech (LibriSpeech) using self-supervised contrastive learning.  It
   learns hierarchical features — from phonemes to prosody to rhythm —
   that handcrafted features (MFCCs, pitch, energy) cannot capture.

2. **Transfer learning**: Fine-tuning on IEMOCAP (12h of emotional speech)
   adapts these rich representations to emotion classification.  Published
   benchmarks show Wav2Vec2 outperforms eGeMAPS features by 10-20%
   weighted accuracy on IEMOCAP (Yang et al., SUPERB 2021).

3. **Robustness**: Deep features are more robust to recording conditions
   (noise, microphone variance) than hand-tuned thresholds.

Design decisions:
  - Uses ``Wav2Vec2ForSequenceClassification`` for direct inference.
  - Falls back to base ``Wav2Vec2Model`` (embeddings only) if the
    fine-tuned model is unavailable.
  - The acoustic baseline (VoiceAnalyzer) is kept for comparison and
    explainability — acoustic features are more interpretable.
  - Audio is resampled to 16 kHz (required by Wav2Vec2).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import librosa

from src.utils.helpers import load_config, setup_logging

logger = setup_logging()


class VoiceAnalyzerDeep:
    """Speech emotion recognition using a fine-tuned Wav2Vec2 model.

    Public API:
      - analyze(audio_path)           -> emotion scores + signals
      - extract_embedding(audio_path) -> 768-d embedding for fusion model
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        deep_cfg = config.get("voice_deep", {})
        self.model_name = deep_cfg.get("model_name", "superb/wav2vec2-base-superb-er")
        self.sr = deep_cfg.get("sample_rate", 16000)
        self.max_duration = deep_cfg.get("max_duration_sec", 30)
        self.embedding_dim = deep_cfg.get("embedding_dim", 768)

        # Label mapping from model-specific labels to shared vocabulary
        self.label_map = deep_cfg.get("label_map", {
            "neu": "neutral", "hap": "joy", "ang": "anger", "sad": "sadness"
        })

        self._has_classifier = False
        self._model = None
        self._feature_extractor = None

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading with graceful fallback
    # ------------------------------------------------------------------

    def _load_model(self):
        """Try loading the emotion-fine-tuned model, fall back to base."""
        try:
            from transformers import (
                Wav2Vec2ForSequenceClassification,
                Wav2Vec2FeatureExtractor,
            )

            logger.info("Loading deep voice model: %s", self.model_name)
            self._model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name
            )
            self._model.eval()
            self._has_classifier = True

            # Build label mapping from model config
            if hasattr(self._model.config, "id2label"):
                self._id2label = self._model.config.id2label
            else:
                self._id2label = {0: "neu", 1: "hap", 2: "ang", 3: "sad"}

            logger.info(
                "Deep voice model ready (classifier + embedding). "
                "Labels: %s", list(self._id2label.values())
            )

        except Exception as e:
            logger.warning(
                "Could not load fine-tuned model (%s). "
                "Trying base Wav2Vec2 for embeddings only...", e
            )
            try:
                from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

                self._model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "facebook/wav2vec2-base"
                )
                self._model.eval()
                self._has_classifier = False
                logger.info("Base Wav2Vec2 loaded (embeddings only, no classifier).")

            except Exception as e2:
                logger.error("Could not load any Wav2Vec2 model: %s", e2)
                self._model = None

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, path: str) -> np.ndarray:
        """Load audio, resample to 16 kHz, normalise, and truncate."""
        y, sr = librosa.load(path, sr=self.sr, mono=True)

        max_samples = self.sr * self.max_duration
        if len(y) > max_samples:
            logger.warning("Audio truncated to %d seconds", self.max_duration)
            y = y[:max_samples]

        # Peak normalisation
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak

        return y

    # ------------------------------------------------------------------
    # Public API: Emotion Classification
    # ------------------------------------------------------------------

    @torch.no_grad()
    def analyze(self, audio_path: str) -> dict:
        """Classify speech emotions using the deep model.

        Parameters
        ----------
        audio_path : str
            Path to a .wav file.

        Returns
        -------
        dict with keys:
            emotions         - {emotion: score} in shared vocabulary
            dominant_emotion - top emotion label
            dominant_score   - its probability
            signals          - explainability evidence
            model_used       - "wav2vec2-emotion" or "wav2vec2-base"
        """
        if self._model is None:
            return self._fallback_result()

        y = self._load_audio(audio_path)
        inputs = self._feature_extractor(
            y, sampling_rate=self.sr, return_tensors="pt", padding=True
        )

        if self._has_classifier:
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

            # Map model labels to shared vocabulary
            emotions = {}
            for idx, prob in enumerate(probs):
                raw_label = self._id2label.get(idx, f"class_{idx}")
                norm_label = self.label_map.get(raw_label, raw_label)
                emotions[norm_label] = round(prob.item(), 4)

            # Ensure all 7 canonical emotions are present (pad missing with 0)
            for emo in ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]:
                emotions.setdefault(emo, 0.0)

            dominant = max(emotions, key=emotions.get)
            dominant_score = emotions[dominant]

            signals = self._build_signals(dominant, dominant_score, "wav2vec2-emotion")

            return {
                "emotions": emotions,
                "dominant_emotion": dominant,
                "dominant_score": round(dominant_score, 4),
                "signals": signals,
                "model_used": "wav2vec2-emotion",
            }
        else:
            # Base model — no classifier head, can only provide embeddings.
            # Return neutral-biased distribution to be honest about uncertainty.
            return self._neutral_result("wav2vec2-base")

    # ------------------------------------------------------------------
    # Public API: Embedding Extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_embedding(self, audio_path: str) -> torch.Tensor:
        """Extract 768-d embedding from the Wav2Vec2 model.

        Returns a (1, 768) tensor: the mean-pooled last hidden state.

        Why mean-pooling?
          Unlike text (fixed [CLS] token) or images (fixed [CLS] patch),
          speech has variable-length sequences.  Mean-pooling over all
          time frames creates a fixed-size representation that captures
          the overall prosodic and emotional characteristics.
        """
        if self._model is None:
            logger.warning("Voice model not ready; returning zero embedding.")
            return torch.zeros(1, self.embedding_dim)

        y = self._load_audio(audio_path)
        inputs = self._feature_extractor(
            y, sampling_rate=self.sr, return_tensors="pt", padding=True
        )

        if self._has_classifier:
            outputs = self._model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # (1, T, 768)
        else:
            outputs = self._model(**inputs)
            hidden_states = outputs.last_hidden_state  # (1, T, 768)

        # Mean-pool over time dimension
        embedding = hidden_states.mean(dim=1)  # (1, 768)
        return embedding.cpu()

    # ------------------------------------------------------------------
    # Status check
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def has_classifier(self) -> bool:
        return self._has_classifier

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_signals(dominant: str, score: float, model_name: str) -> list[dict]:
        """Generate explainability evidence."""
        confidence_word = (
            "strongly" if score > 0.6 else "moderately" if score > 0.35 else "slightly"
        )
        return [{
            "source": "voice",
            "observation": (
                f"Deep speech model ({model_name}) {confidence_word} detects "
                f"{dominant} ({score:.0%} confidence)"
            ),
            "suggests": dominant,
        }]

    @staticmethod
    def _fallback_result() -> dict:
        """Return when no model is available at all."""
        return {
            "emotions": {"neutral": 1.0},
            "dominant_emotion": "neutral",
            "dominant_score": 1.0,
            "signals": [{
                "source": "voice",
                "observation": "Deep voice model unavailable",
                "suggests": "Cannot perform deep speech emotion analysis",
            }],
            "model_used": "none",
        }

    @staticmethod
    def _neutral_result(model_name: str) -> dict:
        """Return when model has no classifier (base model only)."""
        return {
            "emotions": {
                "neutral": 0.4, "joy": 0.1, "anger": 0.1,
                "sadness": 0.1, "fear": 0.1, "surprise": 0.1, "disgust": 0.1,
            },
            "dominant_emotion": "neutral",
            "dominant_score": 0.4,
            "signals": [{
                "source": "voice",
                "observation": (
                    f"Base model ({model_name}) provides embeddings only; "
                    "no emotion classifier available"
                ),
                "suggests": "neutral (default)",
            }],
            "model_used": model_name,
        }
