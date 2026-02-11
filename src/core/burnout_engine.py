"""
Burnout Engine — Unified Burnout Risk Assessment (v2 Core)
===========================================================
The central orchestrator that:

1. Runs per-modality emotion analysis (text, voice, face)
2. Extracts dense embeddings from each modality's model
3. Feeds embeddings into the attention-based fusion network
4. Combines emotion analysis + fusion-based burnout risk into one result

This replaces the simple averaging approach from v1 with a learned,
attention-weighted fusion — while keeping the v1 StateEngine for
emotion-level analysis (backward compatible).

Architecture flow:
  Input(text, audio, face_image)
    -> Per-modality analyzers (emotion scores + signals)
    -> Per-modality embedding extraction
    -> AttentionFusionNetwork -> Burnout Risk (Low/Moderate/High)
    -> StateEngine -> Emotional dimensions (energy/stress/work)
    -> Combined EmotionalState with burnout risk

Why two engines?
  The StateEngine handles emotion-to-dimension mapping (energy, stress,
  work inclination) which is a deterministic, interpretable mapping.
  The BurnoutEngine adds the learned fusion model for cross-modal risk
  prediction.  Keeping them separate follows Single Responsibility and
  lets users choose the level of analysis they need.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

import torch

from src.analyzers.text_analyzer import TextAnalyzer
from src.analyzers.voice_analyzer import VoiceAnalyzer
from src.analyzers.voice_analyzer_deep import VoiceAnalyzerDeep
from src.analyzers.face_analyzer import FaceAnalyzer
from src.core.state_engine import StateEngine
from src.core.emotional_state import EmotionalState
from src.fusion.attention_fusion import AttentionFusionNetwork, load_fusion_model
from src.utils.helpers import load_config, setup_logging

logger = setup_logging()


class BurnoutEngine:
    """End-to-end burnout risk assessment from multimodal inputs.

    Combines emotion analysis (v1) with attention-based fusion (v2).
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.config = config
        self.burnout_labels = config.get("burnout_labels", {
            0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"
        })

        # --- Per-modality analyzers ---
        logger.info("Initialising BurnoutEngine...")
        self.text_analyzer = TextAnalyzer(config)
        self.voice_analyzer_acoustic = VoiceAnalyzer(config)   # baseline (interpretable)
        self.voice_analyzer_deep = VoiceAnalyzerDeep(config)   # deep model (embeddings)
        self.face_analyzer = FaceAnalyzer(config)

        # --- Fusion model ---
        self.fusion_model = load_fusion_model(config)

        # --- State engine for emotion-level analysis ---
        self.state_engine = StateEngine(config)

        logger.info("BurnoutEngine ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        image=None,
    ) -> EmotionalState:
        """Run full burnout risk assessment.

        Parameters
        ----------
        text : str, optional
            Free-form text input.
        audio_path : str, optional
            Path to a .wav audio file.
        image : str or PIL.Image, optional
            Face image (file path or PIL Image).

        Returns
        -------
        EmotionalState with burnout risk, emotion analysis, and explanations.

        At least one input must be provided.
        """
        if text is None and audio_path is None and image is None:
            raise ValueError("At least one input (text, audio, or image) is required.")

        # ================================================================
        # Step 1: Per-modality emotion analysis
        # ================================================================
        text_result = None
        voice_result = None
        face_result = None
        voice_deep_result = None

        if text is not None:
            logger.info("Analyzing text...")
            text_result = self.text_analyzer.analyze(text)

        if audio_path is not None:
            logger.info("Analyzing voice (acoustic baseline)...")
            voice_result = self.voice_analyzer_acoustic.analyze(audio_path)

            if self.voice_analyzer_deep.is_ready:
                logger.info("Analyzing voice (deep model)...")
                voice_deep_result = self.voice_analyzer_deep.analyze(audio_path)

        if image is not None:
            logger.info("Analyzing face...")
            face_result = self.face_analyzer.analyze(image)

        # ================================================================
        # Step 2: Get emotional state from StateEngine (v1 logic)
        # For voice, prefer the deep model's emotion output when available
        # ================================================================
        voice_for_state = voice_result  # acoustic baseline by default
        voice_model_used = "acoustic"

        if voice_deep_result is not None and voice_deep_result.get("model_used") != "none":
            # Enhance the voice result with deep model's emotion prediction
            # while keeping acoustic indicators for energy/stress mapping
            voice_for_state = self._merge_voice_results(voice_result, voice_deep_result)
            voice_model_used = voice_deep_result.get("model_used", "deep")

        state = self.state_engine.assess(
            text_result=text_result,
            voice_result=voice_for_state,
            face_result=face_result,
        )

        # ================================================================
        # Step 3: Extract embeddings and run fusion model
        # ================================================================
        embeddings = {}
        modalities_with_embeddings = []

        if text is not None:
            embeddings["text"] = self.text_analyzer.extract_embedding(text)
            modalities_with_embeddings.append("text")

        if audio_path is not None and self.voice_analyzer_deep.is_ready:
            embeddings["voice"] = self.voice_analyzer_deep.extract_embedding(audio_path)
            modalities_with_embeddings.append("voice")

        if image is not None:
            embeddings["face"] = self.face_analyzer.extract_embedding(image)
            modalities_with_embeddings.append("face")

        # ================================================================
        # Step 4: Fusion-based burnout risk prediction
        # ================================================================
        if embeddings:
            with torch.no_grad():
                fusion_output = self.fusion_model(embeddings)

            predicted_class = fusion_output["predicted_class"].item()
            confidence = fusion_output["confidence"].item()
            probs = fusion_output["probabilities"][0]

            state.burnout_risk = self.burnout_labels.get(predicted_class, "Unknown")
            state.burnout_confidence = round(confidence, 4)
            state.burnout_probabilities = {
                self.burnout_labels.get(i, f"Class {i}"): round(probs[i].item(), 4)
                for i in range(len(probs))
            }

            # Extract attention weights for explainability
            if "modality_weights" in fusion_output:
                weights = fusion_output["modality_weights"]
                state.modality_contributions = {
                    mod: round(weights[mod][0].item(), 4)
                    for mod in weights
                    if mod in modalities_with_embeddings
                }

                # Renormalise to only include used modalities
                total = sum(state.modality_contributions.values())
                if total > 0:
                    state.modality_contributions = {
                        k: round(v / total, 4)
                        for k, v in state.modality_contributions.items()
                    }
        else:
            logger.warning("No embeddings available for fusion model.")
            state.burnout_risk = "N/A"
            state.burnout_confidence = 0.0

        state.voice_model_used = voice_model_used
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_voice_results(acoustic: dict, deep: dict) -> dict:
        """Merge acoustic baseline indicators with deep model emotion scores.

        The acoustic result provides interpretable features (energy, stress,
        pitch, tempo) while the deep model provides more accurate emotion
        classification.  Merging gives us the best of both worlds.
        """
        merged = dict(acoustic)  # keep indicators, features, signals

        # Override the emotion with deep model's prediction
        if deep.get("emotions"):
            merged["deep_emotions"] = deep["emotions"]
            merged["deep_dominant"] = deep.get("dominant_emotion", "neutral")
            # Update the inferred emotion to use the deep model's prediction
            merged["inferred_emotion"] = deep.get("dominant_emotion", "neutral")

        # Add deep model's signals
        deep_signals = deep.get("signals", [])
        merged["signals"] = merged.get("signals", []) + deep_signals

        return merged
