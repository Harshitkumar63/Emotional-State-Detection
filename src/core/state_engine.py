"""
State Engine — Unified Emotional Assessment
=============================================
The central intelligence that combines results from any subset of
{text, voice, face} analysers into a single, coherent EmotionalState.

Key design principles:

1. **Any single modality is sufficient.**  If only text is provided, the
   engine produces a full assessment from text alone — no fusion model,
   no missing-data hacks.

2. **Multiple modalities are averaged, not fused by a neural net.**
   The old approach used an attention-based fusion network with random
   weights.  That produces random output.  Here, we average the emotion
   probability distributions across available modalities — simple,
   transparent, and correct.

3. **Emotion → higher-level dimensions.**  Raw emotion scores are mapped
   to energy level, stress level, and work inclination using a
   config-driven lookup table grounded in psychological research.

4. **Full traceability.**  Every signal from every modality is preserved
   so the Explainer can tell the user *exactly* why the system reached
   its conclusion.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.core.emotional_state import EmotionalState
from src.utils.helpers import load_config, setup_logging

logger = setup_logging()

# Canonical emotion list that all modalities normalise to
_ALL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class StateEngine:
    """Combine per-modality results into a unified EmotionalState."""

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        self.config = config
        self.emotion_map = config["emotion_mapping"]
        self.energy_labels = config["energy_labels"]
        self.stress_labels = config["stress_labels"]
        self.work_labels = config["work_labels"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        text_result: Optional[dict] = None,
        voice_result: Optional[dict] = None,
        face_result: Optional[dict] = None,
    ) -> EmotionalState:
        """Produce a complete emotional assessment from available modality results.

        Parameters
        ----------
        text_result  : output of TextAnalyzer.analyze()  (or None)
        voice_result : output of VoiceAnalyzer.analyze()  (or None)
        face_result  : output of FaceAnalyzer.analyze()   (or None)

        At least one must be provided.
        """
        if text_result is None and voice_result is None and face_result is None:
            raise ValueError("At least one modality result must be provided.")

        modalities_used = []
        all_signals = []

        # ------ Collect emotion distributions from each modality ------
        emotion_dists = []

        if text_result is not None:
            modalities_used.append("text")
            all_signals.extend(text_result.get("signals", []))
            emotion_dists.append(
                self._normalise_emotions(text_result["emotions"])
            )

        if face_result is not None:
            modalities_used.append("face")
            all_signals.extend(face_result.get("signals", []))
            emotion_dists.append(
                self._normalise_emotions(face_result["emotions"])
            )

        if voice_result is not None:
            modalities_used.append("voice")
            all_signals.extend(voice_result.get("signals", []))
            # Voice produces indicators, not a full emotion distribution.
            # Convert its inferred emotion into a soft distribution.
            voice_emotion = voice_result.get("inferred_emotion", "neutral")
            emotion_dists.append(
                self._single_emotion_to_dist(voice_emotion, confidence=0.5)
            )

        # ------ Average emotion distributions across modalities -------
        avg_emotions = self._average_distributions(emotion_dists)
        primary_emotion = max(avg_emotions, key=avg_emotions.get)

        # ------ Map emotions → energy / stress / work -----------------
        energy_score = self._emotion_to_dimension(avg_emotions, "energy")
        stress_score = self._emotion_to_dimension(avg_emotions, "stress")
        work_score = self._emotion_to_dimension(avg_emotions, "work")

        # Voice indicators directly adjust energy & stress when available,
        # because acoustic features are a more direct measure of these
        # dimensions than emotion labels alone.
        if voice_result is not None:
            vi = voice_result["indicators"]
            # Blend: 50 % emotion-derived + 50 % acoustic-derived
            energy_score = 0.5 * energy_score + 0.5 * vi["energy_level"]
            stress_score = 0.5 * stress_score + 0.5 * vi["stress_level"]

        energy_score = round(max(0.0, min(1.0, energy_score)), 3)
        stress_score = round(max(0.0, min(1.0, stress_score)), 3)
        work_score = round(max(0.0, min(1.0, work_score)), 3)

        # ------ Derive labels -----------------------------------------
        energy_label = self._score_to_label(energy_score, self.energy_labels)
        stress_label = self._score_to_label(stress_score, self.stress_labels)
        work_label = self._score_to_label(work_score, self.work_labels)

        # ------ Generate summary & recommendations --------------------
        summary = self._build_summary(
            primary_emotion, energy_label, stress_label, work_label,
            modalities_used,
        )
        recommendations = self._build_recommendations(
            energy_score, stress_score, work_score, primary_emotion,
        )

        return EmotionalState(
            primary_emotion=primary_emotion,
            emotion_scores={k: round(v, 4) for k, v in avg_emotions.items()},
            energy_level=energy_label,
            energy_score=energy_score,
            stress_level=stress_label,
            stress_score=stress_score,
            work_inclination=work_label,
            work_score=work_score,
            mental_summary=summary,
            signals=all_signals,
            modalities_used=modalities_used,
            recommendations=recommendations,
            raw_text_result=text_result,
            raw_voice_result=voice_result,
            raw_face_result=face_result,
        )

    # ------------------------------------------------------------------
    # Emotion distribution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_emotions(emotions: dict[str, float]) -> dict[str, float]:
        """Ensure the dict covers all canonical emotions and sums to ~1."""
        dist = {e: emotions.get(e, 0.0) for e in _ALL_EMOTIONS}
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        return dist

    @staticmethod
    def _single_emotion_to_dist(emotion: str, confidence: float = 0.5) -> dict[str, float]:
        """Convert a single inferred emotion into a soft probability distribution.

        Assigns *confidence* to the target emotion and spreads the rest
        uniformly.  This is more honest than a one-hot vector because
        voice-based emotion inference has inherent uncertainty.
        """
        remainder = (1.0 - confidence) / max(len(_ALL_EMOTIONS) - 1, 1)
        dist = {e: remainder for e in _ALL_EMOTIONS}
        if emotion in dist:
            dist[emotion] = confidence
        else:
            dist["neutral"] = confidence
        return dist

    @staticmethod
    def _average_distributions(dists: list[dict]) -> dict[str, float]:
        """Element-wise average of multiple emotion distributions."""
        if not dists:
            return {e: 1.0 / len(_ALL_EMOTIONS) for e in _ALL_EMOTIONS}

        avg = {e: 0.0 for e in _ALL_EMOTIONS}
        for d in dists:
            for e in _ALL_EMOTIONS:
                avg[e] += d.get(e, 0.0)
        n = len(dists)
        return {e: v / n for e, v in avg.items()}

    # ------------------------------------------------------------------
    # Emotion → dimension mapping
    # ------------------------------------------------------------------

    def _emotion_to_dimension(self, emotions: dict[str, float], dim: str) -> float:
        """Weighted average of a dimension score across all emotions.

        Example: if joy has score 0.7 and joy's energy mapping is 0.85,
        that contributes 0.7 × 0.85 = 0.595 to the energy score.
        """
        score = 0.0
        for emotion, weight in emotions.items():
            mapping = self.emotion_map.get(emotion, {})
            score += weight * mapping.get(dim, 0.5)
        return score

    @staticmethod
    def _score_to_label(score: float, label_table: list[dict]) -> str:
        """Look up the human-readable label for a 0-1 score."""
        for entry in label_table:
            if score <= entry["max"]:
                return entry["label"]
        return label_table[-1]["label"]

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        emotion: str,
        energy: str,
        stress: str,
        work: str,
        modalities: list[str],
    ) -> str:
        """Create a one-paragraph human-readable mental state summary."""
        mod_str = ", ".join(modalities)

        emotion_phrases = {
            "joy": "in a positive and upbeat emotional state",
            "sadness": "experiencing low mood and possible emotional heaviness",
            "anger": "showing signs of frustration or irritation",
            "fear": "feeling anxious or worried",
            "surprise": "in a state of heightened alertness or surprise",
            "disgust": "experiencing displeasure or aversion",
            "neutral": "in a relatively calm and neutral state",
        }
        emotion_desc = emotion_phrases.get(emotion, f"experiencing {emotion}")

        return (
            f"Based on {mod_str} analysis, the person appears to be "
            f"{emotion_desc}. "
            f"Energy level is assessed as **{energy}**, "
            f"stress level is **{stress}**, "
            f"and work inclination suggests **{work}**."
        )

    @staticmethod
    def _build_recommendations(
        energy: float,
        stress: float,
        work: float,
        emotion: str,
    ) -> list[str]:
        """Generate actionable, non-clinical suggestions."""
        recs = []

        if energy < 0.3:
            recs.append(
                "Energy levels appear very low. Prioritise rest, hydration, "
                "and sleep before taking on demanding tasks."
            )
        if stress > 0.7:
            recs.append(
                "Stress indicators are elevated. Consider a short break, "
                "breathing exercises, or talking to someone you trust."
            )
        if work < 0.3:
            recs.append(
                "Signals suggest a low inclination toward work right now. "
                "Forcing productivity may worsen the situation — a change of "
                "pace or environment might help."
            )
        if emotion in ("sadness", "fear") and stress > 0.5:
            recs.append(
                "If these feelings persist, consider reaching out to a "
                "mental-health professional — there is no downside to checking in."
            )
        if energy > 0.6 and stress < 0.4 and emotion == "joy":
            recs.append(
                "Current state looks positive! This is a good window for "
                "creative or challenging work."
            )

        if not recs:
            recs.append(
                "Emotional state appears balanced. Keep monitoring and take "
                "breaks as needed."
            )

        return recs
