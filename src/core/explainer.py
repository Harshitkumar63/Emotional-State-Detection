"""
Explainer — Human-Readable Reasoning for Predictions
======================================================
Transforms the structured EmotionalState into plain-English explanations
that answer **why** the system reached each conclusion.

The Explainer does NOT make predictions — it reads the signals that the
analysers already produced and narrates them in a way that non-technical
users can understand.

Every explanation follows a consistent structure:
  1. What was observed (the evidence)
  2. What it suggests (the interpretation)
  3. How confident we are (honest uncertainty)
"""

from __future__ import annotations

from src.core.emotional_state import EmotionalState


class Explainer:
    """Generate clear, honest explanations for an EmotionalState assessment."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, state: EmotionalState) -> dict:
        """Build a full explanation package.

        Returns
        -------
        dict with keys:
            overall_narrative – 2-3 sentence summary of the assessment
            signal_narratives – list of per-signal explanations
            dimension_explanations – energy / stress / work reasoning
            confidence_note – how much to trust this result
            limitations     – honest list of caveats
            disclaimer      – ethical/legal disclaimer
        """
        return {
            "overall_narrative": self._overall_narrative(state),
            "signal_narratives": self._signal_narratives(state),
            "dimension_explanations": self._dimension_explanations(state),
            "confidence_note": self._confidence_note(state),
            "limitations": self._limitations(state),
            "disclaimer": self._disclaimer(),
        }

    # ------------------------------------------------------------------
    # Narrative builders
    # ------------------------------------------------------------------

    @staticmethod
    def _overall_narrative(state: EmotionalState) -> str:
        """Top-level summary paragraph."""
        n_modalities = len(state.modalities_used)
        mod_str = ", ".join(state.modalities_used)

        intro = (
            f"This assessment is based on **{n_modalities}** input(s): "
            f"**{mod_str}**."
        )

        primary = state.primary_emotion
        score = state.emotion_scores.get(primary, 0)

        emotion_text = (
            f"The dominant emotional signal is **{primary}** "
            f"(score: {score:.0%})."
        )

        return f"{intro}  {emotion_text}  {state.mental_summary}"

    @staticmethod
    def _signal_narratives(state: EmotionalState) -> list[str]:
        """Convert each raw signal dict into a readable sentence."""
        narratives = []
        for sig in state.signals:
            source = sig.get("source", "unknown").title()
            obs = sig.get("observation", "")
            sug = sig.get("suggests", "")
            narratives.append(
                f"**[{source}]** {obs} — this typically suggests *{sug}*."
            )
        return narratives

    @staticmethod
    def _dimension_explanations(state: EmotionalState) -> dict:
        """Explain each higher-level dimension."""
        return {
            "energy": (
                f"Energy is assessed as **{state.energy_level}** "
                f"(score: {state.energy_score:.0%}).  "
                + (
                    "This suggests fatigue — the person may struggle with demanding tasks."
                    if state.energy_score < 0.35
                    else "Energy appears adequate for normal activity."
                    if state.energy_score < 0.65
                    else "Energy is good — the person appears alert and active."
                )
            ),
            "stress": (
                f"Stress is assessed as **{state.stress_level}** "
                f"(score: {state.stress_score:.0%}).  "
                + (
                    "The person appears relatively calm."
                    if state.stress_score < 0.35
                    else "There are moderate stress indicators — worth monitoring."
                    if state.stress_score < 0.65
                    else "Stress signals are elevated — proactive coping is recommended."
                )
            ),
            "work_inclination": (
                f"Work inclination is **{state.work_inclination}** "
                f"(score: {state.work_score:.0%}).  "
                + (
                    "Signals suggest the person would benefit from rest or a change of pace."
                    if state.work_score < 0.35
                    else "The person appears neutral about continuing work."
                    if state.work_score < 0.60
                    else "The person appears ready and willing to engage with work."
                )
            ),
        }

    @staticmethod
    def _confidence_note(state: EmotionalState) -> str:
        """Honest statement about prediction confidence."""
        n = len(state.modalities_used)
        primary_score = state.emotion_scores.get(state.primary_emotion, 0)

        if n >= 2 and primary_score > 0.5:
            return (
                "**Confidence: Moderate to High.**  Multiple input sources agree, "
                "and the primary emotion has a clear lead."
            )
        if n >= 2:
            return (
                "**Confidence: Moderate.**  Multiple inputs were analysed, but "
                "the emotional signals are mixed — interpret with care."
            )
        if primary_score > 0.5:
            return (
                "**Confidence: Moderate.**  Only one input was provided, but "
                "the emotional signal is fairly clear."
            )
        return (
            "**Confidence: Low.**  Only one input was provided and the "
            "emotional signal is ambiguous.  Adding more input modalities "
            "would improve reliability."
        )

    @staticmethod
    def _limitations(state: EmotionalState) -> list[str]:
        """List honest limitations specific to this assessment."""
        limits = []

        if len(state.modalities_used) == 1:
            limits.append(
                "Only one input modality was used.  Cross-modal validation "
                "is not possible — reliability is reduced."
            )

        if "voice" in state.modalities_used:
            limits.append(
                "Voice analysis uses acoustic features (pitch, energy, pace), "
                "not a speech-emotion model.  This is interpretable but less "
                "accurate than dedicated emotion recognition systems."
            )

        if "face" in state.modalities_used:
            raw_face = state.raw_face_result or {}
            if raw_face.get("model_used") == "fallback":
                limits.append(
                    "Face analysis used fallback mode — the pretrained emotion "
                    "model was unavailable.  Facial emotion predictions are unreliable."
                )
            limits.append(
                "Facial expression interpretation varies across cultures and "
                "individuals.  A 'neutral' face does not necessarily mean "
                "neutral emotions."
            )

        limits.append(
            "This system captures a single-moment snapshot.  Emotional state "
            "fluctuates — repeated assessments over time are more informative."
        )

        return limits

    @staticmethod
    def _disclaimer() -> str:
        return (
            "**Disclaimer:** This tool is for self-awareness and educational "
            "purposes only.  It is **not** a clinical or diagnostic instrument.  "
            "It cannot detect or diagnose mental health conditions.  If you are "
            "struggling, please reach out to a qualified mental-health professional.  "
            "All data is processed locally and is never stored or transmitted."
        )
