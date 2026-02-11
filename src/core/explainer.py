"""
Explainer — Human-Readable Reasoning for Burnout Risk Predictions (v2)
=======================================================================
Transforms the structured EmotionalState into plain-English explanations
that answer **why** the system reached each conclusion.

v2 additions:
  - Burnout risk explanation (modality contributions, risk factors)
  - Attention weight narrative (which modality mattered most, and why)
  - Improved confidence assessment using fusion model confidence

The Explainer does NOT make predictions — it reads the signals that the
analysers and fusion model already produced and narrates them for humans.

Every explanation follows:
  1. What was observed (the evidence)
  2. What it suggests (the interpretation)
  3. How confident we are (honest uncertainty)
"""

from __future__ import annotations

from src.core.emotional_state import EmotionalState


class Explainer:
    """Generate clear, honest explanations for burnout risk assessments."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, state: EmotionalState) -> dict:
        """Build a full explanation package.

        Returns
        -------
        dict with keys:
            overall_narrative       - summary of assessment
            burnout_narrative       - burnout risk explanation (NEW v2)
            contribution_narrative  - modality contribution breakdown (NEW v2)
            signal_narratives       - per-signal explanations
            dimension_explanations  - energy / stress / work reasoning
            confidence_note         - how much to trust this result
            limitations             - honest list of caveats
            disclaimer              - ethical/legal disclaimer
        """
        return {
            "overall_narrative": self._overall_narrative(state),
            "burnout_narrative": self._burnout_narrative(state),
            "contribution_narrative": self._contribution_narrative(state),
            "signal_narratives": self._signal_narratives(state),
            "dimension_explanations": self._dimension_explanations(state),
            "confidence_note": self._confidence_note(state),
            "limitations": self._limitations(state),
            "disclaimer": self._disclaimer(),
        }

    # ------------------------------------------------------------------
    # Burnout Risk Narrative (NEW in v2)
    # ------------------------------------------------------------------

    @staticmethod
    def _burnout_narrative(state: EmotionalState) -> str:
        """Explain the burnout risk prediction and what drives it.

        Example output:
          "Burnout risk is assessed as **Moderate Risk** (62% confidence).
           Text showed high exhaustion signals (45%), audio indicated low
           energy (35%), face showed fatigue cues (20%)."
        """
        if state.burnout_risk == "N/A":
            return (
                "Burnout risk could not be assessed. The fusion model requires "
                "at least one modality with a valid embedding."
            )

        risk = state.burnout_risk
        confidence = state.burnout_confidence

        # Risk-specific context
        risk_context = {
            "Low Risk": (
                "The emotional signals across available modalities suggest "
                "the person is in a relatively healthy emotional state. "
                "No strong burnout indicators were detected."
            ),
            "Moderate Risk": (
                "Some concerning patterns were detected across the available "
                "inputs. There are mixed emotional signals that could indicate "
                "early signs of stress accumulation or emotional strain."
            ),
            "High Risk": (
                "Multiple strong negative signals were detected. The emotional "
                "patterns across modalities suggest significant stress, fatigue, "
                "or emotional exhaustion — consistent with burnout risk factors."
            ),
        }

        context = risk_context.get(risk, "")
        probs_str = ""
        if state.burnout_probabilities:
            parts = [f"{k}: {v:.0%}" for k, v in state.burnout_probabilities.items()]
            probs_str = f" (Distribution: {', '.join(parts)})"

        return (
            f"**Burnout Risk: {risk}** (confidence: {confidence:.0%}).{probs_str}\n\n"
            f"{context}"
        )

    # ------------------------------------------------------------------
    # Modality Contribution Narrative (NEW in v2)
    # ------------------------------------------------------------------

    @staticmethod
    def _contribution_narrative(state: EmotionalState) -> str:
        """Explain which modality contributed most and why that matters.

        The attention mechanism assigns weights to each modality per-sample.
        We translate these weights into a human-readable explanation.

        Example:
          "Text showed high exhaustion signals (45%), audio indicated
           low energy (35%), face showed fatigue cues (20%)"
        """
        contributions = state.modality_contributions
        if not contributions:
            return "Modality contribution weights are not available."

        # Sort by contribution (highest first)
        sorted_mods = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

        # Build narrative
        mod_descriptions = {
            "text": "written text analysis",
            "voice": "voice/speech analysis",
            "face": "facial expression analysis",
        }

        parts = []
        for mod, weight in sorted_mods:
            desc = mod_descriptions.get(mod, mod)
            importance = (
                "dominated" if weight > 0.6
                else "contributed significantly to" if weight > 0.35
                else "provided supporting signals for"
            )
            parts.append(f"**{desc.title()}** {importance} the assessment ({weight:.0%})")

        result = "Modality contributions:\n" + "\n".join(f"- {p}" for p in parts)

        # Add insight about what the weighting means
        if len(sorted_mods) > 1:
            top_mod, top_weight = sorted_mods[0]
            if top_weight > 0.6:
                result += (
                    f"\n\nThe fusion model placed most weight on {top_mod}, suggesting "
                    "it provided the clearest emotional signals for this input."
                )

        return result

    # ------------------------------------------------------------------
    # Existing narratives (upgraded)
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
                f"**[{source}]** {obs} -- this typically suggests *{sug}*."
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
                    "This suggests fatigue -- the person may struggle with demanding tasks."
                    if state.energy_score < 0.35
                    else "Energy appears adequate for normal activity."
                    if state.energy_score < 0.65
                    else "Energy is good -- the person appears alert and active."
                )
            ),
            "stress": (
                f"Stress is assessed as **{state.stress_level}** "
                f"(score: {state.stress_score:.0%}).  "
                + (
                    "The person appears relatively calm."
                    if state.stress_score < 0.35
                    else "There are moderate stress indicators -- worth monitoring."
                    if state.stress_score < 0.65
                    else "Stress signals are elevated -- proactive coping is recommended."
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
        burnout_conf = state.burnout_confidence

        # v2: factor in fusion model confidence
        if n >= 2 and burnout_conf > 0.6:
            return (
                "**Confidence: High.**  Multiple input sources were analysed, "
                "and the fusion model shows strong agreement. The burnout risk "
                f"prediction has {burnout_conf:.0%} confidence."
            )
        if n >= 2 and primary_score > 0.5:
            return (
                "**Confidence: Moderate to High.**  Multiple input sources agree, "
                "and the primary emotion has a clear lead."
            )
        if n >= 2:
            return (
                "**Confidence: Moderate.**  Multiple inputs were analysed, but "
                "the emotional signals are mixed -- interpret with care."
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
                "is not possible -- reliability is reduced."
            )

        if "voice" in state.modalities_used:
            if state.voice_model_used == "acoustic":
                limits.append(
                    "Voice analysis used acoustic features only (no deep model). "
                    "This is interpretable but less accurate than dedicated "
                    "speech emotion recognition."
                )
            else:
                limits.append(
                    "Voice deep model was trained on IEMOCAP (4 emotions only). "
                    "It may miss nuances in non-acted emotional speech."
                )

        if "face" in state.modalities_used:
            raw_face = state.raw_face_result or {}
            if raw_face.get("model_used") == "fallback":
                limits.append(
                    "Face analysis used fallback mode -- facial emotion "
                    "predictions are unreliable."
                )
            limits.append(
                "Facial expression interpretation varies across cultures "
                "and individuals."
            )

        if state.burnout_risk != "N/A":
            limits.append(
                "The fusion model was trained on synthetic data.  Its burnout "
                "risk predictions encode domain knowledge but lack validation "
                "on real burnout-annotated datasets."
            )

        limits.append(
            "This system captures a single-moment snapshot.  Emotional state "
            "fluctuates -- repeated assessments over time are more informative."
        )

        return limits

    @staticmethod
    def _disclaimer() -> str:
        return (
            "**Disclaimer:** This system provides early burnout risk signals "
            "for self-awareness purposes only.  It is **not** a medical or "
            "clinical diagnostic tool.  It cannot detect, diagnose, or treat "
            "mental health conditions.  If you are struggling, please reach "
            "out to a qualified mental-health professional.  All data is "
            "processed locally and is never stored or transmitted."
        )

