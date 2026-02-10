"""
Voice Analyzer — Acoustic Feature Extraction & Emotional Mapping
=================================================================
Extracts **interpretable** acoustic features from voice recordings and
maps them to emotional indicators.

Why NOT a black-box Wav2Vec2 embedding?
  The previous implementation mean-pooled Wav2Vec2 hidden states — a 768-d
  vector that captures speech content but tells you nothing specific about
  emotional state.  Without fine-tuning on labelled emotion data, those
  embeddings are useless for emotion detection.

  Instead, we extract well-researched psychoacoustic correlates of emotion:

  | Feature          | High value means         | Low value means            |
  |------------------|--------------------------|----------------------------|
  | RMS energy       | Loud / animated / angry  | Quiet / fatigued / sad     |
  | Pitch (F0) mean  | Aroused / stressed       | Flat / depressed           |
  | Pitch variability| Expressive / anxious     | Monotone / disengaged      |
  | Tempo            | Rushed / anxious         | Slow / lethargic           |
  | Spectral centroid| Bright / alert tone      | Dull / tired tone          |
  | Speech ratio     | Fluent / confident       | Many pauses / hesitation   |

  These features are grounded in decades of psychoacoustics research
  (Scherer 2003, Juslin & Laukka 2003, Eyben et al. 2016).

Design decisions:
  • Uses ``librosa`` — a lightweight, well-tested Python audio library.
    No 380 MB Wav2Vec2 download needed.
  • Every feature has a clear physical meaning, so the explainer can
    generate sentences like "Your vocal energy is below average, which
    often correlates with fatigue."
  • Thresholds are configurable in ``config.yaml``, not hardcoded.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import librosa

from src.utils.helpers import load_config, setup_logging

logger = setup_logging()


class VoiceAnalyzer:
    """Extract emotional indicators from voice recordings via acoustic features.

    Works with any single-channel or multi-channel .wav file.
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        voice_cfg = config["voice"]
        self.sr = voice_cfg["sample_rate"]
        self.max_duration = voice_cfg["max_duration_sec"]
        self.thresholds = voice_cfg["thresholds"]
        logger.info("Voice analyzer ready (librosa, sr=%d)", self.sr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, audio_path: str) -> dict:
        """Extract acoustic features and map them to emotional indicators.

        Parameters
        ----------
        audio_path : str
            Path to a .wav file (mono or stereo, any sample rate).

        Returns
        -------
        dict with keys:
            features   – raw numeric acoustic measurements
            indicators – {energy_level, stress_level, emotional_variability} as 0-1
            inferred_emotion  – best-guess emotion from acoustic profile
            signals    – list of explainability evidence dicts
        """
        y = self._load_audio(audio_path)
        features = self._extract_features(y)
        indicators = self._features_to_indicators(features)
        emotion = self._infer_emotion(indicators)
        signals = self._build_signals(features, indicators)

        return {
            "features": features,
            "indicators": indicators,
            "inferred_emotion": emotion,
            "signals": signals,
        }

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, path: str) -> np.ndarray:
        """Load, resample, normalise, and truncate audio."""
        y, sr = librosa.load(path, sr=self.sr, mono=True)

        # Truncate to max duration to prevent long processing times
        max_samples = self.sr * self.max_duration
        if len(y) > max_samples:
            logger.warning("Audio truncated to %d seconds", self.max_duration)
            y = y[:max_samples]

        # Peak normalisation — makes energy features comparable across clips
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak

        return y

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, y: np.ndarray) -> dict:
        """Compute interpretable acoustic features from the waveform.

        Every feature here has a clear psychoacoustic meaning — this is
        intentional so we can explain predictions in plain language.
        """
        # --- Energy: how loud / animated the speaker is ---
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))

        # --- Pitch (fundamental frequency F0) ---
        # librosa.pyin is more robust than piptrack for single-speaker audio
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=self.sr,
        )
        # Only keep frames where a pitch was actually detected
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        if len(f0_valid) > 0:
            pitch_mean = float(np.mean(f0_valid))
            pitch_std = float(np.std(f0_valid))
        else:
            # No pitched frames detected (silence / noise)
            pitch_mean = 0.0
            pitch_std = 0.0

        # --- Tempo: speaking pace ---
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        tempo_array = librosa.feature.tempo(onset_envelope=onset_env, sr=self.sr)
        tempo = float(tempo_array[0]) if len(tempo_array) > 0 else 0.0

        # --- Spectral centroid: "brightness" of the voice ---
        cent = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        spectral_centroid = float(np.mean(cent))

        # --- Speech ratio: proportion of non-silent frames ---
        # A high silence ratio suggests pauses / hesitation
        frame_energies = librosa.feature.rms(y=y)[0]
        silence_threshold = 0.02 * np.max(frame_energies) if np.max(frame_energies) > 0 else 0.0
        speech_frames = np.sum(frame_energies > silence_threshold)
        total_frames = len(frame_energies)
        speech_ratio = float(speech_frames / total_frames) if total_frames > 0 else 0.0

        duration_sec = float(len(y) / self.sr)

        return {
            "energy_rms": round(energy_mean, 4),
            "pitch_mean_hz": round(pitch_mean, 1),
            "pitch_std_hz": round(pitch_std, 1),
            "tempo_bpm": round(tempo, 1),
            "spectral_centroid_hz": round(spectral_centroid, 1),
            "speech_ratio": round(speech_ratio, 3),
            "duration_sec": round(duration_sec, 1),
        }

    # ------------------------------------------------------------------
    # Feature → Emotional indicator mapping
    # ------------------------------------------------------------------

    def _features_to_indicators(self, feat: dict) -> dict:
        """Map raw acoustic features to 0-1 emotional indicator scores.

        Uses configurable thresholds from config.yaml so the mapping
        can be tuned without changing code.
        """
        t = self.thresholds

        # Energy level: 0 = exhausted, 1 = energetic
        energy = self._scale(feat["energy_rms"], t["energy_low"], t["energy_high"])

        # Stress level: combines pitch height and speaking pace
        pitch_stress = self._scale(feat["pitch_mean_hz"], t["pitch_low_hz"], t["pitch_high_hz"])
        tempo_stress = self._scale(feat["tempo_bpm"], t["tempo_slow_bpm"], t["tempo_fast_bpm"])
        stress = 0.5 * pitch_stress + 0.5 * tempo_stress

        # Emotional variability: pitch variance captures expressiveness
        variability = self._scale(feat["pitch_std_hz"], t["pitch_var_low"], t["pitch_var_high"])

        # Pause factor: low speech ratio → more pauses → hesitation / fatigue
        pause_factor = 1.0 - feat["speech_ratio"]

        return {
            "energy_level": round(max(0.0, min(1.0, energy)), 3),
            "stress_level": round(max(0.0, min(1.0, stress)), 3),
            "emotional_variability": round(max(0.0, min(1.0, variability)), 3),
            "pause_factor": round(max(0.0, min(1.0, pause_factor)), 3),
        }

    # ------------------------------------------------------------------
    # Emotion inference from acoustic profile
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_emotion(ind: dict) -> str:
        """Best-guess emotion label from acoustic indicators.

        This is a simplified rule-based mapping.  It is NOT as accurate
        as a model trained on labelled speech-emotion data (e.g. IEMOCAP),
        and the system acknowledges that in its explanations.
        """
        energy = ind["energy_level"]
        stress = ind["stress_level"]
        variability = ind["emotional_variability"]

        # Decision tree based on psychoacoustic research:
        if energy < 0.3 and stress < 0.4:
            return "sadness"       # quiet, slow, flat → depressed / sad
        if energy < 0.3 and stress >= 0.4:
            return "fear"          # quiet but tense → anxious / fearful
        if energy >= 0.7 and stress >= 0.6:
            return "anger"         # loud, fast, high pitch → angry / frustrated
        if energy >= 0.6 and stress < 0.4 and variability > 0.5:
            return "joy"           # loud, expressive, relaxed → happy
        if variability < 0.3 and energy < 0.5:
            return "neutral"       # monotone, moderate → disengaged / neutral
        if stress >= 0.6:
            return "fear"          # stressed → anxious
        return "neutral"

    # ------------------------------------------------------------------
    # Explainability signals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_signals(feat: dict, ind: dict) -> list[dict]:
        """Generate human-readable evidence from each measured feature."""
        signals = []

        # Energy
        if ind["energy_level"] < 0.3:
            signals.append({
                "source": "voice",
                "observation": f"Vocal energy is low (RMS={feat['energy_rms']:.3f})",
                "suggests": "fatigue or low mood",
            })
        elif ind["energy_level"] > 0.7:
            signals.append({
                "source": "voice",
                "observation": f"Vocal energy is high (RMS={feat['energy_rms']:.3f})",
                "suggests": "animation, possibly stress or excitement",
            })

        # Pitch
        if feat["pitch_mean_hz"] > 0:
            if feat["pitch_mean_hz"] > 220:
                signals.append({
                    "source": "voice",
                    "observation": f"Pitch is elevated ({feat['pitch_mean_hz']:.0f} Hz)",
                    "suggests": "emotional arousal or stress",
                })
            elif feat["pitch_mean_hz"] < 120:
                signals.append({
                    "source": "voice",
                    "observation": f"Pitch is low ({feat['pitch_mean_hz']:.0f} Hz)",
                    "suggests": "flat affect, possible fatigue",
                })

        # Pitch variability
        if ind["emotional_variability"] < 0.25:
            signals.append({
                "source": "voice",
                "observation": f"Voice is monotone (pitch std={feat['pitch_std_hz']:.1f} Hz)",
                "suggests": "disengagement or emotional numbness",
            })

        # Tempo
        if feat["tempo_bpm"] > 160:
            signals.append({
                "source": "voice",
                "observation": f"Speaking pace is fast ({feat['tempo_bpm']:.0f} BPM)",
                "suggests": "anxiety or urgency",
            })
        elif feat["tempo_bpm"] < 90 and feat["tempo_bpm"] > 0:
            signals.append({
                "source": "voice",
                "observation": f"Speaking pace is slow ({feat['tempo_bpm']:.0f} BPM)",
                "suggests": "lethargy or careful deliberation",
            })

        # Pauses
        if ind["pause_factor"] > 0.5:
            signals.append({
                "source": "voice",
                "observation": f"Frequent pauses (speech ratio={feat['speech_ratio']:.0%})",
                "suggests": "hesitation, fatigue, or uncertainty",
            })

        # Default signal if nothing notable
        if not signals:
            signals.append({
                "source": "voice",
                "observation": "Vocal features are within normal ranges",
                "suggests": "a relatively neutral emotional state from voice alone",
            })

        return signals

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _scale(value: float, low: float, high: float) -> float:
        """Linearly scale *value* to [0, 1] given reference bounds."""
        if high <= low:
            return 0.5
        return (value - low) / (high - low)
