"""
Audio Preprocessor — Noise Reduction & Silence Trimming
========================================================
Ensures audio input is clean, normalised, and consistently formatted
before feeding it to any voice analyzer.

Why preprocessing matters for voice emotion analysis:
  1. Background noise biases energy/spectral features upward.
  2. Long silences at start/end dilute speech-related features.
  3. Inconsistent sample rates cause feature extraction bugs.
  4. Clipping distorts pitch estimation.

This module applies:
  - Silence trimming (librosa.effects.trim)
  - Peak normalisation (consistent amplitude scale)
  - Resampling to target rate
  - Duration validation
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import tempfile

import numpy as np
import librosa
import soundfile as sf

from src.utils.helpers import setup_logging

logger = setup_logging()


class AudioPreprocessor:
    """Preprocess audio files for consistent, reliable analysis."""

    def __init__(
        self,
        target_sr: int = 16000,
        max_duration_sec: int = 60,
        trim_top_db: int = 20,
    ):
        self.target_sr = target_sr
        self.max_duration = max_duration_sec
        self.trim_top_db = trim_top_db

    def process(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """Preprocess an audio file and optionally save the result.

        Parameters
        ----------
        audio_path : str
            Path to input audio file.
        output_path : str, optional
            Where to save the preprocessed audio. If None, creates a temp file.

        Returns
        -------
        str : path to the preprocessed audio file.
        """
        # Load and resample
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

        # Trim silence from beginning and end
        # trim_top_db: threshold in dB below peak to consider silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=self.trim_top_db)

        # Only use trimmed version if it has meaningful content
        if len(y_trimmed) > self.target_sr * 0.5:  # at least 0.5 seconds
            y = y_trimmed
            logger.info("Silence trimmed: %.1fs -> %.1fs",
                        len(y) / self.target_sr, len(y_trimmed) / self.target_sr)

        # Truncate if too long
        max_samples = self.target_sr * self.max_duration
        if len(y) > max_samples:
            y = y[:max_samples]
            logger.warning("Audio truncated to %d seconds.", self.max_duration)

        # Peak normalisation
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak * 0.95  # leave 5% headroom to avoid clipping

        # Save result
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
        sf.write(output_path, y, self.target_sr)

        logger.info("Audio preprocessed: %s -> %s (%.1fs, %d Hz)",
                     audio_path, output_path, len(y) / self.target_sr, self.target_sr)

        return output_path

    def validate(self, audio_path: str) -> dict:
        """Check audio file validity without modifying it.

        Returns
        -------
        dict with keys: valid (bool), duration_sec, sample_rate, issues (list)
        """
        issues = []

        if not Path(audio_path).exists():
            return {"valid": False, "duration_sec": 0, "sample_rate": 0,
                    "issues": ["File does not exist"]}

        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
        except Exception as e:
            return {"valid": False, "duration_sec": 0, "sample_rate": 0,
                    "issues": [f"Could not load audio: {e}"]}

        duration = len(y) / sr

        if duration < 1.0:
            issues.append("Audio is very short (< 1 second)")
        if duration > 300:
            issues.append("Audio is very long (> 5 minutes)")
        if sr < 8000:
            issues.append(f"Low sample rate ({sr} Hz), may reduce quality")
        if np.abs(y).max() < 0.01:
            issues.append("Audio appears to be mostly silence")

        return {
            "valid": len(issues) == 0 or all("very" in i for i in issues),
            "duration_sec": round(duration, 1),
            "sample_rate": sr,
            "issues": issues,
        }
