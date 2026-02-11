"""
Text Preprocessor — Language Detection & Cleaning
===================================================
Validates and cleans text input before emotion analysis.

Why text preprocessing matters:
  1. The emotion model is English-only.  Non-English text produces
     unreliable predictions — we should detect and warn.
  2. Very short inputs (< 5 words) lack sufficient context.
  3. Special characters and formatting artifacts can confuse tokenisation.

Graceful handling:
  The preprocessor never blocks analysis.  It returns a cleaned text
  plus a list of warnings.  The system proceeds but surfaces the
  warnings to the user for transparency.
"""

from __future__ import annotations

import re
from typing import Optional

from src.utils.helpers import setup_logging

logger = setup_logging()


class TextPreprocessor:
    """Clean and validate text input for emotion analysis."""

    def __init__(self, min_words: int = 3, max_length: int = 5000):
        self.min_words = min_words
        self.max_length = max_length
        self._langdetect_available = False

        try:
            from langdetect import detect as _detect
            self._detect = _detect
            self._langdetect_available = True
        except ImportError:
            logger.info(
                "langdetect not installed; language detection disabled. "
                "Install with: pip install langdetect"
            )

    def process(self, text: str) -> dict:
        """Clean and validate text input.

        Parameters
        ----------
        text : str
            Raw text input from user.

        Returns
        -------
        dict with keys:
            cleaned_text : str       - preprocessed text ready for analysis
            warnings     : list[str] - any issues found
            language     : str       - detected language code (or "unknown")
            word_count   : int       - number of words in cleaned text
        """
        warnings = []

        # Basic cleaning
        cleaned = text.strip()

        # Remove excessive whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Remove common formatting artifacts
        cleaned = re.sub(r"[<>{}]", "", cleaned)

        # Truncate if too long
        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length]
            warnings.append(
                f"Text was truncated to {self.max_length} characters."
            )

        # Word count check
        words = cleaned.split()
        word_count = len(words)

        if word_count < self.min_words:
            warnings.append(
                f"Very short input ({word_count} words). "
                "Emotion detection works better with longer text."
            )

        # Language detection
        language = "unknown"
        if self._langdetect_available and word_count >= 3:
            try:
                language = self._detect(cleaned)
                if language != "en":
                    warnings.append(
                        f"Detected language: '{language}'. The emotion model is "
                        "optimised for English — results may be less accurate."
                    )
            except Exception:
                language = "unknown"
                warnings.append("Could not detect language.")

        return {
            "cleaned_text": cleaned,
            "warnings": warnings,
            "language": language,
            "word_count": word_count,
        }

    def validate(self, text: str) -> dict:
        """Quick validation without full preprocessing.

        Returns
        -------
        dict with keys: valid (bool), issues (list)
        """
        issues = []
        text = text.strip() if text else ""

        if not text:
            return {"valid": False, "issues": ["Text is empty."]}

        if len(text.split()) < self.min_words:
            issues.append("Text is very short.")

        if len(text) > self.max_length * 2:
            issues.append("Text is extremely long.")

        return {"valid": len(issues) == 0, "issues": issues}
