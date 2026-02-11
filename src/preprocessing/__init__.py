"""
Preprocessing Module — Robust Input Handling
=============================================
Provides preprocessing utilities for each modality to improve reliability:

- AudioPreprocessor: silence trimming, normalisation, resampling
- FacePreprocessor:  face detection + cropping (MTCNN)
- TextPreprocessor:  language detection, cleaning, fallback handling
"""

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.preprocessing.face_preprocessor import FacePreprocessor
from src.preprocessing.text_preprocessor import TextPreprocessor

__all__ = ["AudioPreprocessor", "FacePreprocessor", "TextPreprocessor"]
