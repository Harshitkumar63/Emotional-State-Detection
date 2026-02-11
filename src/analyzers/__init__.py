"""
Analyzers — Per-Modality Emotion Analysis + Embedding Extraction
=================================================================
Each analyzer handles one input modality:
  - TextAnalyzer:       DistilRoBERTa (emotion classification + embeddings)
  - VoiceAnalyzer:      Librosa acoustic features (interpretable baseline)
  - VoiceAnalyzerDeep:  Wav2Vec2 fine-tuned on IEMOCAP (deep speech emotion)
  - FaceAnalyzer:       ViT fine-tuned on FER-2013 (facial emotion + embeddings)
"""

from src.analyzers.text_analyzer import TextAnalyzer
from src.analyzers.voice_analyzer import VoiceAnalyzer
from src.analyzers.voice_analyzer_deep import VoiceAnalyzerDeep
from src.analyzers.face_analyzer import FaceAnalyzer

__all__ = ["TextAnalyzer", "VoiceAnalyzer", "VoiceAnalyzerDeep", "FaceAnalyzer"]
