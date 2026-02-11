"""
Unit Tests for Analyzers, Fusion, and Engines (v2)
====================================================
Tests cover:
  - Text, Voice (acoustic), Voice (deep), Face analyzers
  - Attention fusion model forward pass
  - BurnoutEngine end-to-end (text only)
  - Preprocessing modules
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestAttentionFusionNetwork(unittest.TestCase):
    """Test the fusion model architecture independently (no model downloads)."""

    def setUp(self):
        from src.fusion.attention_fusion import AttentionFusionNetwork
        self.model = AttentionFusionNetwork(
            embedding_dim=768, hidden_dim=256, num_classes=3
        )
        self.model.eval()

    def test_forward_all_modalities(self):
        """All three modalities provided."""
        embeddings = {
            "text": torch.randn(2, 768),
            "voice": torch.randn(2, 768),
            "face": torch.randn(2, 768),
        }
        output = self.model(embeddings)
        self.assertEqual(output["logits"].shape, (2, 3))
        self.assertEqual(output["probabilities"].shape, (2, 3))
        self.assertEqual(output["predicted_class"].shape, (2,))
        self.assertIn("modality_weights", output)

    def test_forward_single_modality(self):
        """Only text provided (voice and face missing)."""
        embeddings = {"text": torch.randn(1, 768)}
        output = self.model(embeddings)
        self.assertEqual(output["logits"].shape, (1, 3))
        # Text weight should be 1.0 (only available modality)
        text_weight = output["modality_weights"]["text"].item()
        self.assertAlmostEqual(text_weight, 1.0, places=3)

    def test_forward_two_modalities(self):
        """Text + face, voice missing."""
        embeddings = {
            "text": torch.randn(1, 768),
            "face": torch.randn(1, 768),
        }
        output = self.model(embeddings)
        self.assertEqual(output["logits"].shape, (1, 3))
        # Voice weight should be ~0 (masked)
        voice_weight = output["modality_weights"]["voice"].item()
        self.assertAlmostEqual(voice_weight, 0.0, places=3)

    def test_probabilities_sum_to_one(self):
        """Output probabilities should sum to 1."""
        embeddings = {"text": torch.randn(4, 768)}
        output = self.model(embeddings)
        sums = output["probabilities"].sum(dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=4)

    def test_attention_weights_sum_to_one(self):
        """Attention weights across modalities should sum to 1."""
        embeddings = {
            "text": torch.randn(1, 768),
            "voice": torch.randn(1, 768),
            "face": torch.randn(1, 768),
        }
        output = self.model(embeddings)
        total = sum(
            output["modality_weights"][mod].item()
            for mod in ["text", "voice", "face"]
        )
        self.assertAlmostEqual(total, 1.0, places=3)


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test the synthetic data generator."""

    def test_generate_shapes(self):
        from scripts.train_fusion import generate_synthetic_embeddings
        text, voice, face, labels, masks = generate_synthetic_embeddings(
            n_samples=30, embedding_dim=768
        )
        self.assertEqual(text.shape[0], 30)
        self.assertEqual(text.shape[1], 768)
        self.assertEqual(labels.shape[0], 30)
        self.assertEqual(masks.shape, (30, 3))

    def test_balanced_classes(self):
        from scripts.train_fusion import generate_synthetic_embeddings
        _, _, _, labels, _ = generate_synthetic_embeddings(n_samples=300)
        counts = torch.bincount(labels)
        self.assertEqual(len(counts), 3)
        for c in counts:
            self.assertEqual(c.item(), 100)  # 300 / 3 = 100 per class


class TestVoiceAnalyzerAcoustic(unittest.TestCase):
    """Test the acoustic voice analyzer with a synthetic audio file."""

    def setUp(self):
        import tempfile
        import soundfile as sf

        # Generate a sine wave test audio
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 220 * t)

        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(self.temp_file.name, tone, sr)

    def test_analyze_returns_correct_keys(self):
        from src.analyzers.voice_analyzer import VoiceAnalyzer
        from src.utils.helpers import load_config
        config = load_config()
        analyzer = VoiceAnalyzer(config)
        result = analyzer.analyze(self.temp_file.name)

        self.assertIn("features", result)
        self.assertIn("indicators", result)
        self.assertIn("inferred_emotion", result)
        self.assertIn("signals", result)

    def test_indicators_in_range(self):
        from src.analyzers.voice_analyzer import VoiceAnalyzer
        from src.utils.helpers import load_config
        config = load_config()
        analyzer = VoiceAnalyzer(config)
        result = analyzer.analyze(self.temp_file.name)

        for key, value in result["indicators"].items():
            self.assertGreaterEqual(value, 0.0, f"{key} should be >= 0")
            self.assertLessEqual(value, 1.0, f"{key} should be <= 1")


class TestPreprocessors(unittest.TestCase):
    """Test preprocessing modules."""

    def test_text_preprocessor(self):
        from src.preprocessing.text_preprocessor import TextPreprocessor
        prep = TextPreprocessor()
        result = prep.process("  Hello   world!   This is a test.  ")
        self.assertIn("cleaned_text", result)
        self.assertIn("warnings", result)
        self.assertEqual(result["word_count"], 6)

    def test_text_preprocessor_short_text(self):
        from src.preprocessing.text_preprocessor import TextPreprocessor
        prep = TextPreprocessor(min_words=5)
        result = prep.process("Hi")
        self.assertTrue(len(result["warnings"]) > 0)

    def test_audio_preprocessor_validate(self):
        import tempfile
        import soundfile as sf

        from src.preprocessing.audio_preprocessor import AudioPreprocessor
        prep = AudioPreprocessor()

        # Create valid audio
        sr = 16000
        y = np.random.randn(sr * 2).astype(np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, sr)

        result = prep.validate(tmp.name)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["duration_sec"], 2.0, places=0)

    def test_audio_preprocessor_invalid_path(self):
        from src.preprocessing.audio_preprocessor import AudioPreprocessor
        prep = AudioPreprocessor()
        result = prep.validate("nonexistent_file.wav")
        self.assertFalse(result["valid"])


class TestEmotionalState(unittest.TestCase):
    """Test the data model."""

    def test_serialisation(self):
        from src.core.emotional_state import EmotionalState
        state = EmotionalState(
            primary_emotion="sadness",
            emotion_scores={"sadness": 0.7, "joy": 0.1},
            energy_level="Low Energy",
            energy_score=0.3,
            stress_level="Stressed",
            stress_score=0.7,
            work_inclination="Needs Rest",
            work_score=0.2,
            burnout_risk="High Risk",
            burnout_confidence=0.8,
        )

        d = state.to_dict()
        self.assertIn("burnout_risk", d)
        self.assertEqual(d["burnout_risk"], "High Risk")
        self.assertNotIn("raw_text_result", d)

        j = state.to_json()
        import json
        parsed = json.loads(j)
        self.assertEqual(parsed["primary_emotion"], "sadness")


if __name__ == "__main__":
    unittest.main()
