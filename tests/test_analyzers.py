"""
Integration Tests — verify each analyser and the state engine
produce correctly structured output.

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.helpers import load_config

_ALL_EMOTIONS = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}


class TestTextAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.analyzers.text_analyzer import TextAnalyzer
        cls.analyzer = TextAnalyzer()

    def test_returns_all_emotions(self):
        result = self.analyzer.analyze("I feel exhausted and hopeless.")
        self.assertTrue(_ALL_EMOTIONS.issubset(result["emotions"].keys()))

    def test_dominant_in_emotions(self):
        result = self.analyzer.analyze("I am so happy today!")
        self.assertIn(result["dominant_emotion"], result["emotions"])

    def test_scores_sum_roughly_to_one(self):
        result = self.analyzer.analyze("Neutral day.")
        total = sum(result["emotions"].values())
        self.assertAlmostEqual(total, 1.0, places=1)

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            self.analyzer.analyze("")


class TestVoiceAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import soundfile as sf
        import tempfile

        cls.tmp = Path(tempfile.mkdtemp())
        sr = 22050
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        cls.audio_path = str(cls.tmp / "test.wav")
        sf.write(cls.audio_path, wave, sr)

        from src.analyzers.voice_analyzer import VoiceAnalyzer
        cls.analyzer = VoiceAnalyzer()

    def test_returns_features(self):
        result = self.analyzer.analyze(self.audio_path)
        self.assertIn("energy_rms", result["features"])
        self.assertIn("pitch_mean_hz", result["features"])

    def test_returns_indicators(self):
        result = self.analyzer.analyze(self.audio_path)
        for key in ("energy_level", "stress_level", "emotional_variability"):
            self.assertIn(key, result["indicators"])
            self.assertGreaterEqual(result["indicators"][key], 0.0)
            self.assertLessEqual(result["indicators"][key], 1.0)

    def test_returns_inferred_emotion(self):
        result = self.analyzer.analyze(self.audio_path)
        self.assertIn(result["inferred_emotion"], _ALL_EMOTIONS)


class TestFaceAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from PIL import Image
        import numpy as np
        import tempfile

        cls.tmp = Path(tempfile.mkdtemp())
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cls.image_path = str(cls.tmp / "test.png")
        Image.fromarray(arr).save(cls.image_path)

        from src.analyzers.face_analyzer import FaceAnalyzer
        cls.analyzer = FaceAnalyzer()

    def test_returns_emotions_dict(self):
        result = self.analyzer.analyze(self.image_path)
        self.assertIn("emotions", result)
        self.assertIn("dominant_emotion", result)

    def test_returns_signals(self):
        result = self.analyzer.analyze(self.image_path)
        self.assertIsInstance(result["signals"], list)
        self.assertGreater(len(result["signals"]), 0)


class TestStateEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.core.state_engine import StateEngine
        cls.engine = StateEngine()

    def test_text_only(self):
        text_result = {
            "emotions": {"joy": 0.7, "neutral": 0.2, "sadness": 0.1},
            "dominant_emotion": "joy",
            "dominant_score": 0.7,
            "signals": [{"source": "text", "observation": "test", "suggests": "joy"}],
        }
        state = self.engine.assess(text_result=text_result)
        self.assertEqual(len(state.modalities_used), 1)
        self.assertIn("text", state.modalities_used)
        self.assertIsInstance(state.energy_score, float)

    def test_voice_only(self):
        voice_result = {
            "features": {"energy_rms": 0.03, "pitch_mean_hz": 150},
            "indicators": {
                "energy_level": 0.4, "stress_level": 0.5,
                "emotional_variability": 0.3, "pause_factor": 0.2,
            },
            "inferred_emotion": "neutral",
            "signals": [{"source": "voice", "observation": "test", "suggests": "neutral"}],
        }
        state = self.engine.assess(voice_result=voice_result)
        self.assertIn("voice", state.modalities_used)

    def test_no_input_raises(self):
        with self.assertRaises(ValueError):
            self.engine.assess()


if __name__ == "__main__":
    unittest.main()
