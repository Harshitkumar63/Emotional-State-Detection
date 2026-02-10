"""
Generate Sample Data — for smoke testing all three modalities.

Creates:
  • sample_journal.txt  — synthetic journal entry
  • sample_audio.wav    — 3 s sine-wave placeholder
  • sample_face.png     — 224×224 gradient image placeholder
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.helpers import load_config, ensure_dir


def generate_text_sample(out_dir: Path) -> Path:
    text = (
        "I've been feeling completely drained for the past few weeks. "
        "No matter how much I sleep, I wake up exhausted. My motivation "
        "at work has dropped significantly — tasks that used to excite me "
        "now feel like an unbearable chore. I've started withdrawing from "
        "colleagues and skipping team lunches. Even on weekends I can't "
        "seem to recharge. I'm worried this might be more than just "
        "regular tiredness."
    )
    path = out_dir / "sample_journal.txt"
    path.write_text(text, encoding="utf-8")
    return path


def generate_audio_sample(out_dir: Path, duration_sec: float = 3.0) -> Path:
    import soundfile as sf

    sr = 22050
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    wave = (0.5 * np.sin(2 * np.pi * 440 * t) * np.linspace(1, 0.3, len(t))).astype(np.float32)
    path = out_dir / "sample_audio.wav"
    sf.write(str(path), wave, sr)
    return path


def generate_image_sample(out_dir: Path, size: int = 224) -> Path:
    from PIL import Image

    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for c in range(3):
        arr[:, :, c] = np.tile(
            np.linspace(60 + c * 40, 200 + c * 15, size, dtype=np.uint8),
            (size, 1),
        )
    Image.fromarray(arr).save(str(out_dir / "sample_face.png"))
    return out_dir / "sample_face.png"


def main() -> None:
    config = load_config()
    out_dir = ensure_dir(_PROJECT_ROOT / config["paths"]["sample_data"])
    print(f"Generating sample data in: {out_dir}\n")

    print(f"  [text]  {generate_text_sample(out_dir)}")
    print(f"  [audio] {generate_audio_sample(out_dir)}")
    print(f"  [image] {generate_image_sample(out_dir)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
