"""
Inference Script — Run Burnout Risk Assessment from CLI
========================================================
Usage:
  python scripts/inference.py --text "I feel exhausted and stressed about work"
  python scripts/inference.py --voice data/samples/sample_audio.wav
  python scripts/inference.py --face data/samples/sample_face.png
  python scripts/inference.py --text "..." --voice audio.wav --face face.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.burnout_engine import BurnoutEngine
from src.core.explainer import Explainer
from src.utils.helpers import load_config, setup_logging, ensure_dir

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Burnout Risk Detection System v2",
    )
    parser.add_argument("--text", type=str, default=None, help="Text input")
    parser.add_argument("--voice", type=str, default=None, help="Path to audio .wav")
    parser.add_argument("--face", type=str, default=None, help="Path to face image")
    parser.add_argument("--output", type=str, default="output/last_assessment.json",
                        help="Where to save the JSON result")

    args = parser.parse_args()

    if args.text is None and args.voice is None and args.face is None:
        parser.error("At least one input is required: --text, --voice, or --face")

    # --- Load engine ---
    config = load_config()
    engine = BurnoutEngine(config)
    explainer = Explainer()

    print("\n" + "=" * 60)
    print("  MULTIMODAL BURNOUT RISK DETECTION SYSTEM v2")
    print("=" * 60)

    # --- Run assessment ---
    state = engine.assess(
        text=args.text,
        audio_path=args.voice,
        image=args.face,
    )

    explanation = explainer.explain(state)

    # --- Display results ---
    print(f"\n--- BURNOUT RISK ---")
    print(f"  Risk Level:  {state.burnout_risk}")
    print(f"  Confidence:  {state.burnout_confidence:.0%}")
    if state.burnout_probabilities:
        for label, prob in state.burnout_probabilities.items():
            print(f"    {label}: {prob:.0%}")

    print(f"\n--- PRIMARY EMOTION ---")
    print(f"  {state.primary_emotion} ({state.emotion_scores.get(state.primary_emotion, 0):.0%})")

    print(f"\n--- DIMENSIONS ---")
    print(f"  Energy:          {state.energy_level} ({state.energy_score:.0%})")
    print(f"  Stress:          {state.stress_level} ({state.stress_score:.0%})")
    print(f"  Work Inclination: {state.work_inclination} ({state.work_score:.0%})")

    if state.modality_contributions:
        print(f"\n--- MODALITY CONTRIBUTIONS ---")
        for mod, weight in state.modality_contributions.items():
            print(f"  {mod}: {weight:.0%}")

    print(f"\n--- SIGNALS ---")
    for sig in state.signals:
        src = sig.get("source", "?")
        obs = sig.get("observation", "")
        sug = sig.get("suggests", "")
        print(f"  [{src}] {obs} -> suggests: {sug}")

    print(f"\n--- SUMMARY ---")
    print(f"  {state.mental_summary}")

    print(f"\n--- RECOMMENDATIONS ---")
    for rec in state.recommendations:
        print(f"  - {rec}")

    print(f"\n--- VOICE MODEL ---")
    print(f"  Used: {state.voice_model_used}")

    print(f"\n--- DISCLAIMER ---")
    print(f"  {explanation['disclaimer']}")

    # --- Save JSON ---
    ensure_dir(str(Path(args.output).parent))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"\n  Full result saved to: {args.output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
