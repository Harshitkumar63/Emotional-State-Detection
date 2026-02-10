"""
CLI Inference — Emotional State Analysis from the Command Line
===============================================================

Examples::

    # Text only
    python scripts/inference.py --text "I feel exhausted and can't focus."

    # Voice only
    python scripts/inference.py --voice data/samples/sample_audio.wav

    # Face only
    python scripts/inference.py --face data/samples/sample_face.png

    # All three
    python scripts/inference.py \
        --text "I'm overwhelmed." \
        --voice data/samples/sample_audio.wav \
        --face data/samples/sample_face.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.helpers import load_config, setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Emotional State Detection — CLI")
    parser.add_argument("--text", type=str, default=None, help="Text input")
    parser.add_argument("--voice", type=str, default=None, help="Path to .wav audio")
    parser.add_argument("--face", type=str, default=None, help="Path to face image")
    args = parser.parse_args()

    if not any([args.text, args.voice, args.face]):
        parser.error("Provide at least one input: --text, --voice, or --face")

    config = load_config()

    # --- Analyse each provided modality --------------------------------
    text_result = voice_result = face_result = None

    if args.text:
        from src.analyzers.text_analyzer import TextAnalyzer
        logger.info("Analysing text …")
        text_result = TextAnalyzer(config).analyze(args.text)

    if args.voice:
        from src.analyzers.voice_analyzer import VoiceAnalyzer
        logger.info("Analysing voice …")
        voice_result = VoiceAnalyzer(config).analyze(args.voice)

    if args.face:
        from src.analyzers.face_analyzer import FaceAnalyzer
        logger.info("Analysing face …")
        face_result = FaceAnalyzer(config).analyze(args.face)

    # --- Build unified assessment ---------------------------------------
    from src.core.state_engine import StateEngine
    from src.core.explainer import Explainer

    engine = StateEngine(config)
    state = engine.assess(text_result, voice_result, face_result)
    explanation = Explainer().explain(state)

    # --- Print results --------------------------------------------------
    print("\n" + "=" * 62)
    print("  EMOTIONAL STATE ASSESSMENT")
    print("=" * 62)
    print(f"  Primary Emotion  : {state.primary_emotion.upper()}")
    print(f"  Energy Level     : {state.energy_level}  ({state.energy_score:.0%})")
    print(f"  Stress Level     : {state.stress_level}  ({state.stress_score:.0%})")
    print(f"  Work Inclination : {state.work_inclination}  ({state.work_score:.0%})")
    print(f"  Modalities       : {', '.join(state.modalities_used)}")
    print("-" * 62)

    print("\n  Evidence:")
    for sig in state.signals:
        print(f"    [{sig['source'].upper():>5s}] {sig['observation']}")
        print(f"           -> suggests: {sig['suggests']}")

    print(f"\n  Summary:\n    {state.mental_summary}")

    print("\n  Recommendations:")
    for rec in state.recommendations:
        print(f"    - {rec}")

    print("\n  " + explanation["disclaimer"])
    print("=" * 62)

    # Save JSON
    out = _PROJECT_ROOT / "output" / "last_assessment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(state.to_json(), encoding="utf-8")
    logger.info("JSON saved to %s", out)


if __name__ == "__main__":
    main()
