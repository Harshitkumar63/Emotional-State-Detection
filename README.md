# Multimodal Emotional State Detection System

> Understand a person's **current mental and emotional state** from text, voice, or facial expression — **any single input is enough**.

---

## What This System Does

Given **any one** (or more) of these inputs:

| Input | What it analyses | Technology |
|-------|-----------------|------------|
| **Text** | Written thoughts, journal entries, chat messages | DistilRoBERTa fine-tuned on 6 emotion datasets |
| **Voice** | Tone, pitch, speed, pauses in a recording | librosa acoustic feature extraction |
| **Face** | Facial expression from a photograph | Vision Transformer fine-tuned on FER-2013 |

The system determines:

- **Primary emotion** — joy, sadness, anger, fear, surprise, disgust, or neutral
- **Energy level** — Exhausted → Low → Moderate → Good → Energetic
- **Stress level** — Calm → Mild → Stressed → High Stress
- **Work inclination** — Needs Rest → Leaning Rest → Neutral → Willing → Motivated
- **Why** — every conclusion is explained with specific evidence from the input

---

## How Emotions Are Actually Detected

### Text → Real Emotion Classification (not just embeddings)

The previous version extracted DistilBERT `[CLS]` embeddings and fed them into a randomly-initialised classifier — producing **random output**.

The fixed version uses `j-hartmann/emotion-english-distilroberta-base`, a model **actually trained on emotion data** (GoEmotions, ISEAR, etc.). It directly outputs probabilities for 7 emotions.

### Voice → Interpretable Acoustic Features (not black-box embeddings)

The previous version mean-pooled Wav2Vec2 hidden states — generic speech representations that contain no emotion-specific information without fine-tuning.

The fixed version extracts research-backed psychoacoustic features:
- **Vocal energy (RMS)** — loud/animated vs. quiet/fatigued
- **Pitch (F0) mean** — aroused/stressed vs. flat/depressed
- **Pitch variability** — expressive/anxious vs. monotone/disengaged
- **Tempo** — rushed/anxious vs. slow/lethargic
- **Spectral centroid** — bright/alert vs. dull/tired voice
- **Speech ratio** — fluent/confident vs. many pauses/hesitation

These features are grounded in decades of psychoacoustics research (Scherer 2003, Juslin & Laukka 2003).

### Face → Actual Facial Emotion Recognition (not ImageNet features)

The previous version used ResNet-18 ImageNet features — trained to recognise dogs, cars, and buildings, not facial emotions.

The fixed version uses a Vision Transformer fine-tuned on FER-2013 (facial expression recognition data), which directly classifies faces into 7 emotions.

### Combining Inputs → Simple Averaging (not random neural net fusion)

The previous version used an attention-based fusion model with **randomly-initialised weights** — mathematically equivalent to random output.

The fixed version **averages emotion probability distributions** across available modalities. This is transparent, correct, and doesn't require training data.

---

## Architecture

```
Input (any one or more)
├── Text  ──→ TextAnalyzer  ──→ {emotion_scores, signals}
├── Voice ──→ VoiceAnalyzer ──→ {acoustic_features, indicators, signals}
└── Face  ──→ FaceAnalyzer  ──→ {emotion_scores, signals}
                    │
                    ▼
            ┌─────────────┐
            │ StateEngine  │  ← averages emotions, maps to dimensions
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────┐
            │  Explainer   │  ← generates WHY explanations
            └──────┬──────┘
                   │
                   ▼
            EmotionalState
            ├── primary_emotion
            ├── energy_level + score
            ├── stress_level + score
            ├── work_inclination + score
            ├── signals (evidence)
            ├── recommendations
            └── limitations
```

---

## Project Structure

```
├── config/config.yaml             # All settings — nothing hardcoded
├── src/
│   ├── analyzers/
│   │   ├── text_analyzer.py       # HuggingFace emotion classification
│   │   ├── voice_analyzer.py      # librosa acoustic features
│   │   └── face_analyzer.py       # HuggingFace facial emotion recognition
│   ├── core/
│   │   ├── emotional_state.py     # Data model (dataclass)
│   │   ├── state_engine.py        # Combines modalities → unified assessment
│   │   └── explainer.py           # Generates human-readable explanations
│   └── utils/
│       └── helpers.py             # Config, logging, file utilities
├── app/streamlit_app.py           # Web frontend (4 pages)
├── scripts/
│   ├── inference.py               # CLI inference
│   └── generate_samples.py        # Dummy data for testing
├── tests/test_analyzers.py        # Unit tests
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
cd "Multimodal Burnout Detection"

# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample test data
python scripts/generate_samples.py

# 3. Run the web app
streamlit run app/streamlit_app.py

# Or use the CLI:
python scripts/inference.py --text "I feel exhausted and can't focus."
python scripts/inference.py --voice data/samples/sample_audio.wav
python scripts/inference.py --face data/samples/sample_face.png
```

---

## Limitations (Honest Assessment)

1. **Text model is English-only** — other languages need a multilingual backbone
2. **Voice analysis is rule-based** — less accurate than a model trained on labelled speech-emotion data (e.g. IEMOCAP)
3. **Face model has cultural bias** — FER-2013 is predominantly Western faces; expressions vary across cultures
4. **Single-moment snapshot** — emotions change rapidly; one assessment isn't a complete picture
5. **No clinical validity** — this is not a validated clinical instrument
6. **Sample data is fake** — the sine wave and gradient image are for testing, not realistic inputs

---

## Future Improvements

- [ ] Fine-tuned speech emotion model (Wav2Vec2 on RAVDESS/IEMOCAP)
- [ ] Multilingual text support (XLM-RoBERTa)
- [ ] Longitudinal tracking (emotions over time)
- [ ] Video input (facial expressions from webcam)
- [ ] Real-time voice analysis during conversation
- [ ] Validated against clinical instruments (PHQ-9, MBI)
- [ ] Face detection preprocessing (MTCNN) for uncropped photos

---

## Ethical Considerations

- This is a **self-awareness tool**, not a surveillance or diagnostic system
- It should never be used for employment, insurance, or academic decisions
- All processing happens **locally** — no data leaves your machine
- The system is transparent about its limitations and confidence level

---

*Built for learning, interviews, and research — because understanding how we feel is the first step to feeling better.*
