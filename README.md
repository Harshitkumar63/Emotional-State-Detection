# Multimodal Burnout Risk Detection System v2

> **Detect early burnout risk using text, voice, and facial expressions via attention-based multimodal fusion.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)](https://streamlit.io/)

---

## Problem Statement

**Burnout is a growing workplace crisis.** The WHO classifies it as an "occupational phenomenon" affecting millions. Yet most people don't recognise the signs until it's too late.

This system provides **early burnout risk signals** by analysing a person's current emotional state through three modalities:

- **Text** (journal entries, written thoughts, feedback)
- **Voice** (tone, pitch, energy, speaking pace)
- **Face** (facial expressions and micro-expressions)

It does **NOT** provide medical diagnosis. It is a self-awareness tool that helps individuals and organisations notice warning patterns.

### Burnout vs. Emotion Detection

| Aspect | Emotion Detection (v1) | Burnout Risk Detection (v2) |
|--------|----------------------|---------------------------|
| **Output** | Single emotion label | Risk level (Low/Moderate/High) |
| **Fusion** | Simple averaging | Attention-based neural fusion |
| **Voice** | Handcrafted features | Deep speech model (Wav2Vec2) |
| **Explainability** | Basic signals | Modality contributions + narratives |
| **Product framing** | "You feel sad" | "Burnout risk signals detected" |

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │           BURNOUT ENGINE (v2)                │
                    │                                             │
    Text ──────►    │  TextAnalyzer (DistilRoBERTa)               │
                    │    ├── Emotion scores (7 classes)            │
                    │    └── [CLS] embedding (768-d) ─────┐       │
                    │                                     │       │
    Voice ─────►    │  VoiceAnalyzer (librosa baseline)   │       │
                    │    └── Acoustic indicators           │       │
                    │  VoiceAnalyzerDeep (Wav2Vec2)       │       │
                    │    ├── Emotion scores (4 classes)    │       │
                    │    └── Mean-pooled embedding ────────┤       │
                    │                                     │       │
    Face ──────►    │  FaceAnalyzer (ViT)                 │       │
                    │    ├── Emotion scores (7 classes)    │       │
                    │    └── [CLS] embedding (768-d) ─────┤       │
                    │                                     ▼       │
                    │              ┌───────────────────────────┐   │
                    │              │  AttentionFusionNetwork    │   │
                    │              │  ┌─ Projector (768→256)   │   │
                    │              │  ├─ Cross-Modal Attention  │   │
                    │              │  └─ Classifier (256→3)    │   │
                    │              │                           │   │
                    │              │  Output: Burnout Risk     │   │
                    │              │  (Low / Moderate / High)  │   │
                    │              └───────────────────────────┘   │
                    │                                             │
                    │  StateEngine: emotion → energy/stress/work  │
                    │  Explainer:   signals → human narratives    │
                    └─────────────────────────────────────────────┘
```

### Why This Architecture?

1. **Modular**: Each component can be tested, debugged, and upgraded independently
2. **Explainable**: Attention weights show which modality the model trusts
3. **Robust**: Missing modalities are handled via attention masking (not zero-padding)
4. **Practical**: Works on CPU, no GPU required

---

## Project Structure

```
Multimodal Burnout Detection/
├── app/
│   └── streamlit_app.py          # Streamlit frontend
├── config/
│   └── config.yaml               # All configuration (no hardcoding)
├── src/
│   ├── analyzers/
│   │   ├── text_analyzer.py      # DistilRoBERTa emotion + embedding
│   │   ├── voice_analyzer.py     # Acoustic baseline (librosa)
│   │   ├── voice_analyzer_deep.py # Wav2Vec2 speech emotion model
│   │   └── face_analyzer.py      # ViT facial emotion + embedding
│   ├── core/
│   │   ├── emotional_state.py    # Data model (EmotionalState)
│   │   ├── state_engine.py       # Emotion → dimensions mapping
│   │   ├── burnout_engine.py     # Main orchestrator (v2)
│   │   └── explainer.py          # Human-readable explanations
│   ├── fusion/
│   │   └── attention_fusion.py   # PyTorch attention fusion network
│   ├── preprocessing/
│   │   ├── audio_preprocessor.py # Silence trimming, normalisation
│   │   ├── face_preprocessor.py  # MTCNN face detection + cropping
│   │   └── text_preprocessor.py  # Language detection, cleaning
│   └── utils/
│       └── helpers.py            # Config loading, logging
├── scripts/
│   ├── inference.py              # CLI assessment
│   ├── train_fusion.py           # Train the fusion model
│   └── generate_samples.py       # Generate test data
├── checkpoints/                  # Trained model weights
├── tests/
│   └── test_analyzers.py         # Unit tests
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Train the Fusion Model

```bash
python scripts/train_fusion.py
```

This trains the `AttentionFusionNetwork` on synthetic data (~30 seconds on CPU).  
The trained model is saved to `checkpoints/fusion_model.pt`.

### 3. Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

### 4. CLI Usage

```bash
# Text only
python scripts/inference.py --text "I feel exhausted and overwhelmed"

# Voice only
python scripts/inference.py --voice path/to/audio.wav

# All modalities
python scripts/inference.py --text "..." --voice audio.wav --face face.jpg
```

---

## How It Works

### Text Analysis
Uses `j-hartmann/emotion-english-distilroberta-base` — a DistilRoBERTa model fine-tuned on 6 emotion datasets (GoEmotions, ISEAR, etc.) for 7-class emotion classification. Also extracts the [CLS] token embedding (768-d) for the fusion model.

### Voice Analysis (Acoustic Baseline)
Extracts interpretable psychoacoustic features via librosa:
- **RMS energy** → vocal animation level
- **Pitch (F0)** → emotional arousal
- **Pitch variability** → expressiveness
- **Tempo** → speaking pace
- **Speech ratio** → pause frequency

### Voice Analysis (Deep Model)
Uses `superb/wav2vec2-base-superb-er` — a Wav2Vec2 model fine-tuned on IEMOCAP for 4-class speech emotion recognition (angry, happy, neutral, sad). Provides both emotion classification and 768-d embeddings for fusion.

**Why deep models outperform acoustic features:**  
Wav2Vec2 learns hierarchical speech representations from 960h of unlabelled speech. When fine-tuned on emotion data, it captures prosodic patterns that handcrafted features miss — achieving 10-20% higher accuracy on IEMOCAP benchmarks.

### Face Analysis
Uses `trpakov/vit-face-expression` — a Vision Transformer fine-tuned on FER-2013 for 7-class facial emotion recognition. Includes MTCNN face detection for robust preprocessing.

### Attention-Based Fusion
The `AttentionFusionNetwork` (~260K parameters):
1. Projects each modality's 768-d embedding to a shared 256-d space
2. Computes attention scores (Bahdanau-style) for each modality
3. Creates a weighted sum of projected embeddings
4. Classifies the fused representation into 3 burnout risk levels

**Advantages over averaging:**
- Learns which modality to trust **per-sample** (not globally)
- Naturally handles missing modalities via masking
- Provides interpretable contribution weights

### Burnout Risk Mapping
Emotions are mapped to burnout indicators:
- **High Risk indicators**: sadness, fear, low energy, high stress
- **Moderate Risk indicators**: anger, mixed emotions, moderate stress
- **Low Risk indicators**: joy, neutral state, good energy, low stress

---

## Models Used

| Model | Task | Parameters | Source |
|-------|------|-----------|--------|
| DistilRoBERTa | Text emotion (7 classes) | 82M | [j-hartmann](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) |
| Wav2Vec2 | Speech emotion (4 classes) | 95M | [SUPERB](https://huggingface.co/superb/wav2vec2-base-superb-er) |
| ViT | Facial emotion (7 classes) | 86M | [trpakov](https://huggingface.co/trpakov/vit-face-expression) |
| AttentionFusion | Burnout risk (3 classes) | ~260K | Custom PyTorch |

---

## Ethical Considerations

### What This System Is
- A **self-awareness tool** that highlights emotional patterns
- An **early warning system** for burnout risk factors
- A **research prototype** demonstrating multimodal fusion

### What This System Is NOT
- A **medical diagnostic tool** — it cannot diagnose burnout, depression, or any condition
- An **HR screening tool** — it must not be used for employment decisions
- A **replacement for professional help** — if struggling, seek a qualified professional

### Known Biases
- **Cultural bias**: Emotion models are primarily trained on Western datasets
- **Gender bias**: FER-2013 and IEMOCAP have demographic imbalances
- **Acted speech**: IEMOCAP uses acted emotional speech, which differs from natural speech
- **Synthetic training**: The fusion model learns from synthetic data, not real burnout cases

### Privacy
- All processing runs locally on the user's machine
- No data is transmitted to external servers (except initial model downloads from HuggingFace)
- Audio and images are processed in temporary memory and immediately discarded

---

## Limitations

1. **Synthetic fusion training**: The attention fusion model is trained on synthetic embeddings. It encodes domain knowledge about emotion-burnout mapping but lacks validation on real clinical data.

2. **IEMOCAP limitations**: The deep voice model was fine-tuned on acted emotional speech (12 hours, 4 emotions). Real-world spontaneous speech has different characteristics.

3. **Single-moment analysis**: Burnout is a longitudinal phenomenon. A single assessment captures current state, not trend. Repeated assessments over time would be more informative.

4. **No personalisation**: The system uses population-level models. Individual baselines (some people naturally speak softly) are not accounted for.

5. **English only**: The text model is optimised for English. Non-English text will produce unreliable results.

---

## Future Improvements

| Priority | Improvement | Impact |
|----------|------------|--------|
| High | Train fusion on MELD/CMU-MOSEI aligned data | More accurate cross-modal prediction |
| High | Longitudinal tracking (trend analysis) | Catch gradual burnout onset |
| Medium | Fine-tune voice model on spontaneous speech | Better real-world accuracy |
| Medium | Confidence calibration | More reliable uncertainty estimates |
| Medium | Bias auditing across demographics | Fairer predictions |
| Low | Video stream analysis (real-time) | Continuous monitoring |
| Low | Personalised baselines | Account for individual differences |

---

## Interview Talking Points

**Q: Why attention-based fusion instead of simple averaging?**  
A: Averaging treats all modalities equally. If someone writes an emotional journal entry but has a neutral resting face, averaging dilutes the text signal. Attention learns per-sample importance — upweighting informative modalities and downweighting noisy ones.

**Q: How do you handle missing modalities?**  
A: We mask absent modalities to -inf before the softmax in the attention layer. This means they receive zero weight — mathematically equivalent to "the model only looks at what's available."

**Q: Why not train end-to-end?**  
A: Modularity. Each component (text/voice/face analyzer) uses a pretrained model with strong standalone performance. The fusion layer only needs to learn cross-modal importance, which requires far less data. We can also upgrade any single component without retraining everything.

**Q: What's the biggest limitation?**  
A: The fusion model is trained on synthetic data. The architecture is sound, but real-world accuracy depends on fine-tuning with aligned, clinically-annotated multimodal datasets — which are rare and expensive to collect.

**Q: How would you deploy this in production?**  
A: Containerise with Docker, serve the models via FastAPI (not Streamlit), add authentication, rate limiting, and logging. Move model inference to a GPU server for speed. Add longitudinal tracking via a database. Most importantly, partner with clinical psychologists for validation.

---

## License

This project is for educational and research purposes. Not intended for clinical use.
