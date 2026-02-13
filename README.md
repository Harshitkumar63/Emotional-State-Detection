# Multimodal Burnout Risk Detection System v3

> **Detect early burnout risk using text, voice, and facial expressions via attention-based multimodal fusion, GRU temporal tracking, and enterprise dashboard.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)](https://streamlit.io/)

---

## Problem Statement

**Burnout is a growing workplace crisis.** The WHO classifies it as an "occupational phenomenon" affecting millions globally. Yet most people don't recognise the signs until it's too late — by then, recovery takes months.

This system provides **early burnout risk signals** by analysing a person's current emotional state through three modalities:

- **Text** (journal entries, written thoughts, feedback)
- **Voice** (tone, pitch, energy, speaking pace)
- **Face** (facial expressions and micro-expressions)

It does **NOT** provide medical diagnosis. It is a self-awareness tool that helps individuals notice warning patterns before burnout sets in.

### Why Multimodal > Unimodal?

| Scenario | Text Only | Voice Only | Face Only | Multimodal |
|----------|-----------|-----------|-----------|-----------|
| Person writes "I'm fine" but looks exhausted | Misses it | N/A | Catches fatigue | Catches contradiction |
| Monotone voice but positive text | Reports positive | Reports low energy | N/A | Weighs both signals |
| Neutral face but anxious writing | N/A | N/A | Reports neutral | Detects hidden anxiety |

A single modality captures one dimension. **Multimodal fusion** detects contradictions and builds a fuller picture — which is exactly how burnout manifests (people mask it in some channels but not others).

### Version History

| Version | What Changed |
|---------|-------------|
| v1 | Basic emotion detection per modality, simple averaging |
| v2 | Attention-based fusion, deep voice model, burnout risk framing |
| **v3** | **GRU temporal tracking, enterprise dashboard, confidence calibration, real-time capture, persistent storage** |

---

## Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │              BURNOUT ENGINE (v3)                      │
                    │                                                      │
    Text ──────►    │  TextAnalyzer (DistilRoBERTa)                        │
                    │    ├── Emotion scores (7 classes)                     │
                    │    └── [CLS] embedding (768-d) ──────────┐           │
                    │                                          │           │
    Voice ─────►    │  VoiceAnalyzer (librosa baseline)        │           │
                    │    └── Acoustic indicators                │           │
                    │  VoiceAnalyzerDeep (Wav2Vec2)            │           │
                    │    ├── Emotion scores (4 classes)         │           │
                    │    └── Mean-pooled embedding ─────────────┤           │
                    │                                          │           │
    Face ──────►    │  FaceAnalyzer (ViT)                      │           │
                    │    ├── Emotion scores (7 classes)         │           │
                    │    └── [CLS] embedding (768-d) ──────────┤           │
                    │                                          ▼           │
                    │              ┌───────────────────────────────────┐    │
                    │              │  AttentionFusionNetwork            │    │
                    │              │  ├─ Per-modality projectors        │    │
                    │              │  │   (768 → 256, LayerNorm+GELU)  │    │
                    │              │  ├─ Bahdanau attention scorer      │    │
                    │              │  │   (learns per-sample weights)   │    │
                    │              │  └─ Classifier (256 → 3)          │    │
                    │              │                                    │    │
                    │              │  Output: Burnout Risk              │    │
                    │              │  (Low / Moderate / High)           │    │
                    │              │  + modality contribution weights   │    │
                    │              └───────────────────────────────────┘    │
                    │                                                      │
                    │  StateEngine: emotion → energy/stress/work dims      │
                    │  Explainer:   signals → human-readable narratives    │
                    └──────────────────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────────────┐
                    │         TEMPORAL TRACKING (v3 NEW)                    │
                    │                                                      │
                    │  SessionStore (SQLite)                                │
                    │    └── Persistent assessment history                  │
                    │                                                      │
                    │  TemporalBurnoutPredictor (GRU)                      │
                    │    ├── Input: sequence of past assessments            │
                    │    └── Output: Improving / Stable / Worsening        │
                    │                                                      │
                    │  ConfidenceCalibration                                │
                    │    ├── Temperature scaling (soften overconfidence)    │
                    │    └── MC Dropout (uncertainty estimation)            │
                    └──────────────────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────────────┐
                    │         ENTERPRISE DASHBOARD (v3 NEW)                 │
                    │                                                      │
                    │  DashboardAnalytics                                   │
                    │    ├── Aggregated stats & risk distribution           │
                    │    ├── Escalation detection (rising stress alerts)    │
                    │    ├── CSV/JSON report export                         │
                    │    └── Trend visualisation (line & bar charts)        │
                    └──────────────────────────────────────────────────────┘
```

### Why This Architecture?

1. **Modular**: Each component can be tested, debugged, and upgraded independently
2. **Explainable**: Attention weights show which modality the model trusts per-sample
3. **Robust**: Missing modalities handled via attention masking (not zero-padding)
4. **Temporal**: GRU captures burnout as a *process*, not a single snapshot
5. **Practical**: Works on CPU, no GPU required
6. **Persistent**: SQLite stores history for longitudinal analysis

---

## Project Structure

```
Multimodal Burnout Detection/
├── app/
│   └── streamlit_app.py              # Streamlit frontend (7 pages)
├── config/
│   └── config.yaml                   # All configuration (no hardcoding)
├── src/
│   ├── analyzers/
│   │   ├── text_analyzer.py          # DistilRoBERTa emotion + embedding
│   │   ├── voice_analyzer.py         # Acoustic baseline (librosa)
│   │   ├── voice_analyzer_deep.py    # Wav2Vec2 speech emotion model
│   │   └── face_analyzer.py          # ViT facial emotion + embedding
│   ├── core/
│   │   ├── emotional_state.py        # Data model (EmotionalState dataclass)
│   │   ├── state_engine.py           # Emotion → dimension mapping
│   │   ├── burnout_engine.py         # Main orchestrator
│   │   └── explainer.py              # Human-readable explanations
│   ├── fusion/
│   │   ├── attention_fusion.py       # PyTorch attention fusion network
│   │   └── confidence.py             # Temperature scaling + MC Dropout
│   ├── temporal/
│   │   ├── temporal_model.py         # GRU-based trend predictor
│   │   └── session_store.py          # SQLite persistent storage
│   ├── dashboard/
│   │   └── analytics.py              # Aggregated stats, alerts, exports
│   ├── preprocessing/
│   │   ├── audio_preprocessor.py     # Silence trimming, normalisation
│   │   ├── face_preprocessor.py      # MTCNN face detection + cropping
│   │   └── text_preprocessor.py      # Language detection, cleaning
│   └── utils/
│       └── helpers.py                # Config loading, logging
├── scripts/
│   ├── inference.py                  # CLI assessment
│   ├── train_fusion.py               # Train the attention fusion model
│   ├── train_temporal.py             # Train the GRU temporal model
│   └── generate_samples.py           # Generate test data
├── checkpoints/
│   ├── fusion_model.pt               # Trained fusion weights
│   └── temporal_model.pt             # Trained GRU weights
├── data/
│   ├── samples/                      # Sample test files
│   └── session_history.db            # SQLite database (auto-created)
├── tests/
│   └── test_analyzers.py             # Unit tests
├── requirements.txt
├── README.md
└── INTERVIEW_NOTES.md                # Interview preparation guide
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

### 2. Train the Models

```bash
# Train the attention fusion network (~30 seconds on CPU)
python scripts/train_fusion.py

# Train the GRU temporal model (~20 seconds on CPU)
python scripts/train_temporal.py
```

Both models train on synthetic data and save checkpoints to `checkpoints/`.

### 3. Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app has 7 pages:
- **Assessment** — standard text/audio/image input
- **Real-Time** — webcam capture + microphone recording
- **Results** — burnout risk, emotions, attention weights, explanations
- **Trends** — GRU trend prediction + time-series charts
- **Dashboard** — aggregated stats, alerts, CSV/JSON export
- **History** — browse past assessments
- **About** — architecture, ethics, technical details

### 4. CLI Usage

```bash
# Text only
python scripts/inference.py --text "I feel exhausted and overwhelmed"

# Voice only
python scripts/inference.py --voice path/to/audio.wav

# All modalities
python scripts/inference.py --text "..." --voice audio.wav --face face.jpg
```

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

---

## How It Works

### Stage 1: Per-Modality Emotion Analysis

**Text Analysis** — Uses `j-hartmann/emotion-english-distilroberta-base`, a DistilRoBERTa model fine-tuned on 6 emotion datasets (GoEmotions, ISEAR, etc.) for 7-class emotion classification. Also extracts the [CLS] token embedding (768-d) for fusion.

**Voice Analysis (Acoustic Baseline)** — Extracts interpretable psychoacoustic features via librosa:
- RMS energy → vocal animation level
- Pitch (F0) → emotional arousal
- Pitch variability → expressiveness
- Tempo → speaking pace
- Speech ratio → pause frequency

**Voice Analysis (Deep Model)** — Uses `superb/wav2vec2-base-superb-er`, a Wav2Vec2 model fine-tuned on IEMOCAP for 4-class speech emotion recognition. Provides both emotion classification and 768-d embeddings.

*Why two voice analysers?* The acoustic baseline gives interpretable features for explainability ("your pitch variability is low, suggesting flat affect"). The deep model gives more accurate emotion predictions. We use both — the deep embeddings for fusion, the acoustic features for explanations.

**Face Analysis** — Uses `trpakov/vit-face-expression`, a Vision Transformer fine-tuned on FER-2013 for 7-class facial emotion recognition.

### Stage 2: Attention-Based Fusion

The `AttentionFusionNetwork` (~260K parameters):
1. Projects each modality's 768-d embedding to a shared 256-d space via separate projectors
2. Computes Bahdanau-style attention scores for each modality
3. Masks absent modalities to `-inf` (they receive zero weight after softmax)
4. Creates a weighted sum of projected embeddings
5. Classifies the fused representation into 3 burnout risk levels

**Why attention over averaging?**
- Learns which modality to trust **per-sample** (not globally)
- Naturally handles missing modalities via masking
- Provides interpretable contribution weights for explainability

### Stage 3: Temporal Trend Prediction (v3 NEW)

A GRU model processes a sequence of past assessments to predict whether burnout risk is **Improving**, **Stable**, or **Worsening**.

*Why temporal matters:* A single "Moderate Risk" might be a bad day. But if stress has been rising for two weeks while energy declines — that's a burnout trajectory. The GRU captures this pattern.

Architecture: `LayerNorm → GRU(10→64, 2 layers) → Classifier(64→32→3)`

Each timestep is encoded as a 10-d feature vector: `[energy, stress, work, risk, anger, disgust, fear, joy, neutral, sadness]`.

### Stage 4: Confidence Calibration (v3 NEW)

**Temperature Scaling** — Divides logits by T=1.5 before softmax, softening overconfident predictions. This is the standard post-hoc calibration method (Guo et al., 2017).

**Monte Carlo Dropout** — Runs multiple forward passes with dropout enabled at inference time. High prediction variance indicates model uncertainty (Gal & Ghahramani, 2016).

Combined, these give users two honest signals: "how confident is the model?" and "how consistent are its predictions?"

---

## Models Used

| Model | Task | Parameters | Source |
|-------|------|-----------|--------|
| DistilRoBERTa | Text emotion (7 classes) | 82M | [j-hartmann](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) |
| Wav2Vec2 | Speech emotion (4 classes) | 95M | [SUPERB](https://huggingface.co/superb/wav2vec2-base-superb-er) |
| ViT | Facial emotion (7 classes) | 86M | [trpakov](https://huggingface.co/trpakov/vit-face-expression) |
| AttentionFusion | Burnout risk (3 classes) | ~260K | Custom PyTorch |
| GRU Temporal | Trend (3 classes) | ~25K | Custom PyTorch |

**Total**: ~263M parameters (pretrained models are frozen; only ~285K are trained).

---

## Design Trade-Offs

| Decision | Alternative | Why We Chose This |
|----------|------------|-------------------|
| DistilRoBERTa for text | BERT-base | 40% smaller, 60% faster, within 3% accuracy |
| Wav2Vec2 for voice | HuBERT | IEMOCAP fine-tuned checkpoint readily available |
| ViT for face | ResNet-FER | Higher accuracy on FER-2013 (~73% vs ~68%) |
| Bahdanau attention | Transformer self-attention | Simpler, fewer params, sufficient for 3 modalities |
| GRU for temporal | LSTM | Fewer parameters, comparable on short sequences |
| SQLite | PostgreSQL | Zero-config, built into Python, perfect for local apps |
| Synthetic training | Real clinical data | Clinical data is rare; synthetic data demonstrates architecture |

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
- **Synthetic training**: Fusion and temporal models learn from synthetic data, not real burnout cases

### Privacy
- All processing runs locally on the user's machine
- No data is transmitted to external servers (except initial model downloads from HuggingFace)
- Audio and images are processed in temporary memory and immediately discarded
- Assessment history stored in local SQLite database — user can delete anytime

---

## Limitations

1. **Synthetic fusion training**: The attention fusion model is trained on synthetic embeddings. It encodes domain knowledge about emotion-burnout mapping but lacks validation on real clinical data.

2. **IEMOCAP limitations**: The deep voice model was fine-tuned on acted emotional speech (12 hours, 4 emotions). Real-world spontaneous speech has different characteristics.

3. **No personalisation**: The system uses population-level models. Individual baselines (some people naturally speak softly) are not accounted for.

4. **English only**: The text model is optimised for English. Non-English text will produce unreliable results.

5. **Synthetic temporal data**: The GRU trend model is trained on generated sequences. Real burnout trajectories are more complex and variable.

6. **No clinical validation**: This system has not been validated against clinical burnout assessments (e.g., MBI — Maslach Burnout Inventory).

---

## Future Improvements

| Priority | Improvement | Impact |
|----------|------------|--------|
| High | Train fusion on MELD/CMU-MOSEI aligned data | Real cross-modal prediction |
| High | Clinical validation with MBI scores | Establish real-world accuracy |
| Medium | Fine-tune voice model on spontaneous speech | Better real-world accuracy |
| Medium | Bias auditing across demographics | Fairer predictions |
| Medium | Personal baselines via few-shot calibration | Account for individual differences |
| Low | Multilingual text model (XLM-RoBERTa) | Non-English support |
| Low | Video stream analysis (real-time) | Continuous monitoring |
| Low | Containerise with Docker + FastAPI | Production deployment |

---

## Interview Talking Points

See [INTERVIEW_NOTES.md](INTERVIEW_NOTES.md) for detailed interview preparation covering:
- Architecture deep-dive
- Why attention-based fusion
- Handling missing modalities
- GRU rationale
- Confidence calibration
- Model trade-offs
- Limitations and ethics
- System design questions

**Quick answers:**

**Q: Why attention-based fusion instead of simple averaging?**
A: Averaging treats all modalities equally. If someone writes an emotional journal entry but has a neutral resting face, averaging dilutes the text signal. Attention learns per-sample importance — upweighting informative modalities and downweighting noisy ones.

**Q: How do you handle missing modalities?**
A: We mask absent modalities to -inf before the softmax in the attention layer. This means they receive zero weight — mathematically equivalent to "the model only looks at what's available."

**Q: Why not train end-to-end?**
A: Modularity. Each component uses a pretrained model with strong standalone performance. The fusion layer only needs to learn cross-modal importance, requiring far less data. We can upgrade any single component without retraining everything.

**Q: What's the biggest limitation?**
A: The fusion and temporal models are trained on synthetic data. The architecture is sound, but real-world accuracy depends on fine-tuning with aligned, clinically-annotated multimodal datasets — which are rare and expensive to collect.

**Q: How would you deploy this in production?**
A: Containerise with Docker, serve models via FastAPI (not Streamlit), add authentication and rate limiting. Move inference to GPU server for speed. Switch to PostgreSQL for multi-user storage. Partner with clinical psychologists for validation.

---

## License

This project is for educational and research purposes. Not intended for clinical use.
