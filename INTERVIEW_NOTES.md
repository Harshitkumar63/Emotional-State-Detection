# Interview Talking Points — Multimodal Burnout Risk Detection System v3

> Use these to explain your project in AI/ML internship interviews.
> Each answer is structured as: **what you did**, **why**, and **trade-offs**.

---

## 1. Project Overview (30-second pitch)

"I built a multimodal burnout risk detection system that analyses text, voice, and facial expressions to estimate burnout risk. It uses three pretrained emotion models, fuses their embeddings with a learned attention mechanism, tracks risk trends over time with a GRU, and presents everything in a Streamlit dashboard. It's framed as a self-awareness tool — not a clinical diagnosis."

---

## 2. Architecture Deep-Dive

### 2.1 Four-Stage Pipeline

```
Input (text / audio / image)
  |
  v
Stage 1: Per-modality emotion analysis
  - TextAnalyzer     -> DistilRoBERTa (7-class emotion)
  - VoiceAnalyzer    -> librosa acoustic features (interpretable)
  - VoiceAnalyzerDeep -> Wav2Vec2 (4-class speech emotion)
  - FaceAnalyzer     -> ViT (7-class facial emotion)
  |
  v
Stage 2: Attention-based fusion
  - Each modality produces a 768-d embedding
  - Projected to shared 256-d space via separate projectors
  - Bahdanau-style attention computes per-sample modality weights
  - Weighted sum -> classifier -> 3-class burnout risk
  |
  v
Stage 3: Temporal trend prediction
  - GRU processes sequence of past assessments (10-d features x N timesteps)
  - Outputs: Improving / Stable / Worsening
  |
  v
Stage 4: Enterprise dashboard
  - SQLite persistent storage
  - Aggregated stats, trend charts, escalation alerts
  - CSV/JSON report export
```

### 2.2 Why Four Stages?

**Single Responsibility:** Each stage is independently testable and upgradable. You can swap the face model from ViT to a CNN without touching the fusion layer. You can replace SQLite with PostgreSQL without changing the temporal model.

**Separation of concerns:** Emotion analysis is deterministic (pretrained models). Fusion is learned (attention network). Trend analysis is sequential (GRU). Dashboard is presentation logic.

---

## 3. Why Attention-Based Fusion?

### The Problem with Simpler Approaches

| Approach | Limitation |
|----------|-----------|
| **Averaging** | Treats all modalities equally. If someone writes an emotional journal but has a neutral resting face, averaging dilutes the text signal. |
| **Concatenation** | Input dimension grows linearly (768 x 3 = 2304). No learned weighting. More parameters to train. |
| **Fixed weights** | Same weights for every input. Can't adapt to which modality is most informative for a specific person. |

### Why Attention Works

- **Per-sample importance**: The model learns to upweight informative modalities and downweight noisy ones, *for each input separately*.
- **Missing modality handling**: Absent modalities are masked to `-inf` before softmax, so they receive exactly zero weight. No imputation or zero-padding needed.
- **Explainability**: The attention weights are interpretable — you can show users "65% of this prediction came from your text, 35% from your voice."
- **Small footprint**: ~260K parameters, trains in seconds on CPU.

### Implementation Details

```
Per-modality projectors: 768 -> 256 (with LayerNorm + GELU + Dropout)
Attention scorer: 256 -> 64 -> 1 (Bahdanau-style, Tanh activation)
Classifier: 256 -> 128 -> 3 (GELU + Dropout)
```

---

## 4. Handling Missing Modalities

This is a critical design decision because users may provide only text, or text + voice, or all three.

### Approach: Attention Masking

1. If a modality is absent, its attention score is set to `-inf`
2. After softmax, `-inf` maps to exactly 0 weight
3. The remaining modalities' weights re-normalise to sum to 1
4. Mathematically equivalent to "the model only looks at what's available"

### Why Not Zero-Padding?

Zero-padding the missing embedding would create a non-zero attention score for an empty signal. The model might learn to associate the zero vector with a specific class, leading to pathological behaviour when modalities are absent.

### Why Not Separate Models per Combination?

That would require 7 models (3 single + 3 pair + 1 triple). Attention masking handles all combinations with a single model, which is far more elegant and maintainable.

---

## 5. Temporal Model (GRU) Rationale

### Why Track Trends?

Burnout is a *process*, not a single event. A single "Moderate Risk" assessment might be a bad day. But if stress has been rising and energy declining over two weeks, that's a clear burnout trajectory.

### Why GRU Over LSTM?

- **Fewer parameters**: GRU has 2 gates vs LSTM's 3. With small training data, fewer parameters mean less overfitting.
- **Comparable performance**: On short sequences (< 50 timesteps), GRU matches LSTM accuracy. Our sequences are 3-10 assessments long.
- **Faster training**: ~30% fewer operations per timestep.

### Architecture

```
Input: (batch, seq_len, 10) — 10 features per timestep
  [energy, stress, work, risk_encoded, anger, disgust, fear, joy, neutral, sadness]

LayerNorm -> GRU (10 -> 64, 2 layers) -> classifier (64 -> 32 -> 3)

Output: Improving (0) / Stable (1) / Worsening (2) + confidence
```

### Feature Encoding

Each assessment is reduced to a 10-d vector that captures the key dimensions of burnout. This is more compact and noise-resistant than feeding raw emotion distributions directly.

---

## 6. Confidence Calibration

### The Problem

Raw softmax probabilities from neural networks are systematically overconfident (Guo et al., 2017). A model might output 95% confidence when it's essentially guessing.

### Our Approach: Temperature Scaling + MC Dropout

**Temperature Scaling:**
- Divides logits by temperature T before softmax
- T = 1.5 (our default) softens overconfident predictions
- Example: raw 92% confidence becomes ~78% after scaling
- Learned post-hoc — doesn't change the model's predictions, only calibrates probabilities

**Monte Carlo Dropout:**
- Runs N forward passes with dropout *enabled* at inference time
- Measures prediction variance across passes
- High variance = model is uncertain (different dropout masks produce different answers)
- This approximates Bayesian inference (Gal & Ghahramani, 2016)

### Combined Reliability Score

```python
is_reliable = (calibrated_confidence > 0.5) AND (mc_uncertainty < 0.5)
```

This gives users two signals: "how confident is the model?" and "how consistent is it?"

---

## 7. Model Trade-Offs

| Decision | Alternative | Why We Chose This |
|----------|------------|-------------------|
| DistilRoBERTa for text | BERT-base | 40% smaller, 60% faster, within 3% accuracy |
| Wav2Vec2 for voice | HuBERT | Wav2Vec2 has IEMOCAP fine-tuned checkpoint readily available |
| ViT for face | ResNet-FER | ViT achieves higher accuracy on FER-2013 (~73% vs ~68%) |
| Bahdanau attention | Transformer self-attention | Simpler, fewer parameters, sufficient for 3 modalities |
| GRU for temporal | LSTM / Transformer | Fewer parameters, comparable on short sequences |
| SQLite for storage | PostgreSQL / MongoDB | Zero-config, built into Python, perfect for local apps |
| Synthetic training data | Real clinical data | Clinical data is expensive and rare; synthetic data demonstrates the architecture |
| Temperature = 1.5 | Learned temperature | No validation set available; 1.5 is a reasonable conservative default |

---

## 8. Models Used

| Model | Parameters | Training Data | Task |
|-------|-----------|---------------|------|
| DistilRoBERTa (j-hartmann) | 82M | GoEmotions, ISEAR, etc. (6 datasets) | 7-class text emotion |
| Wav2Vec2 (SUPERB) | 95M | IEMOCAP (12h acted speech) | 4-class speech emotion |
| ViT (trpakov) | 86M | FER-2013 (35K images) | 7-class facial emotion |
| AttentionFusionNetwork | ~260K | Synthetic (3K samples) | 3-class burnout risk |
| TemporalBurnoutPredictor | ~25K | Synthetic (1.5K sequences) | 3-class trend prediction |

**Total model parameters:** ~263M (dominated by pretrained models, which are frozen).
**Trainable parameters:** ~285K (only the fusion and temporal models are trained).

---

## 9. Limitations (Be Honest)

### Critical Limitations

1. **Synthetic training data**: Both the fusion and temporal models are trained on synthetic embeddings/sequences. The architecture is sound, but real-world accuracy is unknown without clinical validation.

2. **Acted speech**: The Wav2Vec2 model is fine-tuned on IEMOCAP (acted emotional speech). Real spontaneous speech has different prosodic characteristics.

3. **No personalisation**: Population-level models. Some people naturally speak softly (low energy ≠ exhaustion for them). Individual baselines would improve accuracy.

4. **Cultural bias**: All three models are primarily trained on Western datasets. Emotion expression varies across cultures — a neutral face in one culture may indicate different emotions in another.

5. **English only**: The text model is optimised for English. Non-English input produces unreliable results.

6. **Single-point fusion**: Each assessment is independent. The temporal model helps, but true longitudinal modelling would require larger datasets.

---

## 10. Future Improvements

### High Priority

| Improvement | Impact | Difficulty |
|------------|--------|-----------|
| Train fusion on CMU-MOSEI or MELD aligned data | Real cross-modal learning | Medium (data preprocessing) |
| Fine-tune Wav2Vec2 on spontaneous speech (e.g., RAVDESS) | More natural voice understanding | Medium |
| Add personal baselines via few-shot calibration | Account for individual differences | High |

### Medium Priority

| Improvement | Impact | Difficulty |
|------------|--------|-----------|
| Bias auditing across demographics | Fairer predictions | Medium |
| Video stream analysis (continuous) | Real-time monitoring | Medium |
| Containerise with Docker + FastAPI | Production deployment | Low |
| Multilingual text model (XLM-RoBERTa) | Non-English support | Low |

### Research Directions

- **Cross-attention fusion**: Instead of Bahdanau attention over independent embeddings, use cross-attention where modalities attend to each other (like in Perceiver IO).
- **Contrastive learning**: Pre-train embeddings with contrastive loss across modalities before fusion.
- **Graph neural networks**: Model relationships between emotion dimensions as a graph.

---

## 11. Ethics Talking Points

**Q: Isn't this dangerous — predicting mental health from AI?**

A: "We frame this explicitly as *risk detection*, not diagnosis. The system displays confidence levels and explains *why* it made each prediction. High-risk results include suggestions to seek professional help. We never store raw data by default, and all processing is local. The biggest ethical risk is someone treating this as medical advice — which is why every screen includes a disclaimer."

**Q: How do you handle bias?**

A: "We acknowledge the bias openly. The text model is trained on English data, the face model on Western faces (FER-2013), and the voice model on acted American English speech (IEMOCAP). In production, I would add bias auditing — running the system on balanced demographic test sets and reporting performance gaps."

**Q: Should companies use this to monitor employees?**

A: "No. The system is designed for individual self-awareness, not surveillance. Using it for employment decisions would be ethically wrong and potentially illegal. The architecture doesn't include multi-user tracking for this reason — the 'enterprise dashboard' aggregates anonymised self-reports."

---

## 12. System Design Questions

**Q: How would you deploy this in production?**

A: "Containerise with Docker. Replace Streamlit with FastAPI for the backend and React for the frontend. Serve models via ONNX Runtime for 3-5x inference speedup. Add authentication (OAuth2), rate limiting, and structured logging. Move to PostgreSQL for multi-user storage. Most importantly — validate with clinical psychologists before any real deployment."

**Q: How do you handle concurrent users?**

A: "Currently, SQLite handles single-user access. For production, I'd switch to PostgreSQL with connection pooling (asyncpg). Model inference would run on a GPU server behind a load balancer. Each request gets its own inference context — no shared state between users."

**Q: What's the inference latency?**

A: "On CPU, text analysis takes ~200ms, face analysis ~300ms, voice analysis ~500ms. Fusion is <10ms. Total for a single modality: ~200-500ms. All three: ~800ms-1s. This is acceptable for a web app. For real-time video, I'd quantise the ViT model and process every 5th frame."

---

## 13. Code Architecture Questions

**Q: Why config-driven design?**

A: "Every tuneable parameter lives in `config.yaml`. This means I can change model names, thresholds, hidden dimensions, and file paths without touching Python code. It also makes the system reproducible — someone can recreate my exact setup from the config file."

**Q: Why separate acoustic and deep voice analysis?**

A: "The acoustic analyser (librosa) gives interpretable features — RMS energy, pitch, tempo. These are explainable to users. The deep model (Wav2Vec2) gives better emotion classification but is a black box. By running both, we get accurate predictions *and* can explain them in human terms. The fusion uses the deep model's embedding; the explainability uses the acoustic features."

**Q: Why not train end-to-end?**

A: "Modularity. Each component uses a pretrained model with strong standalone performance. The fusion layer only needs to learn cross-modal importance, which requires far less data. I can upgrade any single component (e.g., swap ViT for a newer face model) without retraining everything. End-to-end training requires large aligned multimodal datasets, which are rare for burnout."

---

*Last updated: February 2026*
