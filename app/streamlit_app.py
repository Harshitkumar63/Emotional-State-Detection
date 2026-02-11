"""
Multimodal Burnout Risk Detection System v2 — Streamlit Frontend
=================================================================
A professional, production-quality web interface for burnout risk
assessment using text, voice, and facial expression inputs.

v2 upgrades:
  - Burnout risk display with confidence gauge
  - Modality contribution breakdown (attention weights)
  - Deep voice model indicator
  - Ethical disclaimer section
  - Improved UI with clearer risk framing
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Burnout Risk Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center;
        padding: 1rem 0;
        font-size: 2rem;
        font-weight: 700;
    }

    /* Risk card styling */
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        color: white;
        font-weight: 600;
    }
    .risk-low { background: linear-gradient(135deg, #27ae60, #2ecc71); }
    .risk-moderate { background: linear-gradient(135deg, #f39c12, #e67e22); }
    .risk-high { background: linear-gradient(135deg, #e74c3c, #c0392b); }
    .risk-na { background: linear-gradient(135deg, #95a5a6, #7f8c8d); }

    /* Metric cards */
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        margin: 0.3rem 0;
    }

    /* Contribution bar */
    .contrib-bar {
        height: 24px;
        border-radius: 12px;
        margin: 2px 0;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions (must be defined before they are called)
# ---------------------------------------------------------------------------

def _get_color(score: float, reverse: bool = False) -> str:
    """Get a color based on score (green for good, red for bad)."""
    if reverse:
        score = 1.0 - score
    if score > 0.7:
        return "#27ae60"
    elif score > 0.4:
        return "#f39c12"
    else:
        return "#e74c3c"


# ---------------------------------------------------------------------------
# Model loading (cached across Streamlit reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models... (first time may take a few minutes)")
def load_engine():
    """Load the BurnoutEngine with all analysers and fusion model."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.core.burnout_engine import BurnoutEngine
    from src.core.explainer import Explainer
    from src.utils.helpers import load_config

    config = load_config()
    engine = BurnoutEngine(config)
    explainer = Explainer()
    return engine, explainer, config


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "latest_state" not in st.session_state:
    st.session_state.latest_state = None
if "latest_explanation" not in st.session_state:
    st.session_state.latest_explanation = None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Assessment", "Results", "History", "About"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Burnout Risk Detection v2**\n\n"
    "Attention-based multimodal fusion for early burnout risk signals."
)
st.sidebar.divider()
st.sidebar.caption(
    "This system provides early risk signals, not medical diagnosis."
)


# =====================================================================
# Page: Assessment
# =====================================================================
if page == "Assessment":
    st.markdown('<h1 class="main-title">Burnout Risk Detection System</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color: #666;'>"
        "Provide at least one input. Text is recommended; audio and face are optional."
        "</p>",
        unsafe_allow_html=True,
    )

    # --- Input tabs ---
    tab_text, tab_voice, tab_face = st.tabs(["Text Input", "Voice Input", "Face Input"])

    # Text input
    with tab_text:
        st.markdown("**Write or paste text** (journal entry, thoughts, feedback)")
        text_input = st.text_area(
            "Text input",
            height=150,
            placeholder="I've been feeling overwhelmed at work lately. The deadlines keep piling up and I can't seem to find any motivation...",
            label_visibility="collapsed",
        )

        with st.expander("Load sample text"):
            sample_texts = {
                "High stress": "I can't take this anymore. Every day feels like a battle. I'm exhausted, stressed, and the work never ends. I don't want to go back to the office tomorrow.",
                "Moderate concern": "Work has been okay but I feel disconnected from what I'm doing. Not really excited about anything lately, just going through the motions.",
                "Positive state": "Had a great day! Finally finished the big project and feeling really motivated. Looking forward to what's next.",
            }
            selected = st.selectbox("Choose a sample:", list(sample_texts.keys()))
            if st.button("Use this sample", key="use_text_sample"):
                st.session_state["text_sample"] = sample_texts[selected]
                st.rerun()

        if "text_sample" in st.session_state:
            text_input = st.session_state.pop("text_sample")

    # Voice input
    with tab_voice:
        st.markdown("**Upload a voice recording** (.wav format)")
        audio_file = st.file_uploader(
            "Upload audio", type=["wav"], label_visibility="collapsed",
        )
        if audio_file:
            st.audio(audio_file)

    # Face input
    with tab_face:
        st.markdown("**Upload a face photo** (.jpg, .png)")
        image_file = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed",
        )
        if image_file:
            st.image(image_file, width=250)

    # --- Run analysis ---
    st.divider()

    has_input = bool(text_input) or audio_file is not None or image_file is not None

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button(
            "Run Burnout Assessment",
            type="primary",
            use_container_width=True,
            disabled=not has_input,
        )
    with col_info:
        if not has_input:
            st.info("Provide at least one input to begin.")

    if run_clicked and has_input:
        engine, explainer, config = load_engine()

        # Save audio to temp file if provided
        audio_path = None
        if audio_file:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_file.read())
                audio_path = tmp.name

        # Save image if provided
        image_input = None
        if image_file:
            from PIL import Image
            image_input = Image.open(image_file).convert("RGB")

        # Run assessment
        with st.spinner("Analyzing inputs and computing burnout risk..."):
            try:
                state = engine.assess(
                    text=text_input if text_input else None,
                    audio_path=audio_path,
                    image=image_input,
                )
                explanation = explainer.explain(state)

                st.session_state.latest_state = state
                st.session_state.latest_explanation = explanation

                # Add to history
                st.session_state.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "modalities": state.modalities_used,
                    "burnout_risk": state.burnout_risk,
                    "primary_emotion": state.primary_emotion,
                    "state_dict": state.to_dict(),
                })

                st.success("Assessment complete! Go to the **Results** tab.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")


# =====================================================================
# Page: Results
# =====================================================================
elif page == "Results":
    state = st.session_state.latest_state
    explanation = st.session_state.latest_explanation

    if state is None:
        st.info("No assessment results yet. Go to the **Assessment** page first.")
    else:
        st.markdown('<h1 class="main-title">Assessment Results</h1>', unsafe_allow_html=True)

        # --- Hero: Burnout Risk ---
        risk = state.burnout_risk
        risk_class = {
            "Low Risk": "risk-low",
            "Moderate Risk": "risk-moderate",
            "High Risk": "risk-high",
        }.get(risk, "risk-na")

        st.markdown(
            f'<div class="risk-card {risk_class}">'
            f'<h2 style="margin:0">{risk}</h2>'
            f'<p style="margin:0.3rem 0 0 0">Confidence: {state.burnout_confidence:.0%}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Burnout probability breakdown
        if state.burnout_probabilities:
            cols = st.columns(3)
            for i, (label, prob) in enumerate(state.burnout_probabilities.items()):
                with cols[i]:
                    st.metric(label, f"{prob:.0%}")

        st.divider()

        # --- Emotion + Dimensions ---
        col_emo, col_dim = st.columns(2)

        with col_emo:
            st.subheader("Emotion Breakdown")
            st.markdown(f"**Primary:** {state.primary_emotion}")

            for emotion, score in sorted(
                state.emotion_scores.items(), key=lambda x: x[1], reverse=True
            ):
                st.progress(min(score, 1.0), text=f"{emotion}: {score:.0%}")

        with col_dim:
            st.subheader("Wellness Dimensions")

            # Energy
            energy_color = _get_color(state.energy_score, reverse=False)
            st.markdown(f"**Energy:** {state.energy_level} ({state.energy_score:.0%})")
            st.progress(state.energy_score)

            # Stress
            stress_color = _get_color(state.stress_score, reverse=True)
            st.markdown(f"**Stress:** {state.stress_level} ({state.stress_score:.0%})")
            st.progress(state.stress_score)

            # Work
            st.markdown(f"**Work Inclination:** {state.work_inclination} ({state.work_score:.0%})")
            st.progress(state.work_score)

        st.divider()

        # --- Modality Contributions ---
        if state.modality_contributions:
            st.subheader("Modality Contributions (Attention Weights)")
            st.caption(
                "How much each input source influenced the burnout risk prediction. "
                "Higher = the fusion model found stronger signals in that modality."
            )

            contrib_cols = st.columns(len(state.modality_contributions))
            mod_icons = {"text": "📝", "voice": "🎤", "face": "👤"}

            for i, (mod, weight) in enumerate(
                sorted(state.modality_contributions.items(), key=lambda x: x[1], reverse=True)
            ):
                with contrib_cols[i]:
                    icon = mod_icons.get(mod, "")
                    st.metric(f"{icon} {mod.title()}", f"{weight:.0%}")

            st.markdown(explanation.get("contribution_narrative", ""))

        st.divider()

        # --- Signals (Evidence) ---
        st.subheader("Evidence & Signals")
        for narrative in explanation.get("signal_narratives", []):
            st.markdown(f"- {narrative}")

        st.divider()

        # --- Narrative Explanation ---
        st.subheader("Burnout Risk Explanation")
        st.markdown(explanation.get("burnout_narrative", ""))

        st.subheader("Summary")
        st.markdown(state.mental_summary)

        # --- Dimension Explanations ---
        with st.expander("Detailed Dimension Explanations"):
            dims = explanation.get("dimension_explanations", {})
            for dim_name, dim_text in dims.items():
                st.markdown(f"**{dim_name.title()}:** {dim_text}")

        # --- Recommendations ---
        st.subheader("Recommendations")
        for rec in state.recommendations:
            st.markdown(f"- {rec}")

        # --- Confidence ---
        st.markdown(explanation.get("confidence_note", ""))

        # --- Limitations ---
        with st.expander("Limitations"):
            for limit in explanation.get("limitations", []):
                st.markdown(f"- {limit}")

        # --- Disclaimer ---
        st.markdown(
            f'<div class="disclaimer-box">{explanation.get("disclaimer", "")}</div>',
            unsafe_allow_html=True,
        )

        # --- Download ---
        st.divider()
        json_data = state.to_json(indent=2)
        st.download_button(
            label="Download Full Report (JSON)",
            data=json_data,
            file_name=f"burnout_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# =====================================================================
# Page: History
# =====================================================================
elif page == "History":
    st.markdown('<h1 class="main-title">Assessment History</h1>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No assessments recorded yet in this session.")
    else:
        for i, entry in enumerate(reversed(st.session_state.history)):
            risk = entry["burnout_risk"]
            risk_emoji = {
                "Low Risk": "🟢", "Moderate Risk": "🟡", "High Risk": "🔴"
            }.get(risk, "⚪")

            with st.expander(
                f"{risk_emoji} #{len(st.session_state.history) - i}  |  "
                f"{risk}  |  {entry['primary_emotion']}  |  "
                f"{', '.join(entry['modalities'])}  |  "
                f"{entry['timestamp'][:19]}"
            ):
                st.json(entry["state_dict"])

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


# =====================================================================
# Page: About
# =====================================================================
elif page == "About":
    st.markdown('<h1 class="main-title">About This System</h1>', unsafe_allow_html=True)

    tab_how, tab_ethics, tab_tech = st.tabs(["How It Works", "Ethics & Privacy", "Technical Details"])

    with tab_how:
        st.markdown("""
### Architecture Overview

This system uses a **three-stage pipeline**:

**Stage 1: Per-Modality Emotion Analysis**
- **Text**: DistilRoBERTa fine-tuned on 6 emotion datasets classifies text into 7 emotions
- **Voice (Acoustic)**: Librosa extracts interpretable features (pitch, energy, tempo)
- **Voice (Deep)**: Wav2Vec2 fine-tuned on IEMOCAP for speech emotion recognition
- **Face**: Vision Transformer (ViT) fine-tuned on FER-2013 for facial emotion recognition

**Stage 2: Attention-Based Fusion**
- Embeddings from each modality (768-d) are projected to a shared space
- A cross-modal attention mechanism learns which modality to trust per-sample
- The fused representation is classified into 3 burnout risk levels

**Stage 3: Burnout Risk Assessment**
- Emotion patterns are mapped to burnout indicators
- Energy, stress, and work inclination are derived from emotion distributions
- The system generates human-readable explanations with confidence scores

### Why Attention Fusion?
Simple averaging treats all modalities equally. If someone writes an emotional journal
entry but has a neutral face (common when writing), averaging dilutes the text signal.
Attention learns to upweight the informative modality per-sample.
        """)

    with tab_ethics:
        st.markdown("""
### Ethics & Privacy

**This system is NOT a diagnostic tool.** It provides early risk signals for
self-awareness purposes only.

**Privacy:**
- All processing happens locally on your machine
- No data is sent to external servers (except for initial model downloads)
- No user data is stored, logged, or transmitted
- Audio and images are processed in temporary memory and discarded

**Limitations:**
- Emotion detection has inherent biases (cultural, gender, age)
- Facial expression analysis varies significantly across demographics
- Voice emotion models are trained primarily on acted speech (IEMOCAP)
- The burnout risk model uses synthetic training data, not clinical data

**Responsible Use:**
- Do not use this system to make employment, health, or legal decisions
- Do not use results to judge, evaluate, or penalize individuals
- Always treat results as conversation starters, not conclusions
- If you or someone you know is struggling, seek professional help
        """)

    with tab_tech:
        st.markdown("""
### Technical Stack

| Component | Technology |
|-----------|-----------|
| Text Emotion | `j-hartmann/emotion-english-distilroberta-base` (DistilRoBERTa) |
| Voice (Acoustic) | librosa (pitch, energy, tempo, spectral centroid) |
| Voice (Deep) | `superb/wav2vec2-base-superb-er` (Wav2Vec2 + IEMOCAP) |
| Face Emotion | `trpakov/vit-face-expression` (ViT + FER-2013) |
| Fusion | Custom PyTorch AttentionFusionNetwork (~260K params) |
| Framework | PyTorch, HuggingFace Transformers |
| Frontend | Streamlit |

### Fusion Model Architecture
```
text_embedding (768-d)  -> Projector (256-d) --|
voice_embedding (768-d) -> Projector (256-d) --|--> Attention --> Classifier --> Risk (3 classes)
face_embedding (768-d)  -> Projector (256-d) --|
```

### Interview Talking Points
- **Why attention fusion?** It learns per-sample modality importance and handles missing modalities naturally
- **Why not end-to-end?** Modular design allows each component to be tested, debugged, and upgraded independently
- **Why synthetic training data?** No aligned multimodal burnout dataset exists; the architecture is designed to be fine-tuned when real data becomes available
- **What would you improve?** Real MELD/CMU-MOSEI training, longitudinal tracking, confidence calibration, bias auditing
        """)


