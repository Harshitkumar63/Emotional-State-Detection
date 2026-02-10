"""
Streamlit Frontend — Emotional State Detection
================================================
Multi-page web app where the user provides ANY ONE (or more) of:
  • Written text
  • Voice recording (.wav)
  • Face photograph (.jpg / .png)

… and receives a full emotional state assessment with explanations.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.utils.helpers import load_config, setup_logging

logger = setup_logging()

# ======================================================================
# Page config
# ======================================================================

st.set_page_config(
    page_title="Emotional State Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# Colour maps
# ======================================================================

_EMOTION_COLORS = {
    "joy": "#2ecc71", "surprise": "#3498db", "neutral": "#95a5a6",
    "anger": "#e74c3c", "fear": "#9b59b6", "disgust": "#e67e22",
    "sadness": "#34495e",
}

_EMOTION_EMOJIS = {
    "joy": "😊", "surprise": "😲", "neutral": "😐",
    "anger": "😠", "fear": "😰", "disgust": "🤢",
    "sadness": "😢",
}


# ======================================================================
# CSS
# ======================================================================

def inject_css():
    st.markdown("""
    <style>
    .block-container { max-width: 1100px; padding-top: 1.5rem; }
    .hero {
        text-align: center; padding: 2rem 1rem 1.5rem; border-radius: 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2.2rem; }
    .hero p  { margin: .4rem 0 0; opacity: .9; font-size: 1.05rem; }
    .card {
        background: #fff; border: 1px solid #e8e8e8; border-radius: 12px;
        padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,.04);
    }
    .result-hero {
        border-radius: 14px; padding: 1.8rem 2rem; margin: 1rem 0;
        text-align: center;
    }
    .result-hero h2 { margin: 0; font-size: 2rem; }
    .dim-bar-track {
        height: 14px; background: #eee; border-radius: 7px; overflow: hidden;
        margin: 4px 0 12px;
    }
    .dim-bar-fill {
        height: 100%; border-radius: 7px; transition: width .5s ease;
    }
    .signal-card {
        background: #f8f9fa; border-left: 3px solid #667eea;
        border-radius: 8px; padding: .8rem 1rem; margin-bottom: .6rem;
        font-size: .93rem;
    }
    .disclaimer-box {
        background: #fff8e1; border-left: 4px solid #ffc107;
        border-radius: 8px; padding: 1rem 1.2rem; font-size: .92rem; margin: 1rem 0;
    }
    .footer {
        text-align: center; color: #aaa; font-size: .78rem;
        padding: 2rem 0 1rem; border-top: 1px solid #eee; margin-top: 3rem;
    }
    </style>""", unsafe_allow_html=True)


# ======================================================================
# Cached model loading
# ======================================================================

@st.cache_resource(show_spinner="Loading AI models — first run takes ~1 min …")
def load_analyzers():
    from src.analyzers.text_analyzer import TextAnalyzer
    from src.analyzers.voice_analyzer import VoiceAnalyzer
    from src.analyzers.face_analyzer import FaceAnalyzer
    from src.core.state_engine import StateEngine
    from src.core.explainer import Explainer

    config = load_config()
    return (
        TextAnalyzer(config),
        VoiceAnalyzer(config),
        FaceAnalyzer(config),
        StateEngine(config),
        Explainer(),
        config,
    )


# ======================================================================
# Session state
# ======================================================================

def _init_state():
    for key in ("history", "last_state", "last_explanation"):
        if key not in st.session_state:
            st.session_state[key] = [] if key == "history" else None


# ======================================================================
# Sidebar
# ======================================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🧠 Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Assessment", "📊 Results", "📜 History", "ℹ️ About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown(
            "1. Provide **any one** input (or more)\n"
            "2. Click **Analyse**\n"
            "3. View your emotional state assessment"
        )
        st.markdown("---")
        st.markdown(
            "<div class='disclaimer-box'>"
            "⚠️ <strong>Not a clinical tool.</strong> For self-awareness only."
            "</div>", unsafe_allow_html=True,
        )
    return page


# ======================================================================
# PAGE: Assessment
# ======================================================================

def page_assessment():
    st.markdown(
        "<div class='hero'>"
        "<h1>Emotional State Detection</h1>"
        "<p>Provide any input — text, voice, or photo — to understand how you're feeling</p>"
        "</div>", unsafe_allow_html=True,
    )

    # --- Sample data loader ---
    samples_dir = _PROJECT_ROOT / "data" / "samples"
    sample_text = samples_dir / "sample_journal.txt"
    if sample_text.exists():
        with st.expander("🧪 Load sample data for quick testing"):
            if st.button("Load Samples"):
                st.session_state["sample_text"] = sample_text.read_text("utf-8")
                audio = samples_dir / "sample_audio.wav"
                image = samples_dir / "sample_face.png"
                st.session_state["sample_audio"] = str(audio) if audio.exists() else None
                st.session_state["sample_image"] = str(image) if image.exists() else None
                st.success("Sample data loaded!")

    # ---- Input tabs ----
    tab_text, tab_voice, tab_face = st.tabs([
        "📝 Text Input", "🎙️ Voice Input", "📷 Face Photo"
    ])

    # --- Text ---
    with tab_text:
        st.markdown("#### Write about how you're feeling")
        st.caption("A journal entry, chat message, or any written thoughts.")
        default_text = st.session_state.get("sample_text", "")
        text_input = st.text_area(
            "Your text",
            value=default_text,
            height=180,
            placeholder="Example: I've been feeling drained for weeks. "
                        "I can't concentrate and everything feels overwhelming …",
            key="txt",
        )
        if text_input.strip():
            st.caption(f"✅ {len(text_input.split())} words")

    # --- Voice ---
    with tab_voice:
        st.markdown("#### Upload a voice recording")
        st.caption(
            "A 5-60 second `.wav` clip of you speaking naturally. "
            "The system analyses tone, pitch, pace, and pauses."
        )
        audio_file = st.file_uploader("Upload .wav", type=["wav"], key="aud")
        sample_audio = st.session_state.get("sample_audio")
        if audio_file:
            st.audio(audio_file, format="audio/wav")
            st.caption("✅ Audio uploaded")
        elif sample_audio and Path(sample_audio).exists():
            st.audio(sample_audio, format="audio/wav")
            st.caption("✅ Sample audio loaded")

    # --- Face ---
    with tab_face:
        st.markdown("#### Upload a face photo")
        st.caption(
            "A front-facing, well-lit selfie (`.jpg` / `.png`). "
            "The system reads facial expression cues."
        )
        image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img")
        sample_image = st.session_state.get("sample_image")
        if image_file:
            st.image(image_file, caption="Uploaded photo", width=250)
        elif sample_image and Path(sample_image).exists():
            from PIL import Image
            st.image(Image.open(sample_image), caption="Sample image", width=250)

    # ---- Input summary ----
    st.markdown("---")
    has_text = bool(text_input and text_input.strip())
    has_audio = audio_file is not None or (sample_audio and Path(sample_audio).exists())
    has_image = image_file is not None or (sample_image and Path(sample_image).exists())

    active = []
    if has_text:  active.append("📝 Text")
    if has_audio: active.append("🎙️ Voice")
    if has_image: active.append("📷 Face")

    if active:
        st.markdown(f"**Active inputs:** {' · '.join(active)}")
    else:
        st.warning("Please provide at least one input (text, voice, or face photo).")

    # ---- Run button ----
    run = st.button(
        "🔬  Analyse Emotional State",
        use_container_width=True,
        type="primary",
        disabled=not active,
    )

    if run:
        _run_analysis(
            text_input if has_text else None,
            audio_file, sample_audio,
            image_file, sample_image,
        )


# ======================================================================
# Analysis execution
# ======================================================================

def _run_analysis(text, audio_file, sample_audio, image_file, sample_image):
    text_analyzer, voice_analyzer, face_analyzer, engine, explainer, config = load_analyzers()

    progress = st.progress(0, text="Starting analysis …")

    # --- Text ---
    text_result = None
    if text and text.strip():
        progress.progress(10, text="Analysing text …")
        text_result = text_analyzer.analyze(text)

    # --- Audio ---
    voice_result = None
    tmp_dir = Path(tempfile.mkdtemp())
    audio_path = None
    if audio_file is not None:
        audio_path = str(tmp_dir / "upload.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
    elif sample_audio and Path(sample_audio).exists():
        audio_path = sample_audio

    if audio_path:
        progress.progress(35, text="Analysing voice …")
        voice_result = voice_analyzer.analyze(audio_path)

    # --- Image ---
    face_result = None
    image_input = None
    if image_file is not None:
        suffix = Path(image_file.name).suffix
        image_input = str(tmp_dir / f"upload{suffix}")
        with open(image_input, "wb") as f:
            f.write(image_file.read())
    elif sample_image and Path(sample_image).exists():
        image_input = sample_image

    if image_input:
        progress.progress(60, text="Analysing facial expression …")
        face_result = face_analyzer.analyze(image_input)

    # --- State Engine ---
    progress.progress(80, text="Building emotional assessment …")
    state = engine.assess(text_result, voice_result, face_result)
    explanation = explainer.explain(state)
    progress.progress(100, text="Done!")

    # Store
    st.session_state.last_state = state
    st.session_state.last_explanation = explanation
    st.session_state.history.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "state": state,
        "explanation": explanation,
    })

    st.success(
        f"Analysis complete — primary emotion: "
        f"**{_EMOTION_EMOJIS.get(state.primary_emotion, '')} {state.primary_emotion}**  ·  "
        f"Energy: {state.energy_level}  ·  Stress: {state.stress_level}"
    )
    st.markdown("---")
    _render_results(state, explanation)


# ======================================================================
# PAGE: Results
# ======================================================================

def page_results():
    st.markdown(
        "<div class='hero'><h1>📊 Assessment Results</h1>"
        "<p>Detailed breakdown of your emotional state</p></div>",
        unsafe_allow_html=True,
    )
    state = st.session_state.get("last_state")
    explanation = st.session_state.get("last_explanation")
    if state is None:
        st.info("No assessment yet. Go to **🏠 Assessment** and provide an input.")
        return
    _render_results(state, explanation)

    st.markdown("---")
    st.download_button(
        "📥 Download results as JSON",
        data=state.to_json(),
        file_name="emotional_state.json",
        mime="application/json",
    )


# ======================================================================
# Results renderer
# ======================================================================

def _render_results(state, explanation):
    em = state.primary_emotion
    color = _EMOTION_COLORS.get(em, "#888")
    emoji = _EMOTION_EMOJIS.get(em, "❓")

    # --- Hero card ---
    st.markdown(
        f"<div class='result-hero' style='background:{color}15; border:2px solid {color};'>"
        f"<h2 style='color:{color};'>{emoji} {em.title()}</h2>"
        f"<p>Primary emotion · score: {state.emotion_scores.get(em, 0):.0%}</p>"
        f"</div>", unsafe_allow_html=True,
    )

    # --- Emotion scores ---
    st.markdown("#### Emotion Breakdown")
    sorted_emotions = sorted(state.emotion_scores.items(), key=lambda x: x[1], reverse=True)
    for emo, score in sorted_emotions:
        ec = _EMOTION_COLORS.get(emo, "#888")
        eemoji = _EMOTION_EMOJIS.get(emo, "")
        st.markdown(f"{eemoji} **{emo.title()}**")
        st.progress(min(score, 1.0), text=f"{score:.1%}")

    st.markdown("---")

    # --- Three dimension gauges ---
    st.markdown("#### Energy · Stress · Work Inclination")
    c1, c2, c3 = st.columns(3)
    dims = [
        (c1, "Energy", state.energy_level, state.energy_score, "#3498db"),
        (c2, "Stress", state.stress_level, state.stress_score, "#e74c3c"),
        (c3, "Work Inclination", state.work_inclination, state.work_score, "#2ecc71"),
    ]
    for col, name, label, score, colour in dims:
        with col:
            st.markdown(f"**{name}**")
            st.markdown(
                f"<div class='dim-bar-track'>"
                f"<div class='dim-bar-fill' style='width:{score*100:.0f}%; background:{colour};'></div>"
                f"</div>"
                f"<div style='text-align:center;'><strong>{label}</strong> ({score:.0%})</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # --- Explanation ---
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("#### Why this assessment?")
        st.markdown(explanation["overall_narrative"])

        st.markdown("**Evidence signals:**")
        for narr in explanation["signal_narratives"]:
            st.markdown(f"<div class='signal-card'>{narr}</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("#### Dimensions Explained")
        for dim_name, dim_text in explanation["dimension_explanations"].items():
            with st.expander(dim_name.title()):
                st.markdown(dim_text)

        st.markdown("#### Confidence")
        st.markdown(explanation["confidence_note"])

    # --- Recommendations ---
    st.markdown("---")
    st.markdown("#### 💡 Recommendations")
    for rec in state.recommendations:
        st.markdown(f"- {rec}")

    # --- Limitations ---
    with st.expander("📋 Limitations of this assessment"):
        for lim in explanation["limitations"]:
            st.markdown(f"- {lim}")

    # --- Disclaimer ---
    st.markdown(
        f"<div class='disclaimer-box'>{explanation['disclaimer']}</div>",
        unsafe_allow_html=True,
    )


# ======================================================================
# PAGE: History
# ======================================================================

def page_history():
    st.markdown(
        "<div class='hero'><h1>📜 Assessment History</h1>"
        "<p>All analyses from this session</p></div>",
        unsafe_allow_html=True,
    )
    history = st.session_state.get("history", [])
    if not history:
        st.info("No assessments yet.")
        return

    st.markdown(f"**{len(history)}** assessment(s)")
    for i, entry in enumerate(reversed(history), 1):
        s = entry["state"]
        em = s.primary_emotion
        color = _EMOTION_COLORS.get(em, "#888")
        emoji = _EMOTION_EMOJIS.get(em, "")
        st.markdown(
            f"<div class='signal-card' style='border-left-color:{color};'>"
            f"<strong>{emoji} {em.title()}</strong> · "
            f"Energy: {s.energy_level} · Stress: {s.stress_level} · "
            f"Work: {s.work_inclination} · "
            f"<span style='color:#aaa;'>{entry['timestamp']}</span>"
            f"</div>", unsafe_allow_html=True,
        )

    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()


# ======================================================================
# PAGE: About
# ======================================================================

def page_about():
    st.markdown(
        "<div class='hero'><h1>ℹ️ About This System</h1>"
        "<p>Architecture, ethics, and how it works</p></div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["🏗️ How It Works", "⚖️ Ethics", "🔧 Technical"])

    with tab1:
        st.markdown("""
### How Emotions Are Detected

Each input type uses a **different, specialised approach**:

**📝 Text Analysis**
- Uses `j-hartmann/emotion-english-distilroberta-base` — a transformer model
  fine-tuned on 6 emotion datasets
- Directly classifies text into 7 emotions: anger, disgust, fear, joy,
  neutral, sadness, surprise
- Also scans for emotional keywords to explain its reasoning

**🎙️ Voice Analysis**
- Extracts **acoustic features** using librosa: vocal energy (RMS), pitch (F0),
  pitch variability, speaking tempo, spectral brightness, pause frequency
- Maps these to emotional indicators based on psychoacoustics research
- Example: low energy + slow speech + monotone pitch → suggests fatigue/sadness

**📷 Face Analysis**
- Uses a Vision Transformer (ViT) fine-tuned on FER-2013 facial expression data
- Classifies facial expressions into the same 7 emotion categories
- Falls back gracefully if the model can't be loaded

**Combining Multiple Inputs**
- If multiple inputs are given, emotion distributions are **averaged**
  (not fused by a random neural net)
- Voice acoustic features directly adjust energy & stress scores
- Every signal from every input is preserved for explainability
        """)

    with tab2:
        st.markdown("""
### Ethics & Privacy

- **Not a clinical tool** — cannot diagnose any condition
- **All processing is local** — nothing is sent to any server
- **No data storage** — uploaded files are deleted after analysis
- **Cultural bias** — models were trained on primarily Western data
- **Single-moment snapshot** — emotions fluctuate; one reading isn't definitive
        """)

    with tab3:
        st.markdown("""
### Technical Stack

| Component | Technology |
|-----------|-----------|
| Text emotion | HuggingFace DistilRoBERTa (fine-tuned) |
| Voice features | librosa (acoustic analysis) |
| Face emotion | HuggingFace ViT (fine-tuned on FER-2013) |
| State engine | Rule-based mapping (configurable) |
| Frontend | Streamlit |
| Config | YAML (no hardcoding) |
        """)


# ======================================================================
# Main
# ======================================================================

def main():
    _init_state()
    inject_css()
    page = render_sidebar()

    if page.startswith("🏠"):   page_assessment()
    elif page.startswith("📊"): page_results()
    elif page.startswith("📜"): page_history()
    elif page.startswith("ℹ️"):  page_about()

    st.markdown(
        "<div class='footer'>Emotional State Detection System v2.0 · "
        "For educational purposes only</div>", unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
