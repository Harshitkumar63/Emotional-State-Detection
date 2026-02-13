"""
Multimodal Burnout Risk Detection System v3 — Streamlit Frontend
=================================================================
Production-grade, demo-ready web application with:

  - Standard assessment (text + file upload)
  - Real-time capture (webcam + microphone)
  - Temporal trend tracking (GRU predictions)
  - Enterprise dashboard (charts, exports, alerts)
  - Ethical safeguards throughout
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
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
    .main-title { text-align:center; padding:1rem 0; font-size:2rem; font-weight:700; }
    .risk-card { padding:1.5rem; border-radius:12px; text-align:center; margin:0.5rem 0;
                 color:white; font-weight:600; }
    .risk-low { background:linear-gradient(135deg,#27ae60,#2ecc71); }
    .risk-moderate { background:linear-gradient(135deg,#f39c12,#e67e22); }
    .risk-high { background:linear-gradient(135deg,#e74c3c,#c0392b); }
    .risk-na { background:linear-gradient(135deg,#95a5a6,#7f8c8d); }
    .trend-improving { background:linear-gradient(135deg,#27ae60,#2ecc71); }
    .trend-stable { background:linear-gradient(135deg,#3498db,#2980b9); }
    .trend-worsening { background:linear-gradient(135deg,#e74c3c,#c0392b); }
    .disclaimer-box { background:#fff3cd; border:1px solid #ffc107; border-radius:8px;
                      padding:1rem; margin:1rem 0; font-size:0.85rem; }
    .alert-high { background:#f8d7da; border:1px solid #f5c6cb; border-radius:8px;
                  padding:0.8rem; margin:0.5rem 0; }
    .alert-medium { background:#fff3cd; border:1px solid #ffc107; border-radius:8px;
                    padding:0.8rem; margin:0.5rem 0; }
    .alert-info { background:#d1ecf1; border:1px solid #bee5eb; border-radius:8px;
                  padding:0.8rem; margin:0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_color(score: float, reverse: bool = False) -> str:
    if reverse:
        score = 1.0 - score
    if score > 0.7:
        return "#27ae60"
    elif score > 0.4:
        return "#f39c12"
    else:
        return "#e74c3c"


def _risk_class(risk: str) -> str:
    return {"Low Risk": "risk-low", "Moderate Risk": "risk-moderate",
            "High Risk": "risk-high"}.get(risk, "risk-na")


# ---------------------------------------------------------------------------
# Lazy model loading (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models... (first time takes a few minutes)")
def load_engine():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.core.burnout_engine import BurnoutEngine
    from src.core.explainer import Explainer
    from src.utils.helpers import load_config

    config = load_config()
    engine = BurnoutEngine(config)
    explainer = Explainer()
    return engine, explainer, config


@st.cache_resource(show_spinner="Loading temporal model...")
def load_temporal():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.temporal.temporal_model import load_temporal_model
    return load_temporal_model()


@st.cache_resource
def load_session_store():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.temporal.session_store import SessionStore
    return SessionStore()


@st.cache_resource
def load_dashboard_analytics():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.dashboard.analytics import DashboardAnalytics
    store = load_session_store()
    return DashboardAnalytics(store)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
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
    ["Assessment", "Real-Time", "Results", "Trends", "Dashboard", "History", "About"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown("**Burnout Risk Detection v3**")
st.sidebar.caption("Attention fusion + temporal tracking + enterprise dashboard")
st.sidebar.divider()
st.sidebar.markdown(
    '<div class="disclaimer-box" style="font-size:0.75rem;">'
    "This system provides early risk signals, <b>not</b> medical diagnosis."
    "</div>",
    unsafe_allow_html=True,
)


# =====================================================================
# Helper: run assessment and save to store
# =====================================================================
def _run_assessment(text=None, audio_path=None, image_input=None):
    """Run the burnout engine and persist the result."""
    engine, explainer, config = load_engine()
    store = load_session_store()

    state = engine.assess(text=text, audio_path=audio_path, image=image_input)
    explanation = explainer.explain(state)

    st.session_state.latest_state = state
    st.session_state.latest_explanation = explanation

    # Persist to SQLite
    store.save_assessment(state.to_dict())

    return state, explanation


# =====================================================================
# Page: Assessment
# =====================================================================
if page == "Assessment":
    st.markdown('<h1 class="main-title">Burnout Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#666;'>"
                "Provide at least one input. Text is recommended.</p>",
                unsafe_allow_html=True)

    tab_text, tab_voice, tab_face = st.tabs(["Text Input", "Voice Input", "Face Input"])

    with tab_text:
        st.markdown("**Write or paste text** (journal entry, thoughts, feedback)")
        text_input = st.text_area(
            "Text input", height=150,
            placeholder="I've been feeling overwhelmed at work lately...",
            label_visibility="collapsed",
        )
        with st.expander("Load sample text"):
            samples = {
                "High stress": "I can't take this anymore. Every day feels like a battle. I'm exhausted, stressed, and the work never ends. I don't want to go back to the office tomorrow.",
                "Moderate concern": "Work has been okay but I feel disconnected from what I'm doing. Not really excited about anything lately, just going through the motions.",
                "Positive state": "Had a great day! Finally finished the big project and feeling really motivated. Looking forward to what's next.",
            }
            sel = st.selectbox("Choose a sample:", list(samples.keys()))
            if st.button("Use this sample", key="use_sample"):
                st.session_state["_sample_text"] = samples[sel]
                st.rerun()
        if "_sample_text" in st.session_state:
            text_input = st.session_state.pop("_sample_text")

    with tab_voice:
        st.markdown("**Upload a voice recording** (.wav)")
        audio_file = st.file_uploader("Upload audio", type=["wav"], label_visibility="collapsed")
        if audio_file:
            st.audio(audio_file)

    with tab_face:
        st.markdown("**Upload a face photo** (.jpg, .png)")
        image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"],
                                       label_visibility="collapsed")
        if image_file:
            st.image(image_file, width=250)

    st.divider()
    has_input = bool(text_input) or audio_file is not None or image_file is not None

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("Run Burnout Assessment", type="primary",
                                use_container_width=True, disabled=not has_input)
    with col_info:
        if not has_input:
            st.info("Provide at least one input to begin.")

    if run_clicked and has_input:
        audio_path = None
        if audio_file:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_file.read())
                audio_path = tmp.name
        image_input = None
        if image_file:
            from PIL import Image
            image_input = Image.open(image_file).convert("RGB")

        with st.spinner("Analyzing inputs and computing burnout risk..."):
            try:
                state, explanation = _run_assessment(
                    text=text_input if text_input else None,
                    audio_path=audio_path,
                    image_input=image_input,
                )
                st.success("Assessment complete! Go to **Results** or **Trends**.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")


# =====================================================================
# Page: Real-Time Capture
# =====================================================================
elif page == "Real-Time":
    st.markdown('<h1 class="main-title">Real-Time Capture</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#666;'>"
                "Use your webcam or microphone for instant assessment.</p>",
                unsafe_allow_html=True)

    rt_tab_cam, rt_tab_mic = st.tabs(["Webcam Capture", "Microphone"])

    with rt_tab_cam:
        st.markdown("**Take a photo with your webcam** for facial emotion analysis.")
        cam_image = st.camera_input("Capture face photo")

        if cam_image is not None:
            st.image(cam_image, caption="Captured photo", width=300)
            if st.button("Analyze Face", type="primary", key="rt_face"):
                from PIL import Image
                pil_image = Image.open(cam_image).convert("RGB")
                with st.spinner("Running facial emotion analysis..."):
                    try:
                        state, explanation = _run_assessment(image_input=pil_image)
                        st.success("Done! Go to **Results** to see the assessment.")
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    with rt_tab_mic:
        st.markdown("**Record a voice sample** for speech emotion analysis.")

        # Try to use streamlit-audiorec if installed
        try:
            from st_audiorec import st_audiorec
            st.markdown("Click the microphone button below to start recording. "
                        "Click again to stop. Then click **Analyze Voice**.")
            audio_bytes = st_audiorec()
            if audio_bytes is not None:
                st.audio(audio_bytes, format="audio/wav")
                if st.button("Analyze Voice", type="primary", key="rt_voice"):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_bytes)
                        audio_path = tmp.name
                    with st.spinner("Running speech emotion analysis..."):
                        try:
                            state, explanation = _run_assessment(audio_path=audio_path)
                            st.success("Done! Go to **Results** to see the assessment.")
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
            else:
                st.caption("No audio recorded yet. Use the recorder above.")
        except ImportError:
            st.warning(
                "Microphone recording requires the `streamlit-audiorec` package.\n\n"
                "Install it by running:\n"
                "```\npip install streamlit-audiorec\n```\n\n"
                "Or upload a .wav file in the **Assessment** tab instead."
            )


# =====================================================================
# Page: Results
# =====================================================================
elif page == "Results":
    state = st.session_state.latest_state
    explanation = st.session_state.latest_explanation

    if state is None:
        st.info("No results yet. Run an assessment first.")
    else:
        st.markdown('<h1 class="main-title">Assessment Results</h1>', unsafe_allow_html=True)

        # Hero: Burnout Risk
        risk = state.burnout_risk
        st.markdown(
            f'<div class="risk-card {_risk_class(risk)}">'
            f'<h2 style="margin:0">{risk}</h2>'
            f'<p style="margin:0.3rem 0 0 0">Confidence: {state.burnout_confidence:.0%}</p>'
            f'</div>', unsafe_allow_html=True)

        if state.burnout_probabilities:
            cols = st.columns(3)
            for i, (label, prob) in enumerate(state.burnout_probabilities.items()):
                with cols[i]:
                    st.metric(label, f"{prob:.0%}")

        st.divider()

        # Emotion + Dimensions
        col_emo, col_dim = st.columns(2)
        with col_emo:
            st.subheader("Emotion Breakdown")
            st.markdown(f"**Primary:** {state.primary_emotion}")
            for emo, score in sorted(state.emotion_scores.items(),
                                     key=lambda x: x[1], reverse=True):
                st.progress(min(score, 1.0), text=f"{emo}: {score:.0%}")

        with col_dim:
            st.subheader("Wellness Dimensions")
            st.markdown(f"**Energy:** {state.energy_level} ({state.energy_score:.0%})")
            st.progress(state.energy_score)
            st.markdown(f"**Stress:** {state.stress_level} ({state.stress_score:.0%})")
            st.progress(state.stress_score)
            st.markdown(f"**Work Inclination:** {state.work_inclination} ({state.work_score:.0%})")
            st.progress(state.work_score)

        st.divider()

        # Modality Contributions
        if state.modality_contributions:
            st.subheader("Modality Contributions (Attention Weights)")
            st.caption("How much each input influenced the prediction.")
            icons = {"text": "📝", "voice": "🎤", "face": "👤"}
            contrib_cols = st.columns(len(state.modality_contributions))
            for i, (mod, wt) in enumerate(
                sorted(state.modality_contributions.items(), key=lambda x: x[1], reverse=True)
            ):
                with contrib_cols[i]:
                    st.metric(f"{icons.get(mod, '')} {mod.title()}", f"{wt:.0%}")
            st.markdown(explanation.get("contribution_narrative", ""))
            st.divider()

        # Evidence
        st.subheader("Evidence & Signals")
        for n in explanation.get("signal_narratives", []):
            st.markdown(f"- {n}")

        st.divider()
        st.subheader("Burnout Risk Explanation")
        st.markdown(explanation.get("burnout_narrative", ""))

        st.subheader("Summary")
        st.markdown(state.mental_summary)

        with st.expander("Detailed Dimension Explanations"):
            for name, text in explanation.get("dimension_explanations", {}).items():
                st.markdown(f"**{name.title()}:** {text}")

        st.subheader("Recommendations")
        for rec in state.recommendations:
            st.markdown(f"- {rec}")

        st.markdown(explanation.get("confidence_note", ""))

        with st.expander("Limitations"):
            for lim in explanation.get("limitations", []):
                st.markdown(f"- {lim}")

        st.markdown(
            f'<div class="disclaimer-box">{explanation.get("disclaimer", "")}</div>',
            unsafe_allow_html=True)

        st.divider()
        st.download_button(
            "Download Full Report (JSON)", state.to_json(indent=2),
            file_name=f"burnout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json")


# =====================================================================
# Page: Trends (Temporal Analysis)
# =====================================================================
elif page == "Trends":
    st.markdown('<h1 class="main-title">Burnout Risk Trends</h1>', unsafe_allow_html=True)

    store = load_session_store()
    n_records = store.count()

    if n_records < 2:
        st.info(f"Need at least 2 assessments for trends. You have {n_records}. "
                "Run more assessments first.")
    else:
        st.markdown(f"**{n_records} assessments** recorded.")

        # GRU Trend Prediction
        if n_records >= 3:
            temporal_model = load_temporal()
            feature_seq = store.get_feature_sequence(seq_length=10)

            if len(feature_seq) >= 3:
                trend_result = temporal_model.predict_trend(feature_seq)
                trend_label = trend_result["trend_label"]
                trend_conf = trend_result["trend_confidence"]
                trend_class = {
                    "Improving": "trend-improving",
                    "Stable": "trend-stable",
                    "Worsening": "trend-worsening",
                }.get(trend_label, "risk-na")

                st.markdown(
                    f'<div class="risk-card {trend_class}">'
                    f'<h2 style="margin:0">Trend: {trend_label}</h2>'
                    f'<p style="margin:0.3rem 0 0 0">'
                    f'Confidence: {trend_conf:.0%}</p></div>',
                    unsafe_allow_html=True)

                if trend_result.get("trend_probabilities"):
                    cols = st.columns(3)
                    for i, (lbl, p) in enumerate(trend_result["trend_probabilities"].items()):
                        with cols[i]:
                            st.metric(lbl, f"{p:.0%}")

                if trend_result.get("note"):
                    st.caption(trend_result["note"])

                st.divider()

        # Time-series charts
        trend_data = store.get_trend_data(days=30)
        if trend_data:
            import pandas as pd

            df = pd.DataFrame(trend_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

            st.subheader("Stress & Energy Over Time")
            st.line_chart(df[["stress_score", "energy_score"]], color=["#e74c3c", "#27ae60"])

            st.subheader("Work Inclination Over Time")
            st.line_chart(df[["work_score"]], color=["#3498db"])

            st.subheader("Risk Level History")
            risk_map = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2}
            df["risk_numeric"] = df["burnout_risk"].map(risk_map)
            st.bar_chart(df[["risk_numeric"]])
            st.caption("0 = Low Risk, 1 = Moderate Risk, 2 = High Risk")


# =====================================================================
# Page: Dashboard (Enterprise)
# =====================================================================
elif page == "Dashboard":
    st.markdown('<h1 class="main-title">Enterprise Dashboard</h1>', unsafe_allow_html=True)

    analytics = load_dashboard_analytics()
    store = load_session_store()

    days = st.selectbox("Time period:", [7, 14, 30, 90], index=2)
    overview = analytics.get_overview(days=days)
    stats = overview["stats"]

    if stats["total_assessments"] == 0:
        st.info("No assessment data yet. Run some assessments first.")
    else:
        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Assessments", stats["total_assessments"])
        c2.metric("Avg Energy", f"{stats['avg_energy']:.0%}")
        c3.metric("Avg Stress", f"{stats['avg_stress']:.0%}")
        c4.metric("Most Common Emotion", stats["most_common_emotion"].title())

        st.divider()

        # Alerts
        st.subheader("Alerts & Warnings")
        for alert in overview["alerts"]:
            sev = alert["severity"]
            st.markdown(
                f'<div class="alert-{sev}">'
                f'<b>{"⚠️" if sev == "high" else "💡" if sev == "medium" else "ℹ️"} '
                f'{alert["message"]}</b><br>'
                f'<small>{alert["suggestion"]}</small></div>',
                unsafe_allow_html=True)

        st.divider()

        # Risk distribution
        st.subheader("Risk Distribution")
        risk_dist = overview["risk_distribution"]
        if risk_dist:
            import pandas as pd
            risk_df = pd.DataFrame(
                list(risk_dist.items()), columns=["Risk Level", "Count"]
            )
            st.bar_chart(risk_df.set_index("Risk Level"))

        st.divider()

        # Export
        st.subheader("Export Reports")
        col_csv, col_json = st.columns(2)
        with col_csv:
            csv_data = analytics.export_csv(days=days)
            st.download_button(
                "Download CSV Report", csv_data,
                file_name=f"burnout_report_{days}d.csv", mime="text/csv")
        with col_json:
            json_data = analytics.export_report(days=days)
            st.download_button(
                "Download JSON Report", json_data,
                file_name=f"burnout_report_{days}d.json", mime="application/json")


# =====================================================================
# Page: History
# =====================================================================
elif page == "History":
    st.markdown('<h1 class="main-title">Assessment History</h1>', unsafe_allow_html=True)

    store = load_session_store()
    records = store.get_history(limit=50)

    if not records:
        st.info("No assessments recorded yet.")
    else:
        st.markdown(f"Showing last **{len(records)}** assessments.")
        for i, entry in enumerate(records):
            risk = entry.get("burnout_risk", "N/A")
            emoji = {"Low Risk": "🟢", "Moderate Risk": "🟡",
                     "High Risk": "🔴"}.get(risk, "⚪")
            ts = entry.get("timestamp", "")[:19]
            emo = entry.get("primary_emotion", "")

            with st.expander(f"{emoji} {risk}  |  {emo}  |  {ts}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Energy", f"{entry.get('energy_score', 0):.0%}")
                c2.metric("Stress", f"{entry.get('stress_score', 0):.0%}")
                c3.metric("Work", f"{entry.get('work_score', 0):.0%}")

        st.divider()
        if st.button("Clear All History", type="secondary"):
            store.clear_history()
            st.rerun()


# =====================================================================
# Page: About
# =====================================================================
elif page == "About":
    st.markdown('<h1 class="main-title">About This System</h1>', unsafe_allow_html=True)

    tab_how, tab_ethics, tab_tech = st.tabs(
        ["How It Works", "Ethics & Privacy", "Technical Details"])

    with tab_how:
        st.markdown("""
### Architecture (v3)

**Stage 1: Per-Modality Emotion Analysis**
- **Text**: DistilRoBERTa fine-tuned on 6 emotion datasets (7 emotions)
- **Voice (Acoustic)**: Librosa extracts interpretable features
- **Voice (Deep)**: Wav2Vec2 fine-tuned on IEMOCAP (4 emotions)
- **Face**: ViT fine-tuned on FER-2013 (7 emotions)

**Stage 2: Attention-Based Fusion**
- 768-d embeddings projected to shared 256-d space
- Cross-modal attention learns per-sample modality importance
- Classifies into 3 burnout risk levels
- Confidence calibration via temperature scaling + MC Dropout

**Stage 3: Temporal Trend Prediction (NEW)**
- GRU model processes sequence of past assessments
- Predicts risk trend: Improving / Stable / Worsening
- Enables early detection of burnout progression

**Stage 4: Enterprise Dashboard (NEW)**
- Persistent SQLite storage for assessment history
- Aggregated statistics, trend charts, escalation alerts
- CSV/JSON report export
        """)

    with tab_ethics:
        st.markdown("""
### Ethics & Privacy

**This system is NOT a diagnostic tool.** It provides early risk signals only.

**Privacy:**
- All processing happens locally - no cloud, no tracking
- Data stored in local SQLite database (user can delete anytime)
- Audio and images processed in memory and discarded
- No personal identifiers collected

**Responsible Use:**
- Do NOT use for employment, health, or legal decisions
- Do NOT use to judge or penalize individuals
- Always treat results as conversation starters, not conclusions
- If struggling, please seek professional help

**Known Biases:**
- Models trained primarily on Western datasets
- IEMOCAP uses acted (not natural) emotional speech
- FER-2013 has demographic imbalances
- Fusion model trained on synthetic data
        """)

    with tab_tech:
        st.markdown("""
### Technical Stack

| Component | Technology |
|-----------|-----------|
| Text Emotion | DistilRoBERTa (j-hartmann) |
| Voice (Acoustic) | librosa features |
| Voice (Deep) | Wav2Vec2 (SUPERB/IEMOCAP) |
| Face Emotion | ViT (FER-2013) |
| Fusion | Attention network (~260K params) |
| Temporal | GRU (2-layer, 64-d hidden) |
| Calibration | Temperature scaling + MC Dropout |
| Storage | SQLite (local, persistent) |
| Dashboard | Streamlit + Pandas |
| Framework | PyTorch, HuggingFace Transformers |

### New in v3
- **Real-time webcam capture** via `st.camera_input()`
- **GRU temporal predictor** for burnout trend forecasting
- **Enterprise dashboard** with alerts, charts, CSV/JSON export
- **Confidence calibration** (temperature scaling + MC Dropout)
- **Persistent SQLite storage** for longitudinal tracking
- **Escalation detection** comparing recent vs. older assessments
        """)
