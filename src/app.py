"""
BiasSpectra – Streamlit Application
====================================
Premium dark-themed UI for political bias detection in Indian news.
Uses the BiasPredictor engine for inference.
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from inference.predictor import BiasPredictor, BiasResult

# ─── Cached Predictor (must be defined before use) ───────────

@st.cache_resource
def get_predictor(model_type: str) -> BiasPredictor:
    """Cache the predictor so the model loads only once."""
    return BiasPredictor(model_type=model_type)


# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="BiasSpectra – Political Bias Detector",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.05rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    .result-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-label {
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .result-reasoning {
        text-align: center;
        color: #9ca3af;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    .conf-bar-container { margin: 0.6rem 0; }
    .conf-bar-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
        font-size: 0.85rem;
    }
    .conf-bar-track {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s ease-out;
    }
    .bar-left    { background: linear-gradient(90deg, #ef4444, #f87171); }
    .bar-neutral { background: linear-gradient(90deg, #eab308, #facc15); }
    .bar-right   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }

    .gate-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 1rem;
    }
    .gate-nonpol  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .gate-neutral { background: rgba(234,179,8,0.15);   color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
    .gate-model   { background: rgba(102,126,234,0.15); color: #667eea; border: 1px solid rgba(102,126,234,0.3); }

    .history-item {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }

    .sidebar-section-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #667eea;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ─── Color & Emoji Maps ──────────────────────────────────────
LABEL_COLORS = {"Left": "#ef4444", "Neutral": "#eab308", "Right": "#3b82f6"}
LABEL_EMOJI  = {"Left": "🟥", "Neutral": "🟡", "Right": "🟦"}
BAR_CLASSES  = {"Left": "bar-left", "Neutral": "bar-neutral", "Right": "bar-right"}

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">⚙️ Settings</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Model", ["nli", "bert", "baseline"], index=0,
        help="NLI (recommended): Zero-shot DeBERTa, understands framing. BERT: fine-tuned, legacy. Baseline: fast TF-IDF.",
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">📖 About</div>', unsafe_allow_html=True)
    st.markdown("""
    **BiasSpectra** detects political bias in Indian news
    headlines using a hybrid approach:

    1. **Gate 1** — Rule-based filter removes non-political content
    2. **Gate 2** — Political but unbiased → Neutral
    3. **Gate 3** — Zero-shot NLI tests multiple hypotheses
       about political framing and aggregates scores

    Bias classes reflect *linguistic framing*, not factual accuracy.
    """)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">📊 Bias Classes</div>', unsafe_allow_html=True)
    st.markdown("""
    🟥 **Left** – Critical / accountability-focused framing

    🟡 **Neutral** – Descriptive / informational framing

    🟦 **Right** – Supportive / pro-establishment framing
    """)

    st.markdown("---")
    st.caption("Built by Aditya Daksh · BTech CSE (AI)")


# ─── Main UI ─────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🔬 BiasSpectra</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    'ML-powered political bias detection for Indian news headlines'
    '</p>',
    unsafe_allow_html=True,
)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Input ───────────────────────────────────────────────────
headline = st.text_area(
    "Enter a news headline",
    placeholder="e.g., Supreme Court hears plea on new education policy",
    height=100,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔍  Analyze Bias", use_container_width=True, type="primary")

# ─── Prediction ──────────────────────────────────────────────
if analyze_btn:
    if not headline.strip():
        st.warning("⚠️ Please enter a headline to analyze.")
        st.stop()

    with st.spinner("Analyzing..."):
        try:
            predictor = get_predictor(model_choice)
            result: BiasResult = predictor.predict(headline)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

    # Track in history
    st.session_state.history.insert(0, {"headline": headline, "label": result.label})
    st.session_state.history = st.session_state.history[:20]

    color = LABEL_COLORS.get(result.label, "#fff")
    emoji = LABEL_EMOJI.get(result.label, "")

    gate_map = {
        "non_political":    ("Non-Political Filter", "gate-nonpol"),
        "neutral_political": ("Neutral Political",   "gate-neutral"),
        "model":            (f"ML Model ({model_choice.upper()})", "gate-model"),
    }
    gate_text, gate_class = gate_map.get(result.gate, ("Unknown", "gate-model"))

    # Build result card HTML
    card = f"""
    <div class="result-card">
        <div class="result-label" style="color:{color};">{emoji} {result.label}</div>
        <div class="result-reasoning">{result.reasoning}</div>
    """

    if result.is_model_prediction:
        for lbl in ["Left", "Neutral", "Right"]:
            conf = result.confidence.get(lbl, 0)
            pct = conf * 100
            card += f"""
            <div class="conf-bar-container">
                <div class="conf-bar-header">
                    <span>{lbl}</span>
                    <span style="color:{LABEL_COLORS[lbl]};font-weight:600;">{pct:.1f}%</span>
                </div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill {BAR_CLASSES[lbl]}" style="width:{pct}%;"></div>
                </div>
            </div>"""

    card += f"""
        <div style="text-align:center;">
            <span class="gate-badge {gate_class}">⚡ {gate_text}</span>
        </div>
    </div>"""

    st.markdown(card, unsafe_allow_html=True)


# ─── History ─────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("### 📜 Recent Analyses")
    for item in st.session_state.history[:10]:
        emoji = LABEL_EMOJI.get(item["label"], "")
        color = LABEL_COLORS.get(item["label"], "#fff")
        short = item["headline"][:80] + ("..." if len(item["headline"]) > 80 else "")
        st.markdown(
            f'<div class="history-item">'
            f'<span style="color:{color};font-weight:600;">{emoji} {item["label"]}</span>'
            f' &nbsp;·&nbsp; {short}</div>',
            unsafe_allow_html=True,
        )

# ─── Footer ──────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    'BiasSpectra · Built with Streamlit, HuggingFace Transformers & PyTorch<br>'
    'Predictions reflect linguistic framing, not factual correctness.'
    '</div>',
    unsafe_allow_html=True,
)
