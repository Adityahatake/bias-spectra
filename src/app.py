import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# import your filter functions
from political_filter import is_non_political, is_political


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Indian Political Bias Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Indian Political Bias Detector")
st.write(
    "Detect **political bias** in Indian news headlines using "
    "**BERT + rule-based filtering**."
)

st.markdown(
    """
**Bias Classes**
- 🟥 Left  
- 🟡 Neutral  
- 🟦 Right  

⚠️ Non-political content (weather, sports, lifestyle, etc.)  
is automatically classified as **Neutral**.
"""
)


# =========================================================
# LOAD MODEL (SAFE FOR WINDOWS)
# =========================================================

MODEL_NAME = "bert-base-multilingual-cased"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "indicbert_bias",
    "checkpoint-124"   # 🔴 CHANGE IF YOUR CHECKPOINT NUMBER IS DIFFERENT
)

LABEL_MAP = {
    0: "Left",
    1: "Neutral",
    2: "Right"
}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


# =========================================================
# INPUT
# =========================================================

headline = st.text_area(
    "Enter a news headline:",
    placeholder="Example: Supreme Court hears plea on new education policy",
    height=100
)


# =========================================================
# PREDICTION BUTTON (ONLY ONE BUTTON!)
# =========================================================

if st.button("Analyze Bias", key="analyze_bias_button"):

    if not headline.strip():
        st.warning("Please enter a headline.")
        st.stop()

    # ---------- GATE 1: NON-POLITICAL ----------
    if is_non_political(headline):
        st.success("🟢 Prediction: **Neutral (Non-Political Content)**")
        st.info("Detected topic: weather / sports / lifestyle / general information.")
        st.stop()

    # ---------- GATE 2: POLITICAL BUT UNBIASED ----------
    if not is_political(headline):
        st.success("🟡 Prediction: **Neutral (Political but Unbiased)**")
        st.info("No clear ideological framing detected.")
        st.stop()

    # ---------- GATE 3: BERT INFERENCE ----------
    with torch.no_grad():
        inputs = tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    # ---------- OUTPUT ----------
    st.success(f"🔍 Prediction: **{LABEL_MAP[pred_class]}**")

    st.subheader("Confidence Scores")
    for i, label in LABEL_MAP.items():
        st.write(f"{label}: **{probs[i] * 100:.2f}%**")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption(
    "Built using Streamlit, HuggingFace Transformers & Indian news data"
)
