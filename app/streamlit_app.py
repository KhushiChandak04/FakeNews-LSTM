"""
streamlit_app.py - Fake News Detection Web Interface

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import re
import json
import pickle
import time

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("models", "fake_news_lstm.keras")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")
SCORES_PATH = os.path.join("models", "scores.json")
MAX_SEQUENCE_LEN = 300
TRANSFORMER_MODEL_ID = os.getenv("TRANSFORMER_MODEL_ID", "jy46604790/Fake-News-Bert-Detect")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Serif:wght@500;700&display=swap');

    :root {
        --paper: #f4f6fb;
        --ink: #101828;
        --accent: #1d4ed8;
        --muted: #475467;
        --card: #ffffff;
        --ok: #047857;
        --warn: #b42318;
        --line: #d0d5dd;
    }

    .stApp {
        background: radial-gradient(900px 420px at 15% 0%, #e6edff 0%, var(--paper) 48%),
                    linear-gradient(180deg, #f6f8ff 0%, #f3f5fb 100%);
        color: var(--ink);
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        color: var(--ink) !important;
    }

    .main * {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--ink);
    }

    .hero-wrap {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 20px 22px;
        background: linear-gradient(120deg, #f8faff 0%, #ffffff 100%);
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.08);
        animation: reveal 0.5s ease;
    }

    .hero-title {
        font-family: 'IBM Plex Serif', serif;
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0;
        color: var(--ink);
        letter-spacing: 0.2px;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 1.02rem;
        margin: 6px 0 16px;
    }

    .chip-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 8px;
    }
    .chip {
        border: 1px solid var(--line);
        background: #f8faff;
        color: #0f172a;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        animation: reveal 0.65s ease;
    }

    .surface {
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--card);
        padding: 18px;
        box-shadow: 0 8px 22px rgba(16, 24, 40, 0.06);
    }

    .result-card {
        border-radius: 16px;
        padding: 26px 22px;
        text-align: center;
        margin: 12px 0;
        animation: reveal 0.5s ease;
    }

    @keyframes reveal {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .result-real {
        background: linear-gradient(135deg, #047857 0%, #065f46 100%);
        color: white;
    }
    .result-fake {
        background: linear-gradient(135deg, #b42318 0%, #912018 100%);
        color: white;
    }
    .result-uncertain {
        background: linear-gradient(135deg, #475467 0%, #344054 100%);
        color: white;
    }
    .result-label { font-size: 1.9rem; font-weight: 700; margin: 0; }
    .result-conf  { font-size: 1.05rem; opacity: 0.93; margin-top: 4px; }

    .metrics-row {
        display: flex;
        gap: 12px;
        margin: 14px 0 4px;
        flex-wrap: wrap;
    }
    .metric-box {
        background: #f8faff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
        min-width: 112px;
    }
    .metric-val { font-size: 1.35rem; font-weight: 700; color: #0f172a; }
    .metric-lbl { font-size: 0.72rem; color: var(--muted); margin-top: 2px; }

    .section-title {
        font-family: 'IBM Plex Serif', serif;
        font-size: 1.35rem;
        margin-bottom: 10px;
        color: #0f172a;
    }

    .note {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 8px;
    }

    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.78rem;
        padding: 20px 0 2px;
    }

    /* Streamlit native widget contrast fixes */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #eaf0ff;
        color: #1e293b !important;
        border: 1px solid #c7d7fe;
        border-radius: 10px;
        padding: 8px 14px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #1d4ed8 !important;
        color: #ffffff !important;
        border-color: #1d4ed8 !important;
    }

    .stTextArea textarea,
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #98a2b3 !important;
    }

    .stTextArea label,
    .stSelectbox label,
    .stMetric label,
    .stMarkdown,
    p,
    span,
    li,
    h1,
    h2,
    h3,
    h4 {
        color: #0f172a !important;
    }

    .stButton > button {
        background: #1d4ed8 !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        font-weight: 700;
    }
    .stButton > button:hover {
        background: #1e40af !important;
        border-color: #1e40af !important;
    }

    .stMetric {
        background: #ffffff;
        border: 1px solid #d0d5dd;
        border-radius: 12px;
        padding: 10px 12px;
    }

    .stAlert {
        color: #0f172a !important;
    }

    @media (max-width: 900px) {
        .hero-title { font-size: 1.8rem; }
        .result-label { font-size: 1.55rem; }
    }
</style>
"""


# ─── Text cleaning ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Cached model loading ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


@st.cache_resource
def load_transformer_fallback():
    """Load transformer classifier for out-of-domain fallback."""
    try:
        from transformers import pipeline
        clf = pipeline(
            "text-classification",
            model=TRANSFORMER_MODEL_ID,
            tokenizer=TRANSFORMER_MODEL_ID,
            truncation=True,
        )
        return clf, None
    except Exception as exc:  # pragma: no cover - depends on runtime env/network
        return None, str(exc)


@st.cache_data
def load_scores():
    if not os.path.exists(SCORES_PATH):
        return {}
    with open(SCORES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Prediction ──────────────────────────────────────────────────────────────
def _lstm_probability(text: str, model, tokenizer):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    tokens = seq[0]
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")
    prob = model.predict(padded, verbose=0)[0][0]

    oov_id = tokenizer.word_index.get("<OOV>")
    token_count = len(tokens)
    oov_ratio = 0.0
    if token_count > 0 and oov_id is not None:
        oov_ratio = sum(1 for t in tokens if t == oov_id) / token_count

    return float(prob), token_count, float(oov_ratio), cleaned


def _transformer_real_probability(text: str, transformer_clf):
    if transformer_clf is None:
        return None, None, None

    out = transformer_clf(text[:4000], truncation=True)[0]
    label = str(out.get("label", "")).strip().lower()
    score = float(out.get("score", 0.5))

    # Most fake-news classifiers expose labels like fake/real or LABEL_0/LABEL_1.
    if "real" in label or "true" in label:
        p_real = score
    elif "fake" in label or "false" in label:
        p_real = 1.0 - score
    elif "label_1" in label:
        p_real = score
    elif "label_0" in label:
        p_real = 1.0 - score
    else:
        p_real = 0.5

    return float(p_real), label, score


def _final_label(prob_real: float):
    if prob_real >= 0.60:
        return "Real", float(prob_real)
    if prob_real <= 0.40:
        return "Fake", float(1.0 - prob_real)
    return "Uncertain", float(1.0 - abs(prob_real - 0.5) * 2.0)


def predict_news(text: str, model, tokenizer, transformer_clf=None) -> tuple:
    start = time.time()
    lstm_prob_real, token_count, oov_ratio, cleaned = _lstm_probability(text, model, tokenizer)
    tf_prob_real, tf_raw_label, tf_raw_score = _transformer_real_probability(cleaned, transformer_clf)

    in_domain = max(0.0, min(1.0, 1.0 - oov_ratio)) * min(1.0, token_count / 40.0)
    if tf_prob_real is None:
        lstm_weight = 1.0
        final_prob_real = lstm_prob_real
    else:
        # LSTM dominates in-domain; transformer gains weight as text moves out-of-domain.
        lstm_weight = max(0.20, min(0.90, 0.20 + 0.70 * in_domain))
        final_prob_real = lstm_weight * lstm_prob_real + (1.0 - lstm_weight) * tf_prob_real

    label, confidence = _final_label(final_prob_real)
    latency = time.time() - start

    return {
        "label": label,
        "confidence": confidence,
        "latency": latency,
        "word_count": len(cleaned.split()),
        "cleaned": cleaned,
        "oov_ratio": float(oov_ratio),
        "lstm_prob_real": float(lstm_prob_real),
        "tf_prob_real": tf_prob_real,
        "lstm_weight": float(lstm_weight),
        "final_prob_real": float(final_prob_real),
        "tf_raw_label": tf_raw_label,
        "tf_raw_score": tf_raw_score,
    }


# ─── App ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Fake News Detection System",
        page_icon="News",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    scores = load_scores()

    st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
    st.markdown('<p class="hero-title">Fake News Detection System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Bidirectional LSTM based news classification with real-time inference and evaluation diagnostics.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="chip-row">'
        '<span class="chip">TensorFlow and Keras</span>'
        '<span class="chip">Bidirectional LSTM</span>'
        '<span class="chip">Train, Validation, Test Split</span>'
        '<span class="chip">WSL GPU Enabled</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("Test Accuracy", f"{scores.get('test_accuracy', 0):.2f}%")
    with top2:
        st.metric("Test AUC", f"{scores.get('test_auc', 0):.2f}%")
    with top3:
        st.metric("Train Samples", f"{scores.get('train_samples', 0):,}")
    with top4:
        st.metric("Test Samples", f"{scores.get('test_samples', 0):,}")

    tab_demo, tab_dashboard, tab_about = st.tabs(["Live Inference", "Performance Dashboard", "Project Overview"])

    with tab_demo:
        st.info("Ensemble mode: LSTM is primary for in-domain text, with transformer fallback for out-of-domain snippets.")

        try:
            model, tokenizer = load_artifacts()
        except FileNotFoundError:
            st.error("Model artifacts were not found. Please run python src/train_model.py first.")
            st.stop()

        transformer_clf, transformer_error = load_transformer_fallback()
        if transformer_clf is None:
            st.warning("Transformer fallback could not be loaded. The app is currently using LSTM only.")
            with st.expander("Fallback loader details"):
                st.code(transformer_error or "Unknown transformer loading error", language=None)

        left, right = st.columns([1.3, 1.0], gap="large")

        with left:
            st.markdown('<div class="section-title">Inference Input</div>', unsafe_allow_html=True)
            st.markdown('<div class="surface">', unsafe_allow_html=True)

            news_text = st.text_area(
                "News article text",
                height=250,
                placeholder="Paste complete article text for classification",
                key="demo_input",
            )

            analyze_btn = st.button("Run Inference", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-title">Inference Output</div>', unsafe_allow_html=True)

            if analyze_btn:
                if not news_text.strip():
                    st.warning("Please paste article text before running inference.")
                else:
                    with st.spinner("Running model inference..."):
                        result = predict_news(news_text, model, tokenizer, transformer_clf)

                    label = result["label"]
                    confidence = result["confidence"]
                    latency = result["latency"]
                    word_count = result["word_count"]
                    cleaned = result["cleaned"]
                    oov_ratio = result["oov_ratio"]
                    final_prob_real = result["final_prob_real"]
                    lstm_prob_real = result["lstm_prob_real"]
                    tf_prob_real = result["tf_prob_real"]
                    lstm_weight = result["lstm_weight"]

                    if label == "Real":
                        st.markdown(
                            '<div class="result-card result-real">'
                            '<p class="result-label">REAL NEWS</p>'
                            f'<p class="result-conf">Confidence: {confidence:.1%}</p>'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                    elif label == "Fake":
                        st.markdown(
                            '<div class="result-card result-fake">'
                            '<p class="result-label">FAKE NEWS</p>'
                            f'<p class="result-conf">Confidence: {confidence:.1%}</p>'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="result-card result-uncertain">'
                            '<p class="result-label">UNCERTAIN</p>'
                            f'<p class="result-conf">Low in-domain signal · Confidence: {confidence:.1%}</p>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        '<div class="metrics-row">'
                        f'<div class="metric-box"><div class="metric-val">{confidence:.1%}</div><div class="metric-lbl">CONFIDENCE</div></div>'
                        f'<div class="metric-box"><div class="metric-val">{latency*1000:.0f}ms</div><div class="metric-lbl">LATENCY</div></div>'
                        f'<div class="metric-box"><div class="metric-val">{word_count}</div><div class="metric-lbl">TOKENS</div></div>'
                        f'<div class="metric-box"><div class="metric-val">{final_prob_real:.2f}</div><div class="metric-lbl">ENSEMBLE P(REAL)</div></div>'
                        f'<div class="metric-box"><div class="metric-val">{lstm_prob_real:.2f}</div><div class="metric-lbl">LSTM P(REAL)</div></div>'
                        f'<div class="metric-box"><div class="metric-val">{(tf_prob_real if tf_prob_real is not None else 0.5):.2f}</div><div class="metric-lbl">TRANSFORMER P(REAL)</div></div>'
                        f'<div class="metric-box"><div class="metric-val">{lstm_weight:.2f}</div><div class="metric-lbl">LSTM WEIGHT</div></div>'
                        f'<div class="metric-box"><div class="metric-val">{oov_ratio:.0%}</div><div class="metric-lbl">OOV RATIO</div></div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    with st.expander("Show preprocessed text"):
                        st.code(cleaned[:1500], language=None)
            else:
                st.markdown('<div class="surface"><p class="note">Use the input panel to run a live classification and display confidence, latency, and token metrics.</p></div>', unsafe_allow_html=True)

    with tab_dashboard:
        st.markdown('<div class="section-title">Model Evaluation Dashboard</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fake F1", f"{scores.get('fake_f1', 0):.2f}%")
        m2.metric("Real F1", f"{scores.get('real_f1', 0):.2f}%")
        m3.metric("Macro F1", f"{scores.get('macro_f1', 0):.2f}%")
        m4.metric("Validation Samples", f"{scores.get('val_samples', 0):,}")

        img_left, img_right = st.columns(2, gap="large")
        with img_left:
            if os.path.exists(os.path.join("models", "training_curves.png")):
                st.image(os.path.join("models", "training_curves.png"), caption="Training Curves")
            if os.path.exists(os.path.join("models", "roc_curve.png")):
                st.image(os.path.join("models", "roc_curve.png"), caption="ROC Curve")
        with img_right:
            if os.path.exists(os.path.join("models", "confusion_matrix.png")):
                st.image(os.path.join("models", "confusion_matrix.png"), caption="Confusion Matrix")
            if os.path.exists(os.path.join("models", "prediction_distribution.png")):
                st.image(os.path.join("models", "prediction_distribution.png"), caption="Prediction Distribution")

        st.markdown('<div class="note">All visualizations are loaded from the latest training run in the models directory.</div>', unsafe_allow_html=True)

    with tab_about:
        st.markdown('<div class="section-title">Project Summary</div>', unsafe_allow_html=True)
        st.markdown(
            "- Models: Bidirectional LSTM primary model with transformer fallback\n"
            "- Pipeline: leakage-free train/validation/test split with tokenizer fit on train only\n"
            "- Ensemble: domain-aware weighted probability calibration for final label\n"
            "- Validation controls: early stopping, learning rate scheduling, class weighting\n"
            "- Outputs: metrics JSON, confusion matrix, ROC, distribution, and training curves"
        )
        st.markdown("Dataset source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")


if __name__ == "__main__":
    main()
