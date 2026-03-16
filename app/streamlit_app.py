"""
streamlit_app.py - Fake News Detection Web Interface

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import re
import pickle
import time

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("models", "fake_news_lstm.keras")
LEGACY_MODEL_PATH = os.path.join("models", "fake_news_lstm.h5")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")
MAX_SEQUENCE_LEN = 300

# ─── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 24px;
    }
    .badge-row {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 28px;
    }
    .badge {
        background: #1e1e2e;
        color: #ccc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Result cards */
    .result-card {
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        margin: 16px 0;
        animation: fadeIn 0.5s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-real {
        background: linear-gradient(135deg, #0d9488 0%, #059669 100%);
        color: white;
    }
    .result-fake {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
    }
    .result-label { font-size: 2rem; font-weight: 700; margin: 0; }
    .result-conf  { font-size: 1.1rem; opacity: 0.9; margin-top: 4px; }
    .result-icon  { font-size: 3rem; margin-bottom: 8px; }

    /* Metrics row */
    .metrics-row {
        display: flex;
        justify-content: center;
        gap: 24px;
        margin: 16px 0 8px;
    }
    .metric-box {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px 28px;
        text-align: center;
        min-width: 120px;
    }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #a78bfa; }
    .metric-lbl { font-size: 0.75rem; color: #888; margin-top: 2px; }

    /* Footer */
    .footer {
        text-align: center;
        color: #555;
        font-size: 0.78rem;
        padding-top: 32px;
        border-top: 1px solid #2a2a3a;
        margin-top: 48px;
    }
    .footer a { color: #667eea; text-decoration: none; }
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
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else LEGACY_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = load_model(model_path)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


# ─── Prediction ──────────────────────────────────────────────────────────────
def predict_news(text: str, model, tokenizer) -> tuple:
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")
    start = time.time()
    prob = model.predict(padded, verbose=0)[0][0]
    latency = time.time() - start
    label = "Real" if prob >= 0.5 else "Fake"
    confidence = float(prob) if prob >= 0.5 else float(1 - prob)
    return label, confidence, latency, len(cleaned.split())


# ─── App ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Fake News Detector · LSTM",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero ──
    st.markdown('<p class="hero-title">🧠 Fake News Detector</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Bidirectional LSTM · Real-time NLP Classification</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="badge-row">'
        '<span class="badge">TensorFlow</span>'
        '<span class="badge">LSTM</span>'
        '<span class="badge">NLP</span>'
        '<span class="badge">Streamlit</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Load model ──
    try:
        model, tokenizer = load_artifacts()
    except FileNotFoundError:
        st.error(
            "**Model not found.** Train the model first:  \n"
            "`python src/train_model.py`"
        )
        st.stop()

    # ── Input ──
    news_text = st.text_area(
        "📝 Paste a news article",
        height=220,
        placeholder="Enter the full news article text here…",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("⚡ Analyze", type="primary", use_container_width=True)

    # ── Prediction ──
    if analyze_btn:
        if not news_text.strip():
            st.warning("Please paste some text first.")
            return

        with st.spinner("Running inference…"):
            label, confidence, latency, word_count = predict_news(
                news_text, model, tokenizer
            )

        # Result card
        if label == "Real":
            st.markdown(
                '<div class="result-card result-real">'
                '<div class="result-icon">✅</div>'
                '<p class="result-label">REAL NEWS</p>'
                f'<p class="result-conf">Confidence: {confidence:.1%}</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card result-fake">'
                '<div class="result-icon">🚨</div>'
                '<p class="result-label">FAKE NEWS</p>'
                f'<p class="result-conf">Confidence: {confidence:.1%}</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Metrics row
        st.markdown(
            '<div class="metrics-row">'
            f'<div class="metric-box"><div class="metric-val">{confidence:.1%}</div>'
            '<div class="metric-lbl">CONFIDENCE</div></div>'
            f'<div class="metric-box"><div class="metric-val">{latency*1000:.0f}ms</div>'
            '<div class="metric-lbl">LATENCY</div></div>'
            f'<div class="metric-box"><div class="metric-val">{word_count}</div>'
            '<div class="metric-lbl">TOKENS</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Expandable details
        with st.expander("🔎 Preprocessed text"):
            st.code(clean_text(news_text)[:600], language=None)

    # ── Sidebar — About ──
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown(
            "This app uses a **Bidirectional LSTM** neural network "
            "trained on ~44 000 news articles to classify text as "
            "**Fake** or **Real** in real time."
        )
        st.markdown("---")
        st.markdown("**Model:** BiLSTM (TensorFlow/Keras)")
        st.markdown("**Accuracy:** ~99%")
        st.markdown("**Dataset:** [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)")

    # ── Footer ──
    st.markdown(
        '<div class="footer">'
        'Built with <strong>TensorFlow</strong> · <strong>LSTM</strong> · '
        '<strong>Streamlit</strong>&nbsp;&nbsp;|&nbsp;&nbsp;'
        '<a href="https://github.com/yourusername/FakeNews-LSTM">GitHub</a>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
