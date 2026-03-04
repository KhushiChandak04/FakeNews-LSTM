"""
predict.py - Load the trained LSTM model and predict on new news text.

Usage (CLI):
    python src/predict.py "Paste a news article text here..."

Usage (import):
    from predict import predict_news
    label, confidence = predict_news("Some news article text...")
"""

import os
import sys
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Paths (relative to project root) ────────────────────────────────────────
MODEL_PATH = os.path.join("models", "fake_news_lstm.h5")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")
MAX_SEQUENCE_LEN = 300


def _load_artifacts():
    """Load the trained model and tokenizer from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Train the model first with:\n"
            "  python src/train_model.py"
        )
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer not found at '{TOKENIZER_PATH}'. Train the model first."
        )

    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


# Module-level lazy cache
_model = None
_tokenizer = None


def _get_artifacts():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = _load_artifacts()
    return _model, _tokenizer


# ─── Prediction ──────────────────────────────────────────────────────────────
def predict_news(text: str) -> tuple:
    """
    Predict whether a piece of news text is Fake or Real.

    Parameters
    ----------
    text : str
        The raw news article text.

    Returns
    -------
    label : str
        "Fake" or "Real"
    confidence : float
        Model confidence (0-1). Values > 0.5 → Real, else Fake.
    """
    from preprocess import clean_text  # local import to avoid circular deps

    model, tokenizer = _get_artifacts()

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")

    prob = model.predict(padded, verbose=0)[0][0]

    label = "Real" if prob >= 0.5 else "Fake"
    confidence = float(prob) if prob >= 0.5 else float(1 - prob)

    return label, confidence


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"<news text>\"")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    label, confidence = predict_news(input_text)

    print(f"\n{'='*50}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence:.2%}")
    print(f"{'='*50}")
