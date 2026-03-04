"""
preprocess.py - Data loading and text preprocessing for Fake News Detection.

Handles:
  - Loading raw CSV datasets (Fake.csv, True.csv)
  - Text cleaning (lowercasing, removing special characters, stopwords)
  - Tokenization and sequence padding
  - Train/test splitting
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_WORDS = 10_000        # vocabulary size
MAX_SEQUENCE_LEN = 300    # max tokens per article
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")


# ─── Text cleaning ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip URLs, special characters, and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)             # keep only letters
    text = re.sub(r"\s+", " ", text).strip()             # collapse whitespace
    return text


# ─── Load & merge datasets ───────────────────────────────────────────────────
def load_data(fake_path: str = "data/raw/Fake.csv",
              true_path: str = "data/raw/True.csv") -> pd.DataFrame:
    """
    Load Fake.csv and True.csv, assign labels, merge, and shuffle.

    Labels:
        0 → Fake
        1 → Real
    """
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    df_fake["label"] = 0
    df_true["label"] = 1

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Combine title + text for richer features
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content"] = df["content"].apply(clean_text)

    return df[["content", "label"]]


# ─── Tokenize & pad ──────────────────────────────────────────────────────────
def tokenize_and_pad(texts,
                     max_words: int = MAX_WORDS,
                     max_len: int = MAX_SEQUENCE_LEN,
                     tokenizer=None,
                     fit: bool = True):
    """
    Convert raw text to padded integer sequences.

    Parameters
    ----------
    texts : array-like of str
    max_words : int – vocabulary size
    max_len : int – fixed sequence length
    tokenizer : Tokenizer or None – supply a fitted tokenizer for inference
    fit : bool – whether to fit the tokenizer on `texts`

    Returns
    -------
    X_pad : np.ndarray of shape (n_samples, max_len)
    tokenizer : fitted Tokenizer instance
    """
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

    if fit:
        tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    X_pad = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    return X_pad, tokenizer


# ─── Prepare full pipeline ───────────────────────────────────────────────────
def prepare_data(test_size: float = 0.2):
    """
    End-to-end: load → clean → tokenize → split.

    Returns
    -------
    X_train, X_test, y_train, y_test, tokenizer
    """
    df = load_data()

    X_pad, tokenizer = tokenize_and_pad(df["content"].values)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=test_size, random_state=42
    )

    # Persist tokenizer for inference
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"✔ Data prepared  |  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"✔ Tokenizer saved to {TOKENIZER_PATH}")

    return X_train, X_test, y_train, y_test, tokenizer


# ─── CLI quick-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, tok = prepare_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"Vocabulary size: {min(MAX_WORDS, len(tok.word_index) + 1)}")
