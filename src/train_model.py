"""
train_model.py - Build, compile, and train the LSTM model for Fake News Detection.

Architecture:
    Embedding → LSTM → Dropout → Dense(sigmoid)

Saves the trained model to models/fake_news_lstm.h5
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt

import tensorflow as tf

# ─── GPU Configuration (RTX 3050) ────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✔ GPU detected: {[g.name for g in gpus]}")
    # Mixed precision for Tensor Cores (RTX 3050+)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("✔ Mixed-precision (float16) enabled — faster on RTX 3050")
else:
    print("⚠ No GPU found — training on CPU (still fast for this dataset)")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocess import prepare_data, MAX_WORDS, MAX_SEQUENCE_LEN

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_lstm.h5")

# ─── Hyper-parameters ────────────────────────────────────────────────────────
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
EPOCHS = 10


# ─── Model builder ───────────────────────────────────────────────────────────
def build_model(vocab_size: int = MAX_WORDS,
                embedding_dim: int = EMBEDDING_DIM,
                input_length: int = MAX_SEQUENCE_LEN,
                lstm_units: int = LSTM_UNITS,
                dropout: float = DROPOUT_RATE) -> Sequential:
    """
    Construct an LSTM-based binary classifier.

    Layers
    ------
    1. Embedding  – learns dense word vectors
    2. Bidirectional LSTM – captures sequential patterns in both directions
    3. Dropout – regularisation
    4. Dense (sigmoid) – binary output (Fake / Real)
    """
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  input_length=input_length),
        Bidirectional(LSTM(lstm_units, return_sequences=False)),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dropout(dropout),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def _print_gpu_info():
    """Print GPU utilisation summary."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            name = details.get("device_name", gpu.name)
            print(f"  GPU : {name}")
        print(f"  Mixed-precision : {tf.keras.mixed_precision.global_policy().name}")
    else:
        print("  Running on CPU")


# ─── Training loop ───────────────────────────────────────────────────────────
def train(save_plot: bool = True):
    """Prepare data, build model, train, evaluate, and save."""

    # 1. Data
    X_train, X_test, y_train, y_test, tokenizer = prepare_data()

    # 2. GPU info
    _print_gpu_info()

    # 3. Model
    model = build_model()
    model.summary()

    # 4. Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1)
    ]

    # 5. Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*50}")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {accuracy:.4f}")
    print(f"{'='*50}")
    print(f"✔ Model saved to {MODEL_PATH}")

    # 7. Plot training curves
    if save_plot:
        _plot_history(history)

    return model, history


def _plot_history(history):
    """Save accuracy & loss curves to models/training_curves.png."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✔ Training curves saved to {plot_path}")


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
