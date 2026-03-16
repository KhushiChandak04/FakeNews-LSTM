"""
train_model.py - Build, compile, and train the LSTM model for Fake News Detection.

Architecture:
    Embedding → LSTM → Dropout → Dense(sigmoid)

Saves the trained model to models/fake_news_lstm.keras
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight

# ─── GPU Configuration (RTX 3050) ────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✔ GPU detected: {[g.name for g in gpus]}")
    # Keep float32 by default for stable cross-platform inference (WSL train, Windows demo).
    if os.getenv("ENABLE_MIXED_PRECISION", "0") == "1":
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("✔ Mixed-precision (float16) enabled")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        print("✔ Using float32 for portable inference")
else:
    print("⚠ No GPU found — training on CPU (still fast for this dataset)")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers

try:
    from preprocess import prepare_data, MAX_WORDS, MAX_SEQUENCE_LEN
except ModuleNotFoundError:
    from src.preprocess import prepare_data, MAX_WORDS, MAX_SEQUENCE_LEN

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_lstm.keras")
SCORES_PATH = os.path.join(MODEL_DIR, "scores.json")

# ─── Hyper-parameters ────────────────────────────────────────────────────────
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE = 0.4
BATCH_SIZE = 128
EPOCHS = 15


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
        Embedding(input_dim=vocab_size + 1,
                  output_dim=embedding_dim),
        SpatialDropout1D(0.2),
        Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=False,
                dropout=0.0,
                recurrent_dropout=0.0,
                kernel_regularizer=regularizers.l2(1e-4),
            )
        ),
        Dropout(dropout),
        Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(dropout),
        Dense(1, activation="sigmoid", dtype="float32")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
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
    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_data()

    # 2. GPU info
    _print_gpu_info()

    # 3. Model
    model = build_model()
    model.summary()

    # 4. Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train,
    )
    class_weight_map = {0: float(class_weights_values[0]), 1: float(class_weights_values[1])}
    print(f"✔ Class weights: {class_weight_map}")

    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 5. Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_map,
        verbose=1
    )

    # 6. Evaluate on untouched test set
    eval_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    test_auc = float(roc_auc_score(y_test, y_prob))

    print(f"\n{'='*50}")
    print(f"  Test Loss     : {eval_metrics['loss']:.4f}")
    print(f"  Test Accuracy : {eval_metrics['accuracy']:.4f}")
    print(f"  Test AUC      : {test_auc:.4f}")
    print(f"{'='*50}")
    print(f"✔ Model saved to {MODEL_PATH}")

    _save_scores(eval_metrics, report, test_auc, y_train, y_val, y_test)
    _plot_diagnostics(y_test, y_prob)

    # 7. Plot training curves
    if save_plot:
        _plot_history(history)

    return model, history


def _plot_history(history):
    """Save accuracy & loss curves to models/training_curves.png."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    axes[0, 0].plot(history.history.get("accuracy", []), label="Train")
    axes[0, 0].plot(history.history.get("val_accuracy", []), label="Validation")
    axes[0, 0].set_title("Model Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(history.history.get("loss", []), label="Train")
    axes[0, 1].plot(history.history.get("val_loss", []), label="Validation")
    axes[0, 1].set_title("Model Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # AUC
    axes[1, 0].plot(history.history.get("auc", []), label="Train")
    axes[1, 0].plot(history.history.get("val_auc", []), label="Validation")
    axes[1, 0].set_title("Model AUC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Precision/Recall
    axes[1, 1].plot(history.history.get("precision", []), label="Train Precision")
    axes[1, 1].plot(history.history.get("val_precision", []), label="Val Precision")
    axes[1, 1].plot(history.history.get("recall", []), label="Train Recall")
    axes[1, 1].plot(history.history.get("val_recall", []), label="Val Recall")
    axes[1, 1].set_title("Precision / Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✔ Training curves saved to {plot_path}")


def _save_scores(eval_metrics, report, test_auc, y_train, y_val, y_test):
    """Persist evaluation metrics to models/scores.json."""
    scores = {
        "test_loss": round(float(eval_metrics["loss"]), 4),
        "test_accuracy": round(float(eval_metrics["accuracy"]) * 100, 2),
        "test_auc": round(test_auc * 100, 2),
        "fake_precision": round(float(report["0"]["precision"]) * 100, 2),
        "fake_recall": round(float(report["0"]["recall"]) * 100, 2),
        "fake_f1": round(float(report["0"]["f1-score"]) * 100, 2),
        "real_precision": round(float(report["1"]["precision"]) * 100, 2),
        "real_recall": round(float(report["1"]["recall"]) * 100, 2),
        "real_f1": round(float(report["1"]["f1-score"]) * 100, 2),
        "macro_f1": round(float(report["macro avg"]["f1-score"]) * 100, 2),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
    }
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"✔ Metrics saved to {SCORES_PATH}")


def _plot_diagnostics(y_true, y_prob):
    """Save confusion matrix, ROC curve, and probability distribution plots."""
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fake", "Real"])
    ax.set_yticklabels(["Fake", "Real"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✔ Confusion matrix saved to {cm_path}")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(MODEL_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"✔ ROC curve saved to {roc_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_prob[y_true == 0], bins=40, alpha=0.6, label="Fake", density=True)
    ax.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="Real", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted probability of Real")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Probability Distribution")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    dist_path = os.path.join(MODEL_DIR, "prediction_distribution.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"✔ Prediction distribution saved to {dist_path}")


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
