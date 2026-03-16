"""
check_gpu.py - Verify TensorFlow/Keras GPU access before training.

Run with:
    python check_gpu.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import keras

print("=" * 50)
print(f"  TensorFlow : {tf.__version__}")
print(f"  Keras      : {keras.__version__}")
print("=" * 50)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            name = details.get("device_name", gpu.name)
        except Exception:
            name = gpu.name
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  OK GPU detected : {name}")
    print("  OK GPU training is READY")
else:
    cpus = tf.config.list_physical_devices("CPU")
    print(f"  No GPU found - training on CPU ({len(cpus)} core(s))")
    print("  NOTE: TF 2.11+ dropped native Windows GPU.")
    print("        Use WSL2 for GPU training on Windows.")
print("=" * 50)
