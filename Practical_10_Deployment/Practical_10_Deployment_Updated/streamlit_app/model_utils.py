from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

def build_model():
    # Same architecture as Part A of Practical 10.
    return tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

def load_mnist_model():
    # The numeric weights are portable across compatible implementations of
    # these Dense layers, even when a whole-model Keras config cannot deserialize.
    weights_path = ROOT / "mnist_weights.npz"
    keras_path = ROOT / "mnist_model.keras"
    if weights_path.is_file():
        model = build_model()
        with np.load(weights_path, allow_pickle=False) as stored:
            keys = sorted(stored.files, key=lambda x: int(x.split('_')[1]))
            model.set_weights([stored[key] for key in keys])
        return model
    if keras_path.is_file():
        return tf.keras.models.load_model(keras_path, compile=False)
    raise FileNotFoundError(
        f"Missing model files in {ROOT}. Run the Practical 10 notebook training "
        "and export cells, then copy mnist_weights.npz (recommended) or "
        "mnist_model.keras into this folder."
    )
