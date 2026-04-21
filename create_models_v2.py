from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing.contracts import HARDWARE_SEQUENCE_FEATURES, NETWORK_FEATURES, PROCESS_FEATURES


def build_network_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(NETWORK_FEATURES),)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def build_process_autoencoder(window_size: int = 10) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, len(PROCESS_FEATURES))),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(PROCESS_FEATURES))),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def build_hardware_autoencoder(window_size: int = 10) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(window_size, len(HARDWARE_SEQUENCE_FEATURES)))
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12, return_sequences=True))(x)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(HARDWARE_SEQUENCE_FEATURES)))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


if __name__ == "__main__":
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    network_model = build_network_model()
    process_model = build_process_autoencoder()
    hardware_model = build_hardware_autoencoder()

    # This script only saves architecture-initialized models to unblock wiring tests.
    # For production/demo, retrain using real industrial datasets before use.
    network_model.save(model_dir / "ram_model.h5")
    process_model.save(model_dir / "lstm_model.h5")
    hardware_model.save(model_dir / "hardware_model.h5")
    np.save(model_dir / "threshold.npy", np.array([0.05], dtype=float))
    np.save(model_dir / "hardware_threshold.npy", np.array([0.05], dtype=float))
    print("Saved untrained network/process/hardware model artifacts to models/")
