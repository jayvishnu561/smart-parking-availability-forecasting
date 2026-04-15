from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf


class ModelService:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        default_model_path = root / "lstm_parking_model.keras"
        model_path = Path(os.getenv("MODEL_PATH", str(default_model_path)))

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = str(model_path)
        dense_init = tf.keras.layers.Dense.__init__

        def _patched_dense_init(layer_self, *args, **kwargs):
            # Some saved models include this key on Dense configs even when not quantized.
            kwargs.pop("quantization_config", None)
            return dense_init(layer_self, *args, **kwargs)

        tf.keras.layers.Dense.__init__ = _patched_dense_init
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
        finally:
            tf.keras.layers.Dense.__init__ = dense_init
        self.input_shape = [d if d is None else int(d) for d in self.model.input_shape]

    def _reshape_for_model(self, sequence: list[float]) -> np.ndarray:
        if len(sequence) == 0:
            raise ValueError("Sequence cannot be empty.")

        shape = self.model.input_shape
        seq = np.asarray(sequence, dtype=np.float32)

        if len(shape) == 3:
            _, time_steps, features = shape

            if time_steps is None:
                time_steps = len(seq)
            if features is None:
                features = 1

            if len(seq) < time_steps:
                pad_value = seq[-1]
                seq = np.pad(seq, (time_steps - len(seq), 0), constant_values=pad_value)
            elif len(seq) > time_steps:
                seq = seq[-time_steps:]

            x = seq.reshape(1, time_steps, 1)
            if features > 1:
                x = np.repeat(x, features, axis=2)
            return x

        if len(shape) == 2:
            _, features = shape
            if features is None:
                features = len(seq)

            if len(seq) < features:
                pad_value = seq[-1]
                seq = np.pad(seq, (features - len(seq), 0), constant_values=pad_value)
            elif len(seq) > features:
                seq = seq[-features:]

            return seq.reshape(1, features)

        raise ValueError(f"Unsupported model input rank: {len(shape)}")

    def predict(self, sequence: list[float], horizon: int) -> list[float]:
        x = self._reshape_for_model(sequence)
        y = self.model.predict(x, verbose=0)
        preds = np.asarray(y).reshape(-1).astype(float).tolist()

        if len(preds) == 0:
            raise ValueError("Model returned an empty prediction.")

        clipped = [float(np.clip(v, 0.0, 1.0)) for v in preds]

        if len(clipped) >= horizon:
            return [round(v, 4) for v in clipped[:horizon]]

        last = clipped[-1]
        padded = clipped + [last] * (horizon - len(clipped))
        return [round(v, 4) for v in padded]