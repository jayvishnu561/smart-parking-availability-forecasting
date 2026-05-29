from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
# Some installations provide Keras as a separate top-level package and
# `tensorflow` may not expose `keras` (environment variations). Prefer
# `tf.keras` but fall back to the standalone `keras` package when needed.
if not hasattr(tf, "keras"):
    try:
        import keras as _keras  # type: ignore

        tf.keras = _keras  # type: ignore[attr-defined]
    except Exception:
        # Let the original import error surface later when loading the model
        pass


class ModelService:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        default_model_path = root / "lstm_parking_model.keras"
        model_path = Path(os.getenv("MODEL_PATH", str(default_model_path)))

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = str(model_path)

        # Attempt to load the Keras model; if loading fails for any reason
        # (environment differences between `tensorflow` and `keras`), fall
        # back to a lightweight heuristic predictor so the API remains usable.
        self.model = None
        self.load_error: str | None = None
        try:
            keras_mod = getattr(tf, "keras", None)
            if keras_mod is None:
                import keras as keras_mod  # type: ignore

            dense_init = keras_mod.layers.Dense.__init__

            def _patched_dense_init(layer_self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                return dense_init(layer_self, *args, **kwargs)

            keras_mod.layers.Dense.__init__ = _patched_dense_init
            try:
                self.model = keras_mod.models.load_model(model_path, compile=False)
            finally:
                keras_mod.layers.Dense.__init__ = dense_init

            self.input_shape = [d if d is None else int(d) for d in self.model.input_shape]
        except Exception as exc:  # pragma: no cover - runtime fallback
            self.load_error = str(exc)
            # Fallback input shape indicates unknown model shape.
            self.input_shape = [None, None]

    def required_sequence_length(self) -> int:
        if self.model is None:
            return 3

        shape = self.model.input_shape

        if len(shape) == 3:
            _, time_steps, features = shape
            t = int(time_steps) if time_steps is not None else 1
            f = int(features) if features is not None else 1
            return max(t * f, 1)

        if len(shape) == 2:
            _, features = shape
            f = int(features) if features is not None else 1
            return max(f, 1)

        raise ValueError(f"Unsupported model input rank: {len(shape)}")

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

            required_values = int(time_steps) * int(features)

            if len(seq) < required_values:
                pad_value = seq[-1]
                seq = np.pad(seq, (required_values - len(seq), 0), constant_values=pad_value)
            elif len(seq) > required_values:
                seq = seq[-required_values:]

            x = seq.reshape(1, int(time_steps), int(features))
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
        # If a real model failed to load, use a simple fallback predictor that
        # repeats or averages recent values so the API stays functional.
        if self.model is None:
            if not sequence:
                raise ValueError("Sequence cannot be empty for fallback prediction.")
            last = float(sequence[-1])
            return [round(float(last), 4) for _ in range(horizon)]

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