from __future__ import annotations

from io import BytesIO
from threading import Lock

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from .config import CLASS_NAMES, MODEL_SPECS, ModelSpec


class ImageValidationError(ValueError):
    """Raised when the received file cannot be used as an image."""


class ModelService:
    """Loads the project models lazily and runs image inference."""

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._lock = Lock()

    def available_models(self) -> list[ModelSpec]:
        return [spec for spec in MODEL_SPECS.values() if spec.path.exists()]

    def get_spec(self, model_key: str) -> ModelSpec:
        try:
            return MODEL_SPECS[model_key]
        except KeyError as exc:
            allowed = ", ".join(MODEL_SPECS)
            raise ValueError(f"Unknown model '{model_key}'. Allowed values: {allowed}.") from exc

    def predict(self, image_bytes: bytes, model_key: str) -> tuple[ModelSpec, np.ndarray]:
        spec = self.get_spec(model_key)
        if not spec.path.exists():
            raise FileNotFoundError(f"Model artifact not found: {spec.path}")

        image = self._read_image(image_bytes)
        batch = self._preprocess(image, spec.input_size)
        model = self._get_model(spec)
        probabilities = self._run_prediction(model, batch)
        return spec, probabilities

    @staticmethod
    def _read_image(image_bytes: bytes) -> Image.Image:
        if not image_bytes:
            raise ImageValidationError("The uploaded file is empty.")
        try:
            with Image.open(BytesIO(image_bytes)) as uploaded:
                uploaded.verify()
            with Image.open(BytesIO(image_bytes)) as uploaded:
                return uploaded.convert("RGB").copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageValidationError("Upload a valid PNG, JPEG, or WEBP image.") from exc

    @staticmethod
    def _preprocess(image: Image.Image, input_size: tuple[int, int]) -> np.ndarray:
        resized = image.resize(input_size, Image.Resampling.LANCZOS)
        normalized = np.asarray(resized, dtype=np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def _get_model(self, spec: ModelSpec) -> object:
        with self._lock:
            if spec.key not in self._models:
                self._models[spec.key] = self._load_model(spec)
            return self._models[spec.key]

    @staticmethod
    def _load_model(spec: ModelSpec) -> object:
        if spec.loader == "keras_model":
            return keras.models.load_model(spec.path)
        if spec.loader == "saved_model":
            return keras.layers.TFSMLayer(spec.path, call_endpoint="serving_default")
        if spec.loader == "mobilenet_weights":
            inputs = keras.Input(shape=(224, 224, 3))
            base_model = keras.applications.MobileNetV2(
                input_tensor=inputs,
                include_top=False,
                weights=None,
            )
            x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(base_model.output)
            x = keras.layers.Dense(256, activation="relu", name="dense")(x)
            x = keras.layers.Dropout(0.5, name="dropout")(x)
            outputs = keras.layers.Dense(9, activation="softmax", name="dense_1")(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.load_weights(spec.path, by_name=True, skip_mismatch=True)
            return model
        raise ValueError(f"Unsupported loader: {spec.loader}")

    @staticmethod
    def _run_prediction(model: object, batch: np.ndarray) -> np.ndarray:
        result = model(batch, training=False)
        if isinstance(result, dict):
            result = next(iter(result.values()))

        probabilities = np.asarray(result.numpy() if hasattr(result, "numpy") else result)
        if probabilities.ndim != 2 or probabilities.shape != (1, len(CLASS_NAMES)):
            raise RuntimeError(
                f"Unexpected prediction shape {probabilities.shape}; expected (1, {len(CLASS_NAMES)})."
            )
        return probabilities[0]


model_service = ModelService()
