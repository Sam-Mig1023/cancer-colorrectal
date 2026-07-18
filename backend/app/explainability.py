from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from .config import CLASS_NAMES
from .inference import model_service


class ExplainabilityUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class GradCamResult:
    original_png: bytes
    overlay_png: bytes
    heatmap_png: bytes
    metadata: dict[str, Any]


def _last_spatial_layer(model: keras.Model) -> keras.layers.Layer:
    for layer in reversed(model.layers):
        shape = getattr(layer, "output_shape", None)
        if shape is None:
            try:
                shape = tuple(layer.output.shape)
            except (AttributeError, ValueError):
                continue
        if len(shape) == 4:
            return layer
    raise ExplainabilityUnavailableError("Grad-CAM requires a compatible Keras model with a spatial convolutional feature layer.")


def _png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def validate_grad_cam_alignment(original_png: bytes, overlay_png: bytes, heatmap_png: bytes) -> dict[str, Any]:
    with Image.open(BytesIO(original_png)) as original, Image.open(BytesIO(overlay_png)) as overlay, Image.open(BytesIO(heatmap_png)) as heatmap:
        same_size = original.size == overlay.size == heatmap.size
        overlay_difference = float(np.mean(np.abs(np.asarray(original.convert("RGB"), dtype=np.int16) - np.asarray(overlay.convert("RGB"), dtype=np.int16))))
        heatmap_range = int(np.ptp(np.asarray(heatmap.convert("L"), dtype=np.uint8)))
    if not same_size:
        raise ExplainabilityUnavailableError("Grad-CAM output dimensions do not match the original image.")
    return {"same_dimensions": same_size, "overlay_mean_absolute_difference": round(overlay_difference, 6), "heatmap_intensity_range": heatmap_range, "visual_validation": "The generated images share identical dimensions. Visual pathology relevance still requires qualified human review."}


def generate_grad_cam(image_bytes: bytes, model_key: str, target_class: str | None = None) -> GradCamResult:
    if target_class is not None and target_class not in CLASS_NAMES:
        raise ValueError("target_class must be one of the configured histology classes.")
    spec = model_service.get_spec(model_key)
    image = model_service._read_image(image_bytes)
    batch = model_service._preprocess(image, spec.input_size)
    model = model_service._get_model(spec)
    if not isinstance(model, keras.Model):
        raise ExplainabilityUnavailableError("Grad-CAM is currently available only for compatible Keras classification models.")
    tensor = tf.convert_to_tensor(batch)
    try:
        _ = model(tensor, training=False)
    except Exception as exc:  # pragma: no cover - Keras surfaces model-specific errors here.
        raise ExplainabilityUnavailableError("The selected model could not be initialized for Grad-CAM.") from exc
    feature_layer = _last_spatial_layer(model)
    try:
        gradient_model = keras.Model(model.inputs, [feature_layer.output, model.outputs[0]])
    except (AttributeError, ValueError) as exc:
        raise ExplainabilityUnavailableError("Grad-CAM could not connect the selected feature layer to this model output.") from exc
    with tf.GradientTape() as tape:
        feature_maps, prediction = gradient_model(tensor, training=False)
        prediction_array = prediction.numpy()[0]
        target_index = CLASS_NAMES.index(target_class) if target_class else int(np.argmax(prediction_array))
        target_score = prediction[:, target_index]
    gradients = tape.gradient(target_score, feature_maps)
    if gradients is None:
        raise ExplainabilityUnavailableError("Grad-CAM gradients could not be computed for this architecture.")
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = float(tf.reduce_max(heatmap).numpy())
    if maximum > 0:
        heatmap = heatmap / maximum
    heatmap_array = np.asarray(heatmap.numpy(), dtype=np.float32)
    heatmap_resized = Image.fromarray(np.uint8(np.clip(heatmap_array, 0, 1) * 255)).resize(image.size, Image.Resampling.BILINEAR)
    normalized = np.asarray(heatmap_resized, dtype=np.float32) / 255.0
    color = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    color[..., 0] = np.uint8(255 * normalized)
    color[..., 1] = np.uint8(255 * np.sqrt(normalized))
    heatmap_color = Image.fromarray(color, mode="RGB")
    alpha = Image.fromarray(np.uint8(150 * normalized), mode="L")
    overlay = Image.blend(image.convert("RGB"), heatmap_color, 0.45)
    overlay.putalpha(alpha)
    composite = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    original_png = _png(image)
    heatmap_png = _png(heatmap_color)
    overlay_png = _png(composite)
    alignment = validate_grad_cam_alignment(original_png, overlay_png, heatmap_png)
    metadata = {
        "source": "reproducible_grad_cam",
        "model_key": model_key,
        "model_name": spec.display_name,
        "target_class": CLASS_NAMES[target_index],
        "predicted_class": CLASS_NAMES[int(np.argmax(prediction_array))],
        "confidence_percent": round(float(prediction_array[target_index] * 100), 4),
        "mean_intensity": round(float(normalized.mean()), 6),
        "max_intensity": round(float(normalized.max()), 6),
        "feature_layer": feature_layer.name,
        "alignment": alignment,
        "interpretation": "The overlay identifies regions that increased the selected model score. It is an academic explanatory aid and is not evidence of malignancy or a clinical diagnosis.",
    }
    return GradCamResult(original_png=original_png, overlay_png=overlay_png, heatmap_png=heatmap_png, metadata=metadata)

