from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow import keras

from .config import CLASS_NAMES, PROJECT_ROOT
from .dataset import dataset_partitions, require_dataset, summarize_dataset
from .evaluation import metrics_from_predictions

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: str | None = None
    train_partition: str = "train"
    validation_partition: str = "validation"
    test_partition: str = "test"
    architecture: str = "mobilenetv2_transfer"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    experiment_name: str = "mobilenetv2_experiment"
    horizontal_flip: bool = True
    rotation_factor: float = 0.05
    early_stopping_patience: int = 3
    pretrained_weights: bool = True


def validate_training_config(config: TrainingConfig) -> dict[str, Any]:
    if config.architecture != "mobilenetv2_transfer":
        raise ValueError("Only the reproducible 'mobilenetv2_transfer' architecture is currently supported.")
    if not 1 <= config.epochs <= 200:
        raise ValueError("epochs must be between 1 and 200.")
    if not 1 <= config.batch_size <= 128:
        raise ValueError("batch_size must be between 1 and 128.")
    if not 0 < config.learning_rate <= 1:
        raise ValueError("learning_rate must be greater than 0 and at most 1.")
    if not 0 <= config.rotation_factor <= 0.5:
        raise ValueError("rotation_factor must be between 0 and 0.5.")
    if not 1 <= config.early_stopping_patience <= 30:
        raise ValueError("early_stopping_patience must be between 1 and 30.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", config.experiment_name):
        raise ValueError("experiment_name must contain 3-80 letters, numbers, underscores, or hyphens.")
    root, partitions = require_dataset(config.dataset_path)
    requested = [config.train_partition, config.validation_partition, config.test_partition]
    if len(set(requested)) != 3:
        raise ValueError("train_partition, validation_partition, and test_partition must be different.")
    missing_partitions = [name for name in requested if name not in partitions]
    if missing_partitions:
        raise ValueError(f"Configured dataset is missing required partitions: {', '.join(missing_partitions)}.")
    summary = summarize_dataset(str(root))
    missing_classes = {name: summary["partitions"][name]["missing_classes"] for name in requested if summary["partitions"][name]["missing_classes"]}
    if missing_classes:
        formatted = "; ".join(f"{partition}: {', '.join(classes)}" for partition, classes in missing_classes.items())
        raise ValueError(f"Each partition must contain all configured classes. Missing: {formatted}.")
    return {"dataset_root": str(root), "dataset_fingerprint_sha256": summary["dataset_fingerprint_sha256"], "available_partitions": list(partitions), "partition_counts": {name: summary["partitions"][name]["images"] for name in requested}}


def training_plan(config: TrainingConfig) -> dict[str, Any]:
    validation = validate_training_config(config)
    return {
        "source": "reproducible_training_plan",
        "config": asdict(config),
        **validation,
        "model": "MobileNetV2 transfer learning with a nine-class softmax output.",
        "augmentation": {"horizontal_flip": config.horizontal_flip, "rotation_factor": config.rotation_factor},
        "artifacts": ["metadata.json", "config.json", "history.json", "history.csv", "training_curves.png", "best_model.keras", "final_model.keras", "test_metrics.json"],
        "interpretation": "The plan uses independent train, validation, and held-out test partitions. Test metrics are generated only after training is complete.",
    }


def _image_dataset(root: Path, config: TrainingConfig, shuffle: bool) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        root,
        labels="inferred",
        label_mode="int",
        class_names=list(CLASS_NAMES),
        image_size=(224, 224),
        batch_size=config.batch_size,
        shuffle=shuffle,
        seed=config.seed if shuffle else None,
    ).prefetch(tf.data.AUTOTUNE)


def _build_model(config: TrainingConfig) -> keras.Model:
    layers: list[keras.layers.Layer] = []
    if config.horizontal_flip:
        layers.append(keras.layers.RandomFlip("horizontal"))
    if config.rotation_factor:
        layers.append(keras.layers.RandomRotation(config.rotation_factor))
    augmentation = keras.Sequential(layers, name="augmentation") if layers else None
    inputs = keras.Input(shape=(224, 224, 3), name="image")
    x = augmentation(inputs) if augmentation else inputs
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet" if config.pretrained_weights else None, input_shape=(224, 224, 3))
    base.trainable = False
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="classification")(x)
    model = keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    model.compile(optimizer=keras.optimizers.Adam(config.learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _write_history_plot(history: dict[str, list[float]], path: Path) -> None:
    width, height, margin = 960, 520, 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    panels = [("loss", "val_loss", "Loss"), ("accuracy", "val_accuracy", "Accuracy")]
    panel_width = (width - 3 * margin) // 2
    for panel_index, (train_key, validation_key, title) in enumerate(panels):
        x0 = margin + panel_index * (panel_width + margin)
        y0, x1, y1 = 42, x0 + panel_width, height - margin
        draw.rectangle((x0, y0, x1, y1), outline="#cbd5e1")
        draw.text((x0, 22), title, fill="#0f172a", font=font)
        train_values = [float(value) for value in history.get(train_key, [])]
        validation_values = [float(value) for value in history.get(validation_key, [])]
        values = train_values + validation_values
        if not values:
            continue
        lower, upper = min(values), max(values)
        if lower == upper:
            lower -= 0.1
            upper += 0.1
        def points(series: list[float]) -> list[tuple[int, int]]:
            if len(series) == 1:
                return [(x0, int(y1 - (series[0] - lower) / (upper - lower) * (y1 - y0)))]
            return [(int(x0 + index * (x1 - x0) / (len(series) - 1)), int(y1 - (value - lower) / (upper - lower) * (y1 - y0))) for index, value in enumerate(series)]
        for series, color in ((train_values, "#0f766e"), (validation_values, "#2563eb")):
            if len(series) > 1:
                draw.line(points(series), fill=color, width=2)
        draw.text((x0, y1 + 8), "train teal | validation blue", fill="#475569", font=font)
    image.save(path, format="PNG")


def _experiment_directory(config: TrainingConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return EXPERIMENTS_DIR / f"{config.experiment_name}_{timestamp}_{uuid.uuid4().hex[:8]}"


def run_training(config: TrainingConfig) -> dict[str, Any]:
    plan = training_plan(config)
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    keras.utils.set_random_seed(config.seed)
    root = Path(plan["dataset_root"])
    partitions = dataset_partitions(root)
    run_dir = _experiment_directory(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "started_at_utc": started_at.isoformat(), "config": asdict(config), "dataset_fingerprint_sha256": plan["dataset_fingerprint_sha256"], "partitions": {"train": config.train_partition, "validation": config.validation_partition, "test": config.test_partition}}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    try:
        train_data = _image_dataset(partitions[config.train_partition], config, shuffle=True)
        validation_data = _image_dataset(partitions[config.validation_partition], config, shuffle=False)
        test_data = _image_dataset(partitions[config.test_partition], config, shuffle=False)
        model = _build_model(config)
        checkpoint_path = run_dir / "best_model.keras"
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
            keras.callbacks.CSVLogger(run_dir / "history.csv"),
        ]
        history = model.fit(train_data, validation_data=validation_data, epochs=config.epochs, callbacks=callbacks, verbose=2)
        model.save(run_dir / "final_model.keras")
        history_payload = {name: [float(value) for value in values] for name, values in history.history.items()}
        (run_dir / "history.json").write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
        _write_history_plot(history_payload, run_dir / "training_curves.png")
        true_labels: list[int] = []
        probabilities: list[Any] = []
        for images, labels in test_data:
            batch_probabilities = model(images, training=False).numpy()
            probabilities.extend(batch_probabilities)
            true_labels.extend(int(value) for value in labels.numpy())
        test_metrics = metrics_from_predictions(true_labels, tf.convert_to_tensor(probabilities).numpy(), bootstrap_iterations=500, seed=config.seed)
        test_metrics["origin"] = {"experiment_id": run_dir.name, "model_name": model.name, "dataset_fingerprint_sha256": plan["dataset_fingerprint_sha256"], "partition": config.test_partition, "seed": config.seed, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "patient_id_regex": None}
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        elapsed_seconds = round(time.perf_counter() - started_clock, 3)
        metadata.update({"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "duration_seconds": elapsed_seconds, "epochs_completed": len(history_payload.get("loss", []))})
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"source": "reproducible_training", "experiment_id": run_dir.name, "output_dir": str(run_dir), "duration_seconds": elapsed_seconds, "history": history_payload, "test_metrics": test_metrics, "interpretation": "Training and held-out test evaluation completed. Use the saved metadata, dataset fingerprint, seed, and test metrics for reproducible comparison."}
    except Exception as exc:
        metadata.update({"status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(), "error": str(exc)})
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise


class TrainingJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self, config: TrainingConfig) -> dict[str, Any]:
        plan = training_plan(config)
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {"job_id": job_id, "status": "queued", "created_at_utc": datetime.now(timezone.utc).isoformat(), "plan": plan}
        thread = threading.Thread(target=self._execute, args=(job_id, config), daemon=True, name=f"training-{job_id[:8]}")
        thread.start()
        return self.get(job_id)

    def _execute(self, job_id: str, config: TrainingConfig) -> None:
        self._update(job_id, status="running", started_at_utc=datetime.now(timezone.utc).isoformat())
        try:
            result = run_training(config)
            self._update(job_id, status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(), result=result)
        except Exception as exc:
            self._update(job_id, status="failed", failed_at_utc=datetime.now(timezone.utc).isoformat(), error=str(exc))

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return dict(self._jobs[job_id])


training_jobs = TrainingJobManager()