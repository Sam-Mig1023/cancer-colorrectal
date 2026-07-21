from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import CLASS_NAMES, MODEL_SPECS
from backend.app.dataset import iter_samples, summarize_dataset
from backend.app.evaluation import metrics_from_predictions
from backend.app.inference import model_service
from backend.app.reporting import persist_prediction_artifact, persist_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate existing project models over the held-out test split.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    parser.add_argument("--partition", default="test", choices=["test"])
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("experiments") / "evaluations_phase4")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("fonts/DejaVuSans-Bold.ttf" if bold else "fonts/DejaVuSans.ttf"),
        Path(__file__).resolve().parents[1] / "fonts" / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


FONT_TITLE = font(30, True)
FONT_BODY = font(18)
FONT_SMALL = font(14)
COLORS = {
    "bg": (249, 251, 255),
    "ink": (28, 33, 40),
    "muted": (91, 101, 116),
    "grid": (220, 225, 232),
    "blue": (47, 97, 214),
    "teal": (22, 145, 121),
    "red": (190, 65, 55),
    "white": (255, 255, 255),
}


def collect_test_samples(partition: str, max_samples: int) -> tuple[list[str], list[str], list[int], list[Path]]:
    sample_ids: list[str] = []
    class_names: list[str] = []
    labels: list[int] = []
    paths: list[Path] = []
    for sample in iter_samples(partition=partition):
        if len(paths) >= max_samples:
            break
        sample_ids.append(sample.path.stem)
        class_names.append(sample.class_name)
        labels.append(CLASS_NAMES.index(sample.class_name))
        paths.append(sample.path)
    return sample_ids, class_names, labels, paths


def load_image_batch(paths: list[Path], input_size: tuple[int, int]) -> np.ndarray:
    images = []
    for path in paths:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize(input_size, Image.Resampling.LANCZOS)
            images.append(np.asarray(resized, dtype=np.float32) / 255.0)
    return np.stack(images, axis=0)


def predict_batches(model: object, paths: list[Path], input_size: tuple[int, int], batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = load_image_batch(batch_paths, input_size)
        result = model(batch, training=False)
        if isinstance(result, dict):
            result = next(iter(result.values()))
        probabilities = np.asarray(result.numpy() if hasattr(result, "numpy") else result, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
            raise RuntimeError(f"Unexpected prediction shape {probabilities.shape}; expected Nx{len(CLASS_NAMES)}.")
        outputs.append(probabilities)
        print(f"  predicted {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.vstack(outputs)


def write_predictions_csv(path: Path, sample_ids: list[str], true_names: list[str], probabilities: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predicted = probabilities.argmax(axis=1)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["sample_id", "true_class", "predicted_class", "confidence", *[f"prob_{name}" for name in CLASS_NAMES]]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            row = {
                "sample_id": sample_id,
                "true_class": true_names[index],
                "predicted_class": CLASS_NAMES[int(predicted[index])],
                "confidence": round(float(probabilities[index, predicted[index]]) * 100, 6),
            }
            row.update({f"prob_{name}": round(float(probabilities[index, class_index]) * 100, 6) for class_index, name in enumerate(CLASS_NAMES)})
            writer.writerow(row)


def draw_confusion_matrix(path: Path, title: str, matrix: list[list[int]]) -> None:
    cell = 78
    left = 150
    top = 120
    width = left + cell * len(CLASS_NAMES) + 70
    height = top + cell * len(CLASS_NAMES) + 90
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((35, 30), title, fill=COLORS["ink"], font=FONT_TITLE)
    values = np.asarray(matrix)
    max_value = max(int(values.max()), 1)
    for i, class_name in enumerate(CLASS_NAMES):
        draw.text((left + i * cell + 15, top - 36), class_name, fill=COLORS["muted"], font=FONT_SMALL)
        draw.text((42, top + i * cell + 28), class_name, fill=COLORS["muted"], font=FONT_SMALL)
    for row in range(len(CLASS_NAMES)):
        for col in range(len(CLASS_NAMES)):
            value = int(values[row, col])
            intensity = value / max_value
            color = tuple(int(COLORS["white"][k] * (1 - intensity) + COLORS["blue"][k] * intensity) for k in range(3))
            x = left + col * cell
            y = top + row * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=color, outline=COLORS["grid"])
            draw.text((x + 12, y + 28), str(value), fill=COLORS["ink"], font=FONT_SMALL)
    image.save(path)


def draw_model_comparison(path: Path, results: list[dict]) -> None:
    metrics = ["accuracy", "macro_f1", "macro_auc_ovr", "mcc"]
    width, height = 1400, 840
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((45, 36), "Comparacion de modelos sobre test", fill=COLORS["ink"], font=FONT_TITLE)
    draw.text((45, 76), "Metricas calculadas desde predicciones reales", fill=COLORS["muted"], font=FONT_BODY)
    x0, y0 = 260, 150
    group_w = 240
    bar_w = 34
    gap = 12
    plot_h = 520
    for m_index, metric in enumerate(metrics):
        x = x0 + m_index * group_w
        draw.text((x, y0 + plot_h + 28), metric, fill=COLORS["muted"], font=FONT_SMALL)
        for r_index, result in enumerate(results):
            value = result.get(metric) or 0
            if metric == "mcc":
                scaled = max(0.0, min(1.0, (value + 1) / 2))
            else:
                scaled = max(0.0, min(1.0, value))
            h = int(plot_h * scaled)
            bx = x + r_index * (bar_w + gap)
            draw.rectangle((bx, y0 + plot_h - h, bx + bar_w, y0 + plot_h), fill=COLORS["teal" if r_index % 2 else "blue"])
            draw.text((bx - 4, y0 + plot_h - h - 22), f"{value:.2f}", fill=COLORS["ink"], font=FONT_SMALL)
    legend_y = 690
    for index, result in enumerate(results):
        y = legend_y + index * 24
        draw.rectangle((45, y + 3, 60, y + 18), fill=COLORS["teal" if index % 2 else "blue"])
        draw.text((70, y), result["model_key"], fill=COLORS["ink"], font=FONT_SMALL)
    image.save(path)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary = summarize_dataset()
    sample_ids, true_names, labels, paths = collect_test_samples(args.partition, args.max_samples)
    if not paths:
        raise SystemExit("No test samples found.")
    print(f"Loaded {len(paths)} test samples from {summary['dataset_root']}", flush=True)

    all_results: list[dict] = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        started = time.perf_counter()
        model_dir = output / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{utc_now()}] Evaluating {model_key}", flush=True)
        model = model_service._get_model(spec)
        probabilities = predict_batches(model, paths, spec.input_size, args.batch_size)
        metrics = metrics_from_predictions(labels, probabilities, bootstrap_iterations=args.bootstrap_iterations, seed=args.seed)
        origin = {
            "model_key": model_key,
            "model_name": spec.display_name,
            "model_artifact": str(spec.path),
            "dataset_fingerprint_sha256": summary["dataset_fingerprint_sha256"],
            "partition": args.partition,
            "evaluated_at_utc": utc_now(),
            "seed": args.seed,
            "experiment_id": f"phase4_{model_key}",
            "max_samples": len(paths),
            "patient_id_regex": None,
            "batch_size": args.batch_size,
            "evaluation_mode": "batch_script_same_backend_preprocessing",
        }
        metrics["origin"] = origin
        artifact = persist_prediction_artifact(origin, labels, probabilities, sample_ids)
        record = persist_result("evaluation", metrics)
        write_predictions_csv(model_dir / "predictions.csv", sample_ids, true_names, probabilities)
        (model_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (model_dir / "artifact_paths.json").write_text(json.dumps({"prediction_artifact": artifact, "record": record}, ensure_ascii=False, indent=2), encoding="utf-8")
        draw_confusion_matrix(model_dir / "confusion_matrix.png", f"Matriz de confusion: {model_key}", metrics["confusion_matrix"])
        elapsed = round(time.perf_counter() - started, 3)
        row = {
            "model_key": model_key,
            "model_name": spec.display_name,
            "samples": metrics["samples"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_auc_ovr": metrics["macro_auc_ovr"],
            "micro_auc_ovr": metrics["micro_auc_ovr"],
            "mcc": metrics["mcc"],
            "elapsed_seconds": elapsed,
            "prediction_artifact_path": artifact["artifact_path"],
            "record_path": record["record_path"],
        }
        all_results.append(row)
        print(json.dumps(row, indent=2), flush=True)

    all_results.sort(key=lambda item: (item["accuracy"], item["macro_f1"] or 0), reverse=True)
    with (output / "model_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_results[0]))
        writer.writeheader()
        writer.writerows(all_results)
    (output / "model_comparison.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_model_comparison(output / "model_comparison.png", all_results)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()



