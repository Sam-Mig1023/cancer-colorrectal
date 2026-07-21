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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import CLASS_NAMES, MODEL_SPECS
from backend.app.evaluation import metrics_from_predictions
from backend.app.inference import model_service
from backend.app.reporting import persist_prediction_artifact, persist_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized batch evaluation from split_manifest.csv.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["best_model", "mobilenetv2", "cnn_simple", "resnet50v2", "hybrid_attention", "hybrid_autoencoder"], choices=list(MODEL_SPECS))
    parser.add_argument("--partition", default="test", choices=["test"])
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--sampling", choices=["sequential", "stratified"], default="stratified")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("experiments") / "evaluations_phase4")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_summary(dataset: Path) -> dict:
    manifest_summary = dataset / "manifests" / "dataset_summary.json"
    if not manifest_summary.is_file():
        raise FileNotFoundError(f"Missing dataset summary: {manifest_summary}")
    return json.loads(manifest_summary.read_text(encoding="utf-8"))


def load_samples(dataset: Path, partition: str, max_samples: int, sampling: str) -> tuple[list[str], list[str], list[int], list[Path]]:
    split_manifest = dataset / "manifests" / "split_manifest.csv"
    if not split_manifest.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_manifest}")
    with split_manifest.open("r", newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == partition]
    if sampling == "stratified":
        per_class = max(1, max_samples // len(CLASS_NAMES))
        selected = []
        for class_name in CLASS_NAMES:
            selected.extend([row for row in rows if row["class_name"] == class_name][:per_class])
        remainder = max_samples - len(selected)
        if remainder > 0:
            selected_ids = {row["sample_id"] for row in selected}
            selected.extend([row for row in rows if row["sample_id"] not in selected_ids][:remainder])
        rows = selected[:max_samples]
    else:
        rows = rows[:max_samples]
    sample_ids: list[str] = []
    true_names: list[str] = []
    labels: list[int] = []
    paths: list[Path] = []
    for row in rows:
        sample_ids.append(row["sample_id"])
        true_names.append(row["class_name"])
        labels.append(CLASS_NAMES.index(row["class_name"]))
        paths.append(Path(row["split_path"]))
    return sample_ids, true_names, labels, paths


def load_image_batch(paths: list[Path], input_size: tuple[int, int]) -> np.ndarray:
    batch = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize(input_size, Image.Resampling.LANCZOS)
            batch.append(np.asarray(rgb, dtype=np.float32) / 255.0)
    return np.stack(batch, axis=0)


def predict(model: object, paths: list[Path], input_size: tuple[int, int], batch_size: int) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        image_batch = load_image_batch(paths[start : start + batch_size], input_size)
        result = model(image_batch, training=False)
        if isinstance(result, dict):
            result = next(iter(result.values()))
        array = np.asarray(result.numpy() if hasattr(result, "numpy") else result, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(CLASS_NAMES):
            raise RuntimeError(f"Unexpected prediction shape {array.shape}; expected Nx{len(CLASS_NAMES)}.")
        probabilities.append(array)
        if start == 0 or (start + batch_size) % 1024 == 0 or start + batch_size >= len(paths):
            print(f"    {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.vstack(probabilities)


def write_predictions_csv(path: Path, sample_ids: list[str], true_names: list[str], probabilities: np.ndarray) -> None:
    predicted = probabilities.argmax(axis=1)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["sample_id", "true_class", "predicted_class", "confidence_percent", *[f"prob_{name}_percent" for name in CLASS_NAMES]]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            row = {
                "sample_id": sample_id,
                "true_class": true_names[index],
                "predicted_class": CLASS_NAMES[int(predicted[index])],
                "confidence_percent": round(float(probabilities[index, predicted[index]]) * 100, 6),
            }
            row.update({f"prob_{name}_percent": round(float(probabilities[index, class_index]) * 100, 6) for class_index, name in enumerate(CLASS_NAMES)})
            writer.writerow(row)


def font(size: int, bold: bool = False):
    for candidate in [
        PROJECT_ROOT / "fonts" / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def draw_confusion_matrix(path: Path, model_key: str, matrix: list[list[int]]) -> None:
    values = np.asarray(matrix)
    cell = 78
    left = 150
    top = 120
    image = Image.new("RGB", (left + cell * 9 + 80, top + cell * 9 + 90), (249, 251, 255))
    draw = ImageDraw.Draw(image)
    draw.text((38, 32), f"Matriz de confusion: {model_key}", fill=(28, 33, 40), font=font(30, True))
    max_value = max(int(values.max()), 1)
    for index, class_name in enumerate(CLASS_NAMES):
        draw.text((left + index * cell + 14, top - 34), class_name, fill=(91, 101, 116), font=font(14))
        draw.text((42, top + index * cell + 28), class_name, fill=(91, 101, 116), font=font(14))
    for row in range(9):
        for col in range(9):
            value = int(values[row, col])
            intensity = value / max_value
            fill = tuple(int(255 * (1 - intensity) + base * intensity) for base in (47, 97, 214))
            x = left + col * cell
            y = top + row * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=fill, outline=(220, 225, 232))
            draw.text((x + 12, y + 28), str(value), fill=(28, 33, 40), font=font(14))
    image.save(path)


def draw_comparison(path: Path, rows: list[dict]) -> None:
    metrics = ["accuracy", "macro_f1", "macro_auc_ovr", "mcc"]
    image = Image.new("RGB", (1420, 860), (249, 251, 255))
    draw = ImageDraw.Draw(image)
    draw.text((45, 35), "Evaluacion reproducible sobre test", fill=(28, 33, 40), font=font(30, True))
    draw.text((45, 75), "Comparacion de modelos con predicciones reales", fill=(91, 101, 116), font=font(18))
    x0, y0, plot_h = 270, 150, 510
    group_w, bar_w, gap = 245, 34, 12
    for metric_index, metric in enumerate(metrics):
        x = x0 + metric_index * group_w
        draw.text((x, y0 + plot_h + 26), metric, fill=(91, 101, 116), font=font(14))
        for row_index, row in enumerate(rows):
            value = float(row.get(metric) or 0)
            scaled = (value + 1) / 2 if metric == "mcc" else value
            scaled = max(0, min(1, scaled))
            h = int(plot_h * scaled)
            bx = x + row_index * (bar_w + gap)
            draw.rectangle((bx, y0 + plot_h - h, bx + bar_w, y0 + plot_h), fill=(47, 97, 214) if row_index % 2 == 0 else (22, 145, 121))
            draw.text((bx - 4, y0 + plot_h - h - 22), f"{value:.2f}", fill=(28, 33, 40), font=font(14))
    for index, row in enumerate(rows):
        y = 700 + index * 24
        draw.rectangle((45, y + 3, 60, y + 18), fill=(47, 97, 214) if index % 2 == 0 else (22, 145, 121))
        draw.text((70, y), row["model_key"], fill=(28, 33, 40), font=font(14))
    image.save(path)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = load_summary(dataset)
    sample_ids, true_names, labels, paths = load_samples(dataset, args.partition, args.max_samples, args.sampling)
    if not paths:
        raise SystemExit("No test samples found.")
    print(f"Evaluating {len(paths)} samples from {dataset}", flush=True)

    comparison_rows: list[dict] = []
    for model_key in args.models:
        started = time.perf_counter()
        spec = MODEL_SPECS[model_key]
        model_output = output / model_key
        model_output.mkdir(parents=True, exist_ok=True)
        print(f"[{utc_now()}] {model_key}", flush=True)
        model = model_service._get_model(spec)
        probabilities = predict(model, paths, spec.input_size, args.batch_size)
        metrics = metrics_from_predictions(labels, probabilities, bootstrap_iterations=args.bootstrap_iterations, seed=args.seed)
        origin = {
            "model_key": model_key,
            "model_name": spec.display_name,
            "model_artifact": str(spec.path),
            "dataset_fingerprint_sha256": summary.get("backend_dataset_fingerprint_sha256") or "4adeec449438176b33beb06f1a6e9e06437e7bda7294fbc92b96ca4618174ee3",
            "manifest_content_fingerprint_sha256": summary["dataset_fingerprint_sha256"],
            "partition": args.partition,
            "evaluated_at_utc": utc_now(),
            "seed": args.seed,
            "experiment_id": f"phase4_{model_key}",
            "max_samples": len(paths),
            "batch_size": args.batch_size,
            "bootstrap_iterations": args.bootstrap_iterations,
            "sampling": args.sampling,
            "patient_id_regex": None,
        }
        metrics["origin"] = origin
        prediction_artifact = persist_prediction_artifact(origin, labels, probabilities, sample_ids)
        record = persist_result("evaluation", metrics)
        (model_output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (model_output / "artifact_paths.json").write_text(json.dumps({"prediction_artifact": prediction_artifact, "record": record}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_predictions_csv(model_output / "predictions.csv", sample_ids, true_names, probabilities)
        draw_confusion_matrix(model_output / "confusion_matrix.png", model_key, metrics["confusion_matrix"])
        row = {
            "model_key": model_key,
            "model_name": spec.display_name,
            "samples": metrics["samples"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_auc_ovr": metrics["macro_auc_ovr"],
            "micro_auc_ovr": metrics["micro_auc_ovr"],
            "mcc": metrics["mcc"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "prediction_artifact_path": prediction_artifact["artifact_path"],
            "record_path": record["record_path"],
        }
        comparison_rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    comparison_rows.sort(key=lambda item: (item["accuracy"], item["macro_f1"] or 0), reverse=True)
    with (output / "model_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    (output / "model_comparison.json").write_text(json.dumps(comparison_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_comparison(output / "model_comparison.png", comparison_rows)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

