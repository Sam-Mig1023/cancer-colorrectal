from __future__ import annotations

import argparse
import csv
import json
import statistics
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast cross-validation demo over precomputed folds using existing models.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["cnn_simple", "best_model"], choices=list(MODEL_SPECS))
    parser.add_argument("--samples-per-class-per-fold", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("experiments") / "cross_validation_phase6_demo")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_folds(dataset: Path, samples_per_class_per_fold: int) -> dict[int, list[dict[str, str]]]:
    folds_csv = dataset / "manifests" / "folds.csv"
    if not folds_csv.is_file():
        raise FileNotFoundError(f"Missing folds.csv: {folds_csv}")
    buckets: dict[int, dict[str, list[dict[str, str]]]] = {}
    with folds_csv.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            fold = int(row["fold"])
            buckets.setdefault(fold, {name: [] for name in CLASS_NAMES})
            if len(buckets[fold][row["class_name"]]) < samples_per_class_per_fold:
                buckets[fold][row["class_name"]].append(row)
    selected = {
        fold: [row for class_name in CLASS_NAMES for row in class_rows[class_name]]
        for fold, class_rows in sorted(buckets.items())
    }
    return selected


def load_image_batch(paths: list[Path], input_size: tuple[int, int]) -> np.ndarray:
    batch = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize(input_size, Image.Resampling.LANCZOS)
            batch.append(np.asarray(rgb, dtype=np.float32) / 255.0)
    return np.stack(batch, axis=0)


def predict(model: object, paths: list[Path], input_size: tuple[int, int], batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        result = model(load_image_batch(paths[start : start + batch_size], input_size), training=False)
        if isinstance(result, dict):
            result = next(iter(result.values()))
        outputs.append(np.asarray(result.numpy() if hasattr(result, "numpy") else result, dtype=np.float64))
    return np.vstack(outputs)


def font(size: int, bold: bool = False):
    for candidate in [
        PROJECT_ROOT / "fonts" / ("DejaVuSans-Bold.ttf" if bold else "fonts/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def draw_summary(path: Path, summary_rows: list[dict]) -> None:
    metrics = ["accuracy_mean", "macro_f1_mean", "macro_auc_ovr_mean", "mcc_mean"]
    image = Image.new("RGB", (1360, 780), (249, 251, 255))
    draw = ImageDraw.Draw(image)
    draw.text((44, 34), "Validacion cruzada demo K=5", fill=(28, 33, 40), font=font(30, True))
    draw.text((44, 74), "Promedio de metricas por fold usando muestras estratificadas", fill=(91, 101, 116), font=font(18))
    x0, y0, plot_h = 250, 145, 450
    group_w, bar_w, gap = 245, 46, 18
    for metric_index, metric in enumerate(metrics):
        x = x0 + metric_index * group_w
        draw.text((x, y0 + plot_h + 28), metric.replace("_mean", ""), fill=(91, 101, 116), font=font(14))
        for row_index, row in enumerate(summary_rows):
            value = float(row[metric] or 0)
            scaled = (value + 1) / 2 if metric == "mcc_mean" else value
            h = int(plot_h * max(0, min(1, scaled)))
            bx = x + row_index * (bar_w + gap)
            fill = (47, 97, 214) if row_index % 2 == 0 else (22, 145, 121)
            draw.rectangle((bx, y0 + plot_h - h, bx + bar_w, y0 + plot_h), fill=fill)
            draw.text((bx - 2, y0 + plot_h - h - 22), f"{value:.2f}", fill=(28, 33, 40), font=font(14))
    for index, row in enumerate(summary_rows):
        y = 650 + index * 28
        fill = (47, 97, 214) if index % 2 == 0 else (22, 145, 121)
        draw.rectangle((44, y + 4, 62, y + 22), fill=fill)
        draw.text((74, y), row["model_key"], fill=(28, 33, 40), font=font(16))
    image.save(path)


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    return round(statistics.mean(clean), 6), round(statistics.pstdev(clean), 6)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    folds = read_folds(dataset, args.samples_per_class_per_fold)
    print(f"Loaded {len(folds)} folds from {dataset}", flush=True)

    fold_rows: list[dict] = []
    for model_key in args.models:
        started = time.perf_counter()
        spec = MODEL_SPECS[model_key]
        model = model_service._get_model(spec)
        print(f"[{utc_now()}] {model_key}", flush=True)
        for fold, rows in folds.items():
            labels = [CLASS_NAMES.index(row["class_name"]) for row in rows]
            paths = [Path(row["path"]) for row in rows]
            probabilities = predict(model, paths, spec.input_size, args.batch_size)
            metrics = metrics_from_predictions(labels, probabilities, bootstrap_iterations=args.bootstrap_iterations, seed=args.seed)
            fold_row = {
                "model_key": model_key,
                "fold": fold,
                "samples": metrics["samples"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_auc_ovr": metrics["macro_auc_ovr"],
                "micro_auc_ovr": metrics["micro_auc_ovr"],
                "mcc": metrics["mcc"],
            }
            fold_rows.append(fold_row)
            print(json.dumps(fold_row), flush=True)
        print(f"{model_key} elapsed_seconds={round(time.perf_counter() - started, 3)}", flush=True)

    summary_rows: list[dict] = []
    for model_key in args.models:
        model_folds = [row for row in fold_rows if row["model_key"] == model_key]
        summary = {"model_key": model_key, "folds": len(model_folds), "samples_per_fold": model_folds[0]["samples"] if model_folds else 0}
        for metric in ("accuracy", "macro_f1", "macro_auc_ovr", "micro_auc_ovr", "mcc"):
            mean, std = mean_std([row[metric] for row in model_folds])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summary_rows.append(summary)
    summary_rows.sort(key=lambda row: (row["accuracy_mean"] or 0, row["macro_f1_mean"] or 0), reverse=True)

    with (output / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    with (output / "cross_validation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "created_at_utc": utc_now(),
        "dataset": str(dataset),
        "mode": "demo_existing_models_no_retraining",
        "limitation": "This validates the cross-validation workflow over stratified folds using existing trained models. Full scientific CV requires retraining inside each fold.",
        "samples_per_class_per_fold": args.samples_per_class_per_fold,
        "fold_metrics": fold_rows,
        "summary": summary_rows,
    }
    (output / "cross_validation_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_summary(output / "cross_validation_summary.png", summary_rows)
    (output / "phase6_cross_validation_report.md").write_text(
        "\n".join(
            [
                "# Fase 6 - Validacion cruzada demo",
                "",
                f"- Dataset: `{dataset}`",
                "- Folds: `5`",
                f"- Muestras por clase y fold: `{args.samples_per_class_per_fold}`",
                "- Modo: modelos ya entrenados, sin reentrenamiento por fold.",
                "- Nota: flujo rapido para presentacion; la version final cientifica debe reentrenar por fold.",
                "",
                "## Archivos",
                "",
                "- `fold_metrics.csv`",
                "- `cross_validation_summary.csv`",
                "- `cross_validation_summary.json`",
                "- `cross_validation_summary.png`",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
