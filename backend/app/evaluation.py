from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import CLASS_NAMES, MODEL_SPECS
from .dataset import iter_samples, summarize_dataset
from .inference import model_service


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _roc_curve(labels: np.ndarray, scores: np.ndarray) -> dict | None:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)
    keep = np.r_[np.where(np.diff(sorted_scores))[0], len(sorted_scores) - 1]
    tpr = np.r_[0.0, tps[keep] / positives, 1.0]
    fpr = np.r_[0.0, fps[keep] / negatives, 1.0]
    thresholds = np.r_[float("inf"), sorted_scores[keep], float("-inf")]
    return {
        "fpr": [round(float(value), 6) for value in fpr],
        "tpr": [round(float(value), 6) for value in tpr],
        "thresholds": [None if not np.isfinite(value) else round(float(value), 6) for value in thresholds],
        "auc": round(float(np.trapz(tpr, fpr)), 6),
    }


def _core_metrics(labels: np.ndarray, scores: np.ndarray, include_roc: bool) -> dict:
    predicted = scores.argmax(axis=1)
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for actual, estimate in zip(labels, predicted):
        matrix[actual, estimate] += 1
    per_class = {}
    for index, class_name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        tn = int(matrix.sum() - tp - fp - fn)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        roc = _roc_curve((labels == index).astype(int), scores[:, index])
        per_class[class_name] = {
            "support": int(matrix[index, :].sum()),
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": _safe_divide(tn, tn + fp),
            "f1": round(2 * precision * recall / (precision + recall), 6) if precision is not None and recall is not None and precision + recall else None,
            "auc": roc["auc"] if roc else None,
        }
        if include_roc:
            per_class[class_name]["roc"] = roc
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total)
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    numerator = np.trace(matrix) * total - np.dot(row_totals, col_totals)
    denominator = np.sqrt((total ** 2 - np.dot(row_totals, row_totals)) * (total ** 2 - np.dot(col_totals, col_totals)))
    auc_values = [item["auc"] for item in per_class.values() if item["auc"] is not None]
    f1_values = [item["f1"] for item in per_class.values() if item["f1"] is not None]
    micro_labels = np.eye(len(CLASS_NAMES), dtype=int)[labels].ravel()
    micro_scores = scores.ravel()
    micro_roc = _roc_curve(micro_labels, micro_scores)
    return {
        "samples": total,
        "accuracy": round(accuracy, 6),
        "macro_f1": round(float(np.mean(f1_values)), 6) if f1_values else None,
        "macro_auc_ovr": round(float(np.mean(auc_values)), 6) if auc_values else None,
        "micro_auc_ovr": micro_roc["auc"] if micro_roc else None,
        "mcc": round(float(numerator / denominator), 6) if denominator else 0.0,
        "classes": list(CLASS_NAMES),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "micro_roc": micro_roc if include_roc else None,
    }


def _bootstrap_intervals(labels: np.ndarray, scores: np.ndarray, group_ids: Sequence[str] | None, iterations: int, seed: int) -> dict:
    generator = np.random.default_rng(seed)
    if group_ids is None:
        groups = [np.asarray([index]) for index in range(len(labels))]
        unit = "image"
        limitation = "No patient identifier was supplied; confidence intervals were bootstrapped at image level and may be optimistic when images from the same patient are correlated."
    else:
        if len(group_ids) != len(labels):
            raise ValueError("patient identifiers must align with evaluated samples.")
        grouped_indices: dict[str, list[int]] = {}
        for index, group_id in enumerate(group_ids):
            grouped_indices.setdefault(group_id, []).append(index)
        groups = [np.asarray(indices) for indices in grouped_indices.values()]
        unit = "patient"
        limitation = None
    if len(groups) < 2:
        return {"unit": unit, "iterations": 0, "limitation": "At least two independent units are required for bootstrap confidence intervals.", "metrics": {}}
    estimates: dict[str, list[float]] = {name: [] for name in ("accuracy", "macro_f1", "mcc", "macro_auc_ovr", "micro_auc_ovr")}
    for _ in range(iterations):
        selected = generator.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        metrics = _core_metrics(labels[indices], scores[indices], include_roc=False)
        for name, values in estimates.items():
            value = metrics[name]
            if value is not None:
                values.append(float(value))
    intervals = {}
    point_estimates = _core_metrics(labels, scores, include_roc=False)
    for name, values in estimates.items():
        if values:
            lower, upper = np.quantile(values, [0.025, 0.975])
            intervals[name] = {"estimate": point_estimates[name], "lower": round(float(lower), 6), "upper": round(float(upper), 6)}
    return {"unit": unit, "iterations": iterations, "limitation": limitation, "metrics": intervals}


def metrics_from_predictions(y_true: Iterable[int], probabilities: np.ndarray, patient_ids: Sequence[str] | None = None, bootstrap_iterations: int = 500, seed: int = 42) -> dict:
    labels = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(CLASS_NAMES) or len(labels) != len(scores):
        raise ValueError("Predictions must be an Nx9 probability matrix aligned with the true labels.")
    if len(labels) == 0:
        raise ValueError("At least one prediction is required for evaluation.")
    if np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
        raise ValueError("True labels must map to the configured histology classes.")
    if not 100 <= bootstrap_iterations <= 5000:
        raise ValueError("bootstrap_iterations must be between 100 and 5000.")
    result = _core_metrics(labels, scores, include_roc=True)
    result.update({
        "source": "reproducible_evaluation",
        "confidence_intervals_95": _bootstrap_intervals(labels, scores, patient_ids, bootstrap_iterations, seed),
        "interpretation": "Metrics and ROC curves were calculated from the selected held-out test partition. Review per-class sensitivity and specificity before comparing models.",
    })
    return result


def _patient_id_from_path(path: Path, pattern: str | None) -> str | None:
    if not pattern:
        return None
    match = re.search(pattern, path.name)
    if not match:
        raise ValueError(f"patient_id_regex did not match evaluated file: {path.name}")
    return match.group(1) if match.groups() else match.group(0)


def evaluate_model(model_key: str, dataset_path: str | None = None, partition: str = "test", max_samples: int = 1000, bootstrap_iterations: int = 500, seed: int = 42, patient_id_regex: str | None = None, experiment_id: str | None = None) -> dict:
    from .reporting import persist_prediction_artifact

    if partition != "test":
        raise ValueError("Medical model evaluation must use the held-out 'test' partition.")
    if not 1 <= max_samples <= 10000:
        raise ValueError("max_samples must be between 1 and 10000.")
    summary = summarize_dataset(dataset_path)
    if partition not in summary["partitions"]:
        raise ValueError("The configured dataset does not provide a held-out 'test' partition.")
    root = Path(summary["dataset_root"])
    true_labels: list[int] = []
    predicted_probabilities: list[np.ndarray] = []
    sample_ids: list[str] = []
    patient_ids: list[str] | None = [] if patient_id_regex else None
    for sample in iter_samples(dataset_path, partition):
        if len(true_labels) >= max_samples:
            break
        _, probabilities = model_service.predict(sample.path.read_bytes(), model_key)
        true_labels.append(CLASS_NAMES.index(sample.class_name))
        predicted_probabilities.append(probabilities)
        sample_ids.append(sample.path.relative_to(root).as_posix())
        if patient_ids is not None:
            patient_ids.append(_patient_id_from_path(sample.path, patient_id_regex))
    result = metrics_from_predictions(true_labels, np.asarray(predicted_probabilities), patient_ids=patient_ids, bootstrap_iterations=bootstrap_iterations, seed=seed)
    spec = MODEL_SPECS[model_key]
    origin = {"model_key": model_key, "model_name": spec.display_name, "model_artifact": str(spec.path), "dataset_fingerprint_sha256": summary["dataset_fingerprint_sha256"], "partition": partition, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "experiment_id": experiment_id, "max_samples": max_samples, "patient_id_regex": patient_id_regex}
    result["origin"] = origin
    result["prediction_artifact"] = persist_prediction_artifact(origin, true_labels, np.asarray(predicted_probabilities), sample_ids, patient_ids)
    return result