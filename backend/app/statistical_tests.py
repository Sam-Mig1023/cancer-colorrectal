from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import CLASS_NAMES
from .evaluation import _core_metrics


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end
    output = np.empty(len(values), dtype=float)
    output[order] = ranks
    return output


def _fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    model_count, sample_count = predictions.shape
    negative_count = sample_count - positive_count
    positives = predictions[:, :positive_count]
    negatives = predictions[:, positive_count:]
    tx = np.empty((model_count, positive_count))
    ty = np.empty((model_count, negative_count))
    tz = np.empty((model_count, sample_count))
    for index in range(model_count):
        tx[index] = _midrank(positives[index])
        ty[index] = _midrank(negatives[index])
        tz[index] = _midrank(predictions[index])
    aucs = tz[:, :positive_count].sum(axis=1) / positive_count / negative_count - (positive_count + 1.0) / (2.0 * negative_count)
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / positive_count + sy / negative_count


def delong_auc_test(labels: Sequence[int], first_scores: Sequence[float], second_scores: Sequence[float]) -> dict[str, Any]:
    actual = np.asarray(labels, dtype=int)
    first = np.asarray(first_scores, dtype=float)
    second = np.asarray(second_scores, dtype=float)
    if len(actual) < 2 or len(actual) != len(first) or len(actual) != len(second):
        raise ValueError("DeLong requires equally sized labels and score vectors with at least two samples.")
    if set(np.unique(actual)) - {0, 1} or actual.sum() in {0, len(actual)}:
        raise ValueError("DeLong requires binary labels with positive and negative observations.")
    positive_count = int(actual.sum())
    order = np.argsort(-actual, kind="mergesort")
    aucs, covariance = _fast_delong(np.vstack([first, second])[:, order], positive_count)
    difference = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    standard_error = math.sqrt(max(variance, 0.0))
    z_score = difference / standard_error if standard_error else 0.0
    p_value = float(math.erfc(abs(z_score) / math.sqrt(2))) if standard_error else 1.0
    return {"method": "delong_auc", "auc_first": round(float(aucs[0]), 6), "auc_second": round(float(aucs[1]), 6), "effect_auc_difference": round(difference, 6), "standard_error": round(standard_error, 6), "ci_95": [round(difference - 1.9599639845 * standard_error, 6), round(difference + 1.9599639845 * standard_error, 6)], "z_score": round(z_score, 6), "p_value": p_value, "significant": p_value < 0.05}


def mcnemar_test(first_correct: Sequence[bool], second_correct: Sequence[bool]) -> dict[str, Any]:
    first = np.asarray(first_correct, dtype=bool)
    second = np.asarray(second_correct, dtype=bool)
    if len(first) != len(second) or not len(first):
        raise ValueError("McNemar requires paired non-empty correctness vectors.")
    b = int(np.sum(first & ~second))
    c = int(np.sum(~first & second))
    chi2 = ((abs(b - c) - 1) ** 2 / (b + c)) if b + c else 0.0
    p_value = float(math.erfc(math.sqrt(chi2 / 2)))
    return {"method": "mcnemar_chi_squared_continuity_correction", "table": [[int(np.sum(first & second)), b], [c, int(np.sum(~first & ~second))]], "discordant_first_only": b, "discordant_second_only": c, "effect_discordant_difference": b - c, "chi2": round(chi2, 6), "p_value": p_value, "significant": p_value < 0.05}


def _bootstrap_groups(sample_count: int, patient_ids: Sequence[str] | None) -> tuple[list[np.ndarray], str, str | None]:
    if patient_ids is None:
        return [np.asarray([index]) for index in range(sample_count)], "image", "No patient identifier was stored; paired bootstrap is performed at image level."
    if len(patient_ids) != sample_count:
        raise ValueError("Patient identifiers must align with the paired predictions.")
    grouped: dict[str, list[int]] = {}
    for index, patient_id in enumerate(patient_ids):
        grouped.setdefault(str(patient_id), []).append(index)
    return [np.asarray(indices) for indices in grouped.values()], "patient", None


def paired_bootstrap_comparison(labels: Sequence[int], first_probabilities: np.ndarray, second_probabilities: np.ndarray, patient_ids: Sequence[str] | None = None, iterations: int = 2000, seed: int = 42) -> dict[str, Any]:
    actual = np.asarray(labels, dtype=int)
    first = np.asarray(first_probabilities, dtype=float)
    second = np.asarray(second_probabilities, dtype=float)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != len(actual) or first.shape[1] != len(CLASS_NAMES):
        raise ValueError("Paired bootstrap requires two aligned Nx9 probability matrices and matching labels.")
    if not 100 <= iterations <= 5000:
        raise ValueError("iterations must be between 100 and 5000.")
    groups, unit, limitation = _bootstrap_groups(len(actual), patient_ids)
    if len(groups) < 2:
        raise ValueError("Paired bootstrap requires at least two independent units.")
    metrics = ("accuracy", "macro_f1", "mcc", "macro_auc_ovr", "micro_auc_ovr")
    observed_first = _core_metrics(actual, first, include_roc=False)
    observed_second = _core_metrics(actual, second, include_roc=False)
    generator = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(iterations):
        selection = generator.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selection])
        first_metrics = _core_metrics(actual[indices], first[indices], include_roc=False)
        second_metrics = _core_metrics(actual[indices], second[indices], include_roc=False)
        for metric in metrics:
            if first_metrics[metric] is not None and second_metrics[metric] is not None:
                estimates[metric].append(float(first_metrics[metric] - second_metrics[metric]))
    result = {}
    for metric, values in estimates.items():
        if not values or observed_first[metric] is None or observed_second[metric] is None:
            continue
        lower, upper = np.quantile(values, [0.025, 0.975])
        difference = float(observed_first[metric] - observed_second[metric])
        result[metric] = {"first": observed_first[metric], "second": observed_second[metric], "effect_difference": round(difference, 6), "ci_95": [round(float(lower), 6), round(float(upper), 6)]}
    return {"method": "paired_nonparametric_bootstrap", "iterations": iterations, "unit": unit, "limitation": limitation, "metrics": result}


def holm_adjustment(p_values: Sequence[float]) -> list[float]:
    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must be between 0 and 1.")
    ordered = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (original_index, p_value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (total - rank) * p_value))
        adjusted[original_index] = round(running_max, 6)
    return adjusted


def compare_paired_predictions(labels: Sequence[int], first_probabilities: np.ndarray, second_probabilities: np.ndarray, class_name: str, first_name: str = "first_model", second_name: str = "second_model", patient_ids: Sequence[str] | None = None, bootstrap_iterations: int = 2000, seed: int = 42) -> dict[str, Any]:
    if class_name not in CLASS_NAMES:
        raise ValueError("class_name must be one of the configured histology classes.")
    actual = np.asarray(labels, dtype=int)
    first = np.asarray(first_probabilities, dtype=float)
    second = np.asarray(second_probabilities, dtype=float)
    if first.shape != second.shape or len(actual) != first.shape[0]:
        raise ValueError("Models must have aligned predictions for the same samples.")
    first_correct = first.argmax(axis=1) == actual
    second_correct = second.argmax(axis=1) == actual
    index = CLASS_NAMES.index(class_name)
    delong = delong_auc_test((actual == index).astype(int), first[:, index], second[:, index])
    mcnemar = mcnemar_test(first_correct, second_correct)
    raw_p_values = [mcnemar["p_value"], delong["p_value"]]
    adjusted = holm_adjustment(raw_p_values)
    mcnemar["p_value_holm"] = adjusted[0]
    mcnemar["significant_holm"] = adjusted[0] < 0.05
    delong["p_value_holm"] = adjusted[1]
    delong["significant_holm"] = adjusted[1] < 0.05
    bootstrap = paired_bootstrap_comparison(actual, first, second, patient_ids=patient_ids, iterations=bootstrap_iterations, seed=seed)
    return {"source": "reproducible_paired_model_comparison", "models": {"first": first_name, "second": second_name}, "sample_count": len(actual), "class_for_delong": class_name, "mcnemar": mcnemar, "delong": delong, "paired_bootstrap": bootstrap, "multiple_comparison_correction": {"method": "holm_bonferroni", "family": ["mcnemar", "delong_auc"], "raw_p_values": raw_p_values, "adjusted_p_values": adjusted}, "interpretation": "The comparison uses paired predictions from the same held-out samples. Effect sizes and confidence intervals should be considered alongside adjusted p-values."}

def compare_prediction_artifacts(first_artifact_path: str, second_artifact_path: str, class_name: str, bootstrap_iterations: int = 2000, seed: int = 42) -> dict[str, Any]:
    from .reporting import load_prediction_artifact

    first = load_prediction_artifact(first_artifact_path)
    second = load_prediction_artifact(second_artifact_path)
    if not np.array_equal(first["labels"], second["labels"]):
        raise ValueError("Prediction artifacts do not contain the same true labels.")
    if first["sample_ids"] != second["sample_ids"]:
        raise ValueError("Prediction artifacts are not aligned to the same ordered samples.")
    first_origin = first["metadata"]
    second_origin = second["metadata"]
    for field in ("dataset_fingerprint_sha256", "partition"):
        if first_origin.get(field) != second_origin.get(field):
            raise ValueError(f"Prediction artifacts have different {field} values and cannot be compared.")
    if first_origin.get("partition") != "test":
        raise ValueError("Only prediction artifacts from the held-out test partition may be compared.")
    patient_ids = first["patient_ids"]
    if patient_ids != second["patient_ids"]:
        raise ValueError("Prediction artifacts have different patient identifiers and cannot be compared.")
    result = compare_paired_predictions(first["labels"], first["probabilities"], second["probabilities"], class_name=class_name, first_name=first_origin.get("model_name", "first_model"), second_name=second_origin.get("model_name", "second_model"), patient_ids=patient_ids, bootstrap_iterations=bootstrap_iterations, seed=seed)
    result["origin"] = {"dataset_fingerprint_sha256": first_origin.get("dataset_fingerprint_sha256"), "partition": "test", "first_artifact": first["path"], "second_artifact": second["path"], "seed": seed}
    return result