from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import CLASS_NAMES, PROJECT_ROOT

SOURCE_LABEL = "legacy_precomputed_from_streamlit"
TRAINING_PLOT_PATH = PROJECT_ROOT / "Graficos de entrenamiento de los modelos.png"
MODEL_KEY_BY_LEGACY_NAME = {
    "CNN Simple": "cnn_simple",
    "ResNet50V2": "resnet50v2",
    "MobileNetV2 Base": "best_model",
    "Hybrid Attention": "hybrid_attention",
    "Hybrid Autoencoder": "hybrid_autoencoder",
}
LEGACY_NAME_BY_MODEL_KEY = {value: key for key, value in MODEL_KEY_BY_LEGACY_NAME.items()}

CONFUSION_MATRICES = {
    "CNN Simple": [[120, 15, 10, 5, 8, 7, 12, 10, 13], [10, 130, 8, 6, 4, 5, 7, 8, 12], [12, 8, 125, 10, 5, 6, 8, 10, 6], [5, 6, 8, 140, 7, 5, 4, 8, 7], [8, 5, 6, 9, 135, 10, 7, 6, 4], [7, 4, 5, 6, 8, 130, 9, 8, 3], [10, 7, 6, 5, 6, 8, 140, 5, 3], [12, 8, 10, 7, 5, 6, 4, 135, 3], [15, 12, 8, 6, 5, 4, 5, 7, 133]],
    "ResNet50V2": [[130, 10, 8, 5, 4, 3, 5, 7, 8], [8, 140, 5, 4, 3, 2, 4, 6, 8], [7, 5, 138, 6, 3, 2, 4, 5, 5], [4, 3, 5, 145, 4, 3, 2, 5, 4], [3, 2, 3, 5, 142, 6, 4, 3, 2], [2, 1, 2, 4, 5, 140, 5, 4, 2], [4, 3, 3, 2, 3, 4, 145, 3, 3], [5, 4, 4, 3, 2, 3, 2, 142, 5], [7, 6, 5, 4, 3, 2, 3, 4, 136]],
    "MobileNetV2 Base": [[145, 3, 2, 1, 1, 1, 1, 2, 4], [2, 148, 1, 1, 1, 1, 1, 2, 3], [1, 1, 147, 2, 1, 1, 1, 2, 4], [1, 1, 2, 149, 1, 1, 1, 1, 3], [1, 1, 1, 1, 148, 2, 1, 1, 2], [1, 1, 1, 1, 2, 147, 1, 1, 1], [1, 1, 1, 1, 1, 1, 149, 1, 2], [2, 1, 1, 1, 1, 1, 1, 148, 2], [3, 2, 2, 2, 1, 1, 1, 2, 141]],
    "Hybrid Attention": [[20, 15, 15, 15, 15, 15, 15, 15, 15], [15, 20, 15, 15, 15, 15, 15, 15, 15], [15, 15, 20, 15, 15, 15, 15, 15, 15], [15, 15, 15, 20, 15, 15, 15, 15, 15], [15, 15, 15, 15, 20, 15, 15, 15, 15], [15, 15, 15, 15, 15, 20, 15, 15, 15], [15, 15, 15, 15, 15, 15, 20, 15, 15], [15, 15, 15, 15, 15, 15, 15, 20, 15], [15, 15, 15, 15, 15, 15, 15, 15, 20]],
    "Hybrid Autoencoder": [[20, 15, 15, 15, 15, 15, 15, 15, 15], [15, 20, 15, 15, 15, 15, 15, 15, 15], [15, 15, 20, 15, 15, 15, 15, 15, 15], [15, 15, 15, 20, 15, 15, 15, 15, 15], [15, 15, 15, 15, 20, 15, 15, 15, 15], [15, 15, 15, 15, 15, 20, 15, 15, 15], [15, 15, 15, 15, 15, 15, 20, 15, 15], [15, 15, 15, 15, 15, 15, 15, 20, 15], [15, 15, 15, 15, 15, 15, 15, 15, 20]],
}
ROC_EXPONENTS = {"CNN Simple": (0.65, 0.85), "ResNet50V2": (0.6, 0.87), "MobileNetV2 Base": (0.4, 0.96), "Hybrid Attention": (1.2, 0.65), "Hybrid Autoencoder": (1.1, 0.68)}
MODEL_DETAILS = {
    "CNN Simple": {"accuracy": 0.5913, "validation_loss": 1.35, "training_time_hours": 4.15, "architecture": {"en": ["Simple convolutional neural network architecture", "3 convolutional layers with max pooling", "Batch normalization", "Dense layer with 256 neurons", "Dropout 50%", "ReLU activation"], "es": ["Arquitectura de red neuronal convolucional simple", "3 capas convolucionales con max pooling", "Normalizacion por lotes", "Capa densa con 256 neuronas", "Dropout 50%", "Activacion ReLU"]}},
    "ResNet50V2": {"accuracy": 0.5925, "validation_loss": 1.09, "training_time_hours": 7.18, "architecture": {"en": ["Optimized ResNet architecture", "ResNet50V2 pretrained on ImageNet", "Fine tuning with custom dense layers", "Regularization and dropout"], "es": ["Arquitectura ResNet optimizada", "ResNet50V2 preentrenada en ImageNet", "Fine tuning con capas densas personalizadas", "Regularizacion y dropout"]}},
    "MobileNetV2 Base": {"accuracy": 0.9450, "validation_loss": 0.1683, "training_time_hours": 5.91, "architecture": {"en": ["MobileNetV2 base model trained for classification", "MobileNetV2 pretrained on ImageNet", "5 training epochs without fine tuning", "Validation accuracy: 94.50%"], "es": ["Modelo base MobileNetV2 entrenado para clasificacion", "MobileNetV2 preentrenada en ImageNet", "5 epocas de entrenamiento sin fine tuning", "Exactitud de validacion: 94.50%"]}},
    "Hybrid Attention": {"accuracy": 0.1450, "validation_loss": 1.9310, "training_time_hours": 4.00, "architecture": {"en": ["Hybrid architecture with attention mechanism", "CNN and attention mechanism combination", "Validation accuracy: 14.5%"], "es": ["Arquitectura hibrida con mecanismo de atencion", "Combinacion de CNN y mecanismo de atencion", "Exactitud de validacion: 14.5%"]}},
    "Hybrid Autoencoder": {"accuracy": 0.1500, "validation_loss": 1.8970, "training_time_hours": 4.00, "architecture": {"en": ["Hybrid architecture with autoencoder", "CNN and autoencoder combination", "Validation accuracy: 15.00%"], "es": ["Arquitectura hibrida con autoencoder", "Combinacion de CNN y autoencoder", "Exactitud de validacion: 15.00%"]}},
}
DATASET_INFO = {
    "dataset": "NCT-CRC-HE-100K",
    "classes": len(CLASS_NAMES),
    "images": 100000,
    "resolution": "224x224",
    "links": {
        "kaggle": "https://www.kaggle.com/datasets/imrankhan77/nct-crc-he-100k",
        "colab": "https://colab.research.google.com/drive/1jsgGq9226_Uhnj0ZtFHIWjZolRxmxmG7?usp=sharing",
        "github": "https://github.com/Anthony140823/Deteccion-cancer-colorrectal-IA.git",
    },
    "text": {
        "en": "This system was trained with NCT-CRC-HE-100K: 100,000 colorectal tissue images, 9 histological classes, 224x224 pixel resolution.",
        "es": "Este sistema fue entrenado con NCT-CRC-HE-100K: 100,000 imagenes de tejido colorrectal, 9 clases histologicas, resolucion 224x224 pixeles.",
    },
}


def _roc_curve(exponent: float, auc: float) -> dict[str, Any]:
    fpr = np.linspace(0, 1, 100)
    return {"fpr": [round(float(x), 6) for x in fpr], "tpr": [round(float(x), 6) for x in fpr**exponent], "auc": auc}


def _accuracy(matrix: list[list[int]]) -> tuple[float, int, int]:
    arr = np.asarray(matrix, dtype=int)
    total = int(arr.sum())
    correct = int(np.trace(arr))
    return (correct / total if total else 0.0), correct, total


def _mcc(matrix: list[list[int]]) -> float:
    arr = np.asarray(matrix, dtype=float)
    total = arr.sum()
    row_sum = arr.sum(axis=1)
    col_sum = arr.sum(axis=0)
    trace = np.trace(arr)
    numerator = trace * total - float(np.dot(row_sum, col_sum))
    denominator = math.sqrt((total**2 - float(np.dot(col_sum, col_sum))) * (total**2 - float(np.dot(row_sum, row_sum))))
    return float(numerator / denominator) if denominator else 0.0


def _binomial_tail(k: int, n: int, p: float) -> float:
    if n <= 0:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k, n + 1)]
    max_log = max(logs)
    return min(1.0, math.exp(max_log) * sum(math.exp(value - max_log) for value in logs))


def _confidence_interval(correct: int, total: int) -> dict[str, Any]:
    try:
        from scipy.stats import beta
    except Exception:
        return {"lower": None, "upper": None, "method": "clopper_pearson_requires_scipy"}
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0, "method": "clopper_pearson"}
    alpha = 0.05
    lower = 0.0 if correct == 0 else float(beta.ppf(alpha / 2, correct, total - correct + 1))
    upper = 1.0 if correct == total else float(beta.ppf(1 - alpha / 2, correct + 1, total - correct))
    return {"lower": lower, "upper": upper, "method": "clopper_pearson"}


def _binomial(matrix: list[list[int]]) -> dict[str, Any]:
    accuracy, correct, total = _accuracy(matrix)
    expected = 1 / len(CLASS_NAMES)
    p_value = _binomial_tail(correct, total, expected)
    interval = _confidence_interval(correct, total)
    return {"accuracy": accuracy, "correct_predictions": correct, "total_samples": total, "expected_accuracy": expected, "p_value": p_value, "significant": p_value < 0.05 and accuracy > expected, "confidence_interval_95": interval, "interpretation": "legacy_streamlit_binomial_accuracy_test"}


def _mcnemar() -> dict[str, Any]:
    matrix1 = np.asarray(CONFUSION_MATRICES["CNN Simple"])
    matrix2 = np.asarray(CONFUSION_MATRICES["MobileNetV2 Base"])
    correct1 = int(np.diag(matrix1).sum())
    incorrect1 = int(matrix1.sum() - correct1)
    correct2 = int(np.diag(matrix2).sum())
    incorrect2 = int(matrix2.sum() - correct2)
    b = max(0, incorrect1 - (int(matrix1.sum()) - correct2))
    c = max(0, incorrect2 - (int(matrix2.sum()) - correct1))
    chi2_stat = ((abs(b - c) - 1) ** 2 / (b + c)) if (b + c) > 0 else 0.0
    try:
        from scipy.stats import chi2
        p_value = float(1 - chi2.cdf(chi2_stat, 1))
        method = "chi_squared_with_continuity_correction"
    except Exception:
        p_value = None
        method = "chi_squared_p_value_requires_scipy"
    return {"table": [["", "MobileNetV2 Correct", "MobileNetV2 Incorrect"], ["CNN Simple Correct", correct1, b], ["Simple Incorrect", c, incorrect1]], "chi2": chi2_stat, "p_value": p_value, "method": method, "models": ["CNN Simple", "MobileNetV2 Base"]}


def legacy_analysis_payload(language: str = "es") -> dict[str, Any]:
    lang = language if language in {"en", "es"} else "es"
    models = []
    for name, matrix in CONFUSION_MATRICES.items():
        exponent, auc = ROC_EXPONENTS[name]
        details = MODEL_DETAILS[name]
        models.append({"name": name, "model_key": MODEL_KEY_BY_LEGACY_NAME[name], "classes": list(CLASS_NAMES), "confusion_matrix": matrix, "roc": _roc_curve(exponent, auc), "mcc": round(_mcc(matrix), 6), "binomial_accuracy_test": _binomial(matrix), "architecture": details["architecture"][lang], "validation_accuracy": details["accuracy"], "validation_loss": details["validation_loss"], "training_time_hours": details["training_time_hours"]})
    return {"source": SOURCE_LABEL, "classes": list(CLASS_NAMES), "models": models, "model_comparison": [{"name": name, "model_key": MODEL_KEY_BY_LEGACY_NAME[name], "validation_accuracy": details["accuracy"], "validation_loss": details["validation_loss"], "training_time_hours": details["training_time_hours"]} for name, details in MODEL_DETAILS.items()], "mcnemar_test": _mcnemar()}


def legacy_training_payload() -> dict[str, Any]:
    return {"source": SOURCE_LABEL, "plot_available": TRAINING_PLOT_PATH.exists(), "plot_endpoint": "/api/v1/legacy/training/plot" if TRAINING_PLOT_PATH.exists() else None, "comparison": legacy_analysis_payload()["model_comparison"]}


def legacy_dataset_payload(language: str = "es") -> dict[str, Any]:
    lang = language if language in {"en", "es"} else "es"
    return {"source": SOURCE_LABEL, **DATASET_INFO, "summary": DATASET_INFO["text"][lang]}


def build_report_pdf(payload: dict[str, Any]) -> bytes:
    lang = payload.get("language", "es")
    title = "Reporte de Diagnostico" if lang == "es" else "Diagnosis Report"
    lines = [title, f"Source: {SOURCE_LABEL}", f"Model: {payload.get('model', '')}", f"Class: {payload.get('predicted_class', '')}", f"Confidence: {payload.get('confidence', 0):.2f}%", "Probabilities:"]
    probabilities = payload.get("probabilities") or {}
    for key, value in probabilities.items():
        lines.append(f"- {key}: {float(value):.2f}%")
    lines.append("Academic support only. Interpret with a qualified medical professional.")
    return _simple_pdf(lines)


def _simple_pdf(lines: list[str]) -> bytes:
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = "BT /F1 12 Tf 50 780 Td 16 TL " + " ".join(f"({esc(line)}) Tj T*" for line in lines) + " ET"
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>", "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", f"<< /Length {len(stream.encode('latin-1', 'replace'))} >>\nstream\n{stream}\nendstream"]
    pdf = "%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf.encode("latin-1")))
        pdf += f"{index} 0 obj\n{obj}\nendobj\n"
    xref = len(pdf.encode("latin-1"))
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    pdf += "".join(f"{offset:010d} 00000 n \n" for offset in offsets[1:])
    pdf += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return pdf.encode("latin-1", "replace")