from __future__ import annotations

import math
import os
from datetime import datetime
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
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0, "method": "wilson_score_95"}
    try:
        from scipy.stats import beta
        alpha = 0.05
        lower = 0.0 if correct == 0 else float(beta.ppf(alpha / 2, correct, total - correct + 1))
        upper = 1.0 if correct == total else float(beta.ppf(1 - alpha / 2, correct + 1, total - correct))
        return {"lower": lower, "upper": upper, "method": "clopper_pearson"}
    except Exception:
        z = 1.959963984540054
        phat = correct / total
        denominator = 1 + z**2 / total
        center = (phat + z**2 / (2 * total)) / denominator
        margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) / denominator
        return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin), "method": "wilson_score_95"}


def _binomial_interpretation(significant: bool, language: str) -> str:
    if language == "en":
        return "The observed accuracy is statistically higher than random classification." if significant else "There is not enough statistical evidence to state that accuracy is higher than random classification."
    return "La exactitud observada es estadisticamente superior a la clasificacion aleatoria." if significant else "No hay evidencia estadistica suficiente para afirmar que la exactitud sea superior a la clasificacion aleatoria."


def _mcnemar_interpretation(p_value: float | None, language: str) -> str:
    significant = p_value is not None and p_value < 0.05
    if language == "en":
        return "There is a statistically significant difference between CNN Simple and MobileNetV2 Base." if significant else "No statistically significant difference was detected between CNN Simple and MobileNetV2 Base."
    return "Existe una diferencia estadisticamente significativa entre CNN Simple y MobileNetV2 Base." if significant else "No se detecto una diferencia estadisticamente significativa entre CNN Simple y MobileNetV2 Base."


def _binomial(matrix: list[list[int]], language: str = "es") -> dict[str, Any]:
    accuracy, correct, total = _accuracy(matrix)
    expected = 1 / len(CLASS_NAMES)
    p_value = _binomial_tail(correct, total, expected)
    significant = p_value < 0.05 and accuracy > expected
    return {
        "accuracy": accuracy,
        "correct_predictions": correct,
        "total_samples": total,
        "expected_accuracy": expected,
        "p_value": p_value,
        "significant": significant,
        "confidence_interval_95": _confidence_interval(correct, total),
        "interpretation": _binomial_interpretation(significant, language),
    }


def _mcnemar(language: str = "es") -> dict[str, Any]:
    matrix1 = np.asarray(CONFUSION_MATRICES["CNN Simple"])
    matrix2 = np.asarray(CONFUSION_MATRICES["MobileNetV2 Base"])
    correct1 = int(np.diag(matrix1).sum())
    incorrect1 = int(matrix1.sum() - correct1)
    correct2 = int(np.diag(matrix2).sum())
    incorrect2 = int(matrix2.sum() - correct2)
    b = max(0, incorrect1 - (int(matrix1.sum()) - correct2))
    c = max(0, incorrect2 - (int(matrix2.sum()) - correct1))
    chi2_stat = ((abs(b - c) - 1) ** 2 / (b + c)) if (b + c) > 0 else 0.0
    p_value = float(math.erfc(math.sqrt(chi2_stat / 2))) if chi2_stat >= 0 else None
    return {
        "table": [["", "MobileNetV2 Correct", "MobileNetV2 Incorrect"], ["CNN Simple Correct", correct1, b], ["CNN Simple Incorrect", c, incorrect1]],
        "chi2": chi2_stat,
        "p_value": p_value,
        "significant": p_value is not None and p_value < 0.05,
        "method": "chi_squared_with_continuity_correction",
        "models": ["CNN Simple", "MobileNetV2 Base"],
        "interpretation": _mcnemar_interpretation(p_value, language),
    }


def _model_payload(name: str, language: str) -> dict[str, Any]:
    matrix = CONFUSION_MATRICES[name]
    exponent, auc = ROC_EXPONENTS[name]
    details = MODEL_DETAILS[name]
    return {
        "name": name,
        "model_key": MODEL_KEY_BY_LEGACY_NAME[name],
        "classes": list(CLASS_NAMES),
        "confusion_matrix": matrix,
        "roc": _roc_curve(exponent, auc),
        "mcc": round(_mcc(matrix), 6),
        "binomial_accuracy_test": _binomial(matrix, language),
        "architecture": details["architecture"][language],
        "validation_accuracy": details["accuracy"],
        "validation_loss": details["validation_loss"],
        "training_time_hours": details["training_time_hours"],
    }


def legacy_analysis_payload(language: str = "es") -> dict[str, Any]:
    lang = language if language in {"en", "es"} else "es"
    models = [_model_payload(name, lang) for name in CONFUSION_MATRICES]
    return {
        "source": SOURCE_LABEL,
        "classes": list(CLASS_NAMES),
        "models": models,
        "roc_comparison": [{"name": item["name"], "model_key": item["model_key"], "roc": item["roc"]} for item in models],
        "model_comparison": [{"name": name, "model_key": MODEL_KEY_BY_LEGACY_NAME[name], "validation_accuracy": details["accuracy"], "validation_loss": details["validation_loss"], "training_time_hours": details["training_time_hours"]} for name, details in MODEL_DETAILS.items()],
        "mcnemar_test": _mcnemar(lang),
    }


def legacy_training_payload() -> dict[str, Any]:
    return {"source": SOURCE_LABEL, "plot_available": TRAINING_PLOT_PATH.exists(), "plot_endpoint": "/api/v1/legacy/training/plot" if TRAINING_PLOT_PATH.exists() else None, "comparison": legacy_analysis_payload()["model_comparison"]}


def legacy_dataset_payload(language: str = "es") -> dict[str, Any]:
    lang = language if language in {"en", "es"} else "es"
    return {"source": SOURCE_LABEL, **DATASET_INFO, "summary": DATASET_INFO["text"][lang]}


def _legacy_for_report(model_key: str | None, language: str) -> dict[str, Any] | None:
    legacy_name = LEGACY_NAME_BY_MODEL_KEY.get(model_key or "")
    if not legacy_name:
        return None
    return _model_payload(legacy_name, language)


def build_report_pdf(
    payload: dict[str, Any],
    image_bytes: bytes | None = None,
    report_id: str | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    lang = payload.get("language", "es") if payload.get("language") in {"es", "en"} else "es"
    try:
        return _professional_report_pdf(payload, lang, image_bytes, report_id, generated_at)
    except Exception:
        return _fallback_report_pdf(payload, lang)


def _report_labels(language: str) -> dict[str, str]:
    if language == "en":
        return {
            "title": "Colorectal Cancer Diagnosis Report",
            "subtitle": "Academic assisted diagnosis system for histopathology images",
            "summary": "Executive Summary",
            "prediction": "Prediction Result",
            "probabilities": "Class Probabilities",
            "legacy": "Legacy Statistical Analysis",
            "matrix": "Confusion Matrix",
            "comparison": "Model Comparison",
            "mcnemar": "McNemar Test",
            "architecture": "Model Architecture",
            "source": "Data source",
            "model": "Model",
            "model_key": "Model key",
            "predicted_class": "Predicted class",
            "confidence": "Confidence",
            "metric": "Metric",
            "value": "Value",
            "interpretation": "Interpretation",
            "note": "This report is academic support and does not replace evaluation by a qualified healthcare professional.",
            "legacy_note": "Statistical values are inherited/precomputed from the original Streamlit application.",
            "institution": "Institution",
            "report_id": "Report ID",
            "generated_at": "Generated at",
            "image": "Analyzed histopathology image",
            "image_unavailable": "The original image was not attached to this report.",
            "roc": "ROC Curve",
            "recommended": "RECOMMENDED MODEL",
            "responsibility": "Academic responsibility statement",
            "responsibility_body": "This document is generated automatically for academic and research support. It is not a clinical diagnosis, does not prescribe treatment, and must be interpreted by qualified healthcare professionals.",
            "signature": "Academic responsible party",
        }
    return {
        "title": "Reporte de Diagnostico de Cancer Colorrectal",
        "subtitle": "Sistema academico de diagnostico asistido con imagenes histopatologicas",
        "summary": "Resumen Ejecutivo",
        "prediction": "Resultado de Prediccion",
        "probabilities": "Probabilidades por Clase",
        "legacy": "Analisis Estadistico Heredado",
        "matrix": "Matriz de Confusion",
        "comparison": "Comparacion de Modelos",
        "mcnemar": "Prueba de McNemar",
        "architecture": "Arquitectura del Modelo",
        "source": "Fuente de datos",
        "model": "Modelo",
        "model_key": "Clave del modelo",
        "predicted_class": "Clase predicha",
        "confidence": "Confianza",
        "metric": "Metrica",
        "value": "Valor",
        "interpretation": "Interpretacion",
        "note": "Este reporte es apoyo academico y no reemplaza la evaluacion de un profesional de salud cualificado.",
        "legacy_note": "Los valores estadisticos son heredados/precalculados desde la aplicacion Streamlit original.",
        "institution": "Institucion",
        "report_id": "ID de reporte",
        "generated_at": "Fecha y hora",
        "image": "Imagen histopatologica analizada",
        "image_unavailable": "La imagen original no fue adjuntada a este reporte.",
        "roc": "Curva ROC",
        "recommended": "MODELO RECOMENDADO",
        "responsibility": "Declaracion de responsabilidad academica",
        "responsibility_body": "Este documento se genera automaticamente como apoyo academico y de investigacion. No constituye un diagnostico clinico, no prescribe tratamientos y debe ser interpretado por profesionales de salud cualificados.",
        "signature": "Responsable academico",
    }


def _professional_report_pdf(
    payload: dict[str, Any],
    language: str,
    image_bytes: bytes | None,
    report_id: str | None,
    generated_at: datetime | None,
) -> bytes:
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    labels = _report_labels(language)
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
        title=labels["title"],
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=colors.white, alignment=TA_LEFT, spaceAfter=5)
    subtitle_style = ParagraphStyle("ReportSubtitle", parent=styles["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#ccfbf1"), alignment=TA_LEFT)
    institution_style = ParagraphStyle("Institution", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=colors.HexColor("#99f6e4"), alignment=TA_LEFT, spaceAfter=12)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor("#0f172a"), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#1f2937"), alignment=TA_LEFT)
    note_style = ParagraphStyle("Note", parent=body_style, textColor=colors.HexColor("#92400e"), backColor=colors.HexColor("#fffbeb"), borderColor=colors.HexColor("#fcd34d"), borderWidth=0.5, borderPadding=6, spaceBefore=8, spaceAfter=8)
    badge_style = ParagraphStyle("Badge", parent=body_style, fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.HexColor("#115e59"), alignment=TA_CENTER)
    model_style = ParagraphStyle("ModelHighlight", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#0f766e"), alignment=TA_CENTER)

    def p(text: object, style: ParagraphStyle = body_style) -> Paragraph:
        return Paragraph(str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), style)

    def styled_table(data: list[list[object]], widths: list[float] | None = None, header: bool = True, font_size: float = 8) -> Table:
        table = Table([[p(cell) for cell in row] for row in data], colWidths=widths, repeatRows=1 if header else 0)
        style = [
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if header:
            style.extend([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ])
        table.setStyle(TableStyle(style))
        return table

    def heatmap(legacy_data: dict[str, Any]) -> Drawing:
        width, height = 11.2 * cm, 8.4 * cm
        drawing = Drawing(width, height)
        matrix = legacy_data["confusion_matrix"]
        class_names = legacy_data["classes"]
        maximum = max(max(row) for row in matrix) or 1
        cell = 0.66 * cm
        origin_x, origin_y = 1.6 * cm, 0.9 * cm
        drawing.add(String(width / 2, height - 12, labels["matrix"], fontName="Helvetica-Bold", fontSize=10, textAnchor="middle", fillColor=colors.HexColor("#0f172a")))
        for row_index, row in enumerate(matrix):
            y = origin_y + (len(matrix) - 1 - row_index) * cell
            drawing.add(String(origin_x - 7, y + cell * 0.35, class_names[row_index], fontName="Helvetica-Bold", fontSize=6.5, textAnchor="end", fillColor=colors.HexColor("#475569")))
            for col_index, value in enumerate(row):
                intensity = 0.12 + 0.88 * (float(value) / maximum)
                fill = colors.Color(0.92 - 0.75 * intensity, 0.98 - 0.50 * intensity, 0.97 - 0.43 * intensity)
                x = origin_x + col_index * cell
                drawing.add(Rect(x, y, cell, cell, fillColor=fill, strokeColor=colors.white, strokeWidth=0.5))
                drawing.add(String(x + cell / 2, y + cell * 0.35, str(value), fontName="Helvetica-Bold", fontSize=5.7, textAnchor="middle", fillColor=colors.HexColor("#0f172a")))
        for col_index, class_name in enumerate(class_names):
            drawing.add(String(origin_x + col_index * cell + cell / 2, origin_y - 9, class_name, fontName="Helvetica-Bold", fontSize=6.5, textAnchor="middle", fillColor=colors.HexColor("#475569")))
        drawing.add(String(origin_x + len(class_names) * cell / 2, 4, "Prediccion", fontSize=6.5, textAnchor="middle", fillColor=colors.HexColor("#64748b")))
        return drawing

    def roc_chart(legacy_data: dict[str, Any]) -> Drawing:
        width, height = 11.2 * cm, 8.4 * cm
        drawing = Drawing(width, height)
        left, bottom, plot_width, plot_height = 1.35 * cm, 0.9 * cm, 8.9 * cm, 6.25 * cm
        drawing.add(String(width / 2, height - 12, f"{labels['roc']} - AUC {legacy_data['roc']['auc']:.2f}", fontName="Helvetica-Bold", fontSize=10, textAnchor="middle", fillColor=colors.HexColor("#0f172a")))
        drawing.add(Rect(left, bottom, plot_width, plot_height, fillColor=colors.HexColor("#f8fafc"), strokeColor=colors.HexColor("#cbd5e1"), strokeWidth=0.7))
        for step in range(1, 5):
            x = left + plot_width * step / 5
            y = bottom + plot_height * step / 5
            drawing.add(Line(x, bottom, x, bottom + plot_height, strokeColor=colors.HexColor("#e2e8f0"), strokeWidth=0.4))
            drawing.add(Line(left, y, left + plot_width, y, strokeColor=colors.HexColor("#e2e8f0"), strokeWidth=0.4))
        drawing.add(Line(left, bottom, left + plot_width, bottom + plot_height, strokeColor=colors.HexColor("#94a3b8"), strokeWidth=0.8, strokeDashArray=[3, 3]))
        points = [(left + float(fpr) * plot_width, bottom + float(tpr) * plot_height) for fpr, tpr in zip(legacy_data["roc"]["fpr"], legacy_data["roc"]["tpr"])]
        drawing.add(PolyLine(points, strokeColor=colors.HexColor("#0f766e"), strokeWidth=2.2))
        drawing.add(String(left + plot_width / 2, 4, "Tasa de falsos positivos", fontSize=6.5, textAnchor="middle", fillColor=colors.HexColor("#64748b")))
        drawing.add(String(4, bottom + plot_height / 2, "TPR", fontSize=6.5, fillColor=colors.HexColor("#64748b")))
        return drawing

    probabilities = payload.get("probabilities") or {}
    legacy = _legacy_for_report(payload.get("model_key"), language)
    analysis = legacy_analysis_payload(language)
    institution = os.getenv("REPORT_INSTITUTION", "Institucion academica")
    program = os.getenv("REPORT_PROGRAM", "Proyecto de Ingenieria de Software")
    responsible = os.getenv("REPORT_RESPONSIBLE", "Equipo academico responsable")
    effective_report_id = report_id or "CCR-SIN-ID"
    timestamp = generated_at.strftime("%d/%m/%Y %H:%M:%S %Z") if generated_at else "N/A"

    cover = Table([[p(institution, institution_style)], [p(labels["title"], title_style)], [p(f"{program} | {labels['subtitle']}", subtitle_style)]], colWidths=[landscape(A4)[0] - doc.leftMargin - doc.rightMargin])
    cover.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0f766e")), ("LEFTPADDING", (0, 0), (-1, -1), 18), ("RIGHTPADDING", (0, 0), (-1, -1), 18), ("TOPPADDING", (0, 0), (-1, 0), 16), ("BOTTOMPADDING", (0, -1), (-1, -1), 16)]))
    story = [cover, Spacer(1, 16)]

    metadata = styled_table([
        [labels["report_id"], effective_report_id, labels["generated_at"], timestamp],
        [labels["institution"], institution, labels["source"], SOURCE_LABEL],
    ], widths=[3.2 * cm, 8.2 * cm, 3.2 * cm, 10.2 * cm], header=False)
    metadata.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0fdfa")), ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f0fdfa")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold")]))
    story.extend([metadata, Spacer(1, 18)])

    recommended = payload.get("model_key") == "best_model"
    model_card = Table([
        [p(labels["recommended"] if recommended else labels["model"], badge_style)],
        [p(payload.get("model", ""), model_style)],
        [p(f"{labels['model_key']}: {payload.get('model_key', '')}", badge_style)],
    ], colWidths=[16 * cm])
    model_card.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#14b8a6")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ccfbf1")), ("BACKGROUND", (0, 1), (-1, -1), colors.white), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.extend([model_card, Spacer(1, 18), p(labels["note"], note_style), PageBreak()])

    story.append(p(labels["summary"], section_style))
    summary_rows = [
        [labels["metric"], labels["value"]],
        [labels["model"], payload.get("model", "")],
        [labels["model_key"], payload.get("model_key", "")],
        [labels["predicted_class"], payload.get("predicted_class", "")],
        [labels["confidence"], f"{float(payload.get('confidence', 0)):.2f}%"],
    ]
    summary_table = styled_table(summary_rows, widths=[4.5 * cm, 8.3 * cm])

    image_flowable: object = p(labels["image_unavailable"], note_style)
    if image_bytes:
        try:
            image_flowable = RLImage(BytesIO(image_bytes))
            image_flowable._restrictSize(11.5 * cm, 8.0 * cm)
        except Exception:
            image_flowable = p(labels["image_unavailable"], note_style)
    image_panel = Table([[p(labels["image"], section_style)], [image_flowable]], colWidths=[12.2 * cm])
    image_panel.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0fdfa")), ("ALIGN", (0, 1), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 1), (-1, -1), 8), ("BOTTOMPADDING", (0, 1), (-1, -1), 8)]))
    summary_panel = Table([[summary_table]], colWidths=[13.0 * cm])
    summary_panel.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(Table([[image_panel, summary_panel]], colWidths=[12.6 * cm, 13.4 * cm], style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")])))

    story.append(p(labels["probabilities"], section_style))
    prob_rows = [["Clase", "Probabilidad"]] + [[key, f"{float(value):.2f}%"] for key, value in sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)]
    story.append(styled_table(prob_rows, widths=[8 * cm, 5 * cm]))

    story.append(PageBreak())
    story.append(p(labels["legacy"], section_style))
    story.append(p(labels["legacy_note"], note_style))
    if legacy:
        binomial = legacy["binomial_accuracy_test"]
        ci = binomial["confidence_interval_95"]
        legacy_rows = [
            [labels["metric"], labels["value"]],
            ["Modelo legacy", legacy["name"]],
            ["AUC", f"{legacy['roc']['auc']:.2f}"],
            ["MCC", f"{legacy['mcc']:.4f}"],
            ["Exactitud observada", f"{binomial['accuracy'] * 100:.2f}%"],
            ["Exactitud aleatoria esperada", f"{binomial['expected_accuracy'] * 100:.2f}%"],
            ["p-value", f"{binomial['p_value']:.6g}"],
            ["IC 95%", f"{ci['lower'] * 100:.2f}% - {ci['upper'] * 100:.2f}% ({ci['method']})"],
            [labels["interpretation"], binomial["interpretation"]],
        ]
        story.append(styled_table(legacy_rows, widths=[6 * cm, 16 * cm]))
        story.append(p(labels["architecture"], section_style))
        story.append(styled_table([["#", "Detalle"]] + [[index + 1, item] for index, item in enumerate(legacy["architecture"])], widths=[1.2 * cm, 18 * cm]))

    if legacy:
        charts = Table([[heatmap(legacy), roc_chart(legacy)]], colWidths=[12.7 * cm, 12.7 * cm])
        charts.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (0, 0), 0.5, colors.HexColor("#cbd5e1")), ("BOX", (1, 0), (1, 0), 0.5, colors.HexColor("#cbd5e1"))]))
        story.extend([Spacer(1, 10), charts])

    story.append(PageBreak())
    story.append(p(labels["comparison"], section_style))
    comparison_rows = [["Modelo", "Exactitud", "Perdida", "Tiempo"]] + [[row["name"], f"{row['validation_accuracy'] * 100:.2f}%", f"{row['validation_loss']:.4f}", f"{row['training_time_hours']:.2f} h"] for row in analysis["model_comparison"]]
    story.append(styled_table(comparison_rows, widths=[7 * cm, 4 * cm, 4 * cm, 4 * cm]))

    story.append(p(labels["mcnemar"], section_style))
    mcnemar = analysis["mcnemar_test"]
    mcnemar_rows = [[labels["metric"], labels["value"]], ["chi2", f"{mcnemar['chi2']:.4f}"], ["p-value", f"{mcnemar['p_value']:.6g}" if mcnemar["p_value"] is not None else "N/A"], ["Significancia", "Si" if mcnemar["significant"] else "No"], [labels["interpretation"], mcnemar["interpretation"]]]
    story.append(styled_table(mcnemar_rows, widths=[5 * cm, 16 * cm]))
    story.append(Spacer(1, 8))
    story.append(styled_table(mcnemar["table"], widths=[6 * cm, 6 * cm, 6 * cm], header=False))
    story.append(Spacer(1, 14))
    responsibility = Table([
        [p(labels["responsibility"], section_style)],
        [p(labels["responsibility_body"], body_style)],
        [Spacer(1, 20)],
        [p("________________________________________", body_style)],
        [p(f"{labels['signature']}: {responsible}", body_style)],
    ], colWidths=[25 * cm])
    responsibility.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#94a3b8")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0fdfa")), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(KeepTogether(responsibility))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(document.leftMargin, 0.55 * cm, f"{effective_report_id} | {labels['note']}")
        canvas.drawRightString(landscape(A4)[0] - document.rightMargin, 0.55 * cm, f"Pagina {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def _fallback_report_pdf(payload: dict[str, Any], language: str) -> bytes:
    labels = _report_labels(language)
    lines = [labels["title"], f"Source: {SOURCE_LABEL}", "", labels["prediction"], f"Model: {payload.get('model', '')}", f"Model key: {payload.get('model_key', '')}", f"Predicted class: {payload.get('predicted_class', '')}", f"Confidence: {float(payload.get('confidence', 0)):.2f}%", "", labels["probabilities"]]
    probabilities = payload.get("probabilities") or {}
    for key, value in sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True):
        lines.append(f"- {key}: {float(value):.2f}%")
    legacy = _legacy_for_report(payload.get("model_key"), language)
    analysis = legacy_analysis_payload(language)
    if legacy:
        binomial = legacy["binomial_accuracy_test"]
        ci = binomial["confidence_interval_95"]
        lines.extend(["", labels["legacy"], f"Legacy model: {legacy['name']}", f"AUC: {legacy['roc']['auc']:.2f}", f"MCC: {legacy['mcc']:.4f}", f"Observed accuracy: {binomial['accuracy'] * 100:.2f}%", f"Expected random accuracy: {binomial['expected_accuracy'] * 100:.2f}%", f"p-value: {binomial['p_value']:.6g}", f"95% CI ({ci['method']}): {ci['lower'] * 100:.2f}% - {ci['upper'] * 100:.2f}%", f"Interpretation: {binomial['interpretation']}"])
    lines.extend(["", labels["comparison"]])
    for row in analysis["model_comparison"]:
        lines.append(f"- {row['name']}: accuracy {row['validation_accuracy'] * 100:.2f}%, loss {row['validation_loss']:.4f}, time {row['training_time_hours']:.2f} h")
    mcnemar = analysis["mcnemar_test"]
    lines.extend(["", labels["mcnemar"], f"chi2: {mcnemar['chi2']:.4f}", f"p-value: {mcnemar['p_value']:.6g}" if mcnemar["p_value"] is not None else "p-value: N/A", f"Interpretation: {mcnemar['interpretation']}", "", labels["note"]])
    return _simple_pdf(lines)
def _simple_pdf(lines: list[str]) -> bytes:
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")[:105]

    lines_per_page = 44
    page_chunks = [lines[index:index + lines_per_page] for index in range(0, len(lines), lines_per_page)] or [[""]]
    objects: list[str] = []
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append("")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_id = 3
    page_ids: list[int] = []

    for chunk in page_chunks:
        content_id = len(objects) + 1
        page_id = len(objects) + 2
        stream = "BT /F1 10 Tf 50 770 Td 14 TL " + " ".join(f"({esc(line)}) Tj T*" for line in chunk) + " ET"
        objects.append(f"<< /Length {len(stream.encode('latin-1', 'replace'))} >>\nstream\n{stream}\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>")
        page_ids.append(page_id)

    objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] /Count {len(page_ids)} >>"
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
