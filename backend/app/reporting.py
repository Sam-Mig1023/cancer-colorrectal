from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import PROJECT_ROOT

REPORTS_DIR = PROJECT_ROOT / "experiments" / "reports"
PREDICTIONS_DIR = PROJECT_ROOT / "experiments" / "predictions"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _validate_kind(kind: str) -> None:
    if not kind.replace("_", "").isalnum():
        raise ValueError("kind must contain only letters, numbers, and underscores.")


def persist_result(kind: str, payload: dict[str, Any]) -> dict[str, str]:
    _validate_kind(kind)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    path = REPORTS_DIR / f"{kind}_{_timestamp()}.json"
    path.write_text(json.dumps({"kind": kind, "created_at": created_at, "payload": payload}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"record_path": str(path), "created_at": created_at}


def persist_prediction_artifact(metadata: dict[str, Any], labels: Sequence[int], probabilities: np.ndarray, sample_ids: Sequence[str], patient_ids: Sequence[str] | None = None) -> dict[str, str]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    sample_ids_array = np.asarray(sample_ids, dtype=str)
    if probabilities_array.ndim != 2 or len(labels_array) != len(probabilities_array) or len(sample_ids_array) != len(labels_array):
        raise ValueError("Prediction artifacts require aligned labels, probabilities, and sample identifiers.")
    if patient_ids is not None and len(patient_ids) != len(labels_array):
        raise ValueError("Patient identifiers must align with prediction artifacts.")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"predictions_{_timestamp()}.npz"
    np.savez_compressed(path, labels=labels_array, probabilities=probabilities_array, sample_ids=sample_ids_array, patient_ids=np.asarray(patient_ids or [], dtype=str), metadata=json.dumps(metadata, ensure_ascii=False))
    return {"artifact_path": str(path), "artifact_type": "paired_prediction_npz"}


def load_prediction_artifact(artifact_path: str) -> dict[str, Any]:
    path = Path(artifact_path).expanduser().resolve()
    root = PREDICTIONS_DIR.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Prediction artifact must be stored in the managed experiments/predictions directory.")
    if not path.is_file():
        raise FileNotFoundError(f"Prediction artifact was not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        patient_ids = archive["patient_ids"].astype(str).tolist()
        return {"path": str(path), "labels": archive["labels"].astype(int), "probabilities": archive["probabilities"].astype(float), "sample_ids": archive["sample_ids"].astype(str).tolist(), "patient_ids": patient_ids or None, "metadata": metadata}


def export_dataset_analysis(analysis: dict[str, Any]) -> dict[str, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp()
    json_path = REPORTS_DIR / f"dataset_eda_{timestamp}.json"
    csv_path = REPORTS_DIR / f"dataset_distribution_{timestamp}.csv"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = analysis["summary"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class_name", "image_count", "dataset_fingerprint_sha256"])
        writer.writeheader()
        for class_name, image_count in summary["class_counts"].items():
            writer.writerow({"class_name": class_name, "image_count": image_count, "dataset_fingerprint_sha256": summary["dataset_fingerprint_sha256"]})
    return {"json_path": str(json_path), "csv_path": str(csv_path), "dataset_fingerprint_sha256": summary["dataset_fingerprint_sha256"]}


def build_grad_cam_pdf(original_image: bytes, overlay_image: bytes, metadata: dict[str, Any], language: str = "es") -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as PdfImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    is_spanish = language == "es"
    title = "Informe de explicabilidad Grad-CAM" if is_spanish else "Grad-CAM explainability report"
    note = "Apoyo academico. El mapa de calor no confirma una lesion ni reemplaza la interpretacion de un profesional de salud." if is_spanish else "Academic support. The heatmap does not confirm a lesion and does not replace qualified clinical interpretation."
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm, title=title)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("GradCamTitle", parent=styles["Title"], textColor=colors.HexColor("#0f766e"), fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=8)
    section_style = ParagraphStyle("GradCamSection", parent=styles["Heading2"], textColor=colors.HexColor("#0f172a"), fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle("GradCamBody", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#334155"))
    note_style = ParagraphStyle("GradCamNote", parent=body_style, textColor=colors.HexColor("#92400e"), backColor=colors.HexColor("#fffbeb"), borderColor=colors.HexColor("#fcd34d"), borderWidth=0.5, borderPadding=6)
    def paragraph(value: object, style: ParagraphStyle) -> Paragraph:
        return Paragraph(str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), style)
    original = PdfImage(BytesIO(original_image), width=8.2 * cm, height=8.2 * cm)
    overlay = PdfImage(BytesIO(overlay_image), width=8.2 * cm, height=8.2 * cm)
    labels = ["Modelo", "Clase objetivo", "Confianza", "Intensidad media", "Intensidad maxima"] if is_spanish else ["Model", "Target class", "Confidence", "Mean intensity", "Maximum intensity"]
    rows = [[labels[0], metadata.get("model_key", "")], [labels[1], metadata.get("target_class", "")], [labels[2], f"{metadata.get('confidence_percent', 0):.2f}%"], [labels[3], f"{metadata.get('mean_intensity', 0):.4f}"], [labels[4], f"{metadata.get('max_intensity', 0):.4f}"]]
    table = Table([[paragraph(left, body_style), paragraph(right, body_style)] for left, right in rows], colWidths=[4.5 * cm, 11 * cm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    caption = "Imagen original y superposicion Grad-CAM alineadas en las mismas dimensiones." if is_spanish else "Original image and Grad-CAM overlay are aligned at the same dimensions."
    report_id = f"grad_cam_{_timestamp()}"
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story = [paragraph(title, title_style), paragraph(f"Report ID: {report_id} | {created_at}", body_style), paragraph(note, note_style), paragraph("Resumen" if is_spanish else "Summary", section_style), table, paragraph("Visualizacion" if is_spanish else "Visualization", section_style), Table([[original, overlay]], colWidths=[8.5 * cm, 8.5 * cm]), Spacer(1, 5), paragraph(caption, body_style), paragraph("Interpretacion" if is_spanish else "Interpretation", section_style), paragraph(metadata.get("interpretation", ""), body_style)]
    document.build(story, onFirstPage=_report_footer, onLaterPages=_report_footer)
    return buffer.getvalue()

def build_user_diagnosis_pdf(payload: dict[str, Any], original_image: bytes | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as PdfImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    is_spanish = payload.get("language", "es") == "es"
    probabilities = payload.get("probabilities") or {}
    confidence = float(payload.get("confidence", 0))
    if confidence >= 70:
        confidence_label = "Alta" if is_spanish else "High"
        confidence_note = "La salida es consistente para la clase principal, siempre como apoyo academico." if is_spanish else "The output is consistent for the main class, always as academic support."
        confidence_color = "#047857"
    elif confidence >= 40:
        confidence_label = "Moderada" if is_spanish else "Moderate"
        confidence_note = "Revise alternativas cercanas y considere contraste con otro modelo o revision experta." if is_spanish else "Review close alternatives and consider another model or expert review."
        confidence_color = "#b45309"
    else:
        confidence_label = "Baja" if is_spanish else "Low"
        confidence_note = "La salida es incierta; requiere revision cuidadosa y no debe usarse como conclusion." if is_spanish else "The output is uncertain; it requires careful review and must not be used as a conclusion."
        confidence_color = "#be123c"

    title = "Reporte academico de apoyo diagnostico" if is_spanish else "Academic diagnostic support report"
    subtitle = "Analisis de imagen histopatologica colorrectal mediante modelo de IA" if is_spanish else "Colorectal histopathology image analysis using an AI model"
    note = "Herramienta academica de apoyo; no sustituye evaluacion medica profesional ni confirma diagnostico clinico." if is_spanish else "Academic support tool; it does not replace professional medical evaluation or confirm a clinical diagnosis."
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.55 * cm, leftMargin=1.55 * cm, topMargin=1.45 * cm, bottomMargin=1.55 * cm, title=title, author="Colorectal AI")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("PatientTitlePro", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor("#0f766e"), spaceAfter=4)
    subtitle_style = ParagraphStyle("PatientSubtitlePro", parent=styles["BodyText"], fontSize=9.5, leading=12, textColor=colors.HexColor("#475569"), spaceAfter=8)
    section_style = ParagraphStyle("PatientSectionPro", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=15, textColor=colors.HexColor("#0f172a"), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("PatientBodyPro", parent=styles["BodyText"], fontSize=8.8, leading=11.5, textColor=colors.HexColor("#334155"))
    note_style = ParagraphStyle("PatientNotePro", parent=body_style, textColor=colors.HexColor("#92400e"), backColor=colors.HexColor("#fffbeb"), borderColor=colors.HexColor("#fcd34d"), borderWidth=0.6, borderPadding=7, leading=12)
    small_style = ParagraphStyle("PatientSmallPro", parent=body_style, fontSize=7.8, leading=10, textColor=colors.HexColor("#64748b"))

    def paragraph(value: object, style: ParagraphStyle) -> Paragraph:
        return Paragraph(str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), style)

    def styled_table(rows: list[list[object]], widths: list[float], header: bool = False) -> Table:
        table = Table([[paragraph(value, body_style) for value in row] for row in rows], colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        else:
            style += [("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]
        table.setStyle(TableStyle(style))
        return table

    report_id = f"diagnosis_{_timestamp()}"
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    predicted = payload.get("predicted_class", "")
    model_name = payload.get("model", "")
    model_key = payload.get("model_key", "") or "N/A"

    summary_rows = [
        ["Modelo" if is_spanish else "Model", model_name],
        ["Clave tecnica" if is_spanish else "Technical key", model_key],
        ["Clase estimada" if is_spanish else "Estimated class", predicted],
        ["Confianza" if is_spanish else "Confidence", f"{confidence:.2f}% ({confidence_label})"],
    ]
    summary_table = styled_table(summary_rows, [4.5 * cm, 11.8 * cm])

    sorted_probabilities = sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)
    probability_rows = [["Clase" if is_spanish else "Class", "Probabilidad" if is_spanish else "Probability", "Lectura" if is_spanish else "Reading"]]
    for index, (class_name, value) in enumerate(sorted_probabilities, start=1):
        marker = "Principal" if is_spanish else "Main"
        if index > 1:
            marker = "Alternativa" if is_spanish else "Alternative"
        probability_rows.append([class_name, f"{float(value):.2f}%", marker])
    probabilities_table = styled_table(probability_rows, [5.2 * cm, 4 * cm, 6.9 * cm], header=True)

    trace_rows = [
        ["ID de reporte" if is_spanish else "Report ID", report_id],
        ["Fecha/hora UTC" if is_spanish else "UTC date/time", created_at],
        ["Origen" if is_spanish else "Source", "FastAPI inference endpoint"],
        ["Tipo" if is_spanish else "Type", "Academic decision-support output"],
    ]

    story: list[Any] = [
        paragraph(title, title_style),
        paragraph(subtitle, subtitle_style),
        paragraph(note, note_style),
        Spacer(1, 8),
        paragraph("Resumen del resultado" if is_spanish else "Result summary", section_style),
        summary_table,
        paragraph("Lectura responsable" if is_spanish else "Responsible reading", section_style),
        paragraph(confidence_note, body_style),
    ]
    if original_image:
        story.extend([
            paragraph("Imagen analizada" if is_spanish else "Analyzed image", section_style),
            PdfImage(BytesIO(original_image), width=12.8 * cm, height=8.8 * cm),
            paragraph("La imagen se incluye solo para trazabilidad del caso analizado." if is_spanish else "The image is included only for traceability of the analyzed case.", small_style),
        ])
    story.extend([
        paragraph("Distribucion de probabilidades" if is_spanish else "Probability distribution", section_style),
        probabilities_table,
        paragraph("Trazabilidad" if is_spanish else "Traceability", section_style),
        styled_table(trace_rows, [4.5 * cm, 11.8 * cm]),
        paragraph("Advertencia final" if is_spanish else "Final warning", section_style),
        paragraph(note, note_style),
    ])
    document.build(story, onFirstPage=_report_footer, onLaterPages=_report_footer)
    return buffer.getvalue()
def load_result_record(record_path: str) -> dict[str, Any]:
    path = Path(record_path).expanduser().resolve()
    root = REPORTS_DIR.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evaluation record must be stored in experiments/reports.")
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation record was not found: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("kind") != "evaluation" or not isinstance(record.get("payload"), dict):
        raise ValueError("The selected record is not a reproducible evaluation result.")
    return {"path": str(path), **record}


def _report_footer(canvas, document) -> None:
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(1.6 * cm, 1.15 * cm, 19.4 * cm, 1.15 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(1.6 * cm, 0.75 * cm, "Colorectal AI - academic support system")
    canvas.drawRightString(19.4 * cm, 0.75 * cm, f"Page {document.page}")
    canvas.restoreState()


def _plot_png(builder) -> BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = builder(plt)
    output = BytesIO()
    figure.savefig(output, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output.seek(0)
    return output


def build_technical_evaluation_pdf(record: dict[str, Any], language: str = "es") -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as PdfImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    payload = record["payload"]
    origin = payload.get("origin", {})
    spanish = language == "es"
    title = "Reporte tecnico de evaluacion" if spanish else "Technical evaluation report"
    note = "Resultados reproducibles de la particion test bloqueada. Uso academico y de investigacion; no es un informe clinico." if spanish else "Reproducible results from the held-out test partition. For academic and research use; not a clinical report."
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.55 * cm, leftMargin=1.55 * cm, topMargin=1.55 * cm, bottomMargin=1.65 * cm, title=title, author="Colorectal AI")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TechTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=colors.HexColor("#0f766e"), spaceAfter=5)
    subtitle_style = ParagraphStyle("TechSubtitle", parent=styles["BodyText"], fontSize=9.5, leading=13, textColor=colors.HexColor("#475569"), spaceAfter=9)
    section_style = ParagraphStyle("TechSection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#0f172a"), spaceBefore=11, spaceAfter=5)
    body_style = ParagraphStyle("TechBody", parent=styles["BodyText"], fontSize=8.4, leading=11.3, textColor=colors.HexColor("#334155"))
    note_style = ParagraphStyle("TechNote", parent=body_style, textColor=colors.HexColor("#92400e"), backColor=colors.HexColor("#fffbeb"), borderColor=colors.HexColor("#fcd34d"), borderWidth=0.5, borderPadding=7)
    def p(value: Any, style: ParagraphStyle = body_style) -> Paragraph:
        return Paragraph(str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), style)
    def table(rows, widths, header=False):
        item = Table([[p(value) for value in row] for row in rows], colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        else:
            style += [("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]
        item.setStyle(TableStyle(style))
        return item
    info = [["Model", origin.get("model_name", origin.get("model_key", ""))], ["Experiment", origin.get("experiment_id") or "Not specified"], ["Dataset fingerprint", origin.get("dataset_fingerprint_sha256", "")], ["Partition", origin.get("partition", "")], ["Evaluated at UTC", origin.get("evaluated_at_utc", record.get("created_at", ""))], ["Samples", payload.get("samples", 0)], ["Seed", origin.get("seed", "")]]
    globals_rows = [["Accuracy", payload.get("accuracy")], ["Macro F1", payload.get("macro_f1")], ["MCC", payload.get("mcc")], ["Macro AUC", payload.get("macro_auc_ovr")], ["Micro AUC", payload.get("micro_auc_ovr")]]
    globals_rows = [[name, "N/A" if value is None else f"{float(value):.4f}"] for name, value in globals_rows]
    def confusion(plt):
        matrix = np.asarray(payload.get("confusion_matrix", []), dtype=float)
        classes = payload.get("classes", [])
        fig, axis = plt.subplots(figsize=(7.1, 5.8))
        image = axis.imshow(matrix, cmap="YlGnBu")
        axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        axis.set_yticks(range(len(classes)), classes)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center", fontsize=7, color="white" if matrix[row, column] > threshold else "#0f172a")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig
    def roc(plt):
        fig, axis = plt.subplots(figsize=(7.1, 5.5))
        for class_name, metrics in payload.get("per_class", {}).items():
            curve = metrics.get("roc")
            if curve and curve.get("auc") is not None:
                axis.plot(curve["fpr"], curve["tpr"], linewidth=1.35, label=f"{class_name} ({float(curve['auc']):.3f})")
        axis.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1, label="Chance")
        axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC curves")
        axis.legend(loc="lower right", fontsize=7, ncol=2)
        axis.grid(alpha=0.2)
        fig.tight_layout()
        return fig
    class_rows = [["Class", "Support", "Precision", "Sensitivity", "Specificity", "F1", "AUC"]]
    for name, values in payload.get("per_class", {}).items():
        display = lambda value: "N/A" if value is None else f"{float(value):.4f}"
        class_rows.append([name, str(values.get("support", 0)), display(values.get("precision")), display(values.get("recall_sensitivity")), display(values.get("specificity")), display(values.get("f1")), display(values.get("auc"))])
    intervals = payload.get("confidence_intervals_95", {})
    interval_rows = [["Metric", "Estimate", "95% CI"]] + [[name, f"{float(values['estimate']):.4f}", f"[{float(values['lower']):.4f}, {float(values['upper']):.4f}]"] for name, values in intervals.get("metrics", {}).items()]
    report_id = Path(record["path"]).stem
    story = [p(title, title_style), p("Colorectal AI - reproducible research evaluation", subtitle_style), p(f"Report ID: {report_id} | Source: {payload.get('source', '')}", subtitle_style), p(note, note_style), Spacer(1, 8), p("Traceability" if spanish else "Traceability", section_style), table(info, [5.1 * cm, 12.2 * cm]), p("Global metrics" if spanish else "Global metrics", section_style), table(globals_rows, [5.1 * cm, 4.1 * cm]), p("Interpretation" if spanish else "Interpretation", section_style), p(payload.get("interpretation", "")), PageBreak(), p("Confusion matrix" if spanish else "Confusion matrix", section_style), PdfImage(_plot_png(confusion), width=16.7 * cm, height=13.6 * cm), p("ROC curves" if spanish else "ROC curves", section_style), PdfImage(_plot_png(roc), width=16.7 * cm, height=12.8 * cm), PageBreak(), p("Metrics by class" if spanish else "Metrics by class", section_style), table(class_rows, [1.65 * cm, 1.65 * cm, 2.25 * cm, 2.4 * cm, 2.3 * cm, 1.75 * cm, 1.75 * cm], header=True), p("Confidence intervals" if spanish else "Confidence intervals", section_style), table(interval_rows, [5.1 * cm, 4.1 * cm, 8.1 * cm], header=True)]
    if intervals.get("limitation"):
        story += [p("Limitation" if spanish else "Limitation", section_style), p(intervals["limitation"], note_style)]
    document.build(story, onFirstPage=_report_footer, onLaterPages=_report_footer)
    return buffer.getvalue()
