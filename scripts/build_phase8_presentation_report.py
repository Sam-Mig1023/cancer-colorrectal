from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle



def configured_dataset_root(project_root: Path) -> Path:
    raw = os.getenv("COLORECTAL_DATASET_DIR")
    env_path = project_root / "backend" / ".env"
    if not raw and env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("COLORECTAL_DATASET_DIR="):
                raw = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    return Path(raw or str(project_root / "data" / "NCT-CRC-HE-100K-SPLIT")).expanduser()

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "presentation"
DATASET = configured_dataset_root(ROOT)
EDA_DIR = DATASET / "eda"
PHASE4_DIR = ROOT / "experiments" / "evaluations_phase4_demo_all"
PHASE5_DIR = PHASE4_DIR / "statistics"
PHASE6_DIR = ROOT / "experiments" / "cross_validation_phase6_demo"
ROC_AUC_CHART = OUTPUT_DIR / "roc_auc_comparison_tum.png"
TRAINING_CHART = ROOT / "Graficos de entrenamiento de los modelos.png"
STAT_MCNEMAR_CHART = OUTPUT_DIR / "statistical_mcnemar_contingency.png"
STAT_DELONG_CHART = OUTPUT_DIR / "statistical_delong_auc.png"
STAT_BOOTSTRAP_CHART = OUTPUT_DIR / "statistical_bootstrap_ci.png"
STAT_PVALUES_CHART = OUTPUT_DIR / "statistical_pvalues_holm.png"
PDF_PATH = OUTPUT_DIR / "reporte_tecnico_fase8_colorectal_ai.pdf"
MD_PATH = OUTPUT_DIR / "resumen_presentacion_fase8.md"
HTML_PATH = OUTPUT_DIR / "resumen_presentacion_fase8.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def percent(value: str | float | None) -> str:
    if value in (None, ""):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def number(value: str | float | None, digits: int = 4) -> str:
    if value in (None, ""):
        return "N/A"
    return f"{float(value):.{digits}f}"


def add_table(story: list, rows: list[list[str]], widths: list[float] | None = None) -> None:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.35 * cm))


def add_image(story: list, path: Path, width_cm: float = 16.5) -> None:
    if not path.is_file():
        story.append(Paragraph(f"No se encontro la grafica: {path}", STYLES["Warn"]))
        return
    image = PdfImage(str(path))
    ratio = image.imageHeight / float(image.imageWidth)
    image.drawWidth = width_cm * cm
    image.drawHeight = image.drawWidth * ratio
    story.append(image)
    story.append(Spacer(1, 0.25 * cm))


def interpretation(story: list, text: str) -> None:
    story.append(Paragraph(f"<b>Interpretacion:</b> {text}", STYLES["Interpretation"]))
    story.append(Spacer(1, 0.25 * cm))


STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name="TitleBlue", parent=STYLES["Title"], textColor=colors.HexColor("#0f172a"), fontSize=22, leading=26, spaceAfter=12))
STYLES.add(ParagraphStyle(name="HeadingBlue", parent=STYLES["Heading1"], textColor=colors.HexColor("#1d4ed8"), fontSize=16, leading=20, spaceBefore=8, spaceAfter=8))
STYLES.add(ParagraphStyle(name="Interpretation", parent=STYLES["BodyText"], backColor=colors.HexColor("#f8fafc"), borderColor=colors.HexColor("#cbd5e1"), borderWidth=0.4, borderPadding=7, leading=13, spaceAfter=6))
STYLES.add(ParagraphStyle(name="Warn", parent=STYLES["BodyText"], textColor=colors.HexColor("#b45309")))


def build_pdf() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    story: list = []
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, rightMargin=1.3 * cm, leftMargin=1.3 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)

    eda = json.loads((EDA_DIR / "eda_summary.json").read_text(encoding="utf-8"))
    comparison = read_csv(PHASE4_DIR / "model_comparison.csv")
    stats = read_csv(PHASE5_DIR / "statistical_comparison.csv")[0]
    cv = read_csv(PHASE6_DIR / "cross_validation_summary.csv")

    story.append(Paragraph("Reporte tecnico de presentacion - Colorectal AI", STYLES["TitleBlue"]))
    story.append(Paragraph("Fase 8: dataset, EDA, evaluacion reproducible, estadistica y validacion cruzada demo", STYLES["BodyText"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("1. Dataset y trazabilidad", STYLES["HeadingBlue"]))
    add_table(
        story,
        [
            ["Elemento", "Valor"],
            ["Dataset", str(DATASET)],
            ["Imagenes", f"{eda['total_images']:,}"],
            ["Split", "train 79,994 | validation 9,995 | test 10,011"],
            ["Resolucion", "224x224 en 100,000 imagenes"],
            ["Formato", "TIFF en 100,000 imagenes"],
            ["Corruptas", str(eda["corrupt_total"])],
            ["Duplicadas", str(eda["duplicate_image_count"])],
            ["Fingerprint backend", eda.get("backend_dataset_fingerprint_sha256", "registrado en backend")],
        ],
        [4.2 * cm, 12 * cm],
    )
    interpretation(story, "El dataset esta particionado y trazado con manifiestos reproducibles. No se detectaron imagenes corruptas ni duplicados exactos, lo que reduce riesgos de errores de lectura y fuga de informacion entre particiones durante la evaluacion.")
    add_image(story, EDA_DIR / "class_distribution.png")
    interpretation(story, "La distribucion por clase no es perfectamente uniforme: algunas categorias tienen mas muestras que otras. Por eso se usan splits estratificados y metricas macro, evitando que las clases con mas imagenes dominen la interpretacion del desempeno.")
    add_image(story, EDA_DIR / "split_distribution.png")
    interpretation(story, "El split conserva proporciones similares por clase en train, validation y test. Esto hace que el conjunto de evaluacion sea comparable al entrenamiento y reduce el riesgo de que una particion quede artificialmente mas facil o mas dificil.")
    add_image(story, EDA_DIR / "representative_montage.png", width_cm=11.5)
    interpretation(story, "El montaje confirma visualmente que el sistema lee parches histopatologicos reales desde disco y que todas las clases configuradas tienen ejemplos. Es una validacion visual complementaria a los conteos y fingerprints.")

    story.append(PageBreak())
    story.append(Paragraph("2. Evaluacion reproducible de modelos actuales", STYLES["HeadingBlue"]))
    eval_rows = [["Modelo", "Muestras", "Accuracy", "Macro F1", "Macro AUC", "MCC"]]
    for row in comparison:
        eval_rows.append([row["model_key"], row["samples"], percent(row["accuracy"]), number(row["macro_f1"]), number(row["macro_auc_ovr"]), number(row["mcc"])])
    add_table(story, eval_rows, [3.2 * cm, 2 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm])
    add_image(story, PHASE4_DIR / "model_comparison.png")
    interpretation(story, "La evaluacion demo usa predicciones reales del conjunto test con una muestra estratificada. En esta corrida rapida, CNN Simple obtiene la mejor combinacion de accuracy, Macro F1, AUC y MCC; el valor principal es demostrar que el pipeline genera artefactos trazables, no cerrar una conclusion clinica final.")
    add_image(story, ROC_AUC_CHART)
    interpretation(story, "La comparacion ROC/AUC muestra la capacidad de separacion para TUM en modo one-vs-rest. Una curva mas cercana a la esquina superior izquierda y un AUC mayor indican mejor discriminacion; AUC cercano a 0.5 equivale aproximadamente a azar.")
    add_image(story, TRAINING_CHART)
    interpretation(story, "Los graficos de entrenamiento heredados muestran la evolucion historica de accuracy y loss. Sirven para explicar convergencia y posible sobreajuste, aunque no corresponden a un reentrenamiento ejecutado durante esta demo urgente.")
    add_image(story, PHASE4_DIR / "cnn_simple" / "confusion_matrix.png", width_cm=15)
    interpretation(story, "La matriz de confusion muestra el patron de errores por clase. Permite detectar si el modelo concentra aciertos en pocas categorias, si confunde tejidos similares o si existe sesgo hacia una clase; por eso complementa las metricas agregadas.")

    story.append(PageBreak())
    story.append(Paragraph("3. Pruebas estadisticas pareadas", STYLES["HeadingBlue"]))
    add_table(
        story,
        [
            ["Prueba", "Resultado", "Interpretacion"],
            ["McNemar", f"p={number(stats['mcnemar_p_value'], 6)}", "Diferencia significativa en aciertos/errores pareados."],
            ["Bootstrap accuracy", f"diff={number(stats['accuracy_difference'])}; IC95 {number(stats['accuracy_diff_ci_low'])} a {number(stats['accuracy_diff_ci_high'])}", "El intervalo no cruza cero en la muestra demo."],
            ["DeLong AUC TUM", f"p={number(stats['delong_p_value'], 6)}; diff={number(stats['auc_difference'])}", "Diferencia significativa de AUC para la clase TUM."],
            ["Holm-Bonferroni", "Aplicado", "Controla error por multiples pruebas."],
        ],
        [4.2 * cm, 5.2 * cm, 7.1 * cm],
    )
    interpretation(story, "Las pruebas son pareadas: ambos modelos se comparan sobre las mismas imagenes, etiquetas, particion y fingerprint. McNemar analiza cambios en acierto/error, DeLong compara AUC para TUM, bootstrap estima intervalos de confianza y Holm-Bonferroni corrige multiples pruebas. Para conclusion final se debe escalar la muestra.")
    add_image(story, STAT_MCNEMAR_CHART)
    interpretation(story, "McNemar se interpreta mirando especialmente los casos discordantes: solo modelo 1 contra solo modelo 2. Un desbalance fuerte entre esos dos grupos respalda diferencia estadistica de aciertos/errores.")
    add_image(story, STAT_DELONG_CHART)
    interpretation(story, "DeLong compara el AUC de dos modelos para la misma clase y las mismas muestras. Un AUC mas alto indica mayor capacidad de separacion, y el p-value permite defender si la diferencia es estadisticamente significativa.")
    add_image(story, STAT_BOOTSTRAP_CHART)
    interpretation(story, "Bootstrap pareado visualiza diferencias de metricas con IC95. Si el intervalo no cruza cero, la diferencia observada es consistente en la muestra evaluada.")
    add_image(story, STAT_PVALUES_CHART)
    interpretation(story, "Holm-Bonferroni ajusta los p-values de McNemar y DeLong para reportar varias pruebas reduciendo el riesgo de falsos positivos.")

    story.append(Paragraph("4. Validacion cruzada demo", STYLES["HeadingBlue"]))
    cv_rows = [["Modelo", "Folds", "Muestras/fold", "Accuracy media", "Accuracy std", "Macro F1 media", "Macro AUC media", "MCC media"]]
    for row in cv:
        cv_rows.append([row["model_key"], row["folds"], row["samples_per_fold"], percent(row["accuracy_mean"]), number(row["accuracy_std"]), number(row["macro_f1_mean"]), number(row["macro_auc_ovr_mean"]), number(row["mcc_mean"])])
    add_table(story, cv_rows, [2.8 * cm, 1.3 * cm, 2.2 * cm, 2.2 * cm, 2 * cm, 2.2 * cm, 2.2 * cm, 2 * cm])
    add_image(story, PHASE6_DIR / "cross_validation_summary.png")
    interpretation(story, "Se generaron folds estratificados reales K=5 y balanceados. La demo evalua modelos existentes para calcular medias y desviaciones rapidamente; la version cientifica final debe reentrenar el modelo dentro de cada fold para medir generalizacion de entrenamiento.")

    story.append(PageBreak())
    story.append(Paragraph("5. Conclusion para exposicion", STYLES["HeadingBlue"]))
    conclusion = (
        "El proyecto ya demuestra un flujo completo: dataset externo sin subir a GitHub, particiones reproducibles, EDA con graficas, evaluacion con predicciones reales, "
        "pruebas estadisticas pareadas y validacion cruzada demo. Para una entrega final cientifica se recomienda ejecutar evaluacion amplia sobre mas muestras y entrenar por fold."
    )
    story.append(Paragraph(conclusion, STYLES["BodyText"]))
    doc.build(story)


def build_markdown_html() -> None:
    comparison = read_csv(PHASE4_DIR / "model_comparison.csv")
    cv = read_csv(PHASE6_DIR / "cross_validation_summary.csv")
    stats = read_csv(PHASE5_DIR / "statistical_comparison.csv")[0]
    lines = [
        "# Resumen de presentacion - Colorectal AI",
        "",
        "## Dataset",
        "- 100,000 imagenes TIFF 224x224.",
        "- Split: train 79,994; validation 9,995; test 10,011.",
        "- Sin corruptas ni duplicados exactos.",
        "",
        "## Evaluacion demo",
        "| Modelo | Accuracy | Macro F1 | Macro AUC | MCC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(f"| {row['model_key']} | {percent(row['accuracy'])} | {number(row['macro_f1'])} | {number(row['macro_auc_ovr'])} | {number(row['mcc'])} |")
    lines += [
        "",
        "Interpretacion: CNN Simple lidera en la muestra demo. Estos valores sirven para mostrar el pipeline, no como conclusion clinica final.",
        "",
        "## Pruebas estadisticas",
        f"- McNemar p-value: `{stats['mcnemar_p_value']}`.",
        f"- DeLong p-value para TUM: `{stats['delong_p_value']}`.",
        f"- Diferencia pareada de accuracy: `{stats['accuracy_difference']}`.",
        "",
        "Interpretacion: en la muestra demo, la diferencia entre CNN Simple y best_model es significativa; debe escalarse para conclusion final.",
        "",
        "## Validacion cruzada demo",
        "| Modelo | Accuracy mean | Accuracy std | Macro F1 mean | Macro AUC mean | MCC mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in cv:
        lines.append(f"| {row['model_key']} | {percent(row['accuracy_mean'])} | {number(row['accuracy_std'])} | {number(row['macro_f1_mean'])} | {number(row['macro_auc_ovr_mean'])} | {number(row['mcc_mean'])} |")
    lines += [
        "",
        "Interpretacion: los folds K=5 existen y estan balanceados. La demo usa modelos ya entrenados; la CV cientifica final debe reentrenar por fold.",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    html = "<html><body>" + "\n".join(f"<p>{line}</p>" for line in lines) + "</body></html>"
    HTML_PATH.write_text(html, encoding="utf-8")


def main() -> None:
    build_pdf()
    build_markdown_html()
    print(json.dumps({"pdf": str(PDF_PATH), "markdown": str(MD_PATH), "html": str(HTML_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
