from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PHASE4_DIR = ROOT / "experiments" / "evaluations_phase4_demo_all"
OUTPUT_DIR = ROOT / "output" / "presentation"
CLASS_NAME = "TUM"


def font(size: int, bold: bool = False):
    candidates = [
        ROOT / "fonts" / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    series = []
    for metrics_path in sorted(PHASE4_DIR.glob("*/metrics.json")):
        model_key = metrics_path.parent.name
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_class = metrics.get("per_class", {})
        roc = per_class.get(CLASS_NAME, {}).get("roc")
        auc = per_class.get(CLASS_NAME, {}).get("auc")
        if roc and roc.get("fpr") and roc.get("tpr"):
            series.append({"model": model_key, "auc": auc, "fpr": roc["fpr"], "tpr": roc["tpr"]})

    width, height = 1320, 900
    left, top, plot = 110, 110, 640
    image = Image.new("RGB", (width, height), (249, 251, 255))
    draw = ImageDraw.Draw(image)
    draw.text((48, 36), f"Comparacion ROC de modelos - clase {CLASS_NAME}", fill=(16, 32, 51), font=font(32, True))
    draw.text((48, 76), "Curvas one-vs-rest calculadas desde predicciones reales de la Fase 4 demo.", fill=(82, 98, 116), font=font(18))
    draw.rectangle((left, top, left + plot, top + plot), outline=(203, 213, 225), width=2)
    for i in range(6):
        x = left + int(plot * i / 5)
        y = top + int(plot * i / 5)
        draw.line((x, top, x, top + plot), fill=(226, 232, 240), width=1)
        draw.line((left, y, left + plot, y), fill=(226, 232, 240), width=1)
        draw.text((x - 10, top + plot + 14), f"{i/5:.1f}", fill=(82, 98, 116), font=font(14))
        draw.text((left - 48, top + plot - int(plot * i / 5) - 8), f"{i/5:.1f}", fill=(82, 98, 116), font=font(14))
    draw.line((left, top + plot, left + plot, top), fill=(148, 163, 184), width=2)
    draw.text((left + plot // 2 - 70, top + plot + 46), "False Positive Rate", fill=(16, 32, 51), font=font(16, True))
    draw.text((18, top + plot // 2), "TPR", fill=(16, 32, 51), font=font(16, True))

    palette = [(37, 99, 235), (20, 184, 166), (225, 29, 72), (124, 58, 237), (217, 119, 6), (71, 85, 105)]
    legend_x = left + plot + 70
    legend_y = top + 20
    for index, item in enumerate(series):
        color = palette[index % len(palette)]
        points = []
        for fpr, tpr in zip(item["fpr"], item["tpr"]):
            x = left + int(float(fpr) * plot)
            y = top + plot - int(float(tpr) * plot)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        y = legend_y + index * 48
        draw.rectangle((legend_x, y + 4, legend_x + 24, y + 28), fill=color)
        draw.text((legend_x + 36, y), f"{item['model']}  AUC {float(item['auc']):.3f}", fill=(16, 32, 51), font=font(18))
    draw.text((legend_x, legend_y + len(series) * 48 + 22), "Interpretacion", fill=(16, 32, 51), font=font(20, True))
    draw.text(
        (legend_x, legend_y + len(series) * 48 + 54),
        "Una curva mas cercana al borde superior izquierdo\nindica mejor separacion para la clase TUM.\nAUC=0.5 equivale aproximadamente a azar.",
        fill=(82, 98, 116),
        font=font(16),
        spacing=6,
    )
    out = OUTPUT_DIR / "roc_auc_comparison_tum.png"
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
