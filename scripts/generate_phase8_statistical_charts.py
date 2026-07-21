from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
STATS_JSON = ROOT / "experiments" / "evaluations_phase4_demo_all" / "statistics" / "best_model_vs_cnn_simple.json"
OUTPUT_DIR = ROOT / "output" / "presentation"


def font(size: int, bold: bool = False):
    for candidate in [
        ROOT / "fonts" / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def canvas(title: str, subtitle: str, size=(1200, 720)):
    image = Image.new("RGB", size, (249, 251, 255))
    draw = ImageDraw.Draw(image)
    draw.text((44, 34), title, fill=(16, 32, 51), font=font(30, True))
    draw.text((44, 76), subtitle, fill=(82, 98, 116), font=font(16))
    return image, draw


def pvalue_chart(result: dict) -> Path:
    correction = result["multiple_comparison_correction"]
    labels = correction["family"]
    raw = correction["raw_p_values"]
    adjusted = correction["adjusted_p_values"]
    image, draw = canvas("P-values y Holm-Bonferroni", "Comparacion de p-values originales y ajustados; linea roja = 0.05.")
    x0, y0, plot_w, plot_h = 220, 150, 780, 360
    threshold = y0 + int(plot_h * (1 - 0.05 / 0.06))
    draw.line((x0, y0 + plot_h, x0 + plot_w, y0 + plot_h), fill=(203, 213, 225), width=2)
    draw.line((x0, threshold, x0 + plot_w, threshold), fill=(225, 29, 72), width=3)
    draw.text((x0 + plot_w + 14, threshold - 10), "0.05", fill=(225, 29, 72), font=font(14, True))
    group_w = plot_w // max(len(labels), 1)
    for i, label in enumerate(labels):
        gx = x0 + i * group_w + 95
        for j, (value, color, tag) in enumerate([(raw[i], (37, 99, 235), "raw"), (adjusted[i], (20, 184, 166), "ajustado")]):
            h = int(plot_h * min(value, 0.06) / 0.06)
            bx = gx + j * 62
            draw.rectangle((bx, y0 + plot_h - h, bx + 44, y0 + plot_h), fill=color)
            draw.text((bx - 8, y0 + plot_h - h - 22), f"{value:.3g}", fill=(16, 32, 51), font=font(13))
            draw.text((bx - 4, y0 + plot_h + 42), tag, fill=(82, 98, 116), font=font(12))
        draw.text((gx - 8, y0 + plot_h + 18), label, fill=(16, 32, 51), font=font(15, True))
    out = OUTPUT_DIR / "statistical_pvalues_holm.png"
    image.save(out)
    return out


def bootstrap_chart(result: dict) -> Path:
    rows = [(k, v["effect_difference"], v["ci_95"][0], v["ci_95"][1]) for k, v in result["paired_bootstrap"]["metrics"].items()]
    image, draw = canvas("Bootstrap pareado - diferencias con IC95", "Diferencia = modelo 1 menos modelo 2. Si el intervalo no cruza cero, la diferencia es consistente.", (1320, 760))
    x0, y0, plot_w = 300, 150, 820
    span = max(abs(v) for row in rows for v in row[1:])
    min_axis, max_axis = -span, span
    zero_x = x0 + int((0 - min_axis) / (max_axis - min_axis) * plot_w)
    draw.line((zero_x, y0 - 26, zero_x, y0 + len(rows) * 88), fill=(225, 29, 72), width=3)
    draw.text((zero_x + 8, y0 - 50), "0", fill=(225, 29, 72), font=font(14, True))
    for i, (name, diff, low, high) in enumerate(rows):
        y = y0 + i * 88
        draw.text((44, y - 12), name, fill=(16, 32, 51), font=font(16, True))
        low_x = x0 + int((low - min_axis) / (max_axis - min_axis) * plot_w)
        high_x = x0 + int((high - min_axis) / (max_axis - min_axis) * plot_w)
        diff_x = x0 + int((diff - min_axis) / (max_axis - min_axis) * plot_w)
        draw.line((low_x, y, high_x, y), fill=(37, 99, 235), width=5)
        draw.ellipse((diff_x - 8, y - 8, diff_x + 8, y + 8), fill=(20, 184, 166))
        draw.text((high_x + 12, y - 12), f"{diff:.3f} [{low:.3f}, {high:.3f}]", fill=(82, 98, 116), font=font(13))
    out = OUTPUT_DIR / "statistical_bootstrap_ci.png"
    image.save(out)
    return out


def delong_chart(result: dict) -> Path:
    delong = result["delong"]
    values = [delong["auc_first"], delong["auc_second"]]
    labels = ["Modelo 1", "Modelo 2"]
    colors = [(37, 99, 235), (20, 184, 166)]
    image, draw = canvas("DeLong AUC - clase TUM", "Comparacion de AUC pareado; mayor AUC indica mejor separacion de la clase.", (1200, 720))
    x0, y0, plot_w, plot_h = 250, 155, 620, 390
    draw.line((x0, y0 + plot_h, x0 + plot_w, y0 + plot_h), fill=(203, 213, 225), width=2)
    draw.line((x0, y0, x0, y0 + plot_h), fill=(203, 213, 225), width=2)
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y0 + plot_h - int(plot_h * tick)
        draw.line((x0 - 8, y, x0 + plot_w, y), fill=(226, 232, 240), width=1)
        draw.text((x0 - 54, y - 10), f"{tick:.2f}", fill=(82, 98, 116), font=font(13))
    bar_w = 120
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        bx = x0 + 130 + i * 250
        h = int(plot_h * value)
        draw.rounded_rectangle((bx, y0 + plot_h - h, bx + bar_w, y0 + plot_h), radius=12, fill=color)
        draw.text((bx + 14, y0 + plot_h - h - 30), f"{value:.3f}", fill=(16, 32, 51), font=font(18, True))
        draw.text((bx + 8, y0 + plot_h + 24), label, fill=(16, 32, 51), font=font(16, True))
    draw.text((900, 230), f"Diff AUC: {delong['effect_auc_difference']:.3f}", fill=(16, 32, 51), font=font(18, True))
    draw.text((900, 266), f"p-value: {delong['p_value']:.3g}", fill=(16, 32, 51), font=font(18, True))
    draw.text((900, 302), "Significativo" if delong.get("significant") else "No significativo", fill=(20, 184, 166) if delong.get("significant") else (100, 116, 139), font=font(18, True))
    out = OUTPUT_DIR / "statistical_delong_auc.png"
    image.save(out)
    return out


def mcnemar_chart(result: dict) -> Path:
    table = result["mcnemar"]["table"]
    labels = ["Ambos correctos", "Solo modelo 1", "Solo modelo 2", "Ambos incorrectos"]
    values = [table[0][0], table[0][1], table[1][0], table[1][1]]
    colors = [(20, 184, 166), (37, 99, 235), (225, 29, 72), (100, 116, 139)]
    image, draw = canvas("McNemar - contingencia visual", "La prueba se enfoca en los casos discordantes: solo modelo 1 vs solo modelo 2.")
    x0, y0, plot_w, bar_h, gap = 310, 155, 720, 58, 38
    max_v = max(values)
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = y0 + i * (bar_h + gap)
        draw.text((44, y + 16), label, fill=(16, 32, 51), font=font(16, True))
        draw.rounded_rectangle((x0, y, x0 + plot_w, y + bar_h), radius=10, fill=(255, 255, 255), outline=(203, 213, 225))
        draw.rounded_rectangle((x0, y, x0 + int(plot_w * value / max_v), y + bar_h), radius=10, fill=color)
        draw.text((x0 + plot_w + 20, y + 16), str(value), fill=(16, 32, 51), font=font(18, True))
    out = OUTPUT_DIR / "statistical_mcnemar_contingency.png"
    image.save(out)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = json.loads(STATS_JSON.read_text(encoding="utf-8"))["result"]
    print(mcnemar_chart(result))
    print(delong_chart(result))
    print(bootstrap_chart(result))
    print(pvalue_chart(result))


if __name__ == "__main__":
    main()
