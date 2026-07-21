from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
SPLITS = ("train", "validation", "test")
COLORS = {
    "blue": (47, 97, 214),
    "teal": (22, 145, 121),
    "amber": (210, 137, 20),
    "red": (190, 65, 55),
    "ink": (28, 33, 40),
    "muted": (92, 101, 116),
    "grid": (220, 225, 232),
    "surface": (250, 252, 255),
    "white": (255, 255, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate optimized EDA artifacts from dataset manifests.")
    parser.add_argument("--dataset", required=True, type=Path, help="Split dataset directory.")
    parser.add_argument("--output", type=Path, help="EDA output directory. Defaults to <dataset>/eda.")
    parser.add_argument("--backend-fingerprint", help="Fingerprint reported by backend/app/dataset.py for the split folder.")
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("fonts/DejaVuSans-Bold.ttf" if bold else "fonts/DejaVuSans.ttf"),
        Path(__file__).resolve().parents[1] / "fonts" / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


FONT_TITLE = font(34, True)
FONT_HEAD = font(24, True)
FONT_BODY = font(20)
FONT_SMALL = font(16)
FONT_MONO = font(15)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fill=COLORS["ink"], fnt=FONT_BODY) -> None:
    draw.text(xy, value, fill=fill, font=fnt)


def save_card(path: Path, title: str, subtitle: str, draw_body) -> None:
    width, height = 1500, 940
    image = Image.new("RGB", (width, height), COLORS["surface"])
    draw = ImageDraw.Draw(image)
    text(draw, (56, 42), title, fnt=FONT_TITLE)
    text(draw, (56, 86), subtitle, fill=COLORS["muted"], fnt=FONT_BODY)
    draw_body(image, draw, 56, 142, width - 112, height - 190)
    image.save(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def draw_horizontal_bars(path: Path, title: str, subtitle: str, values: dict[str, int], color=COLORS["blue"]) -> None:
    def body(_: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, w: int, _h: int) -> None:
        max_value = max(values.values()) if values else 1
        bar_h = 46
        gap = 28
        label_w = 120
        value_w = 145
        chart_w = w - label_w - value_w - 30
        for index, (label, value) in enumerate(values.items()):
            yy = y + index * (bar_h + gap)
            text(draw, (x, yy + 8), label, fnt=FONT_BODY)
            draw.rounded_rectangle((x + label_w, yy, x + label_w + chart_w, yy + bar_h), radius=10, fill=COLORS["white"], outline=COLORS["grid"])
            filled = int(chart_w * value / max_value)
            draw.rounded_rectangle((x + label_w, yy, x + label_w + filled, yy + bar_h), radius=10, fill=color)
            text(draw, (x + label_w + chart_w + 24, yy + 8), f"{value:,}", fnt=FONT_BODY)

    save_card(path, title, subtitle, body)


def draw_split_distribution(path: Path, split_counts: dict[str, dict[str, int]]) -> None:
    split_colors = {"train": COLORS["blue"], "validation": COLORS["teal"], "test": COLORS["amber"]}
    totals = {class_name: sum(split_counts[split][class_name] for split in SPLITS) for class_name in CLASS_NAMES}
    max_total = max(totals.values())

    def body(_: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, w: int, _h: int) -> None:
        bar_h = 42
        gap = 29
        label_w = 120
        value_w = 145
        chart_w = w - label_w - value_w - 30
        legend_x = x
        for split in SPLITS:
            draw.rectangle((legend_x, y - 42, legend_x + 22, y - 20), fill=split_colors[split])
            text(draw, (legend_x + 30, y - 47), split, fnt=FONT_SMALL)
            legend_x += 170
        for index, class_name in enumerate(CLASS_NAMES):
            yy = y + index * (bar_h + gap)
            text(draw, (x, yy + 7), class_name, fnt=FONT_BODY)
            xx = x + label_w
            draw.rounded_rectangle((xx, yy, xx + chart_w, yy + bar_h), radius=10, fill=COLORS["white"], outline=COLORS["grid"])
            cursor = xx
            for split in SPLITS:
                segment = int(chart_w * split_counts[split][class_name] / max_total)
                draw.rectangle((cursor, yy, cursor + segment, yy + bar_h), fill=split_colors[split])
                cursor += segment
            text(draw, (x + label_w + chart_w + 24, yy + 7), f"{totals[class_name]:,}", fnt=FONT_BODY)

    save_card(path, "Distribucion por particion", "Split estratificado train / validation / test por clase", body)


def draw_specs(path: Path, dimensions: Counter[str], formats: Counter[str], corrupt_total: int, duplicate_group_count: int) -> None:
    def body(_: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, w: int, _h: int) -> None:
        items = [
            ("Resolucion dominante", dimensions.most_common(1)[0][0] if dimensions else "N/A"),
            ("Formato dominante", formats.most_common(1)[0][0] if formats else "N/A"),
            ("Imagenes corruptas", str(corrupt_total)),
            ("Grupos duplicados", str(duplicate_group_count)),
        ]
        cols = 2
        card_w = (w - 34) // cols
        card_h = 185
        for index, (label, value) in enumerate(items):
            cx = x + (index % cols) * (card_w + 34)
            cy = y + (index // cols) * (card_h + 34)
            draw.rounded_rectangle((cx, cy, cx + card_w, cy + card_h), radius=14, fill=COLORS["white"], outline=COLORS["grid"])
            text(draw, (cx + 28, cy + 30), label, fill=COLORS["muted"], fnt=FONT_BODY)
            text(draw, (cx + 28, cy + 86), value, fill=COLORS["ink"], fnt=FONT_TITLE)

        table_y = y + 2 * (card_h + 34) + 24
        text(draw, (x, table_y), "Conteo de resoluciones y formatos", fnt=FONT_HEAD)
        row_y = table_y + 50
        for label, count in list(dimensions.most_common(4)) + list(formats.most_common(4)):
            text(draw, (x, row_y), label, fnt=FONT_BODY)
            text(draw, (x + 260, row_y), f"{count:,}", fnt=FONT_BODY)
            row_y += 34

    save_card(path, "Calidad tecnica del dataset", "Resumen derivado del manifiesto reproducible", body)


def draw_file_size_histogram(path: Path, sizes: list[int]) -> None:
    if not sizes:
        return
    sizes_kb = [size / 1024 for size in sizes]
    min_v, max_v = min(sizes_kb), max(sizes_kb)
    bins = 18
    width = (max_v - min_v) / bins if max_v > min_v else 1
    counts = [0 for _ in range(bins)]
    for value in sizes_kb:
        index = min(bins - 1, int((value - min_v) / width))
        counts[index] += 1

    def body(_: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
        plot_h = h - 120
        bar_gap = 8
        bar_w = (w - bar_gap * (bins - 1)) / bins
        max_count = max(counts)
        base_y = y + plot_h
        draw.line((x, base_y, x + w, base_y), fill=COLORS["grid"], width=2)
        for index, count in enumerate(counts):
            bx = int(x + index * (bar_w + bar_gap))
            bh = int(plot_h * count / max_count)
            draw.rounded_rectangle((bx, base_y - bh, int(bx + bar_w), base_y), radius=8, fill=COLORS["teal"])
        text(draw, (x, base_y + 26), f"Min {min_v:.1f} KB", fill=COLORS["muted"], fnt=FONT_BODY)
        text(draw, (x + w - 180, base_y + 26), f"Max {max_v:.1f} KB", fill=COLORS["muted"], fnt=FONT_BODY)
        text(draw, (x, base_y + 64), f"Promedio {sum(sizes_kb) / len(sizes_kb):.1f} KB", fnt=FONT_BODY)

    save_card(path, "Distribucion de tamanos de archivo", "Histograma sobre las 100,000 imagenes del manifiesto", body)


def make_montage(path: Path, representative_paths: dict[str, Path]) -> None:
    thumb = 224
    padding = 34
    label_h = 42
    cols = 3
    rows = math.ceil(len(CLASS_NAMES) / cols)
    width = cols * thumb + (cols + 1) * padding
    height = rows * (thumb + label_h) + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), COLORS["surface"])
    draw = ImageDraw.Draw(canvas)
    for index, class_name in enumerate(CLASS_NAMES):
        row = index // cols
        col = index % cols
        x = padding + col * (thumb + padding)
        y = padding + row * (thumb + label_h + padding)
        image_path = representative_paths.get(class_name)
        if image_path and image_path.exists():
            with Image.open(image_path) as image:
                tile = image.convert("RGB").resize((thumb, thumb))
        else:
            tile = Image.new("RGB", (thumb, thumb), (235, 238, 243))
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb, y + thumb), outline=COLORS["grid"], width=2)
        text(draw, (x, y + thumb + 10), class_name, fnt=FONT_BODY)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = (args.output or dataset / "eda").resolve()
    manifests = dataset / "manifests"
    summary_path = manifests / "dataset_summary.json"
    dataset_manifest_path = manifests / "dataset_manifest.csv"
    split_manifest_path = manifests / "split_manifest.csv"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset_rows = read_csv(dataset_manifest_path)
    split_rows = read_csv(split_manifest_path)
    output.mkdir(parents=True, exist_ok=True)

    class_counts = {name: int(summary["class_counts"][name]) for name in CLASS_NAMES}
    split_counts = {
        split: {name: int(summary["split_counts"][split][name]) for name in CLASS_NAMES}
        for split in SPLITS
    }
    dimensions = Counter(f"{row['image_width']}x{row['image_height']}" for row in dataset_rows if row["image_width"] and row["image_height"])
    formats = Counter(row["image_format"] or "unknown" for row in dataset_rows)
    sizes = [int(row["size_bytes"]) for row in dataset_rows]
    corrupt_rows = [row for row in dataset_rows if row["corrupt"].lower() == "true"]
    duplicate_rows = [row for row in dataset_rows if int(row["duplicate_group_size"]) > 1]

    representative_paths: dict[str, Path] = {}
    for row in split_rows:
        if row["split"] == "train" and row["class_name"] not in representative_paths:
            representative_paths[row["class_name"]] = Path(row["split_path"])

    draw_horizontal_bars(output / "class_distribution.png", "Distribucion por clase", "Conteo total por clase en NCT-CRC-HE-100K", class_counts)
    draw_split_distribution(output / "split_distribution.png", split_counts)
    draw_specs(output / "image_quality_summary.png", dimensions, formats, len(corrupt_rows), int(summary["duplicate_group_count"]))
    draw_file_size_histogram(output / "file_size_distribution.png", sizes)
    make_montage(output / "representative_montage.png", representative_paths)

    corrupt_csv = output / "corrupt_files.csv"
    with corrupt_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["sample_id", "class_name", "source_path", "error"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in corrupt_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    duplicate_csv = output / "duplicate_files.csv"
    with duplicate_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["sample_id", "class_name", "source_path", "checksum_sha256", "assigned_split"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in duplicate_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    eda_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset),
        "source_summary": summary_path.as_posix(),
        "manifest_content_fingerprint_sha256": summary["dataset_fingerprint_sha256"],
        "backend_dataset_fingerprint_sha256": args.backend_fingerprint,
        "total_images": summary["total_source_images"],
        "split_images": summary["total_split_images"],
        "class_counts": class_counts,
        "split_counts": split_counts,
        "dimensions": dict(dimensions),
        "formats": dict(formats),
        "file_size_bytes": {
            "min": min(sizes),
            "max": max(sizes),
            "mean": round(sum(sizes) / len(sizes), 2),
        },
        "corrupt_total": len(corrupt_rows),
        "duplicate_image_count": len(duplicate_rows),
        "artifacts": {
            "class_distribution_png": str(output / "class_distribution.png"),
            "split_distribution_png": str(output / "split_distribution.png"),
            "image_quality_summary_png": str(output / "image_quality_summary.png"),
            "file_size_distribution_png": str(output / "file_size_distribution.png"),
            "representative_montage_png": str(output / "representative_montage.png"),
            "corrupt_files_csv": str(corrupt_csv),
            "duplicate_files_csv": str(duplicate_csv),
        },
    }
    (output / "eda_summary.json").write_text(json.dumps(eda_summary, indent=2), encoding="utf-8")

    report = [
        "# EDA del dataset particionado",
        "",
        f"- Dataset: `{dataset}`",
        f"- Fingerprint de contenido/manifiesto: `{summary['dataset_fingerprint_sha256']}`",
        f"- Fingerprint reportado por backend: `{args.backend_fingerprint or 'no incluido'}`",
        f"- Imagenes: `{summary['total_source_images']:,}`",
        f"- Split: train `{sum(split_counts['train'].values()):,}`, validation `{sum(split_counts['validation'].values()):,}`, test `{sum(split_counts['test'].values()):,}`",
        f"- Resoluciones: `{dict(dimensions)}`",
        f"- Formatos: `{dict(formats)}`",
        f"- Corruptas: `{len(corrupt_rows)}`",
        f"- Duplicadas: `{len(duplicate_rows)}`",
        "",
        "## Artefactos",
        "",
        "- `class_distribution.png`",
        "- `split_distribution.png`",
        "- `image_quality_summary.png`",
        "- `file_size_distribution.png`",
        "- `representative_montage.png`",
        "- `eda_summary.json`",
        "- `corrupt_files.csv`",
        "- `duplicate_files.csv`",
    ]
    (output / "eda_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(eda_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()


