from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from .config import CLASS_NAMES
from .dataset import DatasetNotConfiguredError, DatasetSample, iter_samples, require_dataset, summarize_dataset


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_paths(samples: list[DatasetSample], per_class: int) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {class_name: [] for class_name in CLASS_NAMES}
    for sample in samples:
        if len(selected[sample.class_name]) < per_class:
            selected[sample.class_name].append(str(sample.path))
    return {class_name: paths for class_name, paths in selected.items() if paths}


def _stratified_samples(dataset_path: str | None, sample_limit: int) -> list[DatasetSample]:
    """Select a deterministic, approximately balanced sample across classes."""
    selected: list[DatasetSample] = []
    per_class_limit = max(1, (sample_limit + len(CLASS_NAMES) - 1) // len(CLASS_NAMES))
    class_counts: Counter[str] = Counter()
    for sample in iter_samples(dataset_path):
        if class_counts[sample.class_name] < per_class_limit:
            selected.append(sample)
            class_counts[sample.class_name] += 1
            if len(selected) >= sample_limit:
                break
    return selected[:sample_limit]


def analyze_dataset(dataset_path: str | None = None, sample_limit: int = 1000, samples_per_class: int = 3) -> dict:
    if not 1 <= sample_limit <= 10000:
        raise ValueError("sample_limit must be between 1 and 10000.")
    if not 1 <= samples_per_class <= 12:
        raise ValueError("samples_per_class must be between 1 and 12.")
    summary = summarize_dataset(dataset_path)
    samples = _stratified_samples(dataset_path, sample_limit)
    dimensions: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    file_sizes: list[int] = []
    corrupted: list[str] = []
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        file_sizes.append(sample.path.stat().st_size)
        try:
            with Image.open(sample.path) as image:
                image.verify()
            with Image.open(sample.path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                formats[(image.format or "unknown").upper()] += 1
            duplicate_groups[_file_hash(sample.path)].append(str(sample.path))
        except (UnidentifiedImageError, OSError, ValueError):
            corrupted.append(str(sample.path))
    duplicates = [paths for paths in duplicate_groups.values() if len(paths) > 1]
    dominant_dimension = dimensions.most_common(1)[0][0] if dimensions else None
    size_summary = {"total_bytes": sum(file_sizes), "mean_bytes": round(sum(file_sizes) / len(file_sizes), 2) if file_sizes else 0, "min_bytes": min(file_sizes, default=0), "max_bytes": max(file_sizes, default=0)}
    insights = [
        {
            "section": "image_quality",
            "what_it_measures": "Readability, file format, dimensions, and storage size for the inspected images.",
            "finding": f"{len(corrupted)} unreadable files and dominant resolution {dominant_dimension or 'not available'} in {len(samples)} inspected images.",
            "impact": "Unreadable or inconsistent images can interrupt training and add preprocessing variance.",
            "recommendation": "Exclude or repair unreadable files and document any resizing policy before training.",
        },
        {
            "section": "duplicate_detection",
            "what_it_measures": "Exact duplicate files identified through SHA-256 content hashes in the inspected sample.",
            "finding": f"{len(duplicates)} duplicate groups were detected in the inspected sample.",
            "impact": "Duplicates across train and test partitions can artificially increase reported performance.",
            "recommendation": "Remove duplicates or assign each duplicate group to only one partition before evaluation.",
        },
        *summary["insights"],
    ]
    return {
        "source": "dataset_eda",
        "dataset_fingerprint_sha256": summary["dataset_fingerprint_sha256"],
        "sample_limit": sample_limit,
        "images_inspected": len(samples),
        "dimensions": dict(dimensions),
        "formats": dict(formats),
        "file_sizes": size_summary,
        "corrupted_files": corrupted,
        "duplicate_groups": duplicates,
        "representative_samples": _sample_paths(samples, samples_per_class),
        "summary": summary,
        "insights": insights,
    }


def distribution_chart_png(summary: dict) -> bytes:
    counts = summary["class_counts"]
    width, height, margin = 1040, 520, 58
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((margin, 24), "Dataset class distribution", fill="#0f172a", font=font)
    maximum = max(counts.values(), default=1) or 1
    chart_height = height - 150
    bar_width = 72
    gap = 34
    for index, class_name in enumerate(CLASS_NAMES):
        value = counts.get(class_name, 0)
        x0 = margin + index * (bar_width + gap)
        x1 = x0 + bar_width
        y1 = height - 70
        y0 = y1 - int(chart_height * value / maximum)
        draw.rectangle((x0, y0, x1, y1), fill="#0f766e")
        draw.text((x0, y1 + 10), class_name, fill="#334155", font=font)
        draw.text((x0, max(50, y0 - 14)), str(value), fill="#0f172a", font=font)
    draw.line((margin, height - 70, width - margin, height - 70), fill="#94a3b8", width=1)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def representative_montage_png(analysis: dict) -> bytes:
    paths = [(class_name, path) for class_name, values in analysis["representative_samples"].items() for path in values]
    if not paths:
        raise ValueError("No readable representative samples are available for a montage.")
    tile_size, label_height, columns = 144, 22, 5
    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (class_name, path) in enumerate(paths):
        try:
            with Image.open(path) as source:
                tile = source.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        except (UnidentifiedImageError, OSError, ValueError):
            continue
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + label_height)
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y + tile_size, x + tile_size, y + tile_size + label_height), fill="#0f766e")
        draw.text((x + 6, y + tile_size + 5), class_name, fill="white", font=font)
    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def dataset_status(dataset_path: str | None = None) -> dict:
    try:
        root, partitions = require_dataset(dataset_path)
        return {
            "configured": True,
            "dataset_root": str(root),
            "partitions": list(partitions),
        }
    except DatasetNotConfiguredError as exc:
        return {"configured": False, "detail": str(exc)}
