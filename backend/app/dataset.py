from __future__ import annotations

import hashlib
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import CLASS_NAMES, PROJECT_ROOT

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"})
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "NCT-CRC-HE-100K"


class DatasetNotConfiguredError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class DatasetSample:
    path: Path
    class_name: str
    partition: str


def dataset_root(dataset_path: str | None = None) -> Path:
    raw_path = dataset_path or os.getenv("COLORECTAL_DATASET_DIR")
    return Path(raw_path).expanduser().resolve() if raw_path else DEFAULT_DATASET_DIR


def _class_directories(root: Path) -> dict[str, Path]:
    return {name: root / name for name in CLASS_NAMES if (root / name).is_dir()}


def dataset_partitions(root: Path) -> dict[str, Path]:
    if _class_directories(root):
        return {"dataset": root}
    return {child.name: child for child in root.iterdir() if child.is_dir() and _class_directories(child)}


def require_dataset(dataset_path: str | None = None) -> tuple[Path, dict[str, Path]]:
    root = dataset_root(dataset_path)
    if not root.is_dir():
        raise DatasetNotConfiguredError(f"Dataset directory was not found: {root}. Set COLORECTAL_DATASET_DIR or provide dataset_path.")
    partitions = dataset_partitions(root)
    if not partitions:
        raise DatasetNotConfiguredError(f"No class directories matching {', '.join(CLASS_NAMES)} were found in {root}.")
    return root, partitions


def _image_files(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def iter_samples(dataset_path: str | None = None, partition: str | None = None) -> Iterator[DatasetSample]:
    _, partitions = require_dataset(dataset_path)
    if partition and partition not in partitions:
        raise ValueError(f"Unknown partition '{partition}'. Available: {', '.join(partitions)}.")
    selected = {partition: partitions[partition]} if partition else partitions
    for partition_name, partition_root in selected.items():
        for class_name, class_dir in _class_directories(partition_root).items():
            for image_path in _image_files(class_dir):
                yield DatasetSample(image_path, class_name, partition_name)


def _utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _dataset_fingerprint(samples: list[DatasetSample], root: Path) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        stat = sample.path.stat()
        relative_path = sample.path.relative_to(root).as_posix()
        digest.update(f"{relative_path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def summarize_dataset(dataset_path: str | None = None) -> dict:
    root, partitions = require_dataset(dataset_path)
    samples = list(iter_samples(str(root)))
    partition_summary: dict[str, dict] = {}
    all_counts: Counter[str] = Counter()
    for partition_name, partition_root in partitions.items():
        partition_samples = [sample for sample in samples if sample.partition == partition_name]
        counts = Counter(sample.class_name for sample in partition_samples)
        files = [sample.path.stat() for sample in partition_samples]
        total_bytes = sum(item.st_size for item in files)
        partition_summary[partition_name] = {
            "images": len(partition_samples),
            "class_counts": {class_name: counts.get(class_name, 0) for class_name in CLASS_NAMES},
            "missing_classes": [class_name for class_name in CLASS_NAMES if not counts.get(class_name)],
            "size_bytes": total_bytes,
            "size_mb": round(total_bytes / (1024 * 1024), 3),
            "last_modified_utc": _utc_timestamp(max((item.st_mtime for item in files), default=partition_root.stat().st_mtime)),
        }
        all_counts.update(counts)
    nonzero_counts = [all_counts.get(class_name, 0) for class_name in CLASS_NAMES if all_counts.get(class_name, 0)]
    largest = max(nonzero_counts, default=0)
    smallest = min(nonzero_counts, default=0)
    imbalance_ratio = round(largest / smallest, 4) if smallest else None
    missing_classes = [class_name for class_name in CLASS_NAMES if not all_counts.get(class_name)]
    distribution_finding = "No valid images were found." if not all_counts else f"{len(samples)} images were found across {len(all_counts)} configured classes."
    if missing_classes:
        distribution_recommendation = f"Add or document the missing classes before training: {', '.join(missing_classes)}."
    elif imbalance_ratio is not None and imbalance_ratio > 1.5:
        distribution_recommendation = "Use stratified splits, macro metrics, and consider class weighting during training."
    else:
        distribution_recommendation = "Keep the current class distribution documented and preserve stratification during partitioning."
    return {
        "source": "dataset_scan",
        "dataset_root": str(root),
        "dataset_fingerprint_sha256": _dataset_fingerprint(samples, root),
        "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
        "classes_expected": list(CLASS_NAMES),
        "partitions": partition_summary,
        "total_images": len(samples),
        "total_size_bytes": sum(sample.path.stat().st_size for sample in samples),
        "class_counts": {class_name: all_counts.get(class_name, 0) for class_name in CLASS_NAMES},
        "missing_classes": missing_classes,
        "imbalance_ratio": imbalance_ratio,
        "insights": [
            {
                "section": "class_distribution",
                "what_it_measures": "Number of recognized images per histology class and partition.",
                "finding": distribution_finding,
                "impact": "Class imbalance can inflate overall accuracy while reducing sensitivity for under-represented tissue classes.",
                "recommendation": distribution_recommendation,
            },
            {
                "section": "dataset_version",
                "what_it_measures": "A SHA-256 fingerprint based on image paths, sizes, and modification timestamps.",
                "finding": "The fingerprint identifies the dataset state used for this scan.",
                "impact": "Results cannot be compared reliably when the dataset changes without recording its version.",
                "recommendation": "Store this fingerprint with every training and evaluation record.",
            },
        ],
    }
