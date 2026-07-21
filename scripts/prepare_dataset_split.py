from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError


CLASS_NAMES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SourceSample:
    sample_id: str
    source_path: Path
    class_name: str
    file_name: str
    extension: str
    size_bytes: int
    checksum_sha256: str
    image_width: int | None
    image_height: int | None
    image_format: str | None
    corrupt: bool
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a stratified train/validation/test split for NCT-CRC-HE-100K.")
    parser.add_argument("--source", required=True, type=Path, help="Source dataset directory with class folders at the root.")
    parser.add_argument("--dest", required=True, type=Path, help="Destination directory to create the split dataset.")
    parser.add_argument("--seed", default=42, type=int, help="Deterministic shuffle seed.")
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--validation-ratio", default=0.1, type=float)
    parser.add_argument("--test-ratio", default=0.1, type=float)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing destination files when they are already present.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[int | None, int | None, str | None, bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width, image.height, image.format, False, None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, None, None, True, str(exc)


def require_source(source: Path) -> None:
    if not source.is_dir():
        raise SystemExit(f"Source dataset directory not found: {source}")
    missing = [class_name for class_name in CLASS_NAMES if not (source / class_name).is_dir()]
    if missing:
        raise SystemExit(f"Source dataset is missing class folders: {', '.join(missing)}")


def safe_destination(source: Path, dest: Path) -> None:
    source_resolved = source.resolve()
    dest_resolved = dest.resolve()
    if dest_resolved == source_resolved:
        raise SystemExit("Destination must be different from source.")
    try:
        dest_resolved.relative_to(source_resolved)
    except ValueError:
        return
    raise SystemExit("Destination must not be inside the source dataset.")


def collect_samples(source: Path) -> list[SourceSample]:
    samples: list[SourceSample] = []
    next_id = 1
    for class_name in CLASS_NAMES:
        class_dir = source / class_name
        files = sorted(path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        for path in files:
            checksum = sha256_file(path)
            width, height, image_format, corrupt, error = inspect_image(path)
            samples.append(
                SourceSample(
                    sample_id=f"{next_id:06d}",
                    source_path=path,
                    class_name=class_name,
                    file_name=path.name,
                    extension=path.suffix.lower(),
                    size_bytes=path.stat().st_size,
                    checksum_sha256=checksum,
                    image_width=width,
                    image_height=height,
                    image_format=image_format,
                    corrupt=corrupt,
                    error=error,
                )
            )
            next_id += 1
    return samples


def assign_splits(samples: list[SourceSample], seed: int, train_ratio: float, validation_ratio: float, test_ratio: float) -> dict[str, str]:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 0.000001:
        raise SystemExit("Split ratios must add up to 1.0.")

    assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for class_name in CLASS_NAMES:
        valid = [sample for sample in samples if sample.class_name == class_name and not sample.corrupt]
        rng.shuffle(valid)
        total = len(valid)
        train_count = int(total * train_ratio)
        validation_count = int(total * validation_ratio)
        for index, sample in enumerate(valid):
            if index < train_count:
                split = "train"
            elif index < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            assignments[sample.sample_id] = split
    return assignments


def link_or_copy(source: Path, dest: Path, mode: str, reuse_existing: bool) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if reuse_existing and dest.stat().st_size == source.stat().st_size:
            return "reused"
        raise FileExistsError(f"Destination file already exists: {dest}")
    if mode == "copy":
        shutil.copy2(source, dest)
        return "copied"
    try:
        os.link(source, dest)
        return "hardlinked"
    except OSError:
        shutil.copy2(source, dest)
        return "copied_after_hardlink_failed"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    require_source(source)
    safe_destination(source, dest)

    manifests_dir = dest / "manifests"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{utc_now()}] Inspecting source dataset: {source}", flush=True)
    samples = collect_samples(source)
    corrupt_samples = [sample for sample in samples if sample.corrupt]
    assignments = assign_splits(samples, args.seed, args.train_ratio, args.validation_ratio, args.test_ratio)

    checksum_groups: dict[str, list[SourceSample]] = {}
    for sample in samples:
        checksum_groups.setdefault(sample.checksum_sha256, []).append(sample)
    duplicate_groups = {checksum: group for checksum, group in checksum_groups.items() if len(group) > 1}

    print(f"[{utc_now()}] Creating split files in: {dest}", flush=True)
    split_rows: list[dict] = []
    action_counts: dict[str, int] = {}
    for sample in samples:
        split = assignments.get(sample.sample_id)
        if split is None:
            continue
        relative_name = f"{sample.sample_id}_{sample.file_name}"
        split_path = dest / split / sample.class_name / relative_name
        action = link_or_copy(sample.source_path, split_path, args.copy_mode, args.reuse_existing)
        action_counts[action] = action_counts.get(action, 0) + 1
        split_rows.append(
            {
                "sample_id": sample.sample_id,
                "split": split,
                "class_name": sample.class_name,
                "source_path": str(sample.source_path),
                "split_path": str(split_path),
                "checksum_sha256": sample.checksum_sha256,
                "link_action": action,
            }
        )

    dataset_rows = []
    for sample in samples:
        dataset_rows.append(
            {
                **asdict(sample),
                "source_path": str(sample.source_path),
                "duplicate_group_size": len(checksum_groups[sample.checksum_sha256]),
                "assigned_split": assignments.get(sample.sample_id, ""),
            }
        )

    duplicate_rows = []
    for group_index, (checksum, group) in enumerate(sorted(duplicate_groups.items()), start=1):
        for sample in group:
            duplicate_rows.append(
                {
                    "duplicate_group_id": group_index,
                    "checksum_sha256": checksum,
                    "sample_id": sample.sample_id,
                    "class_name": sample.class_name,
                    "source_path": str(sample.source_path),
                    "assigned_split": assignments.get(sample.sample_id, ""),
                }
            )

    write_csv(
        manifests_dir / "dataset_manifest.csv",
        dataset_rows,
        [
            "sample_id",
            "source_path",
            "class_name",
            "file_name",
            "extension",
            "size_bytes",
            "checksum_sha256",
            "image_width",
            "image_height",
            "image_format",
            "corrupt",
            "error",
            "duplicate_group_size",
            "assigned_split",
        ],
    )
    write_csv(
        manifests_dir / "split_manifest.csv",
        split_rows,
        ["sample_id", "split", "class_name", "source_path", "split_path", "checksum_sha256", "link_action"],
    )
    write_csv(
        manifests_dir / "duplicate_groups.csv",
        duplicate_rows,
        ["duplicate_group_id", "checksum_sha256", "sample_id", "class_name", "source_path", "assigned_split"],
    )

    class_counts = {class_name: sum(1 for sample in samples if sample.class_name == class_name) for class_name in CLASS_NAMES}
    corrupt_counts = {class_name: sum(1 for sample in corrupt_samples if sample.class_name == class_name) for class_name in CLASS_NAMES}
    split_counts = {
        split: {class_name: sum(1 for row in split_rows if row["split"] == split and row["class_name"] == class_name) for class_name in CLASS_NAMES}
        for split in SPLITS
    }
    dataset_fingerprint = hashlib.sha256(
        "\n".join(f"{sample.source_path.relative_to(source).as_posix()}|{sample.size_bytes}|{sample.checksum_sha256}" for sample in samples).encode("utf-8")
    ).hexdigest()
    summary = {
        "created_at_utc": utc_now(),
        "source_dataset": str(source),
        "split_dataset": str(dest),
        "classes": list(CLASS_NAMES),
        "image_extensions": sorted(IMAGE_EXTENSIONS),
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "validation": args.validation_ratio, "test": args.test_ratio},
        "total_source_images": len(samples),
        "total_split_images": len(split_rows),
        "class_counts": class_counts,
        "split_counts": split_counts,
        "corrupt_total": len(corrupt_samples),
        "corrupt_counts": corrupt_counts,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_image_count": sum(len(group) for group in duplicate_groups.values()),
        "link_action_counts": action_counts,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "manifests": {
            "dataset_manifest": str(manifests_dir / "dataset_manifest.csv"),
            "split_manifest": str(manifests_dir / "split_manifest.csv"),
            "duplicate_groups": str(manifests_dir / "duplicate_groups.csv"),
        },
    }
    (manifests_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
