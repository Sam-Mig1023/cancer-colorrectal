from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CLASS_NAMES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
CV_SOURCE_SPLITS = {"train", "validation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stratified K-fold assignments from split_manifest.csv.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    manifests = dataset / "manifests"
    split_manifest = manifests / "split_manifest.csv"
    output_csv = manifests / "folds.csv"
    output_json = manifests / "folds_summary.json"

    rows_by_class: dict[str, list[dict[str, str]]] = {name: [] for name in CLASS_NAMES}
    with split_manifest.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] in CV_SOURCE_SPLITS:
                rows_by_class[row["class_name"]].append(row)

    rng = random.Random(args.seed)
    assigned: list[dict[str, str | int]] = []
    for class_name in CLASS_NAMES:
        rows = rows_by_class[class_name]
        rng.shuffle(rows)
        for index, row in enumerate(rows):
            fold = index % args.folds
            assigned.append(
                {
                    "sample_id": row["sample_id"],
                    "fold": fold,
                    "class_name": row["class_name"],
                    "source_split": row["split"],
                    "path": row["split_path"],
                    "checksum_sha256": row["checksum_sha256"],
                }
            )

    assigned.sort(key=lambda item: (int(item["fold"]), str(item["class_name"]), str(item["sample_id"])))
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["sample_id", "fold", "class_name", "source_split", "path", "checksum_sha256"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assigned)

    fold_counts = {
        str(fold): {
            class_name: sum(1 for row in assigned if row["fold"] == fold and row["class_name"] == class_name)
            for class_name in CLASS_NAMES
        }
        for fold in range(args.folds)
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "folds": args.folds,
        "seed": args.seed,
        "source_splits": sorted(CV_SOURCE_SPLITS),
        "excluded_split": "test",
        "total_samples": len(assigned),
        "fold_totals": {fold: sum(counts.values()) for fold, counts in fold_counts.items()},
        "class_totals": dict(Counter(row["class_name"] for row in assigned)),
        "fold_counts": fold_counts,
        "output_csv": str(output_csv),
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
