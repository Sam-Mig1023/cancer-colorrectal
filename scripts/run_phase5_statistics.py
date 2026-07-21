from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import CLASS_NAMES
from backend.app.reporting import persist_result
from backend.app.statistical_tests import compare_prediction_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase 5 paired statistical tests from phase 4 artifacts.")
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--baseline", default="best_model")
    parser.add_argument("--challenger", default="cnn_simple")
    parser.add_argument("--class-name", default="TUM", choices=list(CLASS_NAMES))
    parser.add_argument("--bootstrap-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def artifact_path(evaluation_dir: Path, model_key: str) -> str:
    path = evaluation_dir / model_key / "artifact_paths.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["prediction_artifact"]["artifact_path"]


def compact_row(name: str, result: dict) -> dict:
    mcnemar = result.get("mcnemar", {})
    bootstrap_metrics = result.get("paired_bootstrap", {}).get("metrics", {})
    accuracy_bootstrap = bootstrap_metrics.get("accuracy", {})
    delong = result.get("delong", {})
    ci = accuracy_bootstrap.get("ci_95") or [None, None]
    return {
        "comparison": name,
        "samples": result.get("sample_count"),
        "class_name": result.get("class_for_delong"),
        "mcnemar_chi2": mcnemar.get("chi2"),
        "mcnemar_p_value": mcnemar.get("p_value"),
        "mcnemar_significant": mcnemar.get("significant"),
        "accuracy_first": accuracy_bootstrap.get("first"),
        "accuracy_second": accuracy_bootstrap.get("second"),
        "accuracy_difference": accuracy_bootstrap.get("effect_difference"),
        "accuracy_diff_ci_low": ci[0],
        "accuracy_diff_ci_high": ci[1],
        "auc_first": delong.get("auc_first"),
        "auc_second": delong.get("auc_second"),
        "auc_difference": delong.get("effect_auc_difference"),
        "delong_p_value": delong.get("p_value"),
        "holm_bonferroni": json.dumps(result.get("multiple_comparison_correction", {}), ensure_ascii=False),
    }


def main() -> None:
    args = parse_args()
    evaluation_dir = args.evaluation_dir.resolve()
    output = (args.output or evaluation_dir / "statistics").resolve()
    output.mkdir(parents=True, exist_ok=True)

    baseline_artifact = artifact_path(evaluation_dir, args.baseline)
    challenger_artifact = artifact_path(evaluation_dir, args.challenger)
    result = compare_prediction_artifacts(
        first_artifact_path=baseline_artifact,
        second_artifact_path=challenger_artifact,
        class_name=args.class_name,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    record = persist_result("paired_model_comparison", result)
    payload = {
        "baseline": args.baseline,
        "challenger": args.challenger,
        "baseline_artifact": baseline_artifact,
        "challenger_artifact": challenger_artifact,
        "record": record,
        "result": result,
    }
    (output / f"{args.baseline}_vs_{args.challenger}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    row = compact_row(f"{args.baseline}_vs_{args.challenger}", result)
    with (output / "statistical_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    report = [
        "# Fase 5 - Pruebas estadisticas pareadas",
        "",
        f"- Comparacion: `{args.baseline}` vs `{args.challenger}`",
        f"- Clase usada para DeLong/AUC: `{args.class_name}`",
        f"- Artefactos: mismas muestras, etiquetas, particion y fingerprint validados por el backend.",
        f"- Registro backend: `{record['record_path']}`",
        "",
        "## Resultado compacto",
        "",
        f"- McNemar p-value: `{row['mcnemar_p_value']}`",
        f"- McNemar significativo: `{row['mcnemar_significant']}`",
        f"- Diferencia de accuracy pareada: `{row['accuracy_difference']}`",
        f"- IC 95% accuracy diff: `{row['accuracy_diff_ci_low']}` a `{row['accuracy_diff_ci_high']}`",
        f"- Diferencia AUC DeLong: `{row['auc_difference']}`",
        f"- DeLong p-value: `{row['delong_p_value']}`",
        "",
        "Nota: esta comparacion usa la muestra demo urgente de Fase 4; sirve para demostrar el flujo estadistico, no para concluir desempeno clinico final.",
    ]
    (output / "phase5_statistics_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
