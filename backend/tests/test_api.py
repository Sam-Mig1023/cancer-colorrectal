from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import backend.app.routers.legacy as legacy_router
import backend.app.routers.system as system_router
from backend.app.config import CLASS_NAMES
from backend.app.inference import ImageValidationError, model_service
from backend.app.main import app


client = TestClient(app)


def png_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), color=(32, 128, 96))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def report_payload() -> dict[str, object]:
    return {
        "language": "es",
        "model": "Best model (MobileNetV2 Base)",
        "model_key": "best_model",
        "predicted_class": "MUS",
        "confidence": 91.0,
        "probabilities": {class_name: (91.0 if class_name == "MUS" else 1.125) for class_name in CLASS_NAMES},
    }


def test_health_and_models_contract() -> None:
    health = client.get("/api/v1/health")
    models = client.get("/api/v1/models")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert models.status_code == 200
    assert any(model["key"] == "best_model" for model in models.json())


def test_predict_rejects_unsupported_media_type() -> None:
    response = client.post(
        "/api/v1/predict",
        files={"image": ("sample.gif", b"GIF89a", "image/gif")},
    )

    assert response.status_code == 415


def test_image_validation_rejects_gif_bytes() -> None:
    image = Image.new("RGB", (2, 2))
    output = BytesIO()
    image.save(output, format="GIF")

    try:
        model_service._read_image(output.getvalue())
    except ImageValidationError:
        return
    raise AssertionError("GIF data must not be accepted as a supported inference image.")


def test_predict_uses_valid_png_without_loading_real_weights(monkeypatch) -> None:
    def fake_predict(image_bytes: bytes, model_key: str):
        probabilities = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.05])
        return model_service.get_spec(model_key), probabilities

    monkeypatch.setattr(model_service, "predict", fake_predict)
    response = client.post(
        "/api/v1/predict?model=best_model",
        files={"image": ("sample.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["predicted_class"] == "MUS"
    assert response.json()["confidence"] == 95.0


def test_legacy_analysis_is_explicitly_marked_as_precomputed() -> None:
    response = client.get("/api/v1/legacy/analysis?language=es")

    assert response.status_code == 200
    assert response.json()["source"] == "legacy_precomputed_from_streamlit"


def test_report_rejects_inconsistent_payload() -> None:
    payload = report_payload()
    payload["probabilities"] = {"MUS": 100.0}
    response = client.post("/api/v1/legacy/report", json=payload)

    assert response.status_code == 422


def test_report_returns_pdf_with_valid_payload(monkeypatch) -> None:
    monkeypatch.setattr(legacy_router, "build_report_pdf", lambda _: b"%PDF-test")
    response = client.post("/api/v1/legacy/report", json=report_payload())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF-")


def test_patient_report_returns_a_user_facing_pdf(monkeypatch) -> None:
    monkeypatch.setattr(system_router, "build_user_diagnosis_pdf", lambda _: b"%PDF-patient")
    response = client.post("/api/v1/patient/report", json=report_payload())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF-")

def test_scientific_dataset_status_is_honest_when_not_configured() -> None:
    response = client.get("/api/v1/scientific/dataset/status", params={"dataset_path": "missing_dataset_for_test"})

    assert response.status_code == 200
    assert response.json()["configured"] is False


def test_reproducible_metrics_are_calculated_from_predictions() -> None:
    from backend.app.evaluation import metrics_from_predictions

    probabilities = np.zeros((2, len(CLASS_NAMES)))
    probabilities[0, 0] = 1.0
    probabilities[1, 1] = 1.0
    result = metrics_from_predictions([0, 1], probabilities)

    assert result["source"] == "reproducible_evaluation"
    assert result["accuracy"] == 1.0
    assert result["confusion_matrix"][0][0] == 1
    assert result["micro_roc"]["auc"] is not None
    assert result["confidence_intervals_95"]["unit"] == "image"

def create_temporary_dataset(root) -> None:
    for class_name, color in (("ADI", (210, 30, 30)), ("BACK", (30, 30, 210))):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        image = Image.new("RGB", (16, 12), color=color)
        image.save(class_dir / "image_1.png")
    (root / "ADI" / "image_duplicate.png").write_bytes((root / "ADI" / "image_1.png").read_bytes())


def test_dataset_recognizes_tiff_images(tmp_path) -> None:
    from backend.app.dataset import summarize_dataset

    class_dir = tmp_path / "TUM"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), (120, 30, 80)).save(class_dir / "sample.tif", format="TIFF")

    summary = summarize_dataset(str(tmp_path))

    assert summary["total_images"] == 1
    assert summary["class_counts"]["TUM"] == 1


def test_dataset_summary_uses_real_metadata(tmp_path) -> None:
    from backend.app.dataset import summarize_dataset

    create_temporary_dataset(tmp_path)
    summary = summarize_dataset(str(tmp_path))

    assert summary["total_images"] == 3
    assert summary["total_size_bytes"] > 0
    assert len(summary["dataset_fingerprint_sha256"]) == 64
    assert summary["partitions"]["dataset"]["size_bytes"] > 0
    assert summary["class_counts"]["ADI"] == 2


def test_eda_detects_duplicates_generates_visuals_and_exports(tmp_path, monkeypatch) -> None:
    from backend.app.eda import analyze_dataset, distribution_chart_png, representative_montage_png
    import backend.app.reporting as reporting

    create_temporary_dataset(tmp_path)
    analysis = analyze_dataset(str(tmp_path), sample_limit=10, samples_per_class=1)
    monkeypatch.setattr(reporting, "REPORTS_DIR", tmp_path / "exports")
    artifacts = reporting.export_dataset_analysis(analysis)

    assert len(analysis["duplicate_groups"]) == 1
    assert set(analysis["representative_samples"]) == {"ADI", "BACK"}
    assert distribution_chart_png(analysis["summary"]).startswith(b"\x89PNG")
    assert representative_montage_png(analysis).startswith(b"\x89PNG")
    assert Path(artifacts["json_path"]).exists()
    assert Path(artifacts["csv_path"]).exists()


def test_dataset_endpoints_return_real_summary_and_chart(tmp_path) -> None:
    create_temporary_dataset(tmp_path)
    params = {"dataset_path": str(tmp_path)}

    summary = client.get("/api/v1/scientific/dataset/summary", params=params)
    chart = client.get("/api/v1/scientific/dataset/distribution-chart", params=params)

    assert summary.status_code == 200
    assert summary.json()["source"] == "dataset_scan"
    assert chart.status_code == 200
    assert chart.content.startswith(b"\x89PNG")

def test_medical_evaluation_requires_test_partition() -> None:
    from backend.app.schemas import ScientificEvaluationRequest

    request = ScientificEvaluationRequest(model_key="best_model")
    assert request.partition == "test"

def test_paired_statistics_use_aligned_real_artifact_format(tmp_path, monkeypatch) -> None:
    import backend.app.reporting as reporting
    from backend.app.statistical_tests import compare_prediction_artifacts

    monkeypatch.setattr(reporting, "PREDICTIONS_DIR", tmp_path / "predictions")
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    first = np.zeros((len(labels), len(CLASS_NAMES)))
    second = np.zeros((len(labels), len(CLASS_NAMES)))
    first[:, 0] = [0.95, 0.90, 0.85, 0.80, 0.20, 0.15, 0.10, 0.05]
    second[:, 0] = [0.70, 0.65, 0.55, 0.45, 0.60, 0.55, 0.45, 0.35]
    first[:, 1] = 1 - first[:, 0]
    second[:, 1] = 1 - second[:, 0]
    metadata = {"dataset_fingerprint_sha256": "fingerprint", "partition": "test"}
    first_artifact = reporting.persist_prediction_artifact({**metadata, "model_name": "first"}, labels, first, [f"sample_{index}" for index in range(len(labels))])
    second_artifact = reporting.persist_prediction_artifact({**metadata, "model_name": "second"}, labels, second, [f"sample_{index}" for index in range(len(labels))])

    result = compare_prediction_artifacts(first_artifact["artifact_path"], second_artifact["artifact_path"], "ADI", bootstrap_iterations=100)

    assert result["source"] == "reproducible_paired_model_comparison"
    assert result["mcnemar"]["method"].startswith("mcnemar")
    assert result["delong"]["method"] == "delong_auc"
    assert result["multiple_comparison_correction"]["method"] == "holm_bonferroni"
    assert "accuracy" in result["paired_bootstrap"]["metrics"]


def test_grad_cam_pdf_and_alignment_contract() -> None:
    from backend.app.explainability import validate_grad_cam_alignment
    from backend.app.reporting import build_grad_cam_pdf

    base = Image.new("RGB", (48, 48), color=(80, 120, 160))
    overlay = Image.new("RGB", (48, 48), color=(140, 80, 50))
    heatmap = Image.new("RGB", (48, 48), color=(255, 180, 0))
    base_buffer, overlay_buffer, heatmap_buffer = BytesIO(), BytesIO(), BytesIO()
    base.save(base_buffer, format="PNG")
    overlay.save(overlay_buffer, format="PNG")
    heatmap.save(heatmap_buffer, format="PNG")
    alignment = validate_grad_cam_alignment(base_buffer.getvalue(), overlay_buffer.getvalue(), heatmap_buffer.getvalue())
    pdf = build_grad_cam_pdf(base_buffer.getvalue(), overlay_buffer.getvalue(), {"model_key": "best_model", "target_class": "MUS", "confidence_percent": 92.5, "mean_intensity": 0.45, "max_intensity": 1.0, "interpretation": "Academic explanatory aid."})

    assert alignment["same_dimensions"] is True
    assert alignment["overlay_mean_absolute_difference"] > 0
    assert pdf.startswith(b"%PDF-")

def test_predict_rejects_oversized_upload() -> None:
    from backend.app.config import MAX_UPLOAD_BYTES

    response = client.post(
        "/api/v1/predict",
        files={"image": ("too_large.jpg", b"0" * (MAX_UPLOAD_BYTES + 1), "image/jpeg")},
    )

    assert response.status_code == 413


def test_professional_pdf_builders_generate_real_documents() -> None:
    from backend.app.evaluation import metrics_from_predictions
    from backend.app.reporting import build_technical_evaluation_pdf, build_user_diagnosis_pdf

    probabilities = np.zeros((2, len(CLASS_NAMES)))
    probabilities[0, 0] = 1.0
    probabilities[1, 1] = 1.0
    evaluation = metrics_from_predictions([0, 1], probabilities, bootstrap_iterations=100)
    evaluation["origin"] = {
        "model_name": "Best model (MobileNetV2 Base)",
        "dataset_fingerprint_sha256": "test-fingerprint",
        "partition": "test",
        "evaluated_at_utc": "2026-07-18T00:00:00+00:00",
        "seed": 42,
        "experiment_id": "pdf_test",
    }
    record = {"path": "evaluation_pdf_test", "created_at": "2026-07-18T00:00:00+00:00", "payload": evaluation}

    diagnosis_pdf = build_user_diagnosis_pdf(report_payload())
    technical_pdf = build_technical_evaluation_pdf(record)

    assert diagnosis_pdf.startswith(b"%PDF-")
    assert technical_pdf.startswith(b"%PDF-")
    assert len(technical_pdf) > 10_000


def test_technical_report_rejects_unmanaged_record_path() -> None:
    response = client.get("/api/v1/scientific/evaluations/report", params={"record_path": "C:/not-managed/evaluation.json"})

    assert response.status_code == 422

def test_patient_report_accepts_analyzed_image() -> None:
    import json

    response = client.post(
        "/api/v1/patient/report",
        data={"payload": json.dumps(report_payload())},
        files={"image": ("sample.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF-")
