from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from ..config import ALLOWED_IMAGE_MEDIA_TYPES, MAX_UPLOAD_BYTES
from ..dataset import DatasetNotConfiguredError, summarize_dataset
from ..eda import analyze_dataset, dataset_status, distribution_chart_png, representative_montage_png
from ..reporting import build_grad_cam_pdf, build_technical_evaluation_pdf, export_dataset_analysis, load_result_record, persist_result
from ..schemas import ScientificEvaluationRequest, StatisticalComparisonRequest, TrainingJobRequest, TrainingPlanRequest

router = APIRouter(prefix="/scientific", tags=["scientific-workflow"])


@router.get("/dataset/status")
def get_dataset_status(dataset_path: str | None = Query(default=None, max_length=500)) -> dict:
    return dataset_status(dataset_path)


@router.get("/dataset/summary")
def get_dataset_summary(dataset_path: str | None = Query(default=None, max_length=500)) -> dict:
    try:
        return summarize_dataset(dataset_path)
    except DatasetNotConfiguredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/dataset/distribution-chart")
def get_distribution_chart(dataset_path: str | None = Query(default=None, max_length=500)) -> Response:
    try:
        summary = summarize_dataset(dataset_path)
    except DatasetNotConfiguredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(content=distribution_chart_png(summary), media_type="image/png", headers={"X-Dataset-Fingerprint": summary["dataset_fingerprint_sha256"]})


@router.get("/eda")
def get_eda(dataset_path: str | None = Query(default=None, max_length=500), sample_limit: int = Query(default=1000, ge=1, le=10000), samples_per_class: int = Query(default=3, ge=1, le=12)) -> dict:
    try:
        return analyze_dataset(dataset_path, sample_limit, samples_per_class)
    except DatasetNotConfiguredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/eda/representative-montage")
def get_representative_montage(dataset_path: str | None = Query(default=None, max_length=500), sample_limit: int = Query(default=1000, ge=1, le=10000), samples_per_class: int = Query(default=3, ge=1, le=12)) -> Response:
    try:
        analysis = analyze_dataset(dataset_path, sample_limit, samples_per_class)
        montage = representative_montage_png(analysis)
    except DatasetNotConfiguredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content=montage, media_type="image/png", headers={"X-Dataset-Fingerprint": analysis["dataset_fingerprint_sha256"]})


@router.post("/eda/export")
def export_eda(dataset_path: str | None = Query(default=None, max_length=500), sample_limit: int = Query(default=1000, ge=1, le=10000), samples_per_class: int = Query(default=3, ge=1, le=12)) -> dict:
    try:
        analysis = analyze_dataset(dataset_path, sample_limit, samples_per_class)
    except DatasetNotConfiguredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"source": "dataset_eda_export", "artifacts": export_dataset_analysis(analysis), "insights": analysis["insights"]}


@router.post("/training/plan")
def create_training_plan(payload: TrainingPlanRequest) -> dict:
    from ..training import TrainingConfig, training_plan

    try:
        return training_plan(TrainingConfig(**payload.model_dump()))
    except (DatasetNotConfiguredError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/training/jobs")
def start_training_job(payload: TrainingJobRequest) -> dict:
    from ..training import TrainingConfig, training_jobs

    try:
        return training_jobs.start(TrainingConfig(**payload.model_dump()))
    except (DatasetNotConfiguredError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/training/jobs/{job_id}")
def get_training_job(job_id: str) -> dict:
    from ..training import training_jobs

    try:
        return training_jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Training job was not found.") from exc


@router.post("/evaluations")
def create_evaluation(payload: ScientificEvaluationRequest) -> dict:
    from ..evaluation import evaluate_model

    try:
        result = evaluate_model(**payload.model_dump())
    except DatasetNotConfiguredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["record"] = persist_result("evaluation", result)
    return result
@router.get("/evaluations/report")
def evaluation_report(record_path: str = Query(..., min_length=1, max_length=700), language: str = Query(default="es", pattern="^(es|en)$")) -> Response:
    try:
        record = load_result_record(record_path)
        pdf_bytes = build_technical_evaluation_pdf(record, language)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="technical_evaluation_report.pdf"'})


@router.post("/statistics/compare")
def compare_statistics(payload: StatisticalComparisonRequest) -> dict:
    from ..statistical_tests import compare_prediction_artifacts

    try:
        result = compare_prediction_artifacts(**payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["record"] = persist_result("paired_model_comparison", result)
    return result


async def _grad_cam_result(image: UploadFile, model: str, target_class: str | None):
    if image.content_type and image.content_type.lower() not in ALLOWED_IMAGE_MEDIA_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or WEBP image.")
    content = await image.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image exceeds the 15 MB limit.")
    from ..explainability import ExplainabilityUnavailableError, generate_grad_cam

    try:
        return generate_grad_cam(content, model, target_class)
    except ExplainabilityUnavailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/explainability/grad-cam")
async def grad_cam(image: UploadFile = File(...), model: str = Query("best_model"), target_class: str | None = Query(default=None, max_length=20)) -> Response:
    result = await _grad_cam_result(image, model, target_class)
    metadata = result.metadata
    return Response(content=result.overlay_png, media_type="image/png", headers={"X-Grad-Cam-Model": metadata["model_key"], "X-Grad-Cam-Class": metadata["target_class"], "X-Grad-Cam-Confidence": str(metadata["confidence_percent"]), "X-Grad-Cam-Alignment": str(metadata["alignment"]["same_dimensions"]).lower()})


@router.post("/explainability/grad-cam/report")
async def grad_cam_report(image: UploadFile = File(...), model: str = Query("best_model"), target_class: str | None = Query(default=None, max_length=20), language: str = Query(default="es", pattern="^(es|en)$")) -> Response:
    result = await _grad_cam_result(image, model, target_class)
    pdf_bytes = build_grad_cam_pdf(result.original_png, result.overlay_png, result.metadata, language)
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="grad_cam_report.pdf"'})