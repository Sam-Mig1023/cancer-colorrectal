from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from ..legacy_analysis import TRAINING_PLOT_PATH, build_report_pdf, legacy_analysis_payload, legacy_dataset_payload, legacy_training_payload
from ..schemas import LegacyReportRequest

router = APIRouter(prefix="/legacy", tags=["legacy-analysis"])


@router.get("/analysis")
def legacy_analysis(language: str = Query("es", pattern="^(es|en)$")) -> dict:
    return legacy_analysis_payload(language=language)


@router.get("/training")
def legacy_training() -> dict:
    return legacy_training_payload()


@router.get("/training/plot")
def legacy_training_plot() -> FileResponse:
    if not TRAINING_PLOT_PATH.exists():
        raise HTTPException(status_code=404, detail="Legacy training plot image was not found.")
    return FileResponse(TRAINING_PLOT_PATH, media_type="image/png", filename=TRAINING_PLOT_PATH.name)


@router.get("/dataset")
def legacy_dataset(language: str = Query("es", pattern="^(es|en)$")) -> dict:
    return legacy_dataset_payload(language=language)


@router.post("/report", tags=["legacy-report"])
def legacy_report(payload: LegacyReportRequest) -> Response:
    pdf_bytes = build_report_pdf(payload.model_dump())
    filename = f"diagnosis_{payload.model.replace(' ', '_')}.pdf"
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})