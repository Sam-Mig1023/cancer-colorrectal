import json

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from ..config import ALLOWED_IMAGE_MEDIA_TYPES, CLASS_NAMES, DEFAULT_MODEL, MAX_UPLOAD_BYTES, MODEL_SPECS
from ..inference import ImageValidationError, model_service
from ..reporting import build_user_diagnosis_pdf
from ..schemas import HealthResponse, LegacyReportRequest, ModelResponse, PredictionResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", models_available=len(model_service.available_models()), models_total=len(MODEL_SPECS))


@router.get("/models", response_model=list[ModelResponse], tags=["models"])
def list_models() -> list[ModelResponse]:
    return [ModelResponse(key=spec.key, name=spec.display_name, input_width=spec.input_size[0], input_height=spec.input_size[1], recommended=spec.is_recommended, available=spec.path.exists()) for spec in MODEL_SPECS.values()]


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(image: UploadFile = File(..., description="Histopathology image in PNG, JPEG, or WEBP format."), model: str = Query(DEFAULT_MODEL, description="Model key returned by GET /api/v1/models.")) -> PredictionResponse:
    if image.content_type and image.content_type.lower() not in ALLOWED_IMAGE_MEDIA_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or WEBP image.")
    content = await image.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image exceeds the 15 MB limit.")
    try:
        spec, probabilities = model_service.predict(content, model)
    except ImageValidationError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail="The model returned an invalid prediction.") from exc
    predicted_index = int(probabilities.argmax())
    return PredictionResponse(model=spec.display_name, model_key=spec.key, predicted_class=CLASS_NAMES[predicted_index], confidence=round(float(probabilities[predicted_index] * 100), 4), probabilities={class_name: round(float(score * 100), 4) for class_name, score in zip(CLASS_NAMES, probabilities)})


@router.post("/patient/report", tags=["patient-report"])
async def patient_report(request: Request) -> Response:
    original_image: bytes | None = None
    try:
        if request.headers.get("content-type", "").lower().startswith("application/json"):
            payload = LegacyReportRequest.model_validate(await request.json())
        else:
            form = await request.form()
            serialized = form.get("payload")
            if not isinstance(serialized, str):
                raise ValueError("Multipart reports require a JSON payload field.")
            payload = LegacyReportRequest.model_validate(json.loads(serialized))
            image = form.get("image")
            if image is not None:
                if not hasattr(image, "read") or not hasattr(image, "content_type"):
                    raise ValueError("The multipart image field is invalid.")
                if image.content_type and image.content_type.lower() not in ALLOWED_IMAGE_MEDIA_TYPES:
                    raise ValueError("Report image must be PNG, JPEG, or WEBP.")
                original_image = await image.read()
                if len(original_image) > MAX_UPLOAD_BYTES:
                    raise ValueError("The report image exceeds the 15 MB limit.")
                model_service._read_image(original_image)
    except (ValueError, json.JSONDecodeError, ImageValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    pdf_bytes = build_user_diagnosis_pdf(payload.model_dump()) if original_image is None else build_user_diagnosis_pdf(payload.model_dump(), original_image=original_image)
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="diagnosis_result.pdf"'})