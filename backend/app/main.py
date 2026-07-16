from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .config import CLASS_NAMES, DEFAULT_MODEL, MODEL_SPECS
from .groq_chat import GroqConnectionError, GroqTimeoutError, GroqUpstreamError, ask_groq
from .inference import ImageValidationError, model_service
from .legacy_analysis import TRAINING_PLOT_PATH, build_report_pdf, legacy_analysis_payload, legacy_dataset_payload, legacy_training_payload
from .schemas import ChatRequest, ChatResponse, HealthResponse, LegacyReportRequest, ModelResponse, PredictionResponse


app = FastAPI(
    title="Colorectal Cancer Detection API",
    version="1.0.0",
    description="API for running the colorectal histopathology models from this project.",
)

# Next.js runs on a separate local port during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_available=len(model_service.available_models()),
        models_total=len(MODEL_SPECS),
    )


@app.get("/api/v1/models", response_model=list[ModelResponse], tags=["models"])
def list_models() -> list[ModelResponse]:
    return [
        ModelResponse(
            key=spec.key,
            name=spec.display_name,
            input_width=spec.input_size[0],
            input_height=spec.input_size[1],
            recommended=spec.is_recommended,
            available=spec.path.exists(),
        )
        for spec in MODEL_SPECS.values()
    ]


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    image: UploadFile = File(..., description="Histopathology image in PNG, JPEG, or WEBP format."),
    model: str = Query(DEFAULT_MODEL, description="Model key returned by GET /api/v1/models."),
) -> PredictionResponse:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="The uploaded file must be an image.")

    content = await image.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The uploaded image exceeds the 15 MB limit.")

    try:
        spec, probabilities = model_service.predict(content, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail="The model returned an invalid prediction.") from exc

    predicted_index = int(probabilities.argmax())
    return PredictionResponse(
        model=spec.display_name,
        model_key=spec.key,
        predicted_class=CLASS_NAMES[predicted_index],
        confidence=round(float(probabilities[predicted_index] * 100), 4),
        probabilities={class_name: round(float(score * 100), 4) for class_name, score in zip(CLASS_NAMES, probabilities)},
    )


@app.get("/api/v1/legacy/analysis", tags=["legacy-analysis"])
def legacy_analysis(language: str = Query("es", pattern="^(es|en)$")) -> dict:
    return legacy_analysis_payload(language=language)


@app.get("/api/v1/legacy/training", tags=["legacy-analysis"])
def legacy_training() -> dict:
    return legacy_training_payload()


@app.get("/api/v1/legacy/training/plot", tags=["legacy-analysis"])
def legacy_training_plot() -> FileResponse:
    if not TRAINING_PLOT_PATH.exists():
        raise HTTPException(status_code=404, detail="Legacy training plot image was not found.")
    return FileResponse(TRAINING_PLOT_PATH, media_type="image/png", filename=TRAINING_PLOT_PATH.name)


@app.get("/api/v1/legacy/dataset", tags=["legacy-analysis"])
def legacy_dataset(language: str = Query("es", pattern="^(es|en)$")) -> dict:
    return legacy_dataset_payload(language=language)


@app.post("/api/v1/legacy/report", tags=["legacy-report"])
def legacy_report(payload: LegacyReportRequest) -> Response:
    pdf_bytes = build_report_pdf(payload.model_dump())
    filename = f"diagnosis_{payload.model.replace(' ', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["assistant"])
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        answer, model = await ask_groq(payload)
    except RuntimeError as exc:
        if str(exc) == "GROQ_API_KEY_NOT_CONFIGURED":
            raise HTTPException(status_code=503, detail="The Groq assistant is not configured.") from exc
        raise
    except GroqTimeoutError as exc:
        raise HTTPException(status_code=504, detail="Groq did not respond in time.") from exc
    except GroqUpstreamError as exc:
        status = exc.status_code
        if status == 401:
            raise HTTPException(status_code=401, detail="The Groq API credentials are not valid.") from exc
        if status == 403:
            raise HTTPException(status_code=403, detail="The configured Groq model is not permitted for this account.") from exc
        if status == 429:
            raise HTTPException(status_code=429, detail="The Groq API request limit was reached.") from exc
        raise HTTPException(status_code=502, detail="Groq could not complete the response.") from exc
    except (GroqConnectionError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Groq returned an invalid response.") from exc
    return ChatResponse(answer=answer, model=model)
