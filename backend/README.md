# Backend FastAPI

Este servicio expone los modelos existentes del proyecto para que el futuro frontend Next.js los consuma. No entrena modelos ni modifica Streamlit.

## Instalacion

Desde la raiz del proyecto:

```powershell
python -m pip install -r backend/requirements.txt
```

## Ejecucion

```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```
```powershell
python -m uvicorn backend.app.main:app --reload --port 8000 --env-file backend/.env
```

La documentacion interactiva queda disponible en `http://127.0.0.1:8000/docs`.

## Endpoints

- `GET /api/v1/health`: confirma que el servicio y los artefactos de modelo estan disponibles.
- `GET /api/v1/models`: lista los cinco modelos originales y `best_model`.
- `POST /api/v1/predict?model=best_model`: recibe un campo multipart llamado `image` y devuelve clase, confianza y probabilidades.

`best_model.h5` es una copia de los pesos de MobileNetV2, porque la aplicacion actual lo identifica como el modelo recomendado. Esta eleccion es provisional hasta reemplazar las metricas manuales por una evaluacion reproducible del pipeline ML.
