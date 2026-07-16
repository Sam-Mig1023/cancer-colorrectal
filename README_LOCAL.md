# Ejecucion local sin Docker

## 1. Backend FastAPI

Desde la raiz del proyecto, instale las dependencias del backend y ejecute el servicio:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

Compruebe `http://127.0.0.1:8000/docs` antes de abrir el frontend.

## 2. Frontend Next.js

En otra terminal:

```powershell
cd frontend
Copy-Item .env.local.example .env.local
npm install
npm run dev
```

Abra `http://localhost:3000`. La variable `NEXT_PUBLIC_API_URL` se configura en `frontend/.env.local` y por defecto debe ser `http://127.0.0.1:8000`.

## Estado de migracion

La pantalla de Next.js consume salud, listado de modelos, prediccion y las secciones heredadas expuestas por FastAPI. La inferencia sigue en Python/FastAPI; Next.js solo envia imagenes y muestra respuestas.

Funciones migradas en este paso:

- Carga de imagen por clic y por arrastrar/soltar, con validacion PNG/JPG/JPEG/WEBP y maximo 15 MB.
- Matrices de confusion, curvas ROC/AUC, MCC, prueba binomial, intervalo de confianza cuando SciPy esta disponible y prueba de McNemar desde `GET /api/v1/legacy/analysis`.
- Comparacion de modelos, arquitectura/informacion de modelos y datos de entrenamiento heredados desde `GET /api/v1/legacy/analysis` y `GET /api/v1/legacy/training`.
- Grafico de entrenamiento original desde `GET /api/v1/legacy/training/plot` si existe `Graficos de entrenamiento de los modelos.png`.
- Informacion de dataset y enlaces Kaggle/Colab/GitHub desde `GET /api/v1/legacy/dataset`.
- Generacion de reporte PDF desde `POST /api/v1/legacy/report`.

Los datos de analisis se devuelven con `source: legacy_precomputed_from_streamlit`. Son valores heredados/precalculados de `app.py`; no deben interpretarse como una evaluacion nueva ni como un pipeline reproducible.

## Verificacion

```powershell
python -m compileall -q backend
cd frontend
npm run build
```