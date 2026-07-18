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
## Paridad funcional migrada

La migracion FastAPI + Next.js cubre actualmente las funciones visibles de `app.py` relacionadas con diagnostico, analisis heredado, entrenamiento, dataset y reporte:

- Diagnostico con modelos TensorFlow/Keras desde `POST /api/v1/predict`.
- Carga por clic y arrastrar/soltar desde Next.js, con validacion de formato y tamano.
- Matriz de confusion por modelo.
- Curva ROC individual y comparacion ROC de todos los modelos.
- AUC, MCC, prueba binomial, p-value e intervalo de confianza del 95%.
- Interpretacion estadistica de la prueba binomial.
- Prueba de McNemar con chi2, p-value, significancia, interpretacion y tabla de contingencia.
- Comparacion de modelos con exactitud, perdida y tiempo de entrenamiento.
- Informacion/arquitectura de modelos.
- Grafico de entrenamiento heredado.
- Informacion de dataset y enlaces Kaggle/Colab/GitHub.
- Reporte PDF generado desde FastAPI con diagnostico, probabilidades, analisis heredado, matriz, comparacion y McNemar.

Los analisis siguen marcados como `legacy_precomputed_from_streamlit`: son valores heredados/precalculados de Streamlit, no un nuevo pipeline de evaluacion reproducible.
## Contrato y estabilizacion de la API

El backend no requiere variables de entorno para el estado actual. El frontend usa `NEXT_PUBLIC_API_URL` desde `frontend/.env.local`; su valor local por defecto es `http://127.0.0.1:8000`.

| Metodo | Ruta | Uso |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Estado del servicio y modelos disponibles. |
| `GET` | `/api/v1/models` | Modelos disponibles para inferencia. |
| `POST` | `/api/v1/predict?model=<model_key>` | Prediccion Python/TensorFlow con multipart `image`; solo PNG, JPEG o WEBP de hasta 15 MB. |
| `GET` | `/api/v1/legacy/analysis?language=es|en` | Analisis heredado marcado como `legacy_precomputed_from_streamlit`. |
| `GET` | `/api/v1/legacy/training` | Metadatos del grafico de entrenamiento heredado. |
| `GET` | `/api/v1/legacy/training/plot` | Grafico heredado si existe en el proyecto. |
| `GET` | `/api/v1/legacy/dataset?language=es|en` | Informacion y enlaces heredados del dataset. |
| `POST` | `/api/v1/legacy/report` | PDF generado a partir de una prediccion validada. |

El endpoint de reporte exige las nueve clases configuradas, porcentajes entre 0 y 100 y un total de 100 por ciento con tolerancia de redondeo. Esto evita emitir documentos con datos incompletos o inconsistentes enviados desde el navegador.

## Pruebas de estabilizacion

Instale las dependencias del backend, incluidas `pytest` y `httpx`, y ejecute:

```powershell
python -m pytest backend/tests -q
```

La suite valida los contratos de salud y modelos, el rechazo de GIF, la ruta de prediccion con una respuesta simulada, la marca de analisis heredado y la validacion/descarga del PDF. No carga pesos reales de TensorFlow, por lo que es rapida y no modifica `models/`.
## Backend cientifico reproducible

El backend incorpora modulos separados para `dataset`, `eda`, `training`, `evaluation`, `statistical_tests`, `explainability` y `reporting`. Las rutas se organizan en `backend/app/routers/` por tres dominios estables: inferencia, migracion heredada y flujo cientifico.

Configure una copia local del dataset sin incluirla en Git. Puede usar la ruta por defecto `data/NCT-CRC-HE-100K` o definir la variable de sesion:

```powershell
$env:COLORECTAL_DATASET_DIR = "E:\ruta\a\NCT-CRC-HE-100K"
```

El dataset debe contener directorios de clase `ADI`, `BACK`, `DEB`, `LYM`, `MUC`, `MUS`, `NORM`, `STR` y `TUM`, directamente o dentro de particiones como `train`, `validation` y `test`.

| Metodo | Ruta | Uso |
| --- | --- | --- |
| `GET` | `/api/v1/scientific/dataset/status` | Confirma si el dataset esta configurado, sin simular resultados. |
| `GET` | `/api/v1/scientific/dataset/summary` | Registra conteos, tamanos, fechas de modificacion, particiones y huella SHA-256 del dataset real. |
| `GET` | `/api/v1/scientific/dataset/distribution-chart` | Genera un grafico PNG de la distribucion real por clase. |
| `GET` | `/api/v1/scientific/eda` | Inspecciona dimensiones, formatos, tamanos, archivos corruptos, duplicados exactos y muestras representativas. |
| `GET` | `/api/v1/scientific/eda/representative-montage` | Genera un montaje PNG de muestras representativas reales. |
| `POST` | `/api/v1/scientific/eda/export` | Exporta el EDA versionado a JSON y la distribucion a CSV. |
| `POST` | `/api/v1/scientific/training/plan` | Valida configuracion de entrenamiento y particiones antes de ejecutar. |
| `POST` | `/api/v1/scientific/evaluations` | Evalua un modelo sobre imagenes reales y versiona el resultado. |
| `POST` | `/api/v1/scientific/explainability/grad-cam` | Genera Grad-CAM para modelos Keras compatibles. |

Cada resumen y EDA incluye `insights` con cuatro campos: que mide el resultado, hallazgo, impacto sobre el entrenamiento/evaluacion y recomendacion. La deteccion de duplicados se basa en SHA-256 de contenido sobre la muestra inspeccionada; antes de declarar que el dataset completo esta libre de duplicados debe aumentarse `sample_limit` hasta cubrirlo.

Los endpoints cientificos no reutilizan datos manuales de Streamlit. Si el dataset no esta configurado, responden con un estado explicito o `409`. El codigo de `run_training` existe en `backend/app/training.py`, pero aun no se expone como una solicitud HTTP de larga duracion; se integrara al panel administrativo mediante un trabajo controlado en el siguiente paso.

Para verificar el backend con el entorno virtual:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend\tests -q
.\backend\.venv\Scripts\python.exe -m compileall -q backend\app backend\tests
```
## Entrenamiento y evaluacion reproducibles

El entrenamiento requiere tres particiones distintas: `train`, `validation` y `test`. FastAPI no ejecuta el entrenamiento dentro de una respuesta HTTP: primero valida el plan y luego inicia un trabajo en segundo plano. Los estados del trabajo se conservan en memoria mientras FastAPI esta activo y los artefactos del experimento permanecen en `experiments/`.

| Metodo | Ruta | Uso |
| --- | --- | --- |
| `POST` | `/api/v1/scientific/training/plan` | Valida arquitectura, semilla, aumento, hiperparametros y tres particiones antes de ejecutar. |
| `POST` | `/api/v1/scientific/training/jobs` | Inicia un trabajo controlado de MobileNetV2 transfer learning. |
| `GET` | `/api/v1/scientific/training/jobs/{job_id}` | Consulta estado `queued`, `running`, `completed` o `failed`. |
| `POST` | `/api/v1/scientific/evaluations` | Evalua un modelo existente exclusivamente sobre la particion `test`. |

Cada entrenamiento guarda `metadata.json`, `config.json`, `history.json`, `history.csv`, `training_curves.png`, `best_model.keras`, `final_model.keras` y `test_metrics.json`. El registro incluye semilla, fecha, duracion, huella del dataset, particiones y configuracion. La curva PNG contiene loss y accuracy de entrenamiento/validacion reales.

La evaluacion calcula accuracy, precision, sensibilidad/recall, especificidad, F1 por clase, matriz de confusion, MCC, curvas ROC one-vs-rest, AUC macro y micro. Tambien genera intervalos de confianza bootstrap al 95 por ciento para accuracy, F1 macro, MCC y AUC. Sin `patient_id_regex`, el backend marca el intervalo como `image` y advierte que podria ser optimista por correlacion entre imagenes del mismo paciente. Solo use `patient_id_regex` cuando el nombre de archivo contenga un identificador de paciente real y documentado.

Ejemplo de plan o trabajo:

```json
{
  "dataset_path": "E:\\ruta\\a\\dataset",
  "train_partition": "train",
  "validation_partition": "validation",
  "test_partition": "test",
  "architecture": "mobilenetv2_transfer",
  "epochs": 20,
  "batch_size": 32,
  "learning_rate": 0.001,
  "seed": 42,
  "experiment_name": "mobilenetv2_v1"
}
```
## Comparacion estadistica y explicabilidad

Los analisis nuevos no usan `legacy_precomputed_from_streamlit`. Una evaluacion reproducible sobre `test` guarda un artefacto comprimido de etiquetas, probabilidades y orden de muestras en `experiments/predictions/`. Solo esos artefactos gestionados se aceptan para comparaciones entre modelos.

| Metodo | Ruta | Uso |
| --- | --- | --- |
| `POST` | `/api/v1/scientific/statistics/compare` | Compara dos artefactos pareados del mismo dataset y particion `test`. |
| `POST` | `/api/v1/scientific/explainability/grad-cam` | Devuelve la superposicion Grad-CAM PNG y metadatos en cabeceras. |
| `POST` | `/api/v1/scientific/explainability/grad-cam/report` | Genera un PDF tecnico con imagen original, superposicion, intensidad y advertencia academica. |

La comparacion requiere misma huella de dataset, misma particion `test`, mismas etiquetas y mismo orden de muestras. Incluye:

- McNemar para comparar aciertos y errores pareados.
- Bootstrap pareado para diferencias de accuracy, F1 macro, MCC y AUC macro/micro, con efecto e intervalo al 95 por ciento.
- DeLong para comparar el AUC one-vs-rest de la clase seleccionada.
- Ajuste Holm-Bonferroni para los p-values de McNemar y DeLong.

DeLong requiere ejemplos positivos y negativos de la clase elegida. El resultado reporta diferencia de AUC, error estandar, intervalo al 95 por ciento y p-value ajustado. No se presenta como una prueba global multicategoria. Las comparaciones por imagen incluyen una limitacion explicita; para bootstrap a nivel de paciente, ambos artefactos deben incluir identificadores de paciente genuinos y alineados.

Grad-CAM se habilita solo para modelos Keras con una capa espacial compatible. El backend valida que imagen original, mapa y superposicion compartan dimensiones, y reporta clase objetivo, confianza e intensidad. La coincidencia anatomopatologica debe seguir siendo revisada por una persona cualificada; el mapa no confirma malignidad ni sustituye un diagnostico clinico.

## Experiencias separadas: usuario y administracion

La migracion mantiene dos interfaces con responsabilidades distintas, ambas conectadas al mismo backend FastAPI.

### Aplicacion de usuario final (Next.js)

La carpeta `frontend/` contiene la interfaz clinica/académica dirigida al usuario final. Solo expone el flujo de diagnostico: carga o arrastre de una imagen PNG/JPG/JPEG/WEBP de hasta 15 MB, seleccion de un modelo disponible, prediccion, confianza, probabilidades, Grad-CAM cuando el modelo es compatible y descarga de un reporte de diagnostico. El reporte usa `POST /api/v1/patient/report` y no incorpora metricas heredadas ni controles cientificos. La interfaz conserva el aviso de uso academico y no ofrece EDA, entrenamiento, evaluacion ni pruebas estadisticas.

Ejecutar desde `frontend/`:

```powershell
npm run dev
```

### Panel tecnico-administrativo (Streamlit)

El nuevo `admin_streamlit.py` es un panel separado para el equipo tecnico. Consume FastAPI por HTTP, sin repetir inferencia, EDA, entrenamiento, evaluacion ni calculos estadisticos en Streamlit. Incluye consulta y validacion de dataset, EDA y exportaciones, inicio/seguimiento de entrenamientos controlados, evaluacion del conjunto de prueba, comparacion estadistica de artefactos reales y Grad-CAM tecnico con reporte.

El `app.py` original se conserva intacto como referencia funcional y no es reemplazado por este panel.

Con FastAPI activo, instalar las dependencias raiz en el entorno que se usara para Streamlit y ejecutar:

```powershell
python -m pip install -r requirements.txt
$env:FASTAPI_API_URL = "http://127.0.0.1:8000"
python -m streamlit run admin_streamlit.py
```

El valor de `FASTAPI_API_URL` es opcional cuando FastAPI escucha en `http://127.0.0.1:8000`.
## Reportes y validacion final

Existen dos reportes separados y ambos llevan identificador, fecha y pie de pagina:

- `POST /api/v1/patient/report`: reporte de resultado para la interfaz Next.js. Incluye la imagen analizada, modelo, clase estimada, confianza, probabilidades y una advertencia visible de uso academico. No muestra metricas heredadas ni afirma un diagnostico clinico.
- `GET /api/v1/scientific/evaluations/report?record_path=...&language=es`: reporte tecnico para el panel Streamlit. Solo acepta un registro `evaluation_*.json` gestionado en `experiments/reports/`, y genera la trazabilidad del experimento, matriz de confusion, ROC reales, metricas por clase, intervalos de confianza e interpretacion de la evaluacion reproducible.
- `POST /api/v1/scientific/explainability/grad-cam/report`: reporte tecnico de explicabilidad con imagen original, superposicion Grad-CAM y validacion de alineacion.

Validacion recomendada antes de una entrega:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend\tests -q
.\backend\.venv\Scripts\python.exe -m compileall -q backend\app backend\tests
Set-Location frontend
npm run build
```

La suite verifica contratos de endpoints, tipos no admitidos, limite de 15 MB, coherencia del payload, analisis heredado marcado, dataset/EDA, metricas y estadistica reproducible, Grad-CAM y generacion real de PDF. La revision manual pendiente de cada entrega debe comprobar los flujos en el navegador: arrastre/clic, respuesta de errores, modo claro/oscuro, idioma, vista movil, descarga de los tres reportes y la consistencia de una prediccion FastAPI frente a una ejecucion controlada del `app.py` original.

## Guia de uso

Consulte [README_FUNCTIONALITY.md](README_FUNCTIONALITY.md) para el flujo del cliente final, los endpoints y las vistas del panel Streamlit, ademas de la comprobacion del alcance implementado y sus limites actuales.
