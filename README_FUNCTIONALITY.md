# Guia funcional del proyecto

## 1. Proposito y arquitectura

El proyecto es un sistema academico de apoyo para el analisis de imagenes histopatologicas colorrectales. No confirma cancer ni sustituye la evaluacion de personal medico cualificado.

La solucion esta dividida en tres capas:

```text
Imagen histopatologica / Dataset local
             |
             v
        FastAPI (backend)
   inferencia + ciencia reproducible
          /                 \
         v                   v
 Next.js - cliente       Streamlit - administracion
```

- `frontend/`: aplicacion final para quien analiza una imagen.
- `backend/app/`: unica fuente de inferencia, dataset, EDA, entrenamiento, evaluacion, estadistica, explicabilidad y reportes.
- `admin_streamlit.py`: panel tecnico que consume la API HTTP. No repite calculos en Streamlit.
- `app.py`: aplicacion Streamlit original conservada como referencia. No fue reemplazada ni debe eliminarse.

## 2. Antes de iniciar

### Backend FastAPI

Use el entorno virtual del backend:

```powershell
.\backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
uvicorn backend.app.main:app --reload
```

La API queda disponible en `http://127.0.0.1:8000` y su documentacion en `http://127.0.0.1:8000/docs`.

### Cliente final Next.js

En otra terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

Abra `http://localhost:3000`. `frontend/.env.local` debe contener:

```text
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

### Panel tecnico Streamlit

Con FastAPI iniciado, use un entorno que tenga las dependencias de raiz:

```powershell
python -m pip install -r requirements.txt
$env:FASTAPI_API_URL = "http://127.0.0.1:8000"
python -m streamlit run admin_streamlit.py
```

`FASTAPI_API_URL` es opcional cuando FastAPI usa la direccion por defecto.

## 3. Uso del cliente final Next.js

El cliente final esta deliberadamente simplificado. No expone configuracion de dataset, entrenamiento ni estadistica.

### Diagnostico de una imagen

1. Compruebe que el indicador de servicio muestre disponibilidad.
2. Seleccione un modelo disponible. El modelo recomendado aparece identificado en la lista.
3. Arrastre una imagen al area de carga o haga clic para seleccionarla.
4. Se aceptan `PNG`, `JPG`, `JPEG` y `WEBP`, con un maximo de 15 MB.
5. Revise la vista previa y presione el boton de analisis.
6. El navegador envia la imagen por `multipart/form-data` a `POST /api/v1/predict`; la inferencia se ejecuta exclusivamente en Python/FastAPI.
7. Revise la clase predicha, confianza y probabilidades por clase.

El selector de idioma cambia el contenido del cliente entre espanol e ingles. El control de tema cambia entre modo claro y oscuro.

### Reporte de diagnostico

Con una prediccion disponible, use la descarga de reporte. Next.js envia el resultado validado y la imagen original a `POST /api/v1/patient/report`. El PDF contiene ID, fecha UTC, imagen analizada, modelo, clase estimada, confianza, probabilidades, pie de pagina y advertencia academica.

### Grad-CAM

Use `Mostrar Grad-CAM` despues de una prediccion. La API genera una superposicion asociada al modelo y clase objetivo. El mapa indica regiones que influyeron en la activacion del modelo; no es una confirmacion de lesion ni una explicacion clinica definitiva.

Tras generarlo puede descargarse un PDF de explicabilidad con imagen original, superposicion, metadatos, interpretacion, ID y fecha.

## 4. FastAPI: que hace cada grupo de rutas

### Sistema y diagnostico

| Ruta | Funcion |
| --- | --- |
| `GET /api/v1/health` | Confirma que la API responde y cuenta modelos disponibles. |
| `GET /api/v1/models` | Devuelve los modelos autorizados para la interfaz. |
| `POST /api/v1/predict` | Valida imagen, ejecuta inferencia Python y devuelve clase, confianza y probabilidades. |
| `POST /api/v1/patient/report` | Genera PDF de diagnostico. Acepta JSON para compatibilidad y multipart con imagen para el cliente actual. |

### Migracion historica

| Ruta | Funcion |
| --- | --- |
| `GET /api/v1/legacy/analysis` | Matrices, ROC, MCC, comparaciones y estadistica originales. Siempre indica `legacy_precomputed_from_streamlit`. |
| `GET /api/v1/legacy/training` y `/training/plot` | Datos y grafico historicos de entrenamiento. |
| `GET /api/v1/legacy/dataset` | Informacion historica de dataset y enlaces. |
| `POST /api/v1/legacy/report` | Reporte compatible con el flujo heredado. |

Los resultados legacy no son evaluaciones nuevas y no se mezclan con los experimentos reproducibles.

### Dataset y EDA reproducibles

Configure el dataset local, sin incluir sus imagenes en Git:

```powershell
$env:COLORECTAL_DATASET_DIR = "E:\ruta\al\dataset"
```

El dataset necesita las nueve clases `ADI`, `BACK`, `DEB`, `LYM`, `MUC`, `MUS`, `NORM`, `STR` y `TUM`; para entrenamiento debe contener particiones independientes `train`, `validation` y `test`.

| Ruta | Funcion |
| --- | --- |
| `GET /api/v1/scientific/dataset/status` | Indica si la ruta esta configurada. |
| `GET /api/v1/scientific/dataset/summary` | Conteos, particiones, tamanos y huella SHA-256. |
| `GET /api/v1/scientific/dataset/distribution-chart` | Grafico PNG de distribucion real. |
| `GET /api/v1/scientific/eda` | Resoluciones, formatos, tamanos, archivos corruptos, duplicados y muestras. |
| `GET /api/v1/scientific/eda/representative-montage` | Montaje de muestras representativas. |
| `POST /api/v1/scientific/eda/export` | Exporta EDA a JSON y CSV versionados. |

Cada analisis reproducible devuelve `insights`: que mide, hallazgo, impacto y recomendacion.

### Entrenamiento, evaluacion y estadistica

| Ruta | Funcion |
| --- | --- |
| `POST /api/v1/scientific/training/plan` | Valida dataset, particiones y configuracion antes de entrenar. |
| `POST /api/v1/scientific/training/jobs` | Inicia un trabajo controlado de MobileNetV2 transfer learning. |
| `GET /api/v1/scientific/training/jobs/{job_id}` | Consulta estados `queued`, `running`, `completed` o `failed`. |
| `POST /api/v1/scientific/evaluations` | Evalua un modelo sobre la particion `test`, guarda metricas y artefacto de predicciones. |
| `GET /api/v1/scientific/evaluations/report` | Genera PDF tecnico desde un registro reproducible gestionado. |
| `POST /api/v1/scientific/statistics/compare` | Compara dos artefactos alineados del mismo test. |

El entrenamiento guarda configuracion, metadatos, pesos, historial, curva y metricas. Los trabajos se guardan en memoria mientras FastAPI esta activo: al reiniciar el proceso no se podra consultar un `job_id` anterior, aunque los artefactos ya escritos en `experiments/` se conservan.

La evaluacion reproducible incluye accuracy, precision, sensibilidad/recall, especificidad, F1 por clase, F1 macro, MCC, matriz de confusion, ROC one-vs-rest, AUC macro/micro e intervalos bootstrap al 95 por ciento. Cuando no se entrega un identificador de paciente valido, el intervalo se marca a nivel de imagen y comunica la limitacion.

La comparacion estadistica usa predicciones pareadas de las mismas muestras: McNemar, DeLong para AUC de una clase, bootstrap pareado para diferencias y correccion Holm-Bonferroni. Exige misma huella de dataset, particion `test`, etiquetas y orden de muestras.

### Explicabilidad y reportes tecnicos

| Ruta | Funcion |
| --- | --- |
| `POST /api/v1/scientific/explainability/grad-cam` | Devuelve PNG Grad-CAM para un modelo Keras compatible. |
| `POST /api/v1/scientific/explainability/grad-cam/report` | Genera PDF tecnico con imagen, superposicion y validacion de alineacion. |

El PDF tecnico de evaluacion incluye trazabilidad, metricas globales y por clase, intervalos, matriz de confusion y ROC reales. Solo acepta registros `evaluation_*.json` dentro de `experiments/reports`, para impedir que un archivo externo se presente como resultado cientifico.

## 5. Panel tecnico Streamlit

El panel se encuentra en `admin_streamlit.py`. Todas sus acciones usan `requests` contra FastAPI. En la barra lateral se configura la URL de la API y se comprueba salud/modelos.

### Dataset and EDA

- Escriba una ruta de dataset o use `COLORECTAL_DATASET_DIR`.
- `Inspect dataset` muestra huella, numero de imagenes, tabla por clase, grafico de distribucion e interpretaciones.
- En `Run reproducible EDA` ajuste el limite de muestra y las muestras por clase. Se muestran dimensiones, formatos, archivos corruptos, duplicados y montaje.
- `Export JSON and CSV` guarda resultados versionados en el backend.

### Training

- Defina ruta, particiones, epocas, batch size, learning rate, semilla, nombre de experimento y rotacion.
- `Validate plan` comprueba que las particiones sean distintas y completas antes de usar recursos.
- `Start controlled job` inicia el entrenamiento en segundo plano.
- Pegue o conserve el ID del trabajo y use `Refresh job status` para revisar avance, finalizacion o error.

### Evaluation

- Seleccione un modelo disponible y configure dataset, maximo de imagenes, bootstrap y, solo si existe de verdad, el patron de identificador de paciente.
- `Evaluate on test` evalua exclusivamente `test` y muestra accuracy, Macro F1, MCC, AUC, tabla por clase, intervalos, interpretacion y artefacto de predicciones.
- Cuando hay un registro de evaluacion, el boton descarga el PDF tecnico reproducible.

### Statistics

- Pegue dos rutas de artefactos de prediccion resultantes de evaluaciones comparables.
- Elija la clase para DeLong y el numero de iteraciones bootstrap.
- `Compare models` solo se ejecuta si dataset, test, etiquetas y orden de muestras coinciden. Revise efecto, IC y p-values ajustados, no solo la significancia.

### Explainability

- Cargue una imagen, seleccione modelo y clase objetivo.
- `Generate Grad-CAM` muestra la superposicion.
- `Generate technical PDF` habilita la descarga del informe de explicabilidad.

Los reportes no tienen una pestana separada: estan disponibles en Evaluation y Explainability para que queden unidos a su origen reproducible.

## 6. Comprobacion contra el alcance solicitado

### Implementado e integrado

- Next.js para diagnostico, PDF, Grad-CAM, idioma y tema.
- FastAPI como unica fuente de inferencia para Next.js.
- Dataset, EDA, entrenamiento controlado, evaluacion sobre test, artefactos versionados e interpretaciones.
- Metricas reales: accuracy, precision, sensibilidad/recall, especificidad, F1 por clase y macro, MCC, ROC y AUC.
- Bootstrap de IC, McNemar, DeLong y Holm-Bonferroni en comparaciones pareadas.
- Grad-CAM, PDF de diagnostico, PDF Grad-CAM y PDF tecnico de evaluacion.
- Streamlit administrativo conectado a FastAPI.
- Funciones historicas de `app.py` expuestas como legacy y rotuladas como precalculadas.

### Pendiente o parcial

No deben describirse como terminados los siguientes puntos de la propuesta inicial:

- Transformer/Vision Transformer/CNN+Transformer no esta implementado como arquitectura experimental.
- No hay saliency maps distintos a Grad-CAM ni attention maps.
- La evaluacion aun no calcula weighted F1, PR-AUC ni un classification report separado.
- La comparacion estadistica no incluye permutation test, FDR/Benjamini-Hochberg, Friedman/Nemenyi, calibracion ni Decision Curve Analysis.
- El entrenamiento actual admite una sola arquitectura (`mobilenetv2_transfer`) y usa Adam/dropout predefinidos; no hay seleccion de optimizador o dropout en el formulario.
- No existe persistencia de estados de trabajo de entrenamiento entre reinicios de FastAPI.
- La revision manual de responsive, accesibilidad y de los PDFs descargados debe hacerse en un navegador/lector PDF local antes de una demostracion.

## 7. Verificacion tecnica ejecutada

En la ultima comprobacion local se verifico:

```text
OpenAPI: 23 rutas bajo /api/v1
GET /api/v1/health: 200
GET /api/v1/models: 200
GET /api/v1/legacy/analysis: source=legacy_precomputed_from_streamlit
GET /api/v1/scientific/dataset/status con ruta inexistente: configured=false
backend tests: 20 passed
frontend: npm run build correcto
```

Las pruebas ejercitan contratos de endpoints, formatos no permitidos, limite de 15 MB, reportes, dataset/EDA, evaluacion, estadistica y Grad-CAM. La inferencia real desde frontend ya debe validarse con FastAPI iniciado y una imagen valida de `PRUEBAS/`.
## Aclaracion: que requiere dataset y que no

No todo el sistema depende del dataset local.

Funciona sin dataset:

- Next.js de usuario final: cargar imagen, seleccionar modelo, predecir, ver probabilidades y descargar PDF.
- Streamlit Admin > Diagnostico tecnico: prueba de inferencia usando los modelos ya entrenados.
- Streamlit Admin > Explicabilidad: Grad-CAM cuando el modelo Keras es compatible, por ejemplo `best_model`.
- Listado de modelos y estado de FastAPI.

Requiere dataset o artefactos reales:

- Dataset y EDA: necesita leer carpetas reales de imagenes.
- Entrenamiento: necesita particiones `train`, `validation` y `test` o una estructura equivalente soportada por el backend.
- Evaluacion reproducible: necesita etiquetas reales del conjunto de prueba.
- Pruebas estadisticas robustas: necesitan dos artefactos de evaluacion generados sobre las mismas muestras.
- Validacion cruzada: requiere dataset completo y aun debe implementarse como extension del pipeline de entrenamiento/evaluacion.

La razon es que los modelos `.h5` ya contienen pesos entrenados y permiten inferencia directa. En cambio, EDA, validacion cruzada, metricas reales y pruebas estadisticas necesitan comparar predicciones contra etiquetas reales.

## Comparacion con `app.py` original

`app.py` original funciona como una aplicacion Streamlit monolitica: en una misma pantalla carga imagenes, predice, muestra matrices de confusion, ROC/AUC, MCC, prueba binomial, McNemar, graficos de entrenamiento, informacion de modelos, enlaces y PDF. Muchas metricas son heredadas/precalculadas manualmente.

La migracion actual separa responsabilidades:

- `frontend/` con Next.js queda como vista de usuario final: diagnostico, probabilidades, Grad-CAM cuando aplica y reporte PDF.
- `backend/app/` con FastAPI queda como motor Python: inferencia, dataset, EDA, entrenamiento, evaluacion, estadistica, explicabilidad y reportes.
- `admin_streamlit.py` queda como panel tecnico: consume FastAPI y permite operar/ver el flujo cientifico sin duplicar calculos en Streamlit.

Diferencia importante: el admin no debe reemplazar al backend. Su rol es mostrar formularios, tablas, graficos, resultados e interpretaciones que vienen de FastAPI.

## Estado de validacion reciente

- `npm run build` en `frontend/`: correcto.
- `python -m pytest backend/tests -q`: 20 pruebas correctas.
- Grad-CAM con `best_model`: responde `200 image/png`.
- Grad-CAM con modelos no compatibles, como el caso probado de `cnn_simple`: responde de forma controlada con `422` en vez de romper con `500`.

## Mejora visual del Streamlit Admin

El panel `admin_streamlit.py` fue reorganizado como una consola administrativa profesional:

- barra superior con estado de API, modelos disponibles y dataset;
- sidebar con identidad del panel, idioma, modo claro/oscuro, URL de FastAPI y navegacion;
- encabezado explicativo por cada modulo;
- tarjetas y paneles con borde para separar configuracion, acciones y resultados;
- diagnostico tecnico en dos columnas: entrada de imagen/modelo y lectura de resultado;
- Dataset/EDA, entrenamiento, evaluacion, estadistica y explicabilidad con estructura consistente de formulario + salida;
- JSON tecnico dentro de secciones colapsables cuando aplica.

La regla de arquitectura se mantiene: Streamlit Admin no duplica los calculos, consume los endpoints de FastAPI.
