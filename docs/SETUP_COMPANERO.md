# Guia rapida para levantar el proyecto en otra maquina

## Que va por Git

- Codigo del backend FastAPI, Streamlit Admin y frontend Next.js.
- Modelos entrenados ubicados en `models/`.
- Scripts reproducibles en `scripts/`.
- Artefactos livianos de demo para la presentacion en `experiments/evaluations_phase4_demo_all`, `experiments/cross_validation_phase6_demo` y `output/presentation`.

## Que NO va por Git

El dataset `NCT-CRC-HE-100K` y el dataset particionado `NCT-CRC-HE-100K-SPLIT` no deben subirse a Git porque tienen muchas imagenes y pesan demasiado. Compartelos por RAR/ZIP, disco externo, Google Drive, OneDrive o Kaggle.

## Ruta recomendada del dataset

Cada integrante puede extraerlo donde quiera. Luego debe configurar la ruta en `backend/.env`:

```env
COLORECTAL_DATASET_DIR=D:/datasets/colorectal/NCT-CRC-HE-100K-SPLIT
```

Si quiere usar la misma ruta que tu maquina:

```env
COLORECTAL_DATASET_DIR=E:/CICLO07/NCT-CRC-HE-100K-SPLIT
```

## Pasos despues de hacer pull

1. Crear entorno e instalar dependencias Python.
2. Copiar `backend/.env.example` a `backend/.env`.
3. Ajustar `COLORECTAL_DATASET_DIR` a la carpeta real del dataset particionado.
4. Levantar FastAPI.
5. Levantar Streamlit Admin.
6. Levantar Next.js si se quiere probar la vista de cliente.

## Si solo tiene el dataset original

Si recibe `NCT-CRC-HE-100K` sin particionar, ejecutar:

```powershell
python scripts/prepare_dataset_split.py --source D:/datasets/colorectal/NCT-CRC-HE-100K --output D:/datasets/colorectal/NCT-CRC-HE-100K-SPLIT
python scripts/generate_dataset_eda_artifacts.py --dataset D:/datasets/colorectal/NCT-CRC-HE-100K-SPLIT
```

Luego actualizar `COLORECTAL_DATASET_DIR` apuntando a la carpeta `NCT-CRC-HE-100K-SPLIT`.

## Para regenerar evidencias rapidas

```powershell
python scripts/evaluate_models_from_manifest.py --dataset D:/datasets/colorectal/NCT-CRC-HE-100K-SPLIT --output experiments/evaluations_phase4_demo_all --max-samples 45
python scripts/run_phase5_statistics.py
python scripts/generate_stratified_folds.py --dataset D:/datasets/colorectal/NCT-CRC-HE-100K-SPLIT
python scripts/run_cross_validation_demo.py --dataset D:/datasets/colorectal/NCT-CRC-HE-100K-SPLIT
python scripts/generate_phase8_extra_charts.py
python scripts/generate_phase8_statistical_charts.py
python scripts/build_phase8_presentation_report.py
```
