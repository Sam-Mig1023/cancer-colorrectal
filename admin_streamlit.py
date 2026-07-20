from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("FASTAPI_API_URL", os.getenv("API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

TEXT = {
    "es": {
        "page_title": "Panel tecnico colorectal AI",
        "hero_title": "Panel tecnico y cientifico",
        "hero_body": "Administra el flujo interno del proyecto desde FastAPI: modelos entrenados, dataset, EDA, entrenamiento, evaluacion, estadistica, explicabilidad y reportes.",
        "api_url": "URL de FastAPI",
        "language": "Idioma",
        "theme": "Modo oscuro",
        "status": "Estado del backend",
        "connected": "Conectado",
        "disconnected": "Sin conexion",
        "models": "Modelos",
        "available": "Disponibles",
        "nav": "Vista administrativa",
        "overview": "Inicio",
        "diagnosis": "Diagnostico tecnico",
        "dataset": "Dataset y EDA",
        "training": "Entrenamiento",
        "evaluation": "Evaluacion",
        "statistics": "Pruebas estadisticas",
        "explainability": "Explicabilidad",
        "reports": "Reportes",
        "needs_dataset": "Requiere dataset configurado",
        "no_dataset_needed": "No requiere dataset",
        "dataset_reason": "Los modelos ya entrenados permiten inferencia sin dataset. El dataset solo es obligatorio para EDA, nuevo entrenamiento, evaluacion reproducible, validacion cruzada y pruebas estadisticas reales.",
        "original_compare": "Comparacion con app.py original",
        "original_body": "app.py original combina diagnostico, metricas heredadas, graficos y PDF en una sola experiencia Streamlit. La migracion separa responsabilidades: Next.js atiende al usuario final, FastAPI ejecuta la logica Python y este panel Streamlit consume FastAPI para tareas tecnicas.",
        "client_ready": "Usuario final listo",
        "client_ready_body": "Puede probarse con una imagen de PRUEBAS y un modelo disponible: prediccion, probabilidades, PDF y Grad-CAM cuando la arquitectura lo permite.",
        "science_ready": "Flujo cientifico",
        "science_ready_body": "Dataset/EDA, entrenamiento, evaluacion, estadistica y reportes tecnicos requieren ruta de dataset o artefactos de evaluacion generados por el backend.",
        "dataset_path": "Ruta local del dataset",
        "dataset_help": "Dejalo vacio si configuraste COLORECTAL_DATASET_DIR. Estructura esperada: particiones como train/validation/test y clases dentro.",
        "inspect": "Inspeccionar dataset",
        "run_eda": "Ejecutar EDA",
        "sample_limit": "Limite de imagenes a inspeccionar",
        "samples_class": "Muestras representativas por clase",
        "summary": "Resumen",
        "distribution": "Distribucion por clase",
        "insights": "Interpretaciones",
        "upload_image": "Imagen histopatologica",
        "select_model": "Modelo",
        "analyze": "Analizar imagen",
        "prediction": "Prediccion",
        "confidence": "Confianza",
        "probabilities": "Probabilidades",
        "download_pdf": "Descargar PDF",
        "gradcam": "Generar Grad-CAM",
        "gradcam_pdf": "Descargar PDF Grad-CAM",
        "plan": "Generar plan de entrenamiento",
        "start_training": "Iniciar entrenamiento controlado",
        "job_id": "ID de job",
        "check_job": "Consultar job",
        "epochs": "Epocas",
        "batch": "Batch size",
        "lr": "Learning rate",
        "seed": "Semilla",
        "experiment": "Nombre del experimento",
        "eval_run": "Ejecutar evaluacion reproducible",
        "max_samples": "Maximo de muestras",
        "bootstrap": "Iteraciones bootstrap",
        "record": "Ruta de artefacto/result record",
        "download_technical": "Descargar reporte tecnico PDF",
        "first_artifact": "Primer artefacto de evaluacion",
        "second_artifact": "Segundo artefacto de evaluacion",
        "class_name": "Clase para comparar",
        "compare": "Comparar modelos",
        "not_available": "Aun no hay datos para mostrar.",
        "technical_note": "Nota: este panel no recalcula nada en Streamlit. Consume FastAPI para mantener una unica fuente de verdad.",
    },
    "en": {
        "page_title": "Colorectal AI technical panel",
        "hero_title": "Technical and scientific panel",
        "hero_body": "Manage the internal FastAPI workflow: trained models, dataset, EDA, training, evaluation, statistics, explainability, and reports.",
        "api_url": "FastAPI URL",
        "language": "Language",
        "theme": "Dark mode",
        "status": "Backend status",
        "connected": "Connected",
        "disconnected": "Disconnected",
        "models": "Models",
        "available": "Available",
        "nav": "Admin view",
        "overview": "Overview",
        "diagnosis": "Technical diagnosis",
        "dataset": "Dataset and EDA",
        "training": "Training",
        "evaluation": "Evaluation",
        "statistics": "Statistical tests",
        "explainability": "Explainability",
        "reports": "Reports",
        "needs_dataset": "Requires configured dataset",
        "no_dataset_needed": "Does not require dataset",
        "dataset_reason": "Already trained models allow inference without the dataset. The dataset is only required for EDA, new training, reproducible evaluation, cross-validation, and real statistical tests.",
        "original_compare": "Comparison with original app.py",
        "original_body": "The original app.py combines diagnosis, legacy metrics, charts, and PDF generation in one Streamlit experience. The migration separates responsibilities: Next.js serves the final user, FastAPI executes Python logic, and this Streamlit panel consumes FastAPI for technical tasks.",
        "client_ready": "Final user ready",
        "client_ready_body": "Can be tested with a PRUEBAS image and an available model: prediction, probabilities, PDF, and Grad-CAM when the architecture supports it.",
        "science_ready": "Scientific workflow",
        "science_ready_body": "Dataset/EDA, training, evaluation, statistics, and technical reports require a dataset path or evaluation artifacts generated by the backend.",
        "dataset_path": "Local dataset path",
        "dataset_help": "Leave empty if COLORECTAL_DATASET_DIR is configured. Expected structure: partitions such as train/validation/test and class folders inside.",
        "inspect": "Inspect dataset",
        "run_eda": "Run EDA",
        "sample_limit": "Images to inspect",
        "samples_class": "Representative samples per class",
        "summary": "Summary",
        "distribution": "Class distribution",
        "insights": "Interpretations",
        "upload_image": "Histopathology image",
        "select_model": "Model",
        "analyze": "Analyze image",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "probabilities": "Probabilities",
        "download_pdf": "Download PDF",
        "gradcam": "Generate Grad-CAM",
        "gradcam_pdf": "Download Grad-CAM PDF",
        "plan": "Generate training plan",
        "start_training": "Start controlled training",
        "job_id": "Job ID",
        "check_job": "Check job",
        "epochs": "Epochs",
        "batch": "Batch size",
        "lr": "Learning rate",
        "seed": "Seed",
        "experiment": "Experiment name",
        "eval_run": "Run reproducible evaluation",
        "max_samples": "Max samples",
        "bootstrap": "Bootstrap iterations",
        "record": "Artifact/result record path",
        "download_technical": "Download technical PDF report",
        "first_artifact": "First evaluation artifact",
        "second_artifact": "Second evaluation artifact",
        "class_name": "Class to compare",
        "compare": "Compare models",
        "not_available": "No data to display yet.",
        "technical_note": "Note: this panel does not recompute inside Streamlit. It consumes FastAPI to keep a single source of truth.",
    },
}

st.set_page_config(page_title="Colorectal AI Admin", page_icon="M", layout="wide")


def tr(key: str) -> str:
    return TEXT[st.session_state.get("lang", "es")].get(key, key)


def apply_theme(dark: bool) -> None:
    bg = "#0f172a" if dark else "#eef3f8"
    surface = "#111827" if dark else "#ffffff"
    surface_2 = "#172033" if dark else "#f8fafc"
    border = "#334155" if dark else "#d5e0ea"
    text = "#e5e7eb" if dark else "#0f172a"
    muted = "#9ca3af" if dark else "#475569"
    success_bg = "#062e2a" if dark else "#ecfdf5"
    warning_bg = "#3b2f12" if dark else "#fffbeb"
    st.markdown(
        f"""
        <style>
        :root {{ --admin-bg:{bg}; --admin-surface:{surface}; --admin-surface-2:{surface_2}; --admin-border:{border}; --admin-text:{text}; --admin-muted:{muted}; --admin-accent:#0f766e; --admin-blue:#2563eb; }}
        .stApp {{ background: var(--admin-bg); color: var(--admin-text); }}
        .block-container {{ padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1440px; }}
        section[data-testid="stSidebar"] {{ background: var(--admin-surface); border-right: 1px solid var(--admin-border); }}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: var(--admin-muted); }}
        div[data-testid="stMetric"] {{ background: var(--admin-surface); border: 1px solid var(--admin-border); border-radius: 8px; padding: 14px 16px; box-shadow: 0 10px 24px rgba(15,23,42,.06); }}
        div[data-testid="stMetricLabel"] p {{ color: var(--admin-muted); font-size:.78rem; text-transform:uppercase; letter-spacing:0; }}
        div[data-testid="stMetricValue"] {{ color: var(--admin-text); font-size:1.45rem; }}
        div[data-testid="stExpander"] {{ border: 1px solid var(--admin-border); border-radius: 8px; background: var(--admin-surface); }}
        .stDataFrame, div[data-testid="stJson"] {{ border: 1px solid var(--admin-border); border-radius:8px; overflow:hidden; }}
        .stButton button, .stDownloadButton button {{ border-radius:8px; min-height:42px; font-weight:700; }}
        .admin-topbar {{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:18px 20px; border:1px solid var(--admin-border); border-radius:8px; background:var(--admin-surface); box-shadow:0 12px 30px rgba(15,23,42,.08); }}
        .brand-lockup {{ display:flex; align-items:center; gap:14px; min-width:0; }}
        .brand-mark {{ width:46px; height:46px; border-radius:8px; display:grid; place-items:center; background:var(--admin-accent); color:white; font-weight:800; font-size:18px; }}
        .brand-title {{ margin:0; color:var(--admin-text); font-size:1.35rem; line-height:1.2; }}
        .brand-subtitle {{ margin:4px 0 0 0; color:var(--admin-muted); font-size:.9rem; }}
        .topbar-actions {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
        .pill {{ display:inline-flex; align-items:center; gap:7px; padding:7px 11px; border-radius:999px; font-size:.82rem; font-weight:700; border:1px solid var(--admin-border); color:var(--admin-text); background:var(--admin-surface-2); }}
        .pill.ok {{ color:#047857; background:{success_bg}; border-color:#8be0c0; }}
        .pill.warn {{ color:#b45309; background:{warning_bg}; border-color:#f5d38a; }}
        .page-hero {{ padding:20px; border:1px solid var(--admin-border); border-radius:8px; background:linear-gradient(135deg,var(--admin-surface),var(--admin-surface-2)); margin-bottom:16px; }}
        .page-hero h1 {{ margin:0; color:var(--admin-text); font-size:1.8rem; line-height:1.18; }}
        .page-hero p {{ margin:8px 0 0 0; color:var(--admin-muted); max-width:900px; }}
        .panel-card {{ padding:16px; border:1px solid var(--admin-border); border-radius:8px; background:var(--admin-surface); height:100%; }}
        .panel-card h3 {{ margin:0 0 8px 0; color:var(--admin-text); font-size:1rem; }}
        .panel-card p {{ margin:0; color:var(--admin-muted); font-size:.9rem; }}
        .note-card {{ padding:14px 16px; border-left:4px solid var(--admin-accent); border-radius:8px; background:var(--admin-surface); border-top:1px solid var(--admin-border); border-right:1px solid var(--admin-border); border-bottom:1px solid var(--admin-border); color:var(--admin-text); }}
        .section-label {{ color:var(--admin-muted); font-size:.78rem; font-weight:800; text-transform:uppercase; letter-spacing:0; margin-bottom:4px; }}
        .sidebar-brand {{ padding:12px 0 4px 0; }}
        .sidebar-brand strong {{ color:var(--admin-text); font-size:1rem; }}
        .sidebar-brand span {{ color:var(--admin-muted); font-size:.82rem; }}
        @media(max-width:900px) {{ .admin-topbar {{ align-items:flex-start; flex-direction:column; }} .topbar-actions {{ justify-content:flex-start; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, body: str, badge: str | None = None) -> None:
    badge_html = f"<span class='pill'>{badge}</span>" if badge else ""
    st.markdown(
        f"<div class='page-hero'><div style='display:flex;justify-content:space-between;gap:12px;align-items:flex-start;'><div><h1>{title}</h1><p>{body}</p></div>{badge_html}</div></div>",
        unsafe_allow_html=True,
    )


def render_topbar(health: dict[str, Any] | None, models: list[dict[str, Any]], dataset_status: dict[str, Any] | None) -> None:
    api_ok = health is not None
    dataset_ok = bool(dataset_status and dataset_status.get("configured"))
    model_count = len([model for model in models if model.get("available")])
    st.markdown(
        f"""
        <div class="admin-topbar">
          <div class="brand-lockup"><div class="brand-mark">AI</div><div><h1 class="brand-title">Colorectal AI Admin</h1><p class="brand-subtitle">{tr('technical_note')}</p></div></div>
          <div class="topbar-actions">
            <span class="pill {'ok' if api_ok else 'warn'}">API: {tr('connected') if api_ok else tr('disconnected')}</span>
            <span class="pill">{tr('models')}: {model_count}</span>
            <span class="pill {'ok' if dataset_ok else 'warn'}">Dataset: {'OK' if dataset_ok else tr('pending')}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str, label: str | None = None) -> None:
    label_html = f"<div class='section-label'>{label}</div>" if label else ""
    st.markdown(f"<div class='panel-card'>{label_html}<h3>{title}</h3><p>{body}</p></div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='note-card'>{text}</div>", unsafe_allow_html=True)

def api_request(method: str, path: str, *, base_url: str, **kwargs: Any) -> Any:
    try:
        response = requests.request(method, f"{base_url}{path}", timeout=180, **kwargs)
    except requests.RequestException as exc:
        st.error(f"FastAPI connection error: {exc}")
        return None
    if not response.ok:
        detail: Any = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        st.warning(f"API {response.status_code}: {detail}")
        return None
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    return response.content


def safe_json(data: Any) -> None:
    if data is None:
        st.caption(tr("not_available"))
    else:
        st.json(data, expanded=False)


def show_insights(insights: list[dict[str, Any]] | None) -> None:
    if not insights:
        return
    st.markdown(f"#### {tr('insights')}")
    for item in insights:
        section = item.get("section", "analysis")
        finding = item.get("finding", "")
        impact = item.get("impact", "")
        recommendation = item.get("recommendation", "")
        st.info(f"**{section}**\n\n{finding}\n\nImpacto/Impact: {impact}\n\nRecomendacion/Recommendation: {recommendation}")


def params_from_path(dataset_path: str) -> dict[str, str]:
    return {"dataset_path": dataset_path} if dataset_path.strip() else {}


def load_models(base_url: str) -> list[dict[str, Any]]:
    models = api_request("GET", "/api/v1/models", base_url=base_url)
    return models if isinstance(models, list) else []


def overview(base_url: str, models: list[dict[str, Any]]) -> None:
    page_header(tr("hero_title"), tr("hero_body"))
    health = api_request("GET", "/api/v1/health", base_url=base_url)
    dataset_status = api_request("GET", "/api/v1/scientific/dataset/status", base_url=base_url)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("status"), tr("connected") if health else tr("disconnected"))
    c2.metric(tr("models"), len([m for m in models if m.get("available")]))
    c3.metric("Dataset", "OK" if dataset_status and dataset_status.get("configured") else tr("pending"))
    c4.metric("Backend", "FastAPI v1")
    note(tr("dataset_reason"))
    st.write("")
    left, middle, right = st.columns(3)
    with left:
        card(tr("client_ready"), tr("client_ready_body"), tr("no_dataset_needed"))
    with middle:
        card(tr("science_ready"), tr("science_ready_body"), tr("needs_dataset"))
    with right:
        card(tr("original_compare"), tr("original_body"), "app.py")
    st.write("")
    st.markdown(f"#### {tr('model_catalog') if 'model_catalog' in TEXT[st.session_state.get('lang', 'es')] else tr('models')}")
    rows = []
    for model in models:
        rows.append({
            "key": model.get("key"),
            "name": model.get("name"),
            "input": f"{model.get('input_width')}x{model.get('input_height')}",
            "recommended": bool(model.get("recommended")),
            "available": bool(model.get("available")),
        })
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info(tr("no_models") if "no_models" in TEXT[st.session_state.get("lang", "es")] else tr("not_available"))
    st.caption(tr("technical_note"))
def technical_diagnosis(base_url: str, models: list[dict[str, Any]]) -> None:
    page_header(tr("diagnosis"), tr("client_ready_body"), tr("no_dataset_needed"))
    available = [m for m in models if m.get("available")]
    if not available:
        st.warning("No hay modelos disponibles desde FastAPI." if st.session_state.get("lang") == "es" else "No models are available from FastAPI.")
        return

    input_col, result_col = st.columns([0.42, 0.58], gap="large")
    with input_col:
        with st.container(border=True):
            st.markdown(f"#### {tr('technical_input') if tr('technical_input') != 'technical_input' else tr('diagnosis')}")
            selected = st.selectbox(tr("select_model"), available, format_func=lambda m: f"{m.get('name')} ({m.get('key')})" + (f" - {tr('recommended')}" if m.get("recommended") else ""))
            image = st.file_uploader(tr("upload_image"), type=["png", "jpg", "jpeg", "webp"], key="admin_predict_image")
            if image:
                st.image(image, caption="Vista previa" if st.session_state.get("lang") == "es" else "Preview", use_container_width=True)
            analyze_clicked = st.button(tr("analyze"), type="primary", disabled=not image or not selected, use_container_width=True)

    with result_col:
        with st.container(border=True):
            st.markdown(f"#### {tr('result') if tr('result') != 'result' else tr('prediction')}")
            if analyze_clicked and image and selected:
                files = {"image": (image.name, image.getvalue(), image.type or "image/png")}
                result = api_request("POST", f"/api/v1/predict?model={selected['key']}", base_url=base_url, files=files)
                if result:
                    st.session_state["admin_prediction"] = result
                    st.session_state["admin_prediction_image"] = {"name": image.name, "bytes": image.getvalue(), "type": image.type or "image/png"}
            result = st.session_state.get("admin_prediction")
            if not result:
                st.info(tr("not_available"))
                return
            c1, c2, c3 = st.columns(3)
            c1.metric(tr("prediction"), result["predicted_class"])
            c2.metric(tr("confidence"), f"{result['confidence']:.2f}%")
            c3.metric(tr("select_model"), result["model_key"])
            st.markdown(f"##### {tr('probabilities')}")
            for class_name, probability in sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True):
                st.progress(min(max(float(probability) / 100, 0), 1), text=f"{class_name}: {probability:.2f}%")
            payload = {**result, "language": st.session_state.get("lang", "es")}
            stored = st.session_state.get("admin_prediction_image")
            if stored:
                files = {"payload": (None, json.dumps(payload), "application/json"), "image": (stored["name"], stored["bytes"], stored["type"])}
                pdf = api_request("POST", "/api/v1/patient/report", base_url=base_url, files=files)
                if isinstance(pdf, bytes):
                    st.download_button(tr("download_pdf"), pdf, file_name="diagnosis_report.pdf", mime="application/pdf", use_container_width=True)
def dataset_eda(base_url: str) -> None:
    page_header(tr("dataset"), tr("dataset_reason"), tr("needs_dataset"))
    config_col, output_col = st.columns([0.34, 0.66], gap="large")
    with config_col:
        with st.container(border=True):
            st.markdown("#### Configuracion" if st.session_state.get("lang") == "es" else "#### Configuration")
            dataset_path = st.text_input(tr("dataset_path"), help=tr("dataset_help"))
            params = params_from_path(dataset_path)
            inspect_clicked = st.button(tr("inspect"), type="primary", use_container_width=True)
            st.divider()
            sample_limit = st.number_input(tr("sample_limit"), min_value=1, max_value=10000, value=1000, step=100)
            samples_per_class = st.number_input(tr("samples_class"), min_value=1, max_value=12, value=3)
            eda_clicked = st.button(tr("run_eda"), use_container_width=True)
    with output_col:
        with st.container(border=True):
            st.markdown("#### Resultados" if st.session_state.get("lang") == "es" else "#### Results")
            if inspect_clicked:
                st.session_state["dataset_summary"] = api_request("GET", "/api/v1/scientific/dataset/summary", base_url=base_url, params=params)
            if eda_clicked:
                st.session_state["eda"] = api_request("GET", "/api/v1/scientific/eda", base_url=base_url, params={**params, "sample_limit": sample_limit, "samples_per_class": samples_per_class})
            summary = st.session_state.get("dataset_summary")
            analysis = st.session_state.get("eda")
            if not summary and not analysis:
                st.info(tr("not_available"))
                return
            if summary:
                c1, c2, c3 = st.columns(3)
                c1.metric("Images", summary.get("total_images", 0))
                c2.metric("Classes", len(summary.get("class_counts", {})))
                c3.metric("Fingerprint", str(summary.get("dataset_fingerprint_sha256", ""))[:10])
                st.dataframe([{"class": k, "images": v} for k, v in summary.get("class_counts", {}).items()], hide_index=True, use_container_width=True)
                chart = api_request("GET", "/api/v1/scientific/dataset/distribution-chart", base_url=base_url, params=params)
                if isinstance(chart, bytes):
                    st.image(chart, caption=tr("distribution"), use_container_width=True)
                show_insights(summary.get("insights"))
            if analysis:
                st.markdown("#### EDA")
                show_insights(analysis.get("insights"))
                montage = api_request("GET", "/api/v1/scientific/eda/representative-montage", base_url=base_url, params={**params, "sample_limit": sample_limit, "samples_per_class": samples_per_class})
                if isinstance(montage, bytes):
                    st.image(montage, caption="Representative samples", use_container_width=True)
                safe_json({k: v for k, v in analysis.items() if k != "insights"})
def training(base_url: str) -> None:
    page_header(tr("training"), tr("science_ready_body"), tr("needs_dataset"))
    form_col, result_col = st.columns([0.4, 0.6], gap="large")
    with form_col:
        with st.container(border=True):
            st.markdown("#### Configuracion de entrenamiento" if st.session_state.get("lang") == "es" else "#### Training configuration")
            dataset_path = st.text_input(tr("dataset_path"), key="training_dataset")
            c1, c2 = st.columns(2)
            epochs = c1.number_input(tr("epochs"), min_value=1, max_value=200, value=10)
            batch = c2.number_input(tr("batch"), min_value=1, max_value=128, value=32)
            c3, c4 = st.columns(2)
            lr = c3.number_input(tr("lr"), min_value=0.000001, max_value=1.0, value=0.001, format="%.6f")
            seed = c4.number_input(tr("seed"), min_value=0, value=42)
            experiment = st.text_input(tr("experiment"), value="mobilenetv2_experiment")
            payload = {"dataset_path": dataset_path or None, "epochs": int(epochs), "batch_size": int(batch), "learning_rate": float(lr), "seed": int(seed), "experiment_name": experiment}
            plan_clicked = st.button(tr("plan"), use_container_width=True)
            start_clicked = st.button(tr("start_training"), type="primary", use_container_width=True)
            st.divider()
            job_id = st.text_input(tr("job_id"))
            check_clicked = st.button(tr("check_job"), disabled=not job_id, use_container_width=True)
    with result_col:
        with st.container(border=True):
            st.markdown("#### Resultado del proceso" if st.session_state.get("lang") == "es" else "#### Process result")
            if plan_clicked:
                st.session_state["training_output"] = api_request("POST", "/api/v1/scientific/training/plan", base_url=base_url, json=payload)
            if start_clicked:
                st.session_state["training_output"] = api_request("POST", "/api/v1/scientific/training/jobs", base_url=base_url, json=payload)
            if check_clicked:
                st.session_state["training_output"] = api_request("GET", f"/api/v1/scientific/training/jobs/{job_id}", base_url=base_url)
            safe_json(st.session_state.get("training_output"))
def evaluation(base_url: str, models: list[dict[str, Any]]) -> None:
    page_header(tr("evaluation"), tr("science_ready_body"), tr("needs_dataset"))
    available = [m for m in models if m.get("available")]
    form_col, result_col = st.columns([0.38, 0.62], gap="large")
    with form_col:
        with st.container(border=True):
            st.markdown("#### Configuracion" if st.session_state.get("lang") == "es" else "#### Configuration")
            selected = st.selectbox(tr("select_model"), available, format_func=lambda m: f"{m.get('name')} ({m.get('key')})", key="eval_model") if available else None
            dataset_path = st.text_input(tr("dataset_path"), key="eval_dataset")
            c1, c2 = st.columns(2)
            max_samples = c1.number_input(tr("max_samples"), min_value=1, max_value=10000, value=1000)
            bootstrap = c2.number_input(tr("bootstrap"), min_value=100, max_value=5000, value=500)
            seed = st.number_input(tr("seed"), min_value=0, value=42, key="eval_seed")
            run_clicked = st.button(tr("eval_run"), type="primary", disabled=not selected, use_container_width=True)
    with result_col:
        with st.container(border=True):
            st.markdown("#### Resultados de evaluacion" if st.session_state.get("lang") == "es" else "#### Evaluation results")
            if run_clicked and selected:
                payload = {"model_key": selected["key"], "dataset_path": dataset_path or None, "max_samples": int(max_samples), "bootstrap_iterations": int(bootstrap), "seed": int(seed)}
                st.session_state["evaluation"] = api_request("POST", "/api/v1/scientific/evaluations", base_url=base_url, json=payload)
            result = st.session_state.get("evaluation")
            if not result:
                st.info(tr("not_available"))
                return
            metrics = result.get("metrics", result) if isinstance(result, dict) else {}
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}" if isinstance(metrics.get("accuracy"), (int, float)) else "-")
            c2.metric("Macro F1", f"{metrics.get('macro_f1', 0):.4f}" if isinstance(metrics.get("macro_f1"), (int, float)) else "-")
            c3.metric("MCC", f"{metrics.get('mcc', 0):.4f}" if isinstance(metrics.get("mcc"), (int, float)) else "-")
            record = result.get("record") if isinstance(result, dict) else None
            if record:
                pdf = api_request("GET", "/api/v1/scientific/evaluations/report", base_url=base_url, params={"record_path": record, "language": st.session_state.get("lang", "es")})
                if isinstance(pdf, bytes):
                    st.download_button(tr("download_technical"), pdf, file_name="technical_evaluation_report.pdf", mime="application/pdf", use_container_width=True)
            safe_json(result)
def statistics(base_url: str) -> None:
    page_header(tr("statistics"), "Requiere dos artefactos reales de evaluacion generados por FastAPI." if st.session_state.get("lang") == "es" else "Requires two real evaluation artifacts generated by FastAPI.", tr("needs_dataset"))
    form_col, result_col = st.columns([0.38, 0.62], gap="large")
    with form_col:
        with st.container(border=True):
            st.markdown("#### Comparacion pareada" if st.session_state.get("lang") == "es" else "#### Paired comparison")
            first = st.text_input(tr("first_artifact"))
            second = st.text_input(tr("second_artifact"))
            cls = st.selectbox(tr("class_name"), CLASS_NAMES)
            bootstrap = st.number_input(tr("bootstrap"), min_value=100, max_value=5000, value=2000, key="stat_bootstrap")
            compare_clicked = st.button(tr("compare"), type="primary", disabled=not first or not second, use_container_width=True)
    with result_col:
        with st.container(border=True):
            st.markdown("#### Resultados estadisticos" if st.session_state.get("lang") == "es" else "#### Statistical results")
            if compare_clicked:
                payload = {"first_artifact_path": first, "second_artifact_path": second, "class_name": cls, "bootstrap_iterations": int(bootstrap), "seed": 42}
                st.session_state["statistics"] = api_request("POST", "/api/v1/scientific/statistics/compare", base_url=base_url, json=payload)
            safe_json(st.session_state.get("statistics"))


def explainability(base_url: str, models: list[dict[str, Any]]) -> None:
    page_header(tr("explainability"), "Grad-CAM depende de que la arquitectura exponga capas compatibles." if st.session_state.get("lang") == "es" else "Grad-CAM depends on the architecture exposing compatible layers.", tr("no_dataset_needed"))
    available = [m for m in models if m.get("available")]
    form_col, output_col = st.columns([0.38, 0.62], gap="large")
    with form_col:
        with st.container(border=True):
            st.markdown("#### Entrada" if st.session_state.get("lang") == "es" else "#### Input")
            selected = st.selectbox(tr("select_model"), available, format_func=lambda m: f"{m.get('name')} ({m.get('key')})", key="xai_model") if available else None
            cls = st.selectbox(tr("class_name"), CLASS_NAMES, index=0, key="xai_class")
            image = st.file_uploader(tr("upload_image"), type=["png", "jpg", "jpeg", "webp"], key="xai_image")
            if image:
                st.image(image, caption="Vista previa" if st.session_state.get("lang") == "es" else "Preview", use_container_width=True)
            gradcam_clicked = st.button(tr("gradcam"), type="primary", disabled=not image or not selected, use_container_width=True)
    with output_col:
        with st.container(border=True):
            st.markdown("#### Grad-CAM")
            if gradcam_clicked and image and selected:
                files = {"image": (image.name, image.getvalue(), image.type or "image/png")}
                st.session_state["gradcam_png"] = api_request("POST", f"/api/v1/scientific/explainability/grad-cam?model={selected['key']}&target_class={cls}", base_url=base_url, files=files)
                st.session_state["gradcam_source"] = {"name": image.name, "bytes": image.getvalue(), "type": image.type or "image/png", "model": selected["key"], "class": cls}
            png = st.session_state.get("gradcam_png")
            if not isinstance(png, bytes):
                st.info(tr("not_available"))
                return
            st.image(png, caption="Grad-CAM", use_container_width=True)
            source = st.session_state.get("gradcam_source")
            if source:
                files = {"image": (source["name"], source["bytes"], source["type"])}
                pdf = api_request("POST", f"/api/v1/scientific/explainability/grad-cam/report?model={source['model']}&target_class={source['class']}&language={st.session_state.get('lang', 'es')}", base_url=base_url, files=files)
                if isinstance(pdf, bytes):
                    st.download_button(tr("gradcam_pdf"), pdf, file_name="grad_cam_report.pdf", mime="application/pdf", use_container_width=True)
def reports(base_url: str) -> None:
    page_header(tr("reports"), "Centro de descarga de reportes tecnicos generados por FastAPI." if st.session_state.get("lang") == "es" else "Download center for technical reports generated by FastAPI.", tr("needs_dataset"))
    record = st.text_input(tr("record"))
    if st.button(tr("download_technical"), disabled=not record):
        pdf = api_request("GET", "/api/v1/scientific/evaluations/report", base_url=base_url, params={"record_path": record, "language": st.session_state.get("lang", "es")})
        if isinstance(pdf, bytes):
            st.download_button(tr("download_technical"), pdf, file_name="technical_evaluation_report.pdf", mime="application/pdf")


def main() -> None:
    with st.sidebar:
        st.markdown("<div class='sidebar-brand'><strong>Colorectal AI</strong><br><span>Scientific admin console</span></div>", unsafe_allow_html=True)
        st.session_state["lang"] = st.selectbox("Idioma / Language", ["es", "en"], index=0)
        dark = st.toggle("Oscuro / Dark", value=False)
        base_url = st.text_input(tr("api_url"), value=DEFAULT_API_URL)
        apply_theme(dark)
        st.divider()
        view = st.radio(tr("nav"), [tr("overview"), tr("diagnosis"), tr("dataset"), tr("training"), tr("evaluation"), tr("statistics"), tr("explainability"), tr("reports")])
        st.divider()
        st.caption(tr("technical_note"))

    health = api_request("GET", "/api/v1/health", base_url=base_url)
    models = load_models(base_url) if health else []
    dataset_status = api_request("GET", "/api/v1/scientific/dataset/status", base_url=base_url) if health else None
    render_topbar(health, models, dataset_status)
    st.write("")
    if not health:
        st.warning("Inicie FastAPI antes de usar el panel." if st.session_state.get("lang") == "es" else "Start FastAPI before using the panel.")

    if view == tr("overview"):
        overview(base_url, models)
    elif view == tr("diagnosis"):
        technical_diagnosis(base_url, models)
    elif view == tr("dataset"):
        dataset_eda(base_url)
    elif view == tr("training"):
        training(base_url)
    elif view == tr("evaluation"):
        evaluation(base_url, models)
    elif view == tr("statistics"):
        statistics(base_url)
    elif view == tr("explainability"):
        explainability(base_url, models)
    else:
        reports(base_url)

if __name__ == "__main__":
    main()
















