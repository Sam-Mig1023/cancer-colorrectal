from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("FASTAPI_API_URL", os.getenv("API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
PROJECT_ROOT = Path(__file__).resolve().parent

def configured_dataset_root(project_root: Path) -> Path:
    raw = os.getenv("COLORECTAL_DATASET_DIR")
    env_path = project_root / "backend" / ".env"
    if not raw and env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("COLORECTAL_DATASET_DIR="):
                raw = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    return Path(raw or str(project_root / "data" / "NCT-CRC-HE-100K-SPLIT")).expanduser()

PRESENTATION_DIR = PROJECT_ROOT / "output" / "presentation"
PHASE4_DIR = PROJECT_ROOT / "experiments" / "evaluations_phase4_demo_all"
PHASE5_DIR = PHASE4_DIR / "statistics"
PHASE6_DIR = PROJECT_ROOT / "experiments" / "cross_validation_phase6_demo"
ROC_AUC_CHART = PRESENTATION_DIR / "roc_auc_comparison_tum.png"
TRAINING_CHART = PROJECT_ROOT / "Graficos de entrenamiento de los modelos.png"
STAT_MCNEMAR_CHART = PRESENTATION_DIR / "statistical_mcnemar_contingency.png"
STAT_DELONG_CHART = PRESENTATION_DIR / "statistical_delong_auc.png"
STAT_BOOTSTRAP_CHART = PRESENTATION_DIR / "statistical_bootstrap_ci.png"
STAT_PVALUES_CHART = PRESENTATION_DIR / "statistical_pvalues_holm.png"
DATASET_ROOT = configured_dataset_root(PROJECT_ROOT)
EDA_DIR = DATASET_ROOT / "eda"

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
        "presentation": "Presentacion",
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
        "presentation": "Presentation",
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
    bg = "#0f172a" if dark else "#f3f7fb"
    surface = "#111827" if dark else "#ffffff"
    surface_2 = "#172033" if dark else "#f8fbff"
    sidebar = "#111827" if dark else "#ffffff"
    sidebar_2 = "#172033" if dark else "#f6f9fc"
    border = "#334155" if dark else "#d8e2ed"
    text = "#e5e7eb" if dark else "#102033"
    muted = "#9ca3af" if dark else "#526274"
    input_bg = "#0b1220" if dark else "#ffffff"
    input_text = "#f8fafc" if dark else "#102033"
    primary = "#14b8a6" if dark else "#2563eb"
    primary_hover = "#0d9488" if dark else "#1d4ed8"
    danger = "#fb7185" if dark else "#e11d48"
    success_bg = "#062e2a" if dark else "#e8f8f1"
    warning_bg = "#3b2f12" if dark else "#fff7df"
    info_bg = "#0f2744" if dark else "#eaf3ff"
    code_bg = "#0b1220" if dark else "#ffffff"
    code_text = "#e5e7eb" if dark else "#102033"
    st.markdown(
        f"""
        <style>
        :root {{
          --admin-bg:{bg}; --admin-surface:{surface}; --admin-surface-2:{surface_2};
          --admin-sidebar:{sidebar}; --admin-sidebar-2:{sidebar_2}; --admin-border:{border};
          --admin-text:{text}; --admin-muted:{muted}; --admin-input-bg:{input_bg}; --admin-input-text:{input_text};
          --admin-primary:{primary}; --admin-primary-hover:{primary_hover}; --admin-danger:{danger};
          --admin-code-bg:{code_bg}; --admin-code-text:{code_text};
        }}
        .stApp {{ background: radial-gradient(circle at top left, rgba(37,99,235,.08), transparent 30%), var(--admin-bg); color: var(--admin-text); }}
        .block-container {{ padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1440px; }}
        h1, h2, h3, h4, h5, h6, p, label, span {{ color: var(--admin-text); }}
        label, .stMarkdown, [data-testid="stMarkdownContainer"] {{ color: var(--admin-text); }}
        [data-testid="stWidgetLabel"] label p {{ color: var(--admin-muted); font-weight:700; }}

        section[data-testid="stSidebar"] {{ background: linear-gradient(180deg,var(--admin-sidebar),var(--admin-sidebar-2)); border-right:1px solid var(--admin-border); box-shadow: 10px 0 28px rgba(15,23,42,.06); }}
        section[data-testid="stSidebar"] > div {{ padding-top:1rem; }}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {{ color: var(--admin-text); }}
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: var(--admin-muted); font-size:.78rem; text-transform:uppercase; font-weight:800; }}
        section[data-testid="stSidebar"] [role="radiogroup"] label {{ padding:7px 9px; border-radius:8px; margin:1px 0; }}
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(37,99,235,.08); }}
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background: rgba(37,99,235,.12); border:1px solid rgba(37,99,235,.20); }}
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color: var(--admin-muted); line-height:1.45; }}

        .sidebar-brand {{ padding:14px 12px 12px 12px; border:1px solid var(--admin-border); border-radius:8px; background:var(--admin-surface-2); margin-bottom:10px; }}
        .sidebar-brand strong {{ color:var(--admin-text); font-size:1.05rem; }}
        .sidebar-brand span {{ color:var(--admin-muted); font-size:.82rem; }}

        input, textarea, select,
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"],
        [data-baseweb="input"] {{ background:var(--admin-input-bg) !important; color:var(--admin-input-text) !important; border-color:var(--admin-border) !important; }}
        input::placeholder, textarea::placeholder {{ color:var(--admin-muted) !important; opacity:1; }}
        div[data-baseweb="input"], div[data-baseweb="select"] {{ border-radius:8px; }}
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {{ border-radius:8px; border-color:var(--admin-border) !important; box-shadow:none !important; }}
        div[data-baseweb="input"]:focus-within > div, div[data-baseweb="select"]:focus-within > div {{ border-color:var(--admin-primary) !important; box-shadow:0 0 0 2px rgba(37,99,235,.14) !important; }}

        .stButton button, .stDownloadButton button {{ border-radius:8px; min-height:42px; font-weight:700; border:1px solid var(--admin-border); background:var(--admin-surface); color:var(--admin-text); }}
        .stButton button:hover, .stDownloadButton button:hover {{ border-color:var(--admin-primary); color:var(--admin-primary); background:var(--admin-surface-2); }}
        .stButton button[kind="primary"], button[kind="primary"] {{ background:var(--admin-primary) !important; color:#ffffff !important; border-color:var(--admin-primary) !important; }}
        .stButton button[kind="primary"]:hover, button[kind="primary"]:hover {{ background:var(--admin-primary-hover) !important; border-color:var(--admin-primary-hover) !important; }}

        div[data-testid="stMetric"] {{ background: var(--admin-surface); border:1px solid var(--admin-border); border-radius:8px; padding:14px 16px; box-shadow:0 10px 24px rgba(15,23,42,.06); }}
        div[data-testid="stMetricLabel"] p {{ color:var(--admin-muted); font-size:.78rem; text-transform:uppercase; font-weight:800; }}
        div[data-testid="stMetricValue"] {{ color:var(--admin-text); font-size:1.45rem; }}
        .stDataFrame, div[data-testid="stJson"] {{ border:1px solid var(--admin-border); border-radius:8px; overflow:hidden; }}
        [data-testid="stAlert"] {{ background:{info_bg}; border-radius:8px; border:1px solid var(--admin-border); color:var(--admin-text); }}
        [data-testid="stAlert"] * {{ color:var(--admin-text) !important; }}

        div[data-testid="stFileUploaderDropzone"] {{ background:var(--admin-surface) !important; border:1px dashed var(--admin-border) !important; border-radius:8px !important; color:var(--admin-text) !important; }}
        div[data-testid="stFileUploaderDropzone"] * {{ color:var(--admin-text) !important; }}
        div[data-testid="stFileUploaderDropzone"] small,
        div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {{ color:var(--admin-muted) !important; }}
        div[data-testid="stFileUploaderDropzone"] button {{ background:var(--admin-surface-2) !important; color:var(--admin-primary) !important; border:1px solid var(--admin-border) !important; }}
        div[data-testid="stFileUploaderDropzone"] button:hover {{ background:rgba(37,99,235,.08) !important; border-color:var(--admin-primary) !important; }}

        [data-testid="stNumberInput"] button {{ background:var(--admin-surface-2) !important; color:var(--admin-text) !important; border-color:var(--admin-border) !important; }}
        [data-testid="stNumberInput"] button:hover {{ background:rgba(37,99,235,.10) !important; color:var(--admin-primary) !important; }}
        [data-testid="stNumberInput"] input {{ background:var(--admin-input-bg) !important; color:var(--admin-input-text) !important; }}

        div[data-testid="stJson"] {{ background:var(--admin-surface) !important; color:var(--admin-text) !important; border:1px solid var(--admin-border); border-radius:8px; }}
        div[data-testid="stJson"] * {{ color:var(--admin-text) !important; }}
        pre, code {{ background:var(--admin-surface-2) !important; color:var(--admin-text) !important; border-radius:8px; }}
        div[data-testid="stExpander"] {{ background:var(--admin-surface) !important; border:1px solid var(--admin-border) !important; border-radius:8px !important; box-shadow:0 8px 22px rgba(15,23,42,.04); }}
        div[data-testid="stExpander"] details summary {{ color:var(--admin-text) !important; background:var(--admin-surface-2) !important; border-radius:8px 8px 0 0; }}
        div[data-testid="stExpander"] details summary * {{ color:var(--admin-text) !important; }}
        .interpretation-box {{ padding:14px 16px; margin:10px 0 18px 0; border:1px solid var(--admin-border); border-left:4px solid var(--admin-primary); border-radius:8px; background:var(--admin-surface); box-shadow:0 8px 22px rgba(15,23,42,.04); }}
        .interpretation-box strong {{ color:var(--admin-text); }}
        .interpretation-box p {{ margin:4px 0 0 0; color:var(--admin-muted); line-height:1.55; }}

        .admin-topbar {{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:18px 20px; border:1px solid var(--admin-border); border-radius:8px; background:var(--admin-surface); box-shadow:0 12px 30px rgba(15,23,42,.08); }}
        .brand-lockup {{ display:flex; align-items:center; gap:14px; min-width:0; }}
        .brand-mark {{ width:46px; height:46px; border-radius:8px; display:grid; place-items:center; background:var(--admin-primary); color:white; font-weight:800; font-size:18px; }}
        .brand-title {{ margin:0; color:var(--admin-text); font-size:1.35rem; line-height:1.2; }}
        .brand-subtitle {{ margin:4px 0 0 0; color:var(--admin-muted); font-size:.9rem; }}
        .topbar-actions {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
        .pill {{ display:inline-flex; align-items:center; gap:7px; padding:7px 11px; border-radius:999px; font-size:.82rem; font-weight:700; border:1px solid var(--admin-border); color:var(--admin-text); background:var(--admin-surface-2); }}
        .pill.ok {{ color:#047857; background:{success_bg}; border-color:#8be0c0; }}
        .pill.warn {{ color:#b45309; background:{warning_bg}; border-color:#f5d38a; }}
        .page-hero {{ padding:20px; border:1px solid var(--admin-border); border-radius:8px; background:linear-gradient(135deg,var(--admin-surface),var(--admin-surface-2)); margin-bottom:16px; box-shadow:0 10px 26px rgba(15,23,42,.05); }}
        .page-hero h1 {{ margin:0; color:var(--admin-text); font-size:1.8rem; line-height:1.18; }}
        .page-hero p {{ margin:8px 0 0 0; color:var(--admin-muted); max-width:900px; }}
        .panel-card {{ padding:16px; border:1px solid var(--admin-border); border-radius:8px; background:var(--admin-surface); height:100%; box-shadow:0 8px 22px rgba(15,23,42,.04); }}
        .panel-card h3 {{ margin:0 0 8px 0; color:var(--admin-text); font-size:1rem; }}
        .panel-card p {{ margin:0; color:var(--admin-muted); font-size:.9rem; }}
        .note-card {{ padding:14px 16px; border-left:4px solid var(--admin-primary); border-radius:8px; background:var(--admin-surface); border-top:1px solid var(--admin-border); border-right:1px solid var(--admin-border); border-bottom:1px solid var(--admin-border); color:var(--admin-text); }}
        .section-label {{ color:var(--admin-muted); font-size:.78rem; font-weight:800; text-transform:uppercase; letter-spacing:0; margin-bottom:4px; }}
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


def interpretation_box(text: str) -> None:
    st.markdown(f"<div class='interpretation-box'><strong>Interpretacion</strong><p>{text}</p></div>", unsafe_allow_html=True)

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

def load_demo_prediction_artifacts() -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if not PHASE4_DIR.is_dir():
        return artifacts
    for model_dir in sorted(PHASE4_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        artifact_file = model_dir / "artifact_paths.json"
        if not artifact_file.is_file():
            continue
        try:
            payload = json.loads(artifact_file.read_text(encoding="utf-8"))
            path = payload.get("prediction_artifact", {}).get("artifact_path")
        except (OSError, json.JSONDecodeError):
            path = None
        if path:
            artifacts[model_dir.name] = path
    return artifacts

def statistics(base_url: str) -> None:
    spanish = st.session_state.get("lang") == "es"
    page_header(
        tr("statistics"),
        "Compara dos modelos usando predicciones guardadas sobre exactamente las mismas imagenes." if spanish else "Compare two models using saved predictions over exactly the same images.",
        tr("needs_dataset"),
    )
    form_col, result_col = st.columns([0.40, 0.60], gap="large")
    artifacts = load_demo_prediction_artifacts()
    artifact_names = list(artifacts)
    with form_col:
        with st.container(border=True):
            st.markdown("#### Comparacion pareada" if spanish else "#### Paired comparison")
            st.caption(
                "Un artefacto de evaluacion es un archivo .npz con etiquetas reales, probabilidades y IDs de muestra. Se genera cuando evaluamos un modelo en Fase 4. Para una comparacion valida, ambos artefactos deben venir de la misma particion test y del mismo orden de imagenes."
                if spanish
                else "An evaluation artifact is an .npz file with true labels, probabilities, and sample IDs. It is generated when a model is evaluated in Phase 4. For a valid comparison, both artifacts must come from the same test partition and image order."
            )
            if artifact_names:
                default_first = artifact_names.index("best_model") if "best_model" in artifact_names else 0
                default_second = artifact_names.index("cnn_simple") if "cnn_simple" in artifact_names else min(1, len(artifact_names) - 1)
                first_name = st.selectbox("Modelo 1 / baseline" if spanish else "Model 1 / baseline", artifact_names, index=default_first, key="stat_first_model")
                second_name = st.selectbox("Modelo 2 / comparado" if spanish else "Model 2 / compared", artifact_names, index=default_second, key="stat_second_model")
                first = artifacts[first_name]
                second = artifacts[second_name]
                st.caption(("Artefactos seleccionados automaticamente desde Fase 4 demo." if spanish else "Artifacts selected automatically from Phase 4 demo."))
            else:
                st.warning("No encontre artefactos demo. Ejecuta primero la Fase 4 o pega rutas manualmente." if spanish else "No demo artifacts found. Run Phase 4 first or paste paths manually.")
                first = ""
                second = ""

            with st.expander("Rutas avanzadas" if spanish else "Advanced paths"):
                first = st.text_input(tr("first_artifact"), value=first, help="Archivo .npz del primer modelo" if spanish else "First model .npz file")
                second = st.text_input(tr("second_artifact"), value=second, help="Archivo .npz del segundo modelo" if spanish else "Second model .npz file")
            cls = st.selectbox(tr("class_name"), CLASS_NAMES, index=CLASS_NAMES.index("TUM"), help="Clase usada para comparar AUC con DeLong" if spanish else "Class used for DeLong AUC comparison")
            bootstrap = st.number_input(tr("bootstrap"), min_value=100, max_value=5000, value=2000, key="stat_bootstrap")
            compare_clicked = st.button(tr("compare"), type="primary", disabled=not first or not second or first == second, use_container_width=True)
            if first == second and first:
                st.warning("Selecciona dos modelos diferentes." if spanish else "Select two different models.")
    with result_col:
        with st.container(border=True):
            st.markdown("#### Resultados estadisticos" if spanish else "#### Statistical results")
            if compare_clicked:
                payload = {"first_artifact_path": first, "second_artifact_path": second, "class_name": cls, "bootstrap_iterations": int(bootstrap), "seed": 42}
                st.session_state["statistics"] = api_request("POST", "/api/v1/scientific/statistics/compare", base_url=base_url, json=payload)
            result = st.session_state.get("statistics")
            if not result:
                st.info("Elige dos modelos y presiona Comparar modelos. El sistema verificara que usen las mismas muestras antes de calcular las pruebas." if spanish else "Choose two models and press Compare models. The system will verify matching samples before computing tests.")
                return
            mcnemar = result.get("mcnemar", {}) if isinstance(result, dict) else {}
            delong = result.get("delong", {}) if isinstance(result, dict) else {}
            bootstrap_metrics = result.get("paired_bootstrap", {}).get("metrics", {}) if isinstance(result, dict) else {}
            c1, c2, c3 = st.columns(3)
            c1.metric("McNemar p", f"{mcnemar.get('p_value', 0):.4g}" if isinstance(mcnemar.get("p_value"), (int, float)) else "-")
            c2.metric("DeLong p", f"{delong.get('p_value', 0):.4g}" if isinstance(delong.get("p_value"), (int, float)) else "-")
            acc_diff = bootstrap_metrics.get("accuracy", {}).get("effect_difference")
            c3.metric("Diff accuracy", f"{acc_diff:.4f}" if isinstance(acc_diff, (int, float)) else "-")
            st.dataframe(
                [
                    {"prueba": "McNemar", "que_evalua": "Diferencia pareada de aciertos/errores", "p_value": mcnemar.get("p_value"), "significativo": mcnemar.get("significant")},
                    {"prueba": "DeLong", "que_evalua": f"Diferencia AUC para {cls}", "p_value": delong.get("p_value"), "significativo": delong.get("significant")},
                    {"prueba": "Bootstrap", "que_evalua": "Intervalos de confianza de diferencias", "p_value": "N/A", "significativo": "ver IC95"},
                    {"prueba": "Holm-Bonferroni", "que_evalua": "Correccion por multiples pruebas", "p_value": "ajustado", "significativo": "ver tabla"},
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("##### McNemar")
            show_image_if_exists(STAT_MCNEMAR_CHART, "McNemar - contingencia visual" if spanish else "McNemar - visual contingency")
            st.dataframe(
                [
                    {"indicador": "tabla_contingencia", "valor": str(mcnemar.get("table"))},
                    {"indicador": "discordantes solo modelo 1", "valor": mcnemar.get("discordant_first_only")},
                    {"indicador": "discordantes solo modelo 2", "valor": mcnemar.get("discordant_second_only")},
                    {"indicador": "chi2", "valor": mcnemar.get("chi2")},
                    {"indicador": "p_value", "valor": mcnemar.get("p_value")},
                    {"indicador": "significativo", "valor": mcnemar.get("significant")},
                ],
                use_container_width=True,
                hide_index=True,
            )
            interpretation_box("McNemar evalua si un modelo acierta significativamente mas imagenes que el otro considerando solo los casos donde difieren. Es la prueba mas directa para comparar errores/aciertos pareados." if spanish else "McNemar tests whether one model gets significantly more images right than the other, using only discordant cases. It is the most direct paired test for correct/incorrect outcomes.")
            st.markdown("##### DeLong AUC")
            show_image_if_exists(STAT_DELONG_CHART, "DeLong AUC - clase comparada" if spanish else "DeLong AUC - compared class")
            st.dataframe(
                [
                    {"indicador": "auc_modelo_1", "valor": delong.get("auc_first")},
                    {"indicador": "auc_modelo_2", "valor": delong.get("auc_second")},
                    {"indicador": "diferencia_auc", "valor": delong.get("effect_auc_difference")},
                    {"indicador": "ic95", "valor": str(delong.get("ci_95"))},
                    {"indicador": "p_value", "valor": delong.get("p_value")},
                    {"indicador": "significativo", "valor": delong.get("significant")},
                ],
                use_container_width=True,
                hide_index=True,
            )
            interpretation_box("DeLong compara la capacidad de separacion medida por AUC para la clase seleccionada. Si el p-value es bajo, la diferencia de AUC entre modelos no parece explicarse solo por ruido muestral." if spanish else "DeLong compares discrimination capacity measured by AUC for the selected class. A low p-value suggests the AUC difference is unlikely to be only sampling noise.")
            st.markdown("##### Bootstrap pareado")
            show_image_if_exists(STAT_BOOTSTRAP_CHART, "Bootstrap pareado - IC95" if spanish else "Paired bootstrap - 95% CI")
            boot_rows = []
            for metric_name, metric_payload in bootstrap_metrics.items():
                boot_rows.append({"metrica": metric_name, "modelo_1": metric_payload.get("first"), "modelo_2": metric_payload.get("second"), "diferencia": metric_payload.get("effect_difference"), "ic95": str(metric_payload.get("ci_95"))})
            if boot_rows:
                st.dataframe(boot_rows, use_container_width=True, hide_index=True)
            interpretation_box("Bootstrap pareado re-muestrea las mismas imagenes para estimar intervalos de confianza de las diferencias. Si el IC95 no cruza cero, la diferencia es consistente en esta muestra." if spanish else "Paired bootstrap resamples the same images to estimate confidence intervals for metric differences. If the 95% CI does not cross zero, the difference is consistent in this sample.")
            st.markdown("##### Holm-Bonferroni")
            show_image_if_exists(STAT_PVALUES_CHART, "P-values ajustados" if spanish else "Adjusted p-values")
            correction = result.get("multiple_comparison_correction", {}) if isinstance(result, dict) else {}
            st.dataframe(
                [{"campo": key, "valor": str(value)} for key, value in correction.items()],
                use_container_width=True,
                hide_index=True,
            )
            interpretation_box("Holm-Bonferroni ajusta los p-values cuando se hacen varias pruebas a la vez. Esto reduce el riesgo de declarar una diferencia significativa por casualidad." if spanish else "Holm-Bonferroni adjusts p-values when multiple tests are run. This reduces the risk of declaring significance by chance.")


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

def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def download_file_button(path: Path, label: str, mime: str) -> None:
    if path.is_file():
        st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime, use_container_width=True)
    else:
        st.warning(f"No se encontro: {path}")


def show_image_if_exists(path: Path, caption: str) -> None:
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"No se encontro la grafica: {path}")


def presentation(_: str) -> None:
    spanish = st.session_state.get("lang", "es") == "es"
    title = "Presentacion y evidencias" if spanish else "Presentation and evidence"
    body = "Resumen listo para exposicion: tablas, graficas, interpretaciones y descargas generadas desde los artefactos reales." if spanish else "Presentation-ready summary: tables, charts, interpretations, and downloads generated from real artifacts."
    page_header(title, body, "Demo reproducible" if spanish else "Reproducible demo")

    pdf_path = PRESENTATION_DIR / "reporte_tecnico_fase8_colorectal_ai.pdf"
    md_path = PRESENTATION_DIR / "resumen_presentacion_fase8.md"
    html_path = PRESENTATION_DIR / "resumen_presentacion_fase8.html"
    c1, c2, c3 = st.columns(3)
    with c1:
        download_file_button(pdf_path, "Descargar reporte PDF" if spanish else "Download PDF report", "application/pdf")
    with c2:
        download_file_button(md_path, "Descargar resumen Markdown" if spanish else "Download Markdown summary", "text/markdown")
    with c3:
        download_file_button(html_path, "Descargar resumen HTML" if spanish else "Download HTML summary", "text/html")

    tab_summary, tab_dataset, tab_eval, tab_stats = st.tabs([
        "Resumen" if spanish else "Summary",
        "Dataset y EDA" if spanish else "Dataset and EDA",
        "Evaluacion" if spanish else "Evaluation",
        "Estadistica y CV" if spanish else "Statistics and CV",
    ])

    comparison = read_csv_dicts(PHASE4_DIR / "model_comparison.csv")
    stats_rows = read_csv_dicts(PHASE5_DIR / "statistical_comparison.csv")
    cv_rows = read_csv_dicts(PHASE6_DIR / "cross_validation_summary.csv")

    with tab_summary:
        st.subheader("Estado del flujo" if spanish else "Workflow status")
        st.dataframe(
            [
                {"fase": "Dataset", "estado": "Completado", "evidencia": "100,000 imagenes, split train/validation/test"},
                {"fase": "EDA", "estado": "Completado", "evidencia": "Graficas, montaje, corruptas=0, duplicados=0"},
                {"fase": "Evaluacion", "estado": "Demo completado", "evidencia": "6 modelos sobre muestra test estratificada"},
                {"fase": "Estadistica", "estado": "Completado", "evidencia": "McNemar, DeLong, bootstrap, Holm-Bonferroni"},
                {"fase": "Validacion cruzada", "estado": "Demo completado", "evidencia": "folds.csv K=5 y metricas agregadas"},
            ],
            use_container_width=True,
            hide_index=True,
        )
        interpretation_box("El flujo ya esta listo para mostrarse de extremo a extremo: dataset externo, particiones, EDA, evaluacion, estadistica, validacion cruzada demo y reportes. Las corridas demo son reproducibles y rapidas para exposicion; para una conclusion cientifica final se deben ampliar muestras y reentrenar por fold." if spanish else "The workflow is ready to present end to end: external dataset, partitions, EDA, evaluation, statistics, cross-validation demo, and reports. Demo runs are reproducible and fast for presentation; final scientific conclusions require larger samples and fold-level retraining.")
        show_image_if_exists(PHASE4_DIR / "model_comparison.png", "Comparacion general de modelos" if spanish else "Overall model comparison")

    with tab_dataset:
        st.subheader("Dataset particionado" if spanish else "Partitioned dataset")
        dataset_summary = EDA_DIR / "eda_summary.json"
        if dataset_summary.is_file():
            summary_payload = json.loads(dataset_summary.read_text(encoding="utf-8"))
            st.dataframe(
                [
                    {"indicador": "Imagenes", "valor": summary_payload.get("total_images")},
                    {"indicador": "Split images", "valor": summary_payload.get("split_images")},
                    {"indicador": "Resolucion", "valor": ", ".join(summary_payload.get("dimensions", {}).keys())},
                    {"indicador": "Formato", "valor": ", ".join(summary_payload.get("formats", {}).keys())},
                    {"indicador": "Corruptas", "valor": summary_payload.get("corrupt_total")},
                    {"indicador": "Duplicadas", "valor": summary_payload.get("duplicate_image_count")},
                ],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Ver detalle tecnico" if spanish else "View technical details"):
                st.markdown("##### Trazabilidad" if spanish else "##### Traceability")
                st.dataframe(
                    [
                        {"campo": "created_at_utc", "valor": summary_payload.get("created_at_utc")},
                        {"campo": "dataset_root", "valor": summary_payload.get("dataset_root")},
                        {"campo": "source_summary", "valor": summary_payload.get("source_summary")},
                        {"campo": "manifest_content_fingerprint_sha256", "valor": summary_payload.get("manifest_content_fingerprint_sha256")},
                        {"campo": "backend_dataset_fingerprint_sha256", "valor": summary_payload.get("backend_dataset_fingerprint_sha256")},
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                left_detail, right_detail = st.columns(2)
                with left_detail:
                    st.markdown("##### Conteo por clase" if spanish else "##### Class counts")
                    st.dataframe(
                        [{"clase": key, "imagenes": value} for key, value in summary_payload.get("class_counts", {}).items()],
                        use_container_width=True,
                        hide_index=True,
                    )
                with right_detail:
                    st.markdown("##### Conteo por particion" if spanish else "##### Split counts")
                    split_rows = []
                    for split_name, counts in summary_payload.get("split_counts", {}).items():
                        split_rows.append({"particion": split_name, "imagenes": sum(counts.values()), "clases": len(counts)})
                    st.dataframe(split_rows, use_container_width=True, hide_index=True)
                st.markdown("##### Archivos generados" if spanish else "##### Generated artifacts")
                st.dataframe(
                    [{"artefacto": key, "ruta": value} for key, value in summary_payload.get("artifacts", {}).items()],
                    use_container_width=True,
                    hide_index=True,
                )
        show_image_if_exists(EDA_DIR / "class_distribution.png", "Distribucion por clase" if spanish else "Class distribution")
        interpretation_box("La grafica muestra que algunas clases tienen mas muestras que otras. Por esa razon no conviene depender solo del accuracy: se usan splits estratificados para conservar proporciones y metricas macro para que las clases minoritarias tambien pesen en la evaluacion." if spanish else "The chart shows that some classes have more samples than others. For that reason, accuracy alone is not enough: stratified splits preserve proportions and macro metrics ensure minority classes also affect evaluation.")
        show_image_if_exists(EDA_DIR / "split_distribution.png", "Distribucion por particion" if spanish else "Split distribution")
        interpretation_box("La particion mantiene una distribucion similar de clases en train, validation y test. Esto evita que un conjunto quede artificialmente mas facil o mas dificil, y permite defender que la evaluacion final se hizo sobre una muestra comparable al entrenamiento." if spanish else "The split keeps a similar class distribution across train, validation, and test. This avoids making one subset artificially easier or harder, and supports a defensible final evaluation.")
        show_image_if_exists(EDA_DIR / "representative_montage.png", "Muestras representativas" if spanish else "Representative samples")
        interpretation_box("El montaje sirve como validacion visual rapida: permite confirmar que el sistema esta leyendo imagenes histopatologicas reales desde disco y que hay ejemplos visibles de cada clase configurada. No reemplaza la validacion cuantitativa, pero ayuda mucho en una exposicion." if spanish else "The montage is a quick visual validation: it confirms the system is reading real histopathology images from disk and that visible examples exist for every configured class. It does not replace quantitative validation, but it helps in a presentation.")

    with tab_eval:
        st.subheader("Evaluacion reproducible demo" if spanish else "Reproducible demo evaluation")
        if comparison:
            st.dataframe(comparison, use_container_width=True, hide_index=True)
        interpretation_box("La tabla resume predicciones reales sobre una muestra estratificada del conjunto test. En esta demo, CNN Simple obtiene los mejores valores de accuracy, Macro F1, AUC y MCC. El punto importante para presentar es que el pipeline ya produce artefactos trazables; los numeros deben escalarse antes de afirmar superioridad clinica final." if spanish else "The table summarizes real predictions over a stratified test sample. In this demo, CNN Simple has the best accuracy, Macro F1, AUC, and MCC. The key presentation point is that the pipeline now produces traceable artifacts; numbers should be scaled before claiming final clinical superiority.")
        show_image_if_exists(PHASE4_DIR / "model_comparison.png", "Metricas comparativas" if spanish else "Comparative metrics")
        interpretation_box("Esta grafica compara metricas globales entre modelos. Accuracy mide aciertos totales, Macro F1 balancea rendimiento por clase, AUC mide separacion y MCC resume calidad de clasificacion incluso con clases desbalanceadas." if spanish else "This chart compares global metrics across models. Accuracy measures total correct predictions, Macro F1 balances per-class performance, AUC measures separation, and MCC summarizes classification quality even with imbalance.")
        show_image_if_exists(ROC_AUC_CHART, "Comparacion ROC/AUC - clase TUM" if spanish else "ROC/AUC comparison - TUM class")
        interpretation_box("La curva ROC muestra sensibilidad frente a falsos positivos para la clase TUM en modo one-vs-rest. Una curva mas cercana a la esquina superior izquierda indica mejor separacion; AUC cercano a 0.5 equivale aproximadamente a azar." if spanish else "The ROC curve shows sensitivity versus false positives for the TUM class in one-vs-rest mode. A curve closer to the upper-left corner indicates better separation; AUC near 0.5 is approximately random.")
        show_image_if_exists(TRAINING_CHART, "Graficos de entrenamiento heredados" if spanish else "Legacy training charts")
        interpretation_box("Los graficos de entrenamiento muestran la evolucion historica de accuracy y loss durante las epocas. Sirven para explicar convergencia y posible sobreajuste, aunque estos datos son heredados y no provienen de un reentrenamiento ejecutado en esta demo." if spanish else "Training charts show historical accuracy and loss across epochs. They help explain convergence and possible overfitting, although these are legacy data and not from a retraining run in this demo.")
        col_a, col_b = st.columns(2)
        with col_a:
            show_image_if_exists(PHASE4_DIR / "cnn_simple" / "confusion_matrix.png", "Matriz CNN Simple" if spanish else "CNN Simple matrix")
        with col_b:
            show_image_if_exists(PHASE4_DIR / "best_model" / "confusion_matrix.png", "Matriz best_model" if spanish else "best_model matrix")
        interpretation_box("Las matrices de confusion muestran el patron de errores por clase. Son utiles para detectar si un modelo concentra aciertos en pocas clases, si confunde tejidos similares o si predice de forma sesgada. Por eso complementan las metricas agregadas como accuracy o AUC." if spanish else "Confusion matrices show the error pattern by class. They help detect whether a model concentrates correct predictions in a few classes, confuses similar tissues, or predicts with bias. That is why they complement aggregate metrics like accuracy or AUC.")

    with tab_stats:
        st.subheader("Pruebas estadisticas" if spanish else "Statistical tests")
        if stats_rows:
            st.dataframe(stats_rows, use_container_width=True, hide_index=True)
        show_image_if_exists(STAT_MCNEMAR_CHART, "McNemar - contingencia visual" if spanish else "McNemar - visual contingency")
        interpretation_box("McNemar se centra en los casos discordantes: cuando un modelo acierta y el otro falla. Si hay muchos mas casos a favor de un modelo, la prueba evidencia diferencia en errores/aciertos pareados." if spanish else "McNemar focuses on discordant cases: when one model is correct and the other is wrong. If many more cases favor one model, the test shows a paired correct/error difference.")
        show_image_if_exists(STAT_DELONG_CHART, "DeLong AUC - clase TUM" if spanish else "DeLong AUC - TUM class")
        interpretation_box("DeLong compara estadisticamente el AUC de dos modelos sobre la misma clase y las mismas muestras. En la grafica, mayor AUC significa mejor capacidad de separacion; el p-value indica si esa diferencia es estadisticamente defendible." if spanish else "DeLong statistically compares the AUC of two models on the same class and samples. In the chart, higher AUC means better discrimination; the p-value indicates whether the difference is statistically defensible.")
        show_image_if_exists(STAT_BOOTSTRAP_CHART, "Bootstrap pareado - diferencias con IC95" if spanish else "Paired bootstrap - differences with 95% CI")
        interpretation_box("Bootstrap muestra la diferencia de metricas con intervalo de confianza. Cuando el intervalo queda completamente a un lado de cero, la diferencia es consistente en la muestra pareada." if spanish else "Bootstrap shows metric differences with confidence intervals. When the interval stays entirely on one side of zero, the difference is consistent in the paired sample.")
        show_image_if_exists(STAT_PVALUES_CHART, "P-values y Holm-Bonferroni" if spanish else "P-values and Holm-Bonferroni")
        interpretation_box("La grafica compara p-values originales y ajustados. Holm-Bonferroni controla el riesgo de falsos positivos cuando se reportan varias pruebas estadisticas a la vez." if spanish else "The chart compares original and adjusted p-values. Holm-Bonferroni controls false-positive risk when several statistical tests are reported together.")
        st.subheader("Validacion cruzada demo" if spanish else "Cross-validation demo")
        if cv_rows:
            st.dataframe(cv_rows, use_container_width=True, hide_index=True)
        show_image_if_exists(PHASE6_DIR / "cross_validation_summary.png", "Resumen CV K=5" if spanish else "K=5 CV summary")
        interpretation_box("La validacion cruzada demo confirma que existen folds K=5 estratificados y balanceados. Aqui se evaluan modelos ya entrenados para demostrar el flujo, calcular medias y desviaciones, y generar graficas rapidamente. La validacion cruzada cientifica final debe entrenar de nuevo el modelo dentro de cada fold." if spanish else "The cross-validation demo confirms that stratified and balanced K=5 folds exist. Here, existing trained models are evaluated to demonstrate the workflow, compute means and standard deviations, and generate charts quickly. Final scientific cross-validation must retrain the model inside each fold.")

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
        view = st.radio(tr("nav"), [tr("overview"), tr("diagnosis"), tr("dataset"), tr("training"), tr("evaluation"), tr("statistics"), tr("explainability"), tr("presentation"), tr("reports")])
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
    elif view == tr("presentation"):
        presentation(base_url)
    else:
        reports(base_url)

if __name__ == "__main__":
    main()

















