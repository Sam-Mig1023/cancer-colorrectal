from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_backend_env() -> None:
    env_path = PROJECT_ROOT / "backend" / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # The project-local backend/.env is the explicit runtime configuration.
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_backend_env()
MODELS_DIR = PROJECT_ROOT / "models"

CLASS_NAMES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
DEFAULT_MODEL = "best_model"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
ALLOWED_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    path: Path
    input_size: tuple[int, int]
    loader: str
    is_recommended: bool = False


MODEL_SPECS = {
    "cnn_simple": ModelSpec(
        key="cnn_simple",
        display_name="CNN Simple",
        path=MODELS_DIR / "cnn_simple_model.h5",
        input_size=(224, 224),
        loader="keras_model",
    ),
    "resnet50v2": ModelSpec(
        key="resnet50v2",
        display_name="ResNet50V2",
        path=MODELS_DIR / "resnet50_model",
        input_size=(224, 224),
        loader="saved_model",
    ),
    "mobilenetv2": ModelSpec(
        key="mobilenetv2",
        display_name="MobileNetV2 Base",
        path=MODELS_DIR / "mobilenetv2_base_only.h5",
        input_size=(224, 224),
        loader="mobilenet_weights",
    ),
    "hybrid_attention": ModelSpec(
        key="hybrid_attention",
        display_name="Hybrid Attention",
        path=MODELS_DIR / "Fast_HybridAttention_final.h5",
        input_size=(96, 96),
        loader="keras_model",
    ),
    "hybrid_autoencoder": ModelSpec(
        key="hybrid_autoencoder",
        display_name="Hybrid Autoencoder",
        path=MODELS_DIR / "Fast_HybridAutoencoder_final.h5",
        input_size=(96, 96),
        loader="keras_model",
    ),
    "best_model": ModelSpec(
        key="best_model",
        display_name="Best model (MobileNetV2 Base)",
        path=MODELS_DIR / "best_model.h5",
        input_size=(224, 224),
        loader="mobilenet_weights",
        is_recommended=True,
    ),
}
