"""Central configuration: paths, environment variables, YAML loading."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CONFIG = REPO_ROOT / "configs" / "features.yaml"
TRAINING_CONFIG = REPO_ROOT / "configs" / "training.yaml"
DATASET_CONFIG = REPO_ROOT / "configs" / "dataset_italian_pvs.yaml"


def load_dotenv_if_present(path: Path | None = None) -> None:
    """Minimal .env reader (no extra dependency). Existing environment variables win."""
    path = path or REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv_if_present()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_data_root(cli_value: str | None = None) -> Path:
    """Dataset root: CLI value > PD_DATA_ROOT env var. Raises a clear error if neither is set."""
    value = cli_value or os.environ.get("PD_DATA_ROOT")
    if not value:
        raise SystemExit("Dataset path not configured. Set PD_DATA_ROOT (see .env.example) or pass --data-root.")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root is not a directory: {root}")
    return root


def get_model_dir(cli_value: str | None = None) -> Path:
    value = cli_value or os.environ.get("PD_MODEL_DIR") or "models/current"
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p
