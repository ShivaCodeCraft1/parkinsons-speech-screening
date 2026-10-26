"""Save / load model artifacts. The API loads these; it never retrains."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

MODEL_FILE, SCHEMA_FILE, META_FILE = "model.joblib", "feature_schema.json", "metadata.json"


class ModelNotAvailable(Exception):
    pass


def save_artifacts(model_dir: Path, pipeline, schema: dict, meta: dict) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / MODEL_FILE)
    (model_dir / SCHEMA_FILE).write_text(json.dumps(schema, indent=2), encoding="utf-8")
    meta = {**meta, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__, "numpy": np.__version__,
                            "pandas": pd.__version__, "platform": platform.system()}}
    (model_dir / META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def load_artifacts(model_dir: Path) -> dict:
    model_dir = Path(model_dir)
    missing = [f for f in (MODEL_FILE, SCHEMA_FILE, META_FILE) if not (model_dir / f).is_file()]
    if missing:
        raise ModelNotAvailable(
            f"Model artifacts missing in {model_dir} ({', '.join(missing)}). Train first: "
            "`python scripts/extract_features.py` then `python scripts/train.py` (see README).")
    return {"pipeline": joblib.load(model_dir / MODEL_FILE),
            "schema": json.loads((model_dir / SCHEMA_FILE).read_text(encoding="utf-8")),
            "meta": json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))}
