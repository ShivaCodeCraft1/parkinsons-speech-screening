#!/usr/bin/env python3
"""Extract acoustic features for every usable recording in the manifest -> data/processed/features.csv

Uses src.models.inference.analyze_audio_bytes, i.e. the SAME code path as the API.
  python scripts/extract_features.py --n-jobs 4
"""
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from src.audio.loading import AudioError  # noqa: E402
from src.config import FEATURES_CONFIG, get_data_root, load_yaml  # noqa: E402
from src.data.dataset import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_MANIFEST,
    DEFAULT_PRIVATE_DIR,
    load_manifest,
    resolve_paths,
    usable_recordings,
)
from src.features.extract import feature_schema  # noqa: E402
from src.models.inference import analyze_audio_bytes  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402


def _one(recording_id: str, path: str, cfg: dict) -> dict:
    try:
        a = analyze_audio_bytes(Path(path).read_bytes(), cfg, with_visuals=False)
        return {"recording_id": recording_id, "status": "ok",
                "warnings": ";".join(w["code"] for w in a["validation"]["warnings"]), **a["features"]}
    except AudioError as e:
        return {"recording_id": recording_id, "status": f"rejected:{e.code}", "warnings": ""}
    except Exception as e:  # keep going; report at the end
        return {"recording_id": recording_id, "status": f"error:{type(e).__name__}", "warnings": ""}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--private-dir", default=str(DEFAULT_PRIVATE_DIR))
    ap.add_argument("--out", default=str(DEFAULT_FEATURES))
    ap.add_argument("--task-families", nargs="*", default=None, help="restrict to these task families (default: all usable)")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N recordings")
    args = ap.parse_args()
    setup_logging()
    log = logging.getLogger("extract")
    cfg = load_yaml(FEATURES_CONFIG)
    root = get_data_root(args.data_root)
    m = usable_recordings(load_manifest(Path(args.manifest)), args.task_families)
    m = resolve_paths(m, root, Path(args.private_dir))
    if args.limit:
        m = m.head(args.limit)
    missing = m["abs_path"].isna() | ~m["abs_path"].map(lambda p: isinstance(p, str) and Path(p).is_file())
    if missing.any():
        log.warning("%d recordings from the manifest were not found under %s (wrong PD_DATA_ROOT?)", int(missing.sum()), root)
    m = m[~missing]
    log.info("Extracting features for %d recordings", len(m))
    rows = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(_one)(r.recording_id, r.abs_path, cfg) for r in m.itertuples())
    feats = pd.DataFrame(rows)
    meta_cols = ["recording_id", "subject_id", "label", "is_pd", "task_code", "task_family", "age_if_available", "sex_if_available"]
    out = m[meta_cols].merge(feats, on="recording_id", how="left")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    schema = feature_schema(cfg)
    (Path(args.out).parent / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    names = schema["feature_names"]
    ok = out[out["status"] == "ok"]
    report = {"n_requested": int(len(out)), "n_ok": int(len(ok)), "status_counts": out["status"].value_counts().to_dict(),
              "undefined_rate_per_feature": {n: round(float(ok[n].isna().mean()), 4) for n in names if ok[n].isna().any()},
              "feature_config_sha256": schema["feature_config_sha256"]}
    (Path(args.out).parent / "feature_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote %s (%d ok / %d). Status counts: %s", args.out, len(ok), len(out), report["status_counts"])
    if report["undefined_rate_per_feature"]:
        log.warning("Features with undefined values (NaN, imputed later inside the training pipeline): %s", report["undefined_rate_per_feature"])
    return 0 if len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
