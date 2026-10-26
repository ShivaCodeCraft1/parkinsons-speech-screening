#!/usr/bin/env python3
"""Train: subject-level holdout split -> grouped-CV model comparison -> threshold from OOF scores -> save artifacts.

The held-out test subjects are NOT used here. Evaluate them once with scripts/evaluate.py.
  python scripts/train.py
  python scripts/train.py --nested-cv          # additionally write a cross-validated (not held-out) estimate
"""
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.config import FEATURES_CONFIG, TRAINING_CONFIG, file_sha256, get_model_dir, load_yaml  # noqa: E402
from src.data.dataset import DEFAULT_FEATURES, DEFAULT_MANIFEST  # noqa: E402
from src.data.leakage import apply_split_groups  # noqa: E402
from src.data.splits import subject_holdout_split, subject_table  # noqa: E402
from src.features.extract import feature_schema  # noqa: E402
from src.models.registry import save_artifacts  # noqa: E402
from src.models.training import nested_grouped_cv, select_and_fit  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402

REPORT_DIR = REPO / "reports" / "training"


def load_training_frame(features_csv: Path, cfg: dict, names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    df = df[df["status"] == "ok"]
    if cfg.get("task_families"):
        df = df[df["task_family"].isin(cfg["task_families"])]
    df = df.dropna(subset=["subject_id", "is_pd"]).copy()
    df["is_pd"] = df["is_pd"].astype(int)
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise SystemExit(f"features.csv lacks schema columns {missing[:5]}...; re-run scripts/extract_features.py")
    return df.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(DEFAULT_FEATURES))
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--nested-cv", action="store_true")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="used to merge subjects that share identical audio")
    ap.add_argument("--skip-duplicate-check", action="store_true", help="NOT recommended: train without the identical-audio leakage guard")
    ap.add_argument("--models", nargs="*", default=None, help="subset of candidate models (default: all in configs/training.yaml)")
    args = ap.parse_args()
    setup_logging()
    log = logging.getLogger("train")
    cfg, fcfg = load_yaml(TRAINING_CONFIG), load_yaml(FEATURES_CONFIG)
    schema = feature_schema(fcfg)
    names = schema["feature_names"]
    df = load_training_frame(Path(args.features), cfg, names)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.features)
    sel = raw[raw["task_family"].isin(cfg["task_families"])] if cfg.get("task_families") else raw
    inclusion = {"rows_in_features_csv_for_selected_tasks": int(len(sel)), "status_counts": sel["status"].value_counts().to_dict(),
                 "rows_used_before_duplicate_guard": int(len(df))}
    manifest = None
    if not args.skip_duplicate_check:
        if not Path(args.manifest).is_file():
            raise SystemExit(f"{args.manifest} not found. It is needed to detect identical audio shared between subjects "
                             "(leakage guard). Re-run scripts/inspect_dataset.py, or pass --skip-duplicate-check to accept the risk.")
        manifest = pd.read_csv(args.manifest)
    df, dup_audit, excluded = apply_split_groups(df, manifest)
    inclusion["duplicate_guard"] = dup_audit
    inclusion["rows_used"] = int(len(df))
    if len(excluded):
        excluded[["recording_id", "subject_id_original", "label", "exclusion_reason"]].to_csv(REPORT_DIR / "excluded_recordings.csv", index=False)
    if dup_audit["merged_groups"]:
        df[df["subject_id"].str.startswith("GRP_")][["subject_id", "subject_id_original"]].drop_duplicates() \
            .to_csv(REPORT_DIR / "merged_subject_groups.csv", index=False)
        log.warning("Identical audio links %d subject_ids into %d merged split groups (kept together in every split). "
                    "Check reports/training/merged_subject_groups.csv and the verifier report.",
                    dup_audit["subjects_in_merged_groups"], dup_audit["merged_groups"])
    (REPORT_DIR / "training_inclusion_audit.json").write_text(json.dumps(inclusion, indent=2, default=str), encoding="utf-8")
    log.info("Inclusion: %s", inclusion)
    st = subject_table(df)
    log.info("Usable: %d recordings from %d subjects %s", len(df), len(st), st["label"].value_counts().to_dict())
    if st["is_pd"].value_counts().min() < cfg["split"]["min_subjects_per_class_for_cv"] * 2:
        raise SystemExit("Too few subjects in a class for a train/test split plus grouped CV. Cannot train responsibly.")

    train, test = subject_holdout_split(df, cfg["split"]["test_subject_fraction"], cfg["seed"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([train.assign(partition="train"), test.assign(partition="test")])[["subject_id", "recording_id", "label", "is_pd", "partition"]] \
        .to_csv(REPORT_DIR / "split_assignment.csv", index=False)
    log.info("Split: train %d subjects / %d recordings; test %d subjects / %d recordings",
             train.subject_id.nunique(), len(train), test.subject_id.nunique(), len(test))

    res = select_and_fit(train, names, cfg, args.models)
    res["comparison"].to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    res["oof_train"].to_csv(REPORT_DIR / "train_oof_scores.csv", index=False)

    version = f"{schema['schema_version']}-{res['model_name']}-{pd.Timestamp.now('UTC').strftime('%Y%m%d%H%M%S')}"
    meta = {"model_version": version, "model_name": res["model_name"], "score_type": res["score_type"],
            "score_is_calibrated_probability": False, "threshold": res["threshold"], "threshold_method": cfg["threshold"]["method"],
            "positive_class": cfg["positive_class"], "task_families": cfg.get("task_families"), "seed": cfg["seed"],
            "grouped_cv_folds": res["n_splits"], "training_config_sha256": file_sha256(TRAINING_CONFIG),
            "feature_config_sha256": schema["feature_config_sha256"],
            "training_data": {"n_subjects": int(train.subject_id.nunique()), "n_recordings": int(len(train)),
                              "subjects_per_group": subject_table(train)["label"].value_counts().to_dict()},
            "test_data_reserved": {"n_subjects": int(test.subject_id.nunique()), "n_recordings": int(len(test))},
            "selection": res["comparison"].to_dict("records"),
            "evaluation_status": "NOT EVALUATED YET: run scripts/evaluate.py (held-out subjects).",
            "intended_use": "Research prototype. Not a diagnostic device; no clinical validity."}
    model_dir = get_model_dir(args.model_dir)
    save_artifacts(model_dir, res["pipeline"], schema, meta)
    (REPORT_DIR / "training_summary.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.info("Best model: %s | threshold %.4f (%s on OOF train scores) | saved to %s", res["model_name"], res["threshold"],
             cfg["threshold"]["method"], model_dir)
    print(res["comparison"].to_string(index=False))

    if args.nested_cv:
        n = nested_grouped_cv(df, names, cfg)
        n["oof"].to_csv(REPORT_DIR / "nested_cv_oof_scores.csv", index=False)
        log.info("Wrote reports/training/nested_cv_oof_scores.csv (summarised by scripts/evaluate.py --nested-cv-summary)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
