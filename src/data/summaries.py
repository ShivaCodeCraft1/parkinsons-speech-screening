"""Participant-level, class-distribution and metadata-validation summaries built from the manifest.
All outputs use pseudonymous IDs only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def participant_summary(m: pd.DataFrame) -> pd.DataFrame:
    g = m[m["subject_id"].notna()].groupby("subject_id")
    out = g.agg(label=("label", "first"), is_pd=("is_pd", "first"), n_recordings=("recording_id", "count"),
                n_tasks=("task_code", "nunique"), age=("age_if_available", "first"), sex=("sex_if_available", "first"),
                n_label_values=("label", "nunique")).reset_index()
    return out


def class_distribution(p: pd.DataFrame, m: pd.DataFrame) -> dict:
    return {
        "subjects_per_label": p["label"].value_counts().to_dict(),
        "recordings_per_label": m[m["label"].notna()]["label"].value_counts().to_dict(),
        "subjects_per_binary_class": {"pd": int((p["is_pd"] == 1).sum()), "non_pd": int((p["is_pd"] == 0).sum())},
        "recordings_per_task_family": m["task_family"].value_counts(dropna=False).to_dict(),
        "subjects_with_conflicting_labels": int((p["n_label_values"] > 1).sum()),
    }


def metadata_validation(p: pd.DataFrame, m: pd.DataFrame) -> dict:
    age = pd.to_numeric(p["age"], errors="coerce")
    return {
        "n_subjects": int(len(p)), "n_recordings": int(len(m)),
        "subjects_with_age": int(age.notna().sum()), "subjects_without_age": int(age.isna().sum()),
        "age_summary_by_label": {k: {"n": int(v.notna().sum()), "mean": None if v.notna().sum() == 0 else round(float(v.mean()), 1),
                                     "min": None if v.notna().sum() == 0 else float(v.min()),
                                     "max": None if v.notna().sum() == 0 else float(v.max())}
                                 for k, v in age.groupby(p["label"])},
        "sex_counts_by_label": p.groupby("label")["sex"].value_counts(dropna=False).rename("n").reset_index().to_dict("records"),
        "unreadable_recordings": int((~m["readable"].astype(bool)).sum()),
        "sample_rates": {str(k): int(v) for k, v in m["sample_rate"].value_counts().items()},
        "channels": {str(k): int(v) for k, v in m["channels"].value_counts().items()},
        "duration_sec": {"min": float(m["duration_sec"].min()), "median": float(m["duration_sec"].median()),
                         "max": float(m["duration_sec"].max())},
        "qc_flag_counts": m["qc_flags"].fillna("").str.split(";").explode().replace("", pd.NA).dropna().value_counts().to_dict(),
        "metadata_link_status": m["metadata_link_status"].value_counts(dropna=False).to_dict(),
    }


def write_summaries(manifest: pd.DataFrame, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = participant_summary(manifest)
    p.to_csv(out_dir / "participant_summary.csv", index=False)
    summary = {"class_distribution": class_distribution(p, manifest), "metadata_validation": metadata_validation(p, manifest)}
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
