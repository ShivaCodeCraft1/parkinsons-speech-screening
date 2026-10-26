#!/usr/bin/env python3
"""Evaluate the saved model on the HELD-OUT test subjects chosen by scripts/train.py.

Writes reports/evaluation/*: metrics.json, test_scores.csv, ROC/PR/confusion/score plots, subgroup + age-confounding
analysis, feature importance, and a Markdown summary. Every number comes from this run.

Use the test set for reporting only. If you change the model after looking at these numbers, they are no longer an
honest held-out estimate.
  python scripts/evaluate.py
  python scripts/evaluate.py --skip-confounding      # faster
  python scripts/evaluate.py --nested-cv-summary     # summarise reports/training/nested_cv_oof_scores.csv
"""
import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.config import FEATURES_CONFIG, TRAINING_CONFIG, get_model_dir, load_yaml  # noqa: E402
from src.data.dataset import DEFAULT_FEATURES  # noqa: E402
from src.data.splits import assert_no_subject_overlap  # noqa: E402
from src.evaluation import plots  # noqa: E402
from src.evaluation.confounding import age_confounding, score_age_correlation, subgroup_report  # noqa: E402
from src.evaluation.metrics import aggregate_by_subject, cluster_bootstrap, full_metrics  # noqa: E402
from src.explainability.explain import global_importance  # noqa: E402
from src.features.extract import feature_schema  # noqa: E402
from src.models.registry import META_FILE, ModelNotAvailable, load_artifacts  # noqa: E402
from src.models.training import score_with  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402

OUT = REPO / "reports" / "evaluation"
TRAIN_DIR = REPO / "reports" / "training"


def fmt_ci(ci: dict) -> str:
    return "n/a" if ci["low"] is None else f"[{ci['low']:.3f}, {ci['high']:.3f}]"


def nested_summary(cfg: dict) -> None:
    f = TRAIN_DIR / "nested_cv_oof_scores.csv"
    if not f.is_file():
        raise SystemExit("Run `python scripts/train.py --nested-cv` first.")
    d = pd.read_csv(f).rename(columns={})
    pooled = {"protocol": "nested subject-grouped CROSS-VALIDATION (not a held-out test)", "n_subjects": int(d.subject_id.nunique()),
              "n_recordings": int(len(d)), "per_fold": []}
    d["pred"] = (d["score"] >= d["fold_threshold"]).astype(int)
    for k, g in d.groupby("fold"):
        m = full_metrics(g["is_pd"], g["score"], float(g["fold_threshold"].iloc[0]))
        pooled["per_fold"].append({"fold": int(k), "n_subjects": int(g.subject_id.nunique()), "model": g["fold_model"].iloc[0], **{x: m[x] for x in ("auroc", "sensitivity", "specificity")}})
    pooled["pooled_auroc"] = float(full_metrics(d["is_pd"], d["score"], 0.0)["auroc"]) if d.is_pd.nunique() == 2 else None
    tp = ((d.pred == 1) & (d.is_pd == 1)).sum(); fn = ((d.pred == 0) & (d.is_pd == 1)).sum()
    tn = ((d.pred == 0) & (d.is_pd == 0)).sum(); fp = ((d.pred == 1) & (d.is_pd == 0)).sum()
    pooled.update(pooled_sensitivity=float(tp / (tp + fn)), pooled_specificity=float(tn / (tn + fp)))
    pooled["note"] = "Pooled AUROC mixes scores from different fold models; use per-fold values for spread."
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "nested_cv_summary.json").write_text(json.dumps(pooled, indent=2), encoding="utf-8")
    print(json.dumps(pooled, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(DEFAULT_FEATURES))
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--skip-confounding", action="store_true")
    ap.add_argument("--nested-cv-summary", action="store_true")
    args = ap.parse_args()
    setup_logging()
    log = logging.getLogger("evaluate")
    cfg, fcfg = load_yaml(TRAINING_CONFIG), load_yaml(FEATURES_CONFIG)
    if args.nested_cv_summary:
        nested_summary(cfg)
        return 0
    model_dir = get_model_dir(args.model_dir)
    try:
        bundle = load_artifacts(model_dir)
    except ModelNotAvailable as e:
        raise SystemExit(str(e)) from None
    names = bundle["schema"]["feature_names"]
    meta = bundle["meta"]
    if feature_schema(fcfg)["feature_config_sha256"] != bundle["schema"]["feature_config_sha256"]:
        raise SystemExit("configs/features.yaml changed since training. Re-extract features and retrain.")
    split_file = TRAIN_DIR / "split_assignment.csv"
    if not split_file.is_file():
        raise SystemExit("reports/training/split_assignment.csv missing; run scripts/train.py first.")
    split = pd.read_csv(split_file)
    feats = pd.read_csv(args.features)
    feats = feats[feats["status"] == "ok"]
    # the split file holds the (possibly merged) split-group id in `subject_id`; use it so groups match training exactly
    df = feats.drop(columns=["subject_id"]).merge(split[["recording_id", "subject_id", "partition"]], on="recording_id", how="inner")
    train_df, test_df = df[df.partition == "train"].copy(), df[df.partition == "test"].copy()
    assert_no_subject_overlap(train_df, test_df)
    test_df["is_pd"] = test_df["is_pd"].astype(int)
    pipe, thr = bundle["pipeline"], float(meta["threshold"])
    test_df["score"] = score_with(pipe, test_df[names].to_numpy(float))
    OUT.mkdir(parents=True, exist_ok=True)
    test_df[["subject_id", "recording_id", "label", "is_pd", "task_code", "score"]].to_csv(OUT / "test_scores.csv", index=False)

    boot_cfg = cfg["bootstrap"]
    rec = full_metrics(test_df["is_pd"], test_df["score"], thr)
    rec_ci = cluster_bootstrap(test_df, thr, boot_cfg["n_resamples"], boot_cfg["confidence"], cfg["seed"])
    subj = aggregate_by_subject(test_df)
    sub = full_metrics(subj["is_pd"], subj["score"], thr)
    sub_ci = cluster_bootstrap(subj, thr, boot_cfg["n_resamples"], boot_cfg["confidence"], cfg["seed"])
    n_pd = int(subj.is_pd.sum())
    result = {
        "protocol": "HELD-OUT test subjects (subject-level split; threshold fixed from out-of-fold training scores)",
        "positive_class": meta["positive_class"], "metric_type": "binary (PD vs non-PD; non-PD = young + elderly controls)",
        "model_version": meta["model_version"], "model_name": meta["model_name"], "threshold": thr,
        "threshold_method": meta["threshold_method"], "score_type": meta["score_type"],
        "partitions": {"train": {"subjects": int(train_df.subject_id.nunique()), "recordings": int(len(train_df))},
                       "test": {"subjects": int(subj.shape[0]), "recordings": int(len(test_df)),
                                "pd_subjects": n_pd, "non_pd_subjects": int(len(subj) - n_pd),
                                "subjects_per_group": subj["label"].value_counts().to_dict()}},
        "recording_level": {**rec, "ci": rec_ci}, "subject_level_mean_score": {**sub, "ci": sub_ci},
        "ci_method": f"subject-cluster bootstrap, {boot_cfg['n_resamples']} resamples, {boot_cfg['confidence']:.0%}",
        "limitations": ["Small number of test subjects: confidence intervals are wide and metrics are unstable.",
                        "Single dataset/site; no external validation.", "Scores are not calibrated probabilities.",
                        "Not a clinical validation."]}
    if len(subj) < 20:
        result["limitations"].insert(0, f"Only {len(subj)} test subjects ({n_pd} PD).")

    for tag, d, m in (("recording", test_df, rec), ("subject", subj, sub)):
        if d["is_pd"].nunique() == 2:
            plots.roc_plot(d["is_pd"], d["score"], OUT / f"roc_{tag}_level.png", f"ROC ({tag}-level, held-out)", m["auroc"])
            plots.pr_plot(d["is_pd"], d["score"], OUT / f"pr_{tag}_level.png", f"Precision-recall ({tag}-level)")
        plots.confusion_plot(m, OUT / f"confusion_{tag}_level.png", f"{tag}-level @ threshold")
    plots.score_hist(test_df["is_pd"], test_df["score"], thr, OUT / "score_distribution.png", "Held-out scores")

    msr = cfg["confounding"]["min_subjects_for_subgroup_report"]
    result["subgroups_held_out"] = {"by_group": subgroup_report(test_df, thr, msr, "label"),
                                    "by_task": subgroup_report(test_df, thr, msr, "task_code"),
                                    "by_sex": subgroup_report(test_df.assign(sex=test_df["sex_if_available"].fillna("unknown")), thr, msr, "sex")}
    if not args.skip_confounding:
        allx = df.copy(); allx["is_pd"] = allx["is_pd"].astype(int)
        allx["score"] = score_with(pipe, allx[names].to_numpy(float))
        result["age_confounding"] = age_confounding(allx, names, pipe, cfg)
        # note: allx scores for TRAIN rows are in-sample and are NOT used for any reported metric; only test rows enter score_age_correlation
        result["score_vs_age_in_test_controls"] = score_age_correlation(test_df, msr)
    log.info("Computing permutation importance on held-out test rows (description only; not used for selection)")
    imp = global_importance(pipe, test_df, test_df["is_pd"].to_numpy(), names, cfg["seed"], cfg["explainability"]["shap_max_rows"])
    for k, t in imp.items():
        t.to_csv(OUT / f"feature_importance_{k}.csv", index=False)
    key = next((k for k in ("shap", "permutation", "coefficients") if k in imp), None)
    if key:
        import matplotlib.pyplot as plt
        t = imp[key].head(15).iloc[::-1]
        col = [c for c in t.columns if c != "feature"][0]
        fig, ax = plt.subplots(figsize=(6, 5)); ax.barh(t["feature"], t[col]); ax.set_xlabel(col)
        ax.set_title(f"Top features ({key}); describes the model, not causation"); fig.tight_layout()
        fig.savefig(OUT / "feature_importance.png", dpi=150); plt.close(fig)

    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    p = result["partitions"]["test"]
    md = ["# Held-out evaluation (generated by scripts/evaluate.py)", "",
          f"Model `{meta['model_version']}`; positive class = {meta['positive_class']}; threshold {thr:.4f} ({meta['threshold_method']}).", "",
          f"Test partition: {p['subjects']} subjects ({p['pd_subjects']} PD) / {p['recordings']} recordings; train: "
          f"{result['partitions']['train']['subjects']} subjects.", "",
          "| level | AUROC (95% CI) | sensitivity (95% CI) | specificity (95% CI) |", "|---|---|---|---|"]
    for name, m in (("recording", result["recording_level"]), ("subject (mean score)", result["subject_level_mean_score"])):
        md.append(f"| {name} | {m['auroc']:.3f} {fmt_ci(m['ci']['auroc'])} | {m['sensitivity']:.3f} {fmt_ci(m['ci']['sensitivity'])} | "
                  f"{m['specificity']:.3f} {fmt_ci(m['ci']['specificity'])} |")
    md += ["", "## Limitations", *[f"- {x}" for x in result["limitations"]], "",
           "Research prototype; not clinically validated. Compare with any resume/README claims only using these numbers."]
    (OUT / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")

    # record the real held-out result in the model metadata so /model-info can show it (and only then)
    meta["evaluation_status"] = "HELD-OUT EVALUATION RUN"
    meta["evaluation"] = {"auroc_recording": rec["auroc"], "auroc_subject": sub["auroc"], "sensitivity_recording": rec["sensitivity"],
                          "specificity_recording": rec["specificity"], "test_subjects": p["subjects"], "test_pd_subjects": p["pd_subjects"],
                          "ci_method": result["ci_method"], "report": "reports/evaluation/metrics.json"}
    (model_dir / META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
