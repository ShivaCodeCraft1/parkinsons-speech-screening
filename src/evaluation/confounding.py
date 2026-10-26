"""Age / group / task confounding checks. Everything here is descriptive and cross-validated; small subgroups are suppressed."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.splits import grouped_cv, safe_n_splits
from src.models.training import score_method


def _cv_auroc(pipe, X, y, groups, n_splits: int, seed: int) -> float | None:
    if len(np.unique(y)) < 2 or pd.Series(groups).nunique() < 4:
        return None
    cv = grouped_cv(n_splits, seed)
    method = score_method(pipe)
    s = cross_val_predict(clone(pipe), X, y, groups=groups, cv=cv, method=method)
    s = s[:, 1] if method == "predict_proba" else s
    return float(roc_auc_score(y, s))


def age_confounding(df: pd.DataFrame, feature_names: list[str], tuned_pipeline, cfg: dict, control_label: str = "elderly_healthy_control") -> dict:
    """Cross-validated (NOT held-out) comparison on all usable subjects: age-only vs acoustic vs acoustic+age,
    plus acoustic PD vs age-closer controls only. Requires age metadata; otherwise reports why it cannot run."""
    seed = int(cfg["seed"])
    out: dict = {"protocol": "subject-grouped CV on ALL usable subjects (cross-validation, not held-out test)"}
    age = pd.to_numeric(df["age_if_available"], errors="coerce")
    d = df.assign(age=age)
    n_age = d.loc[d["age"].notna(), "subject_id"].nunique()
    out["subjects_with_age"] = int(n_age)
    n_splits = safe_n_splits(d, int(cfg["split"]["inner_cv_folds"]))
    X = d[feature_names].to_numpy(float)
    y, g = d["is_pd"].to_numpy(int), d["subject_id"].to_numpy()
    out["acoustic_cv_auroc"] = _cv_auroc(tuned_pipeline, X, y, g, n_splits, seed)
    if n_age < 0.9 * d["subject_id"].nunique():
        out["age_analysis"] = "SKIPPED: age is missing for more than 10% of subjects (metadata not linked unambiguously)."
    else:
        age_pipe = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced"))])
        out["age_only_cv_auroc"] = _cv_auroc(age_pipe, d[["age"]].to_numpy(float), y, g, n_splits, seed)
        out["interpretation_note"] = ("If age-only AUROC is close to the acoustic AUROC, the acoustic model may be exploiting age. "
                                      "Adding age as a feature does not remove confounding.")
    sub = d[d["label"].isin([control_label, cfg["positive_class"]])]
    if sub["label"].nunique() == 2 and sub["subject_id"].nunique() >= 8:
        ns = safe_n_splits(sub, int(cfg["split"]["inner_cv_folds"]))
        out["acoustic_cv_auroc_pd_vs_" + control_label] = _cv_auroc(tuned_pipeline, sub[feature_names].to_numpy(float),
                                                                    sub["is_pd"].to_numpy(int), sub["subject_id"].to_numpy(), ns, seed)
        if sub["age"].notna().sum() > 0:
            ag = sub.groupby("label")["age"].agg(["count", "mean"]).round(1).to_dict("index")
            out["age_by_group_pd_vs_" + control_label] = ag
    else:
        out["pd_vs_" + control_label] = "SKIPPED: not enough subjects in both groups."
    return out


def score_age_correlation(scored: pd.DataFrame, min_subjects: int) -> dict:
    """Spearman correlation between the model score and age among NON-PD subjects (subject-mean scores)."""
    c = scored[(scored["is_pd"] == 0)].copy()
    c["age"] = pd.to_numeric(c["age_if_available"], errors="coerce")
    s = c.groupby("subject_id").agg(score=("score", "mean"), age=("age", "first")).dropna()
    if len(s) < min_subjects:
        return {"status": f"SKIPPED: only {len(s)} non-PD subjects with age (< {min_subjects})."}
    rho, p = spearmanr(s["score"], s["age"])
    return {"status": "ok", "n_subjects": int(len(s)), "spearman_rho": float(rho), "p_value": float(p),
            "note": "Among healthy controls only. A clear positive correlation suggests the score tracks age, not disease."}


def subgroup_report(scored: pd.DataFrame, thr: float, min_subjects: int, by: str) -> list[dict]:
    """Rate of scores above threshold per subgroup (sensitivity for PD rows, 1-specificity for control rows).
    Metrics for groups with fewer than `min_subjects` subjects are suppressed."""
    rows = []
    for key, g in scored.groupby(by, dropna=False):
        n_sub = g["subject_id"].nunique()
        row = {by: str(key), "n_subjects": int(n_sub), "n_recordings": int(len(g)),
               "n_pd_subjects": int(g[g["is_pd"] == 1]["subject_id"].nunique())}
        if n_sub < min_subjects:
            row["note"] = f"suppressed: fewer than {min_subjects} subjects"
        else:
            row["fraction_above_threshold"] = float((g["score"] >= thr).mean())
            if g["is_pd"].nunique() == 2:
                row["auroc"] = float(roc_auc_score(g["is_pd"], g["score"]))
        rows.append(row)
    return rows
