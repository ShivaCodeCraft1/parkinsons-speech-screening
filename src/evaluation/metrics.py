"""Binary metrics, subject-level aggregation and subject-cluster bootstrap CIs. Positive class = Parkinson's (is_pd=1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, roc_auc_score


def threshold_metrics(y, score, thr: float) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"threshold": float(thr), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "sensitivity": float(sens), "specificity": float(spec),
            "precision": float(precision_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float((pred == y).mean())}


def full_metrics(y, score, thr: float) -> dict:
    y = np.asarray(y)
    out = threshold_metrics(y, score, thr)
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, score))
        out["average_precision"] = float(average_precision_score(y, score))
    else:
        out["auroc"] = out["average_precision"] = float("nan")
    return out


def youden_threshold(y, score) -> float:
    """Threshold maximising sensitivity+specificity-1 (Youden J); first maximiser on the ROC curve."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, score)
    j = tpr - fpr
    best = int(np.argmax(j))
    t = float(thr[best])
    return t if np.isfinite(t) else float(np.max(score)) + 1e-9


def aggregate_by_subject(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Mean recording score per subject (subject-level view). Label is constant within a subject."""
    return df.groupby("subject_id").agg(is_pd=("is_pd", "first"), label=("label", "first"), score=(score_col, "mean"),
                                        n_recordings=(score_col, "size")).reset_index()


def cluster_bootstrap(df: pd.DataFrame, thr: float, n: int, conf: float, seed: int, score_col: str = "score") -> dict:
    """Resample SUBJECTS with replacement (all their recordings together) -> CIs for AUROC/sens/spec at a fixed threshold."""
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in df.groupby("subject_id")}
    ids = np.array(list(groups))
    stats = {"auroc": [], "sensitivity": [], "specificity": []}
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        b = pd.concat([groups[s] for s in pick])
        if b["is_pd"].nunique() < 2:
            continue
        m = full_metrics(b["is_pd"], b[score_col], thr)
        for k in stats:
            stats[k].append(m[k])
    lo, hi = (1 - conf) / 2 * 100, (1 + conf) / 2 * 100
    return {k: {"low": float(np.nanpercentile(v, lo)), "high": float(np.nanpercentile(v, hi)), "n_valid_resamples": len(v)}
            if v else {"low": None, "high": None, "n_valid_resamples": 0} for k, v in stats.items()}
