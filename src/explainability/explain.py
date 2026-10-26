"""Feature-importance / explanation for the TRAINED pipeline.

Important: importance describes what THIS model uses. It is not causal evidence or a validated biomarker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression


def _clf_kind(pipe) -> str:
    clf = pipe.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        return "linear"
    if isinstance(clf, RandomForestClassifier) or clf.__class__.__name__ == "XGBClassifier":
        return "tree"
    return "other"


def _transform(pipe, X: np.ndarray) -> np.ndarray:
    return pipe[:-1].transform(X)


def _tree_values(pipe, Xt: np.ndarray) -> np.ndarray:
    import shap
    sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def global_importance(pipe, X_df: pd.DataFrame, y, names: list[str], seed: int, max_rows: int = 300) -> dict:
    """Permutation importance (any model, AUROC drop) + mean |SHAP| (tree) or |coef| (linear) when available."""
    X = X_df[names].to_numpy(float)
    out: dict = {}
    if len(np.unique(y)) == 2:
        r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=20, random_state=seed, n_jobs=1)
        out["permutation"] = pd.DataFrame({"feature": names, "auroc_drop_mean": r.importances_mean, "auroc_drop_std": r.importances_std}) \
            .sort_values("auroc_drop_mean", ascending=False).reset_index(drop=True)
    kind = _clf_kind(pipe)
    if kind == "tree":
        Xt = _transform(pipe, X[:max_rows])
        out["shap"] = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(_tree_values(pipe, Xt)).mean(axis=0)}) \
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    elif kind == "linear":
        coef = pipe.named_steps["clf"].coef_[0]
        out["coefficients"] = pd.DataFrame({"feature": names, "standardized_coef": coef, "abs": np.abs(coef)}) \
            .sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    return out


def explain_one(pipe, row: np.ndarray, names: list[str], top_k: int = 8) -> dict | None:
    """Per-prediction contributions (log-odds scale for LR; SHAP for trees). Returns None for models without a cheap exact method."""
    kind = _clf_kind(pipe)
    Xt = _transform(pipe, row)
    if kind == "linear":
        contrib = pipe.named_steps["clf"].coef_[0] * Xt[0]
        method = "coefficient x standardised value (relative to the training mean)"
    elif kind == "tree":
        contrib = _tree_values(pipe, Xt)[0]
        method = "TreeSHAP"
    else:
        return None
    idx = np.argsort(-np.abs(contrib))[:top_k]
    return {"method": method, "top_features": [{"feature": names[i], "contribution": float(contrib[i])} for i in idx],
            "note": "Describes this model's behaviour only; not causal and not a validated biomarker."}
