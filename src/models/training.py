"""Leakage-safe model comparison, tuning, threshold selection and final fit.

Protocol (see README):
  1. Hold out a fraction of SUBJECTS as a test set (never touched here).
  2. On the training subjects: GridSearchCV with StratifiedGroupKFold (groups = subject_id) for every candidate model.
     Imputer + scaler live INSIDE the Pipeline so they are re-fitted on each training fold only.
  3. The winner is chosen by grouped-CV AUROC. Its OUT-OF-FOLD training scores define the operating threshold (Youden J).
  4. The winner is refitted on all training subjects and saved. The test set is used only by scripts/evaluate.py.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.splits import assert_cv_is_grouped, grouped_cv, safe_n_splits
from src.evaluation.metrics import youden_threshold

log = logging.getLogger(__name__)


def _classifier(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    if name == "svm_rbf":
        return SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss", verbosity=0)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name: str, seed: int) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scaler", StandardScaler()), ("clf", _classifier(name, seed))])


def available_models(cfg: dict) -> list[str]:
    names = list(cfg["models"])
    try:
        import xgboost  # noqa: F401
    except Exception:
        names = [n for n in names if n != "xgboost"]
        log.warning("xgboost not importable; skipped")
    return names


def score_method(pipe: Pipeline) -> str:
    return "predict_proba" if hasattr(pipe, "predict_proba") else "decision_function"


def score_with(pipe: Pipeline, X) -> np.ndarray:
    """Positive-class score. predict_proba[:,1] if the classifier offers it, else decision_function (uncalibrated)."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def _oof_scores(pipe, X, y, groups, cv) -> np.ndarray:
    method = score_method(pipe)
    out = cross_val_predict(clone(pipe), X, y, groups=groups, cv=cv, method=method)
    return out[:, 1] if method == "predict_proba" else out


def select_and_fit(train: pd.DataFrame, feature_names: list[str], cfg: dict, models: list[str] | None = None) -> dict:
    """Compare candidates with grouped CV on `train`, pick the best, derive threshold from OOF scores, refit."""
    seed = int(cfg["seed"])
    X, y, groups = train[feature_names].to_numpy(float), train["is_pd"].to_numpy(int), train["subject_id"].to_numpy()
    n_splits = safe_n_splits(train, int(cfg["split"]["inner_cv_folds"]))
    cv = grouped_cv(n_splits, seed)
    assert_cv_is_grouped(cv, X, y, groups)
    rows, fitted = [], {}
    for name in models or available_models(cfg):
        t0 = time.time()
        gs = GridSearchCV(build_pipeline(name, seed), cfg["models"][name]["grid"], scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
        gs.fit(X, y, groups=groups)
        best_idx = gs.best_index_
        rows.append({"model": name, "cv_auroc_mean": float(gs.best_score_),
                     "cv_auroc_std_across_folds": float(gs.cv_results_["std_test_score"][best_idx]),
                     "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in gs.best_params_.items()},
                     "seconds": round(time.time() - t0, 1)})
        fitted[name] = gs.best_estimator_
        log.info("%s: grouped-CV AUROC %.3f", name, gs.best_score_)
    table = pd.DataFrame(rows).sort_values("cv_auroc_mean", ascending=False).reset_index(drop=True)
    best_name = str(table.loc[0, "model"])
    best = fitted[best_name]
    oof = _oof_scores(best, X, y, groups, cv)
    thr = youden_threshold(y, oof)
    return {"model_name": best_name, "pipeline": best, "threshold": thr, "comparison": table, "oof_train": train[["subject_id", "recording_id", "is_pd"]].assign(oof_score=oof),
            "n_splits": n_splits, "score_type": score_method(best)}


def nested_grouped_cv(df: pd.DataFrame, feature_names: list[str], cfg: dict, n_outer: int = 5) -> dict:
    """Optional CROSS-VALIDATED estimate (NOT a held-out test): outer subject-grouped folds, full model selection
    and threshold choice repeated inside each outer training set."""
    seed = int(cfg["seed"])
    n_outer = safe_n_splits(df, n_outer)
    outer = grouped_cv(n_outer, seed + 1)
    X, y, g = df[feature_names].to_numpy(float), df["is_pd"].to_numpy(int), df["subject_id"].to_numpy()
    parts = []
    for k, (tr, te) in enumerate(outer.split(X, y, g)):
        res = select_and_fit(df.iloc[tr].reset_index(drop=True), feature_names, cfg)
        te_df = df.iloc[te].copy()
        te_df["score"] = score_with(res["pipeline"], te_df[feature_names].to_numpy(float))
        te_df["fold"], te_df["fold_threshold"], te_df["fold_model"] = k, res["threshold"], res["model_name"]
        parts.append(te_df[["subject_id", "recording_id", "is_pd", "label", "score", "fold", "fold_threshold", "fold_model"]])
        log.info("outer fold %d/%d done (%s)", k + 1, n_outer, res["model_name"])
    return {"oof": pd.concat(parts, ignore_index=True), "n_outer": n_outer}
