"""Subject-level (participant-grouped) splitting. A subject NEVER appears in more than one partition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def subject_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby("subject_id").agg(label=("label", "first"), n_labels=("label", "nunique"), is_pd=("is_pd", "first")).reset_index()
    if (t["n_labels"] > 1).any():
        raise ValueError("A subject has more than one label; cannot split safely.")
    return t


def assert_no_subject_overlap(*partitions: pd.DataFrame) -> None:
    seen: set = set()
    for p in partitions:
        ids = set(p["subject_id"])
        if ids & seen:
            raise AssertionError(f"Subject leakage: {len(ids & seen)} subject(s) appear in more than one partition")
        seen |= ids


def subject_holdout_split(df: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fraction of SUBJECTS (stratified by the 3-way group label when possible)."""
    st = subject_table(df)
    strat = st["label"] if st["label"].value_counts().min() >= 2 else st["is_pd"]
    train_ids, test_ids = train_test_split(st["subject_id"], test_size=test_fraction, stratify=strat, random_state=seed)
    train, test = df[df["subject_id"].isin(train_ids)], df[df["subject_id"].isin(test_ids)]
    assert_no_subject_overlap(train, test)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def grouped_cv(n_splits: int, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def safe_n_splits(df: pd.DataFrame, requested: int) -> int:
    st = subject_table(df)
    return int(max(2, min(requested, st["is_pd"].value_counts().min())))


def assert_cv_is_grouped(cv, X, y, groups) -> None:
    for tr, te in cv.split(X, y, groups):
        if set(np.asarray(groups)[tr]) & set(np.asarray(groups)[te]):
            raise AssertionError("Subject appears in both CV train and validation folds")
