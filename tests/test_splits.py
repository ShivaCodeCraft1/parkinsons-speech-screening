import numpy as np
import pandas as pd
import pytest

from src.data.splits import assert_cv_is_grouped, assert_no_subject_overlap, grouped_cv, subject_holdout_split


def frame(n_per=8, rec=4):
    rows = []
    for lab, is_pd in (("young", 0), ("elderly", 0), ("pd", 1)):
        for i in range(n_per):
            for r in range(rec):
                rows.append({"subject_id": f"{lab}{i}", "recording_id": f"{lab}{i}_{r}", "label": lab, "is_pd": is_pd})
    return pd.DataFrame(rows)


def test_holdout_split_never_shares_subjects_and_keeps_recordings_together():
    df = frame()
    tr, te = subject_holdout_split(df, 0.25, seed=0)
    assert not set(tr.subject_id) & set(te.subject_id)
    assert len(tr) + len(te) == len(df)
    for sid, g in df.groupby("subject_id"):
        assert (sid in set(tr.subject_id)) != (sid in set(te.subject_id))
    assert te.label.nunique() == 3                                  # stratified by group
    tr2, te2 = subject_holdout_split(df, 0.25, seed=0)
    assert set(te.subject_id) == set(te2.subject_id)                # reproducible


def test_overlap_assertion_fires():
    df = frame()
    with pytest.raises(AssertionError):
        assert_no_subject_overlap(df.iloc[:10], df.iloc[5:15])


def test_grouped_cv_has_no_subject_leakage():
    df = frame()
    assert_cv_is_grouped(grouped_cv(4, 0), np.zeros((len(df), 1)), df.is_pd.to_numpy(), df.subject_id.to_numpy())


def test_conflicting_subject_labels_are_rejected():
    df = frame()
    df.loc[0, "label"] = "pd"
    with pytest.raises(ValueError):
        subject_holdout_split(df, 0.25, seed=0)
