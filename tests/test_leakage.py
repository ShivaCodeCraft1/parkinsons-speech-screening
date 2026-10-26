import pandas as pd

from src.data.leakage import apply_split_groups, build_subject_groups
from src.data.splits import assert_no_subject_overlap, subject_holdout_split


def manifest():
    rows = []
    for i in range(12):
        lab = "parkinsons_disease" if i % 2 else "elderly_healthy_control"
        for t in range(3):
            rows.append({"subject_id": f"S{i}", "recording_id": f"S{i}_{t}", "label": lab, "is_pd": int(i % 2),
                         "file_sha256": f"f{i}_{t}", "audio_sha256": f"a{i}_{t}"})
    m = pd.DataFrame(rows)
    # S2 and S4 (both controls) share identical audio in a DIFFERENT task row -> same person suspected
    m.loc[m.recording_id == "S4_1", ["file_sha256", "audio_sha256"]] = ["f2_0", "a2_0"]
    # S1 (PD) and S3 (PD) share identical audio too
    m.loc[m.recording_id == "S3_2", "audio_sha256"] = "a1_1"
    return m


def test_shared_audio_merges_subjects_and_keeps_them_together():
    m = manifest()
    sg = build_subject_groups(m).set_index("subject_id")
    assert sg.loc["S2", "split_group"] == sg.loc["S4", "split_group"] != sg.loc["S6", "split_group"]
    assert sg.loc["S1", "split_group"] == sg.loc["S3", "split_group"]
    df, audit, excluded = apply_split_groups(m.copy(), m)
    assert audit["merged_groups"] == 2 and audit["subjects_in_merged_groups"] == 4 and excluded.empty
    for seed in range(15):
        tr, te = subject_holdout_split(df, 0.34, seed)
        assert_no_subject_overlap(tr, te)
        assert not ({"S2", "S4"} & set(tr.subject_id_original)) or not ({"S2", "S4"} & set(te.subject_id_original))
    assert len(df) == len(m)          # nothing deleted


def test_conflicting_labels_in_shared_audio_are_excluded_and_listed():
    m = manifest()
    m.loc[m.recording_id == "S5_0", ["file_sha256", "audio_sha256"]] = ["f2_0", "a2_0"]   # PD S5 shares audio with control S2
    df, audit, excluded = apply_split_groups(m.copy(), m)
    assert audit["excluded_recordings_label_conflict"] == len(excluded) > 0
    assert set(excluded["exclusion_reason"]) == {"identical audio shared by subjects with different labels"}
    assert not ({"S2", "S4", "S5"} & set(df.subject_id_original))


def test_no_manifest_is_reported_not_silent():
    m = manifest()
    _, audit, _ = apply_split_groups(m.copy(), None)
    assert audit["duplicate_check"].startswith("skipped")
