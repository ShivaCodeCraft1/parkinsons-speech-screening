"""Conservative duplicate-identity guard for splitting.

If recordings with IDENTICAL content (byte hash or decoded-audio hash) belong to more than one subject_id, those
subject_ids may be the same person stored under different folders. We do NOT decide that. We only guarantee they can
never be separated by a train/test or CV split: all linked subject_ids are merged into one split group.
Recordings are never deleted from disk. Merged groups whose members have conflicting labels are excluded from
training (the label itself is unreliable) and listed in the audit.
"""
from __future__ import annotations

import hashlib

import pandas as pd

HASH_COLS = ("file_sha256", "audio_sha256")


def build_subject_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return one row per subject_id: subject_id, split_group, n_subjects_in_group, group_labels_conflict.
    Uses ALL manifest rows (every task), because identical audio in any task links two subject_ids."""
    m = manifest[manifest["subject_id"].notna()]
    parent: dict[str, str] = {s: s for s in m["subject_id"].unique()}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for col in HASH_COLS:
        if col not in m.columns:
            continue
        for _, g in m.dropna(subset=[col]).groupby(col):
            subs = list(g["subject_id"].unique())
            for s in subs[1:]:
                parent[find(s)] = find(subs[0])
    comp: dict[str, list[str]] = {}
    for s in parent:
        comp.setdefault(find(s), []).append(s)
    label = m.groupby("subject_id")["label"].agg(lambda x: set(x.dropna()))
    rows = []
    for members in comp.values():
        members = sorted(members)
        gid = members[0] if len(members) == 1 else "GRP_" + hashlib.sha1("|".join(members).encode()).hexdigest()[:10]
        labels = set().union(*(label[s] for s in members))
        for s in members:
            rows.append({"subject_id": s, "split_group": gid, "n_subjects_in_group": len(members), "group_labels_conflict": len(labels) > 1})
    return pd.DataFrame(rows)


def apply_split_groups(df: pd.DataFrame, manifest: pd.DataFrame | None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Replace df['subject_id'] by the merged split group (original kept in 'subject_id_original').
    Returns (df_kept, audit, excluded_rows)."""
    df = df.copy()
    df["subject_id_original"] = df["subject_id"]
    audit = {"duplicate_check": "skipped (no manifest)", "merged_groups": 0, "subjects_in_merged_groups": 0,
             "excluded_recordings_label_conflict": 0}
    if manifest is None:
        return df, audit, df.iloc[0:0]
    sg = build_subject_groups(manifest)
    df = df.merge(sg, left_on="subject_id_original", right_on="subject_id", how="left", suffixes=("", "_sg"))
    df["split_group"] = df["split_group"].fillna(df["subject_id_original"])
    df["group_labels_conflict"] = df["group_labels_conflict"].fillna(False).astype(bool)
    excluded = df[df["group_labels_conflict"]].copy()
    excluded["exclusion_reason"] = "identical audio shared by subjects with different labels"
    df = df[~df["group_labels_conflict"]].copy()
    df["subject_id"] = df["split_group"]
    merged = df[df["n_subjects_in_group"].fillna(1) > 1]
    audit.update(duplicate_check="done (file_sha256 + audio_sha256 across ALL manifest recordings)",
                 merged_groups=int(merged["split_group"].nunique()),
                 subjects_in_merged_groups=int(merged["subject_id_original"].nunique()),
                 excluded_recordings_label_conflict=int(len(excluded)))
    keep = [c for c in df.columns if c not in ("subject_id_sg", "split_group", "n_subjects_in_group", "group_labels_conflict")]
    return df[keep].reset_index(drop=True), audit, excluded.reset_index(drop=True)
