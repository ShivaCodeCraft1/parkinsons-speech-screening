"""Helpers that connect the Phase-1 manifest to the real audio files on the local machine.

The manifest is pseudonymised (no names). The mapping recording_id -> real relative path lives ONLY in
data/interim/private/path_lookup_PRIVATE.csv (created by the verifier, git-ignored).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT

DEFAULT_MANIFEST = REPO_ROOT / "data" / "interim" / "manifest.csv"
DEFAULT_PRIVATE_DIR = REPO_ROOT / "data" / "interim" / "private"
DEFAULT_FEATURES = REPO_ROOT / "data" / "processed" / "features.csv"


def load_manifest(path: Path = DEFAULT_MANIFEST) -> pd.DataFrame:
    if not Path(path).is_file():
        raise SystemExit(f"Manifest not found: {path}. Run `python scripts/inspect_dataset.py` first.")
    return pd.read_csv(path)


def usable_recordings(manifest: pd.DataFrame, task_families: list[str] | None = None) -> pd.DataFrame:
    """Rows that can enter training: readable audio, a subject, a label, a parsed task code."""
    m = manifest.copy()
    m = m[m["readable"].astype(bool) & m["subject_id"].notna() & m["is_pd"].notna() & (m["task_parse_status"] == "ok")]
    if task_families:
        m = m[m["task_family"].isin(task_families)]
    return m.reset_index(drop=True)


def resolve_paths(manifest: pd.DataFrame, data_root: Path, private_dir: Path = DEFAULT_PRIVATE_DIR) -> pd.DataFrame:
    lookup_file = Path(private_dir) / "path_lookup_PRIVATE.csv"
    if not lookup_file.is_file():
        raise SystemExit(f"{lookup_file} not found. Re-run `python scripts/inspect_dataset.py` on this machine.")
    lk = pd.read_csv(lookup_file)
    m = manifest.merge(lk, on="recording_id", how="left")
    m["abs_path"] = [str(Path(data_root) / r) if isinstance(r, str) else None for r in m["relative_path"]]
    return m
