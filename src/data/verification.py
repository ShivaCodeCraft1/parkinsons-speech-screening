#!/usr/bin/env python3
"""
verify_dataset.py -- Phase 1 dataset verification for the Italian Parkinson's
Voice and Speech corpus (research prototype; NOT a diagnostic tool).

What it does (and only this):
  * recursively scans the local dataset folder
  * derives labels from the FOLDER STRUCTURE, participants from PARTICIPANT FOLDERS
  * creates stable pseudonymous subject_id / recording_id values (salted hash)
  * decodes task codes from WAV filenames
  * inspects every WAV (readable, sample rate, channels, duration, silence, clipping)
  * hashes every file to find duplicates
  * inspects every Excel file (sheets, columns, missing values) and tests whether
    metadata can be joined to participants WITHOUT ambiguity
  * writes manifest.csv, dataset_report.json, dataset_report.md
  * exits non-zero if any hard-fail check fails

It never extracts ML features, never trains models, and never fills in, guesses
or imputes a missing value. Participant names are never written to the reports
or the manifest; the only name-bearing outputs go to a separate PRIVATE folder.

Usage (via the thin CLI wrapper):
  python scripts/verify_dataset.py --data-root "/path/to/extracted/dataset"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import secrets
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

SCRIPT_VERSION = "1.1.0"

OTHER_AUDIO_EXT = {".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".opus", ".amr"}
EXCEL_EXT = {".xlsx", ".xlsm", ".xls"}
FILE_CODES_RE = re.compile(r"file[\s_\-]*codes?", re.IGNORECASE)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def name_keys(s: str) -> dict[str, str]:
    """Order/format-insensitive keys used ONLY to test metadata linkage."""
    norm = strip_accents(str(s)).casefold()
    tokens = re.findall(r"[a-z0-9]+", norm)
    return {
        "full": "".join(tokens),
        "tokens_sorted": " ".join(sorted(tokens)),
        "tokens_nodigits": " ".join(sorted(t for t in tokens if not t.isdigit())),
    }


def filename_shape(stem: str) -> str:
    """Name-free description of a filename: digits -> 9, letter runs -> A."""
    return re.sub(r"[A-Za-z]+", "A", re.sub(r"\d+", "9", stem))


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if (math.isnan(float(o)) or math.isinf(float(o))) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def clean_nan(o):
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean_nan(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def stats(values) -> dict:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    a = np.asarray(v, dtype=float)
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "max": float(a.max())}


def load_or_create_salt(path: Path) -> tuple[str, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text().strip(), False
    salt = secrets.token_hex(16)
    path.write_text(salt + "\n")
    return salt, True


def pseudo_id(prefix: str, salt: str, *parts: str, n: int = 10) -> str:
    h = hashlib.sha256((salt + "|" + "|".join(parts)).encode("utf-8")).hexdigest()
    return f"{prefix}_{h[:n]}"


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# folder scan, labels, participants
# ----------------------------------------------------------------------------
def scan_tree(root: Path) -> dict:
    wavs, other_audio, excel, filecodes = [], [], [], []
    other_ext = Counter()
    skipped = Counter()
    dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__MACOSX" and not d.startswith("."))
        rel_dir = Path(dirpath).relative_to(root)
        if rel_dir.parts:
            dirs.append(rel_dir.parts)
        for fn in sorted(filenames):
            if fn.startswith("._") or fn == ".DS_Store" or fn.startswith("~$") or fn.lower() == "thumbs.db":
                skipped["system_or_temp_file"] += 1
                continue
            p = Path(dirpath) / fn
            rel = p.relative_to(root)
            ext = p.suffix.lower()
            if FILE_CODES_RE.search(p.stem):
                filecodes.append(rel)
            elif ext == ".wav":
                wavs.append(rel)
            elif ext in OTHER_AUDIO_EXT:
                other_audio.append(rel)
            elif ext in EXCEL_EXT:
                excel.append(rel)
            else:
                other_ext[ext or "<no extension>"] += 1
    return {"wavs": wavs, "other_audio": other_audio, "excel": excel, "filecodes": filecodes,
            "other_ext": other_ext, "skipped": skipped, "dirs": dirs}


def detect_group(dir_parts: tuple[str, ...], group_res: dict[str, re.Pattern]) -> dict:
    """Resolve group and participant folder, including PD range-folder nesting.

    Typical control path: group / participant / recording.wav
    PD path: group / range (e.g. 1-5) / participant / recording.wav
    A numeric range component immediately after the PD group is treated as an
    organizational folder, never as a participant.
    """
    hits = []
    for i, comp in enumerate(dir_parts):
        labels = [lab for lab, rx in group_res.items() if rx.search(comp)]
        if labels:
            hits.append((i, labels))
    empty = {"status": "no_label", "label": None, "group_idx": None,
             "participant_idx": None, "participant": None}
    if not hits:
        return empty
    if any(len(labs) > 1 for _, labs in hits):
        return {**empty, "status": "ambiguous_label"}
    first_idx, first_labels = hits[0]
    if any(labs != first_labels for _, labs in hits[1:]):
        return {**empty, "status": "ambiguous_label"}
    label = first_labels[0]
    participant_idx = first_idx + 1
    if label == "parkinsons_disease" and participant_idx < len(dir_parts):
        if re.fullmatch(r"\d+\s*[-–]\s*\d+", dir_parts[participant_idx].strip()):
            participant_idx += 1
    if participant_idx >= len(dir_parts):
        return {"status": "no_participant", "label": label, "group_idx": first_idx,
                "participant_idx": participant_idx, "participant": None}
    return {"status": "ok", "label": label, "group_idx": first_idx,
            "participant_idx": participant_idx, "participant": dir_parts[participant_idx]}


# ----------------------------------------------------------------------------
# task-code decoding
# ----------------------------------------------------------------------------
def extract_task_code(stem: str, known: set[str]) -> tuple[str | None, str, str | None]:
    """Return (code, status, detail). status: ok / no_code / ambiguous_code / unknown_code."""
    s = stem.upper()
    tokens = [t for t in re.split(r"[^A-Z0-9]+", s) if t]
    strategies = []
    strategies.append(("whole_stem", {s} & known))
    strategies.append(("token", {t for t in tokens if t in known}))
    pre = {c for c in known if s.startswith(c) and (len(s) == len(c) or not s[len(c)].isdigit())}
    pre = {c for c in pre if not any(o != c and o.startswith(c) for o in pre)}
    strategies.append(("prefix", pre))
    suf = {c for c in known if s.endswith(c) and (len(s) == len(c) or not s[-len(c) - 1].isdigit())}
    suf = {c for c in suf if not any(o != c and o.endswith(c) for o in suf)}
    strategies.append(("suffix", suf))
    for name, found in strategies:
        if len(found) == 1:
            return next(iter(found)), "ok", name
        if len(found) > 1:
            return None, "ambiguous_code", f"{name}:{sorted(found)}"
    shaped = re.findall(r"(?<![A-Z0-9])([A-Z]{1,2}\d{1,2})(?![A-Z0-9])", s)
    m = re.match(r"^([A-Z]{1,2}\d{1,2})(?!\d)", s)
    if m:
        shaped.append(m.group(1))
    shaped = sorted(set(shaped))
    if shaped:
        return None, "unknown_code", ",".join(shaped[:5])
    return None, "no_code", None


def parse_file_codes(path: Path) -> dict:
    """Best-effort parse of the dataset's own FILE CODES file. Cross-check only."""
    info = {"parsed": False, "format": path.suffix.lower(), "codes_found": [], "error_type": None}
    text = None
    try:
        if path.suffix.lower() in EXCEL_EXT:
            xf = pd.ExcelFile(path)
            lines = []
            for sh in xf.sheet_names:
                df = xf.parse(sh, header=None, dtype=str)
                for row in df.fillna("").values.tolist():
                    lines.append("\t".join(str(c) for c in row))
            text = "\n".join(lines)
        elif path.suffix.lower() in {".txt", ".csv", ".tsv", ".md", ""}:
            raw = path.read_bytes()
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            info["error_type"] = "unsupported_format"
            return info
    except Exception as e:  # noqa: BLE001
        info["error_type"] = type(e).__name__
        return info
    codes = set()
    for line in (text or "").splitlines():
        left = re.split(r"[:=\u2192\-\u2013\u2014\t,]|\s{2,}", line.strip(), maxsplit=1)[0]
        for c in re.findall(r"[A-Za-z]{1,2}\d", left):
            codes.add(c.upper())
    info["parsed"] = bool(codes)
    info["codes_found"] = sorted(codes)
    return info


# ----------------------------------------------------------------------------
# audio inspection
# ----------------------------------------------------------------------------
def inspect_audio(path: Path, qc: dict) -> dict:
    out = {"readable": False, "error_type": None, "sample_rate": None, "channels": None,
           "num_frames": None, "duration_sec": None, "subtype": None, "format": None,
           "peak_abs": None, "rms_dbfs": None, "clip_fraction": None, "silence_fraction": None,
           "audio_sha256": None, "flags": []}
    try:
        info = sf.info(str(path))
        out.update(sample_rate=int(info.samplerate), channels=int(info.channels),
                   num_frames=int(info.frames), subtype=info.subtype, format=info.format)
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        out["error_type"] = type(e).__name__
        out["flags"].append("unreadable")
        return out
    out["readable"] = True
    n = data.shape[0]
    out["num_frames"] = int(n)
    out["duration_sec"] = float(n / sr) if sr else None
    out["audio_sha256"] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    if n == 0:
        out["flags"].append("empty")
        return out
    mono = data.mean(axis=1)
    out["peak_abs"] = float(np.abs(data).max())
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    out["rms_dbfs"] = float(20 * np.log10(rms + 1e-12))
    out["clip_fraction"] = float(np.mean(np.abs(data) >= qc["clip_sample_threshold"]))
    fl = int(sr * qc["silence_frame_ms"] / 1000)
    if fl > 0 and n >= fl:
        k = n // fl
        fr = mono[: k * fl].reshape(k, fl).astype(np.float64)
        fdb = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12)
        out["silence_fraction"] = float(np.mean(fdb < qc["silence_frame_dbfs"]))
    if out["rms_dbfs"] < qc["near_silent_rms_dbfs"]:
        out["flags"].append("near_silent")
    if out["clip_fraction"] > qc["clip_fraction_flag"]:
        out["flags"].append("clipping_suspected")
    if out["silence_fraction"] is not None and out["silence_fraction"] > qc["high_silence_fraction_flag"]:
        out["flags"].append("high_silence_fraction")
    if out["duration_sec"] < qc["min_duration_sec_flag"]:
        out["flags"].append("very_short")
    return out


# ----------------------------------------------------------------------------
# Excel metadata inspection + participant linkage
# ----------------------------------------------------------------------------
def inspect_excel(path: Path, meta_cfg: dict, participant_keys: list[dict], root_label: str) -> dict:
    sens_re = re.compile(meta_cfg["sensitive_column_regex"], re.IGNORECASE)
    full_keys = {k["full"] for k in participant_keys if k["full"]}
    res = {"file": root_label, "readable": False, "error_type": None, "sheets": []}
    try:
        xf = pd.ExcelFile(path)
    except Exception as e:  # noqa: BLE001
        res["error_type"] = type(e).__name__
        return res
    res["readable"] = True
    for sh in xf.sheet_names:
        sheet = {"sheet": sh, "readable": False, "error_type": None}
        try:
            df = xf.parse(sh, header=0, dtype=object)
        except Exception as e:  # noqa: BLE001
            sheet["error_type"] = type(e).__name__
            res["sheets"].append(sheet)
            continue
        sheet["readable"] = True
        sheet["n_rows"] = int(df.shape[0])
        sheet["n_columns"] = int(df.shape[1])
        sheet["n_fully_empty_rows"] = int(df.isna().all(axis=1).sum())
        cols = []
        redacted = 0
        for i, c in enumerate(df.columns):
            cname = str(c)
            ckey = name_keys(cname)["full"]
            sensitive = bool(sens_re.search(cname))
            if ckey and ckey in full_keys:
                cname_out, sensitive = f"<REDACTED_header_matches_participant_{i}>", True
                redacted += 1
            else:
                cname_out = cname
            s = df[c]
            n_missing = int((s.isna() | (s.astype(str).str.strip() == "")).sum())
            non_null = s.dropna()
            numeric = pd.to_numeric(non_null, errors="coerce")
            frac_numeric = float(numeric.notna().mean()) if len(non_null) else 0.0
            col = {"index": i, "name": cname_out, "n_missing": n_missing,
                   "frac_missing": float(n_missing / len(df)) if len(df) else None,
                   "n_unique": int(non_null.astype(str).nunique()),
                   "numeric_fraction": round(frac_numeric, 4), "sensitive_values_hidden": sensitive,
                   "unnamed_header": cname.startswith("Unnamed")}
            if not sensitive:
                if frac_numeric == 1.0 and len(non_null):
                    col["numeric_summary"] = stats(numeric.tolist())
                if 0 < col["n_unique"] <= 12:
                    vc = non_null.astype(str).str.strip().value_counts()
                    col["value_counts"] = {str(k): int(v) for k, v in vc.items()}
            cols.append(col)
        sheet["columns"] = cols
        sheet["n_redacted_headers"] = redacted
        sheet["_df"] = df  # removed before serialisation
        res["sheets"].append(sheet)
    return res


def build_linkage(excel_results: list[dict], participants: list[dict], meta_cfg: dict, wav_stem_keys: set[str]) -> dict:
    """Test participant-folder <-> Excel-row linkage. Reports counts only."""
    age_re = re.compile(meta_cfg["age_column_regex"], re.IGNORECASE)
    sex_re = re.compile(meta_cfg["sex_column_regex"], re.IGNORECASE)
    grp_re = re.compile(meta_cfg["group_column_regex"], re.IGNORECASE)
    n_p = len(participants)
    sheet_links = []
    per_participant = defaultdict(lambda: {"age": [], "sex": [], "link_sheets": 0, "multirow_inconsistent": False})
    for er in excel_results:
        for sh in er["sheets"]:
            df = sh.get("_df")
            if df is None or df.empty:
                continue
            best = None
            for ci, c in enumerate(df.columns):
                col = df.iloc[:, ci]
                strs = col.dropna().astype(str)
                if strs.empty:
                    continue
                for ktype in ("full", "tokens_sorted", "tokens_nodigits"):
                    index = defaultdict(list)
                    for ridx, v in col.items():
                        if pd.isna(v):
                            continue
                        k = name_keys(str(v))[ktype]
                        if k:
                            index[k].append(ridx)
                    matches = {}
                    for pi, p in enumerate(participants):
                        k = p["keys"][ktype]
                        if k and k in index:
                            matches[pi] = index[k]
                    n_unique = sum(1 for r in matches.values() if len(r) == 1)
                    n_multi = sum(1 for r in matches.values() if len(r) > 1)
                    rows_shared = 0
                    row_owner = defaultdict(set)
                    for pi, rows in matches.items():
                        for r in rows:
                            row_owner[r].add(pi)
                    rows_shared = sum(1 for s in row_owner.values() if len(s) > 1)
                    stem_hits = len({name_keys(str(v))["full"] for v in strs} & wav_stem_keys)
                    cand = {"file": er["file"], "sheet": sh["sheet"], "column_index": ci,
                            "column_name": sh["columns"][ci]["name"], "key_type": ktype,
                            "n_participants_matched": len(matches), "n_matched_exactly_one_row": n_unique,
                            "n_matched_multiple_rows": n_multi, "n_rows_matched_by_multiple_participants": rows_shared,
                            "n_cells_equal_to_a_wav_filename_stem": stem_hits, "_matches": matches}
                    if best is None or (cand["n_participants_matched"], cand["n_matched_exactly_one_row"]) > \
                            (best["n_participants_matched"], best["n_matched_exactly_one_row"]):
                        best = cand
            if best is None or best["n_participants_matched"] == 0:
                sheet_links.append({"file": er["file"], "sheet": sh["sheet"], "linked": False,
                                    "n_participants_matched": 0})
                continue
            # attribute columns in this sheet
            names = [str(c) for c in df.columns]
            age_cols = [i for i, n in enumerate(names) if age_re.search(n.strip())]
            sex_cols = [i for i, n in enumerate(names) if sex_re.search(n.strip())]
            grp_cols = [i for i, n in enumerate(names) if grp_re.search(n.strip())]
            link = {k: v for k, v in best.items() if k != "_matches"}
            link.update(linked=True, age_columns_found=[sh["columns"][i]["name"] for i in age_cols],
                        sex_columns_found=[sh["columns"][i]["name"] for i in sex_cols],
                        group_like_columns_found=[sh["columns"][i]["name"] for i in grp_cols])
            link["age_used"] = len(age_cols) == 1
            link["sex_used"] = len(sex_cols) == 1
            sheet_links.append(link)
            for pi, rows in best["_matches"].items():
                rec = per_participant[pi]
                rec["link_sheets"] += 1
                if link["age_used"]:
                    vals = pd.to_numeric(df.iloc[rows, age_cols[0]], errors="coerce").dropna().unique().tolist()
                    if len(vals) > 1:
                        rec["multirow_inconsistent"] = True
                    elif len(vals) == 1:
                        rec["age"].append(float(vals[0]))
                if link["sex_used"]:
                    vals = df.iloc[rows, sex_cols[0]].dropna().astype(str).str.strip().unique().tolist()
                    vals = [v for v in vals if v != ""]
                    if len(vals) > 1:
                        rec["multirow_inconsistent"] = True
                    elif len(vals) == 1:
                        rec["sex"].append(vals[0])
    # participant-level resolution
    resolved, status_counts = {}, Counter()
    for pi in range(n_p):
        rec = per_participant.get(pi)
        if rec is None or rec["link_sheets"] == 0:
            resolved[pi] = {"age": None, "sex": None, "status": "not_linked"}
            status_counts["not_linked"] += 1
            continue
        age_set, sex_set = set(rec["age"]), set(rec["sex"])
        if rec["multirow_inconsistent"] or len(age_set) > 1 or len(sex_set) > 1:
            resolved[pi] = {"age": None, "sex": None, "status": "ambiguous"}
            status_counts["ambiguous"] += 1
            continue
        resolved[pi] = {"age": next(iter(age_set)) if age_set else None,
                        "sex": next(iter(sex_set)) if sex_set else None, "status": "linked_unambiguous"}
        status_counts["linked_unambiguous"] += 1
    if n_p == 0 or status_counts["not_linked"] == n_p:
        overall = "NOT_JOINABLE"
    elif status_counts["ambiguous"] > 0:
        overall = "AMBIGUOUS"
    elif status_counts["not_linked"] > 0:
        overall = "PARTIAL"
    else:
        overall = "UNAMBIGUOUS"
    # metadata group-like column vs folder label cross-tab (category counts only)
    crosstabs = []
    for er in excel_results:
        for sh in er["sheets"]:
            df = sh.get("_df")
            link = next((sl for sl in sheet_links if sl["file"] == er["file"] and sl["sheet"] == sh["sheet"] and sl.get("linked")), None)
            if df is None or not link or len(link["group_like_columns_found"]) != 1:
                continue
            gname = link["group_like_columns_found"][0]
            gi = [c["name"] for c in sh["columns"]].index(gname)
            ct = Counter()
            # rebuild matches for the chosen column/key
            col = df.iloc[:, link["column_index"]]
            index = defaultdict(list)
            for ridx, v in col.items():
                if not pd.isna(v):
                    k = name_keys(str(v))[link["key_type"]]
                    if k:
                        index[k].append(ridx)
            for pi, p in enumerate(participants):
                k = p["keys"][link["key_type"]]
                rows = index.get(k, []) if k else []
                if len(rows) >= 1:
                    vals = df.iloc[rows, gi].dropna().astype(str).str.strip().unique().tolist()
                    ct[(p["label"], vals[0] if len(vals) == 1 else "<multiple_values>")] += 1
            if ct and len({v for (_, v) in ct}) <= 12:
                crosstabs.append({"file": er["file"], "sheet": sh["sheet"], "group_column": gname,
                                  "counts": [{"folder_label": a, "metadata_value": b, "n_participants": n}
                                             for (a, b), n in sorted(ct.items())]})
    return {"overall_status": overall, "participants_by_status": dict(status_counts),
            "sheet_links": sheet_links, "folder_label_vs_metadata_group": crosstabs,
            "resolved": resolved}


# ----------------------------------------------------------------------------
# markdown rendering
# ----------------------------------------------------------------------------
def md_table(headers: list[str], rows: list[list]) -> str:
    def f(x):
        if x is None:
            return "n/a"
        if isinstance(x, float):
            return f"{x:.4g}"
        return str(x)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(f(c) for c in r) + " |")
    return "\n".join(out)


def render_markdown(rep: dict) -> str:
    L = []
    a = L.append
    verdict = "PASS" if rep["all_hard_checks_passed"] else "FAIL"
    a("# Dataset verification report\n")
    a(f"**Overall hard-fail verdict: {verdict}**  ")
    a(f"Run (UTC): {rep['meta']['run_timestamp_utc']} | script v{rep['meta']['script_version']} | "
      f"git commit: {rep['meta']['git_commit'] or 'not available'}\n")
    a("> Every number below was measured from the local files on this run. "
      "Nothing is taken from papers. Participant names are not included.\n")
    a("## 1. Hard-fail checks\n")
    a(md_table(["Check", "Status", "Affected", "Details"],
               [[c["id"], c["status"], c["n_affected"], c["details"]] for c in rep["hard_checks"]]))
    a("\n## 2. Warnings\n")
    if rep["warnings"]:
        for w in rep["warnings"]:
            a(f"- {w}")
    else:
        a("None.")
    c = rep["counts"]
    a("\n## 3. Counts\n")
    a(f"- WAV files found: **{c['n_wav_files']}**")
    a(f"- Participant folders found (structure): **{c['n_participant_folders']}**")
    a(f"- Subjects with at least one WAV: **{c['n_subjects_with_audio']}**")
    a(f"- Other audio files (unexpected formats): {c['n_other_audio_files']}")
    a(f"- Excel files: {c['n_excel_files']} | FILE CODES files: {c['n_file_codes_files']}")
    a(f"- Other file extensions present: {c['other_extensions'] or 'none'}\n")
    a(md_table(["Label", "Participant folders", "Subjects with audio", "Recordings", "Your manual count"],
               [[lab, d["participant_folders"], d["subjects_with_audio"], d["recordings"], d["user_confirmed"]]
                for lab, d in c["by_label"].items()]))
    a("\n## 4. Recordings per subject\n")
    rps = rep["recordings_per_subject"]
    a(md_table(["Group", "n subjects", "min", "median", "mean", "max"],
               [[g, s["n"], s["min"], s["median"], s["mean"], s["max"]] for g, s in rps["by_group"].items()]))
    a(f"\nHistogram of recordings-per-subject (all groups): {rps['histogram']}")
    a("\n## 5. Recordings per task\n")
    tasks = rep["recordings_per_task"]
    labels = list(rep["counts"]["by_label"].keys())
    a(md_table(["Task code", "Task name"] + labels + ["Total", "Subjects having it"],
               [[t["code"], t["name"]] + [t["by_label"].get(lab, 0) for lab in labels] + [t["total"], t["n_subjects"]]
                for t in tasks["by_code"]]))
    a(f"\nUnparsed / unknown task-code files: **{tasks['n_unparsed']}** "
      f"(status counts: {tasks['parse_status_counts']})")
    a(f"\nSubjects by number of distinct known task codes covered "
      f"(of {tasks['n_expected_codes']}): {tasks['subjects_by_codes_covered']}")
    a("\n## 6. Filename shapes (names removed: digits->9, letters->A)\n")
    a(md_table(["Shape", "Files"], [[k, v] for k, v in rep["filename_shapes_top"]]))
    a("\n## 7. FILE CODES cross-check\n")
    fc = rep["file_codes_check"]
    a(f"- FILE CODES files found: {fc['n_files']}")
    for it in fc["files"]:
        a(f"- format `{it['format']}`: parsed={it['parsed']}, codes found={it['codes_found']}, error={it['error_type']}")
    a(f"- Codes in FILE CODES but not in config: {fc['in_file_not_config']}")
    a(f"- Codes in config but not in FILE CODES: {fc['in_config_not_file']}")
    a("\n## 8. Audio inspection (all WAV files)\n")
    au = rep["audio"]
    a(f"- Readable: {au['n_readable']} | unreadable: {au['n_unreadable']} | empty: {au['n_empty']}")
    a(f"- Sample rates (Hz): {au['sample_rate_counts']}")
    a(f"- Channels: {au['channel_counts']}")
    a(f"- Encodings (subtype): {au['subtype_counts']}")
    a(f"- QC flags (heuristic, thresholds in config): {au['flag_counts']}")
    a(f"- Duration (s) overall: {au['duration_overall']}\n")
    a(md_table(["Task", "n", "min s", "median s", "mean s", "max s"],
               [[t, s["n"], s["min"], s["median"], s["mean"], s["max"]] for t, s in au["duration_by_task"].items()]))
    a("\n## 9. Duplicates and leakage proxies\n")
    d = rep["duplicates"]
    a(f"- Byte-identical file groups: {d['n_file_hash_groups']} (files involved: {d['n_files_in_file_hash_groups']})")
    a(f"- Identical decoded-audio groups: {d['n_audio_hash_groups']} (files involved: {d['n_files_in_audio_hash_groups']})")
    a(f"- Duplicate-content groups spanning more than one subject: {d['n_groups_spanning_subjects']}")
    a(f"- Subject+task slots with more than one file: {d['n_duplicate_subject_task_slots']}")
    a("\n## 10. Excel metadata\n")
    md = rep["metadata"]
    a(f"- Files: {len(md['files'])}")
    for f in md["files"]:
        a(f"\n### `{f['file']}` (readable: {f['readable']}, error: {f['error_type']})")
        for sh in f["sheets"]:
            if not sh.get("readable"):
                a(f"- Sheet `{sh['sheet']}`: unreadable ({sh.get('error_type')})")
                continue
            a(f"\n**Sheet `{sh['sheet']}`** -- {sh['n_rows']} rows x {sh['n_columns']} columns "
              f"(fully-empty rows: {sh['n_fully_empty_rows']}, redacted headers: {sh['n_redacted_headers']})\n")
            rows = []
            for col in sh["columns"]:
                summ = ""
                if "numeric_summary" in col:
                    ns = col["numeric_summary"]
                    summ = f"min {ns['min']:.4g}, median {ns['median']:.4g}, max {ns['max']:.4g}"
                elif "value_counts" in col:
                    summ = str(col["value_counts"])
                elif col["sensitive_values_hidden"]:
                    summ = "values hidden"
                rows.append([col["index"], col["name"], col["n_missing"], col["n_unique"], summ])
            a(md_table(["#", "Column", "Missing", "Unique", "Summary (only non-sensitive)"], rows))
    a("\n### Linkage to participants\n")
    lk = md["linkage"]
    a(f"**Join status: {lk['overall_status']}**  participants by status: {lk['participants_by_status']}\n")
    a(md_table(["File", "Sheet", "Linked", "Key column", "Key type", "Matched (any)", "Exactly 1 row",
                "Multiple rows", "Age col(s)", "Sex col(s)"],
               [[sl["file"], sl["sheet"], sl["linked"], sl.get("column_name"), sl.get("key_type"),
                 sl.get("n_participants_matched"), sl.get("n_matched_exactly_one_row"),
                 sl.get("n_matched_multiple_rows"), sl.get("age_columns_found"), sl.get("sex_columns_found")]
                for sl in lk["sheet_links"]]))
    if lk["folder_label_vs_metadata_group"]:
        a("\n**Folder label vs metadata group-like column (counts only; NEEDS HUMAN CHECK):**")
        for ct in lk["folder_label_vs_metadata_group"]:
            a(f"\n`{ct['file']}` / `{ct['sheet']}` / column `{ct['group_column']}`\n")
            a(md_table(["Folder label", "Metadata value", "Participants"],
                       [[x["folder_label"], x["metadata_value"], x["n_participants"]] for x in ct["counts"]]))
    a("\n## 11. Manifest summary\n")
    ms = rep["manifest_summary"]
    a(f"- Rows: {ms['n_rows']} | columns: {ms['columns']}")
    a(f"- Rows with age: {ms['n_with_age']} | rows with sex: {ms['n_with_sex']}")
    a(f"- audio_path column contains: **{ms['audio_path_mode']}**")
    a("\n## 12. What YOU must verify by hand\n")
    for x in rep["human_checks"]:
        a(f"- [ ] {x}")
    a("")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the local Italian PVS dataset (Phase 1).")
    ap.add_argument("--data-root", required=True, help="Folder where the dataset zip was extracted")
    ap.add_argument("--config", default="configs/dataset_italian_pvs.yaml")
    ap.add_argument("--report-dir", default="reports/dataset_verification")
    ap.add_argument("--manifest-out", default="data/interim/manifest.csv")
    ap.add_argument("--excluded-out", default=None,
                    help="Pseudonymous audit manifest for recordings with unparsed/unknown task codes (defaults beside --manifest-out)")
    ap.add_argument("--private-dir", default="data/interim/private",
                    help="Name-bearing lookup files + salt go here. NEVER commit or share this folder.")
    ap.add_argument("--real-paths-in-manifest", action="store_true",
                    help="Put real relative paths (which may contain participant names) in manifest.csv")
    ap.add_argument("--print-tree-depth", type=int, default=0,
                    help="Console-only: print folder names to this depth (may show participant names; do not paste)")
    args = ap.parse_args(argv)
    if args.excluded_out is None:
        args.excluded_out = str(Path(args.manifest_out).with_name("excluded_recordings.csv"))

    root = Path(args.data_root).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: --data-root is not a directory: {root}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg_sha = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    report_dir, private_dir = Path(args.report_dir), Path(args.private_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)

    group_res = {lab: re.compile(rx, re.IGNORECASE) for lab, rx in cfg["groups"].items()}
    known_codes = set(cfg["task_codes"].keys())
    salt, salt_new = load_or_create_salt(private_dir / "subject_id_salt.txt")
    warnings: list[str] = []
    if salt_new:
        warnings.append("A new salt file was created in the private folder. Back it up: without it subject_id values change.")

    print("[1/6] Scanning folder tree ...")
    tree = scan_tree(root)
    if args.print_tree_depth > 0:
        print("--- CONSOLE-ONLY folder listing (may include participant names; do not paste) ---")
        seen = set()
        for parts in tree["dirs"]:
            if len(parts) <= args.print_tree_depth and parts not in seen:
                seen.add(parts)
                print("  " * (len(parts) - 1) + parts[-1])
        print("--- end listing ---")
    print(f"      WAV: {len(tree['wavs'])} | Excel: {len(tree['excel'])} | FILE CODES: {len(tree['filecodes'])} "
          f"| other audio: {len(tree['other_audio'])}")

    # ---- participants from folder structure ----
    participants: dict[tuple[str, str], dict] = {}
    participant_dirs_by_label = defaultdict(set)
    for parts in tree["dirs"]:
        g = detect_group(parts, group_res)
        if g["status"] == "ok" and g["participant_idx"] == len(parts) - 1:
            participant_dirs_by_label[g["label"]].add(g["participant"])
    for lab, names in participant_dirs_by_label.items():
        for nm in names:
            participants[(lab, nm)] = {"label": lab, "folder": nm, "wavs": 0}

    # ---- classify every wav ----
    rec_rows = []
    label_problem = Counter()
    shapes = Counter()
    for rel in tree["wavs"]:
        dir_parts = rel.parts[:-1]
        stem = rel.stem
        shapes[filename_shape(stem)] += 1
        g = detect_group(dir_parts, group_res)
        code, cstatus, cdetail = extract_task_code(stem, known_codes)
        row = {"rel": rel, "stem": stem, "label_status": g["status"], "label": g["label"],
               "participant": g["participant"], "task_code": code, "task_parse_status": cstatus,
               "task_parse_detail": cdetail if cstatus != "ok" else None,
               "nested_below_participant": (g["status"] == "ok" and len(dir_parts) > g["participant_idx"] + 1)}
        label_problem[g["status"]] += 1
        if g["status"] == "ok":
            key = (g["label"], g["participant"])
            if key not in participants:
                participants[key] = {"label": g["label"], "folder": g["participant"], "wavs": 0}
            participants[key]["wavs"] += 1
        rec_rows.append(row)

    # ---- subject ids ----
    plist = []
    for (lab, nm), p in sorted(participants.items(), key=lambda kv: (kv[0][0], strip_accents(kv[0][1]).casefold())):
        p["subject_id"] = pseudo_id("SUBJ", salt, lab, nm)
        p["keys"] = name_keys(nm)
        plist.append(p)
    ids = Counter(p["subject_id"] for p in plist)
    id_collisions = [i for i, n in ids.items() if n > 1]
    full_key_groups = defaultdict(list)
    for p in plist:
        if p["keys"]["full"]:
            full_key_groups[p["keys"]["full"]].append(p)
    cross_group_dups = [v for v in full_key_groups.values() if len({x["label"] for x in v}) > 1]
    same_group_norm_dups = [v for v in full_key_groups.values() if len(v) > 1 and len({x["label"] for x in v}) == 1]
    sid_lookup = {(p["label"], p["folder"]): p["subject_id"] for p in plist}

    # ---- inspect audio (the slow part) ----
    print(f"[2/6] Inspecting {len(rec_rows)} WAV files (read, QC, hash) ...")
    qc = cfg["audio_qc"]
    for i, row in enumerate(rec_rows, 1):
        path = root / row["rel"]
        row["file_sha256"] = sha256_file(path)
        row["audio"] = inspect_audio(path, qc)
        row["subject_id"] = sid_lookup.get((row["label"], row["participant"])) if row["label_status"] == "ok" else None
        row["recording_id"] = pseudo_id("REC", salt, str(row["rel"]).replace("\\", "/"), n=12)
        if i % 100 == 0 or i == len(rec_rows):
            print(f"      {i}/{len(rec_rows)}")

    # ---- FILE CODES cross-check ----
    print("[3/6] Checking FILE CODES file ...")
    fc_infos = [parse_file_codes(root / p) for p in tree["filecodes"]]
    found = set(c for it in fc_infos for c in it["codes_found"])
    fc_check = {"n_files": len(fc_infos), "files": fc_infos,
                "in_file_not_config": sorted(found - known_codes) if found else [],
                "in_config_not_file": sorted(known_codes - found) if found else []}
    if not fc_infos:
        warnings.append("No FILE CODES file found by name; task decoding relied on the config mapping you confirmed.")
    elif not found:
        warnings.append("FILE CODES file found but could not be parsed automatically; cross-check skipped (config mapping used).")
    elif fc_check["in_file_not_config"] or fc_check["in_config_not_file"]:
        warnings.append(f"FILE CODES vs config mismatch: in file only={fc_check['in_file_not_config']}, "
                        f"in config only={fc_check['in_config_not_file']}.")

    # ---- Excel metadata ----
    print("[4/6] Inspecting Excel metadata ...")
    pkeys = [p["keys"] for p in plist]
    excel_results = []
    for i, rel in enumerate(tree["excel"]):
        excel_results.append(inspect_excel(root / rel, cfg["metadata"], pkeys, f"excel_file_{i + 1}:{rel.name}"))
    wav_stem_keys = {name_keys(r["stem"])["full"] for r in rec_rows}
    linkage = build_linkage(excel_results, plist, cfg["metadata"], wav_stem_keys)
    resolved = linkage.pop("resolved")
    for p_idx, p in enumerate(plist):
        r = resolved[p_idx]
        p["age"], p["sex"], p["link_status"] = r["age"], r["sex"], r["status"]
    if not excel_results:
        warnings.append("No Excel metadata file found.")
    elif linkage["overall_status"] != "UNAMBIGUOUS":
        warnings.append(f"Metadata join status is {linkage['overall_status']}: age/sex are left EMPTY for participants "
                        "that could not be linked unambiguously (nothing is guessed).")
    meta_by_sid = {p["subject_id"]: p for p in plist}

    # ---- manifest ----
    print("[5/6] Building manifest and checks ...")
    manifest = []
    for r in rec_rows:
        a = r["audio"]
        p = meta_by_sid.get(r["subject_id"]) if r["subject_id"] else None
        tc = cfg["task_codes"].get(r["task_code"]) if r["task_code"] else None
        logical = f"{r['label'] or 'UNLABELED'}/{r['subject_id'] or 'NO_SUBJECT'}/{r['recording_id']}.wav"
        manifest.append({
            "subject_id": r["subject_id"], "recording_id": r["recording_id"], "label": r["label"],
            "is_pd": (None if r["label"] is None else int(r["label"] == cfg["pd_label"])),
            "age_if_available": p["age"] if p else None, "sex_if_available": p["sex"] if p else None,
            "metadata_link_status": p["link_status"] if p else None,
            "task_code": r["task_code"], "task_name": tc["name"] if tc else None,
            "task_family": tc["family"] if tc else None, "task_parse_status": r["task_parse_status"],
            "audio_path": str(r["rel"]).replace("\\", "/") if args.real_paths_in_manifest else logical,
            "duration_sec": a["duration_sec"], "sample_rate": a["sample_rate"], "channels": a["channels"],
            "num_frames": a["num_frames"], "subtype": a["subtype"], "readable": a["readable"],
            "peak_abs": a["peak_abs"], "rms_dbfs": a["rms_dbfs"], "clip_fraction": a["clip_fraction"],
            "silence_fraction": a["silence_fraction"], "qc_flags": ";".join(a["flags"]),
            "file_sha256": r["file_sha256"], "audio_sha256": a["audio_sha256"],
        })
    mdf = pd.DataFrame(manifest)
    mdf = mdf.sort_values(["label", "subject_id", "task_code", "recording_id"], na_position="last").reset_index(drop=True)
    mdf.to_csv(args.manifest_out, index=False)
    # Option B: preserve every recording in the full manifest, while explicitly
    # quarantining recordings whose task code could not be confidently parsed.
    excluded = mdf[mdf["task_parse_status"] != "ok"].copy()
    if len(excluded):
        excluded["exclusion_reason"] = excluded["task_parse_status"].map({
            "unknown_code": "Task code is not present in the confirmed FILE CODES mapping; do not guess.",
            "ambiguous_code": "Filename contains multiple plausible known task codes.",
            "no_code": "No recognised task code could be extracted from filename.",
        }).fillna("Task code could not be confidently parsed.")
    else:
        excluded["exclusion_reason"] = pd.Series(dtype=str)
    Path(args.excluded_out).parent.mkdir(parents=True, exist_ok=True)
    excluded.to_csv(args.excluded_out, index=False)
    pd.DataFrame([{"subject_id": p["subject_id"], "label": p["label"], "participant_folder_name": p["folder"]}
                  for p in plist]).to_csv(private_dir / "subject_key_PRIVATE.csv", index=False)
    pd.DataFrame([{"recording_id": r["recording_id"], "relative_path": str(r["rel"]).replace("\\", "/")}
                  for r in rec_rows]).to_csv(private_dir / "path_lookup_PRIVATE.csv", index=False)

    # ---- duplicates ----
    def dup_groups(field):
        g = defaultdict(list)
        for r in rec_rows:
            h = r["file_sha256"] if field == "file" else r["audio"]["audio_sha256"]
            if h:
                g[h].append(r)
        return [v for v in g.values() if len(v) > 1]
    fdup, adup = dup_groups("file"), dup_groups("audio")
    spanning = [g for g in fdup + adup if len({x["subject_id"] for x in g}) > 1]
    slot = Counter((r["subject_id"], r["task_code"]) for r in rec_rows if r["subject_id"] and r["task_code"])
    dup_slots = {k: n for k, n in slot.items() if n > 1}
    rec_id_dups = [k for k, n in Counter(r["recording_id"] for r in rec_rows).items() if n > 1]

    # ---- hard checks ----
    hard = []

    def check(cid, ok, n, details):
        hard.append({"id": cid, "status": "PASS" if ok else "FAIL", "n_affected": int(n), "details": details})

    check("H0_wav_files_found", len(rec_rows) > 0, 0 if rec_rows else 1, f"{len(rec_rows)} WAV files found")
    n_nolabel = label_problem["no_label"] + label_problem["ambiguous_label"]
    check("H1_every_wav_has_exactly_one_label", n_nolabel == 0, n_nolabel,
          f"label status counts: {dict(label_problem)}. Fix group regexes in the config if wrong.")
    n_noparticipant = label_problem["no_participant"]
    n_ident = n_noparticipant + len(id_collisions) + len(cross_group_dups) + len(same_group_norm_dups)
    check("H2_subject_identity_unambiguous", n_ident == 0, n_ident,
          f"WAVs not inside a participant folder: {n_noparticipant}; subject_id hash collisions: {len(id_collisions)}; "
          f"participant names repeated across groups: {len(cross_group_dups)}; "
          f"different folders with the same normalised name in one group: {len(same_group_norm_dups)}")
    n_leak = len(cross_group_dups) + len(spanning)
    check("H3_subject_leakage_risk", n_leak == 0, n_leak,
          f"same name in different groups: {len(cross_group_dups)}; identical audio shared by different subjects: {len(spanning)}. "
          "Cannot detect one person stored under two differently named folders -> verify by hand.")
    n_unread = sum(1 for r in rec_rows if not r["audio"]["readable"] or "empty" in r["audio"]["flags"])
    check("H4_all_audio_readable_and_non_empty", n_unread == 0, n_unread, f"{n_unread} unreadable or empty WAVs")
    parse_counts = Counter(r["task_parse_status"] for r in rec_rows)
    n_badcode = sum(parse_counts.values()) - parse_counts["ok"]
    # Unknown codes are intentionally quarantined under Option B. Missing or
    # ambiguous codes still need resolution before the manifest is trusted.
    n_unresolved = parse_counts["ambiguous_code"] + parse_counts["no_code"]
    check("H5_task_codes_classified_or_quarantined", n_unresolved == 0, n_unresolved,
          f"task parse status counts: {dict(parse_counts)}; unknown_code rows are written to the exclusion manifest")
    # Multiple distinct takes in a subject/task slot are not duplicates by
    # themselves. H6 checks identical content only; repeated slots are warnings.
    n_dup = len(fdup) + len(adup)
    check("H6_no_duplicate_recording_content", n_dup == 0, n_dup,
          f"byte-identical groups: {len(fdup)}; identical-audio groups: {len(adup)}; "
          f"subject+task slots with >1 file (review only): {len(dup_slots)}")
    check("H7_recording_ids_unique", not rec_id_dups, len(rec_id_dups), f"{len(rec_id_dups)} repeated recording_id values")

    # ---- warnings ----
    if parse_counts["unknown_code"]:
        warnings.append(f"{parse_counts['unknown_code']} recordings have unknown task codes and were preserved in the full manifest but written to the exclusion manifest; no codes were guessed.")
    if dup_slots:
        warnings.append(f"{len(dup_slots)} subject+task slots contain multiple recordings. These are retained and listed for review; only identical content triggers H6.")
    if tree["other_audio"]:
        warnings.append(f"{len(tree['other_audio'])} audio files in non-WAV formats were found and IGNORED.")
    if tree["other_ext"]:
        warnings.append(f"Other file types present (ignored): {dict(tree['other_ext'])}")
    no_wav = [p for p in plist if p["wavs"] == 0]
    if no_wav:
        warnings.append(f"{len(no_wav)} participant folders contain no WAV files.")
    nested = sum(1 for r in rec_rows if r["nested_below_participant"])
    if nested:
        warnings.append(f"{nested} WAVs sit in sub-folders below the participant folder (assigned to that participant).")
    by_label_subjects = Counter(p["label"] for p in plist)
    for lab, exp in cfg.get("user_confirmed_subject_counts", {}).items():
        if by_label_subjects.get(lab, 0) != exp:
            warnings.append(f"Participant-folder count for '{lab}' is {by_label_subjects.get(lab, 0)} but you counted {exp} by hand.")
    sr_counts = Counter(r["audio"]["sample_rate"] for r in rec_rows if r["audio"]["readable"])
    if len(sr_counts) > 1:
        warnings.append(f"More than one sample rate present: {dict(sr_counts)} (resampling policy is a Phase 8 decision).")
    ch_counts = Counter(r["audio"]["channels"] for r in rec_rows if r["audio"]["readable"])
    if len(ch_counts) > 1 or (ch_counts and 1 not in ch_counts):
        warnings.append(f"Channel layout: {dict(ch_counts)} (mono conversion policy is a Phase 8 decision).")
    flag_counts = Counter(f for r in rec_rows for f in r["audio"]["flags"])
    for f_, n_ in flag_counts.items():
        if f_ not in {"unreadable", "empty"}:
            warnings.append(f"QC flag '{f_}' on {n_} files (heuristic; inspect before Phase 6 decisions).")
    # recording-condition shortcut proxy: sample rate / channels / subtype vs label
    for fld in ("sample_rate", "channels", "subtype"):
        tab = defaultdict(set)
        for r in rec_rows:
            if r["audio"]["readable"] and r["label"]:
                tab[r["label"]].add(r["audio"][fld])
        if len(tab) > 1 and len({frozenset(v) for v in tab.values()}) > 1:
            warnings.append(f"SHORTCUT RISK: '{fld}' values differ between label groups "
                            f"({ {k: sorted(map(str, v)) for k, v in tab.items()} }). A model could learn the recording setup.")

    # ---- report assembly ----
    labels_order = list(cfg["groups"].keys())
    by_label = {}
    for lab in labels_order:
        recs = [r for r in rec_rows if r["label"] == lab]
        by_label[lab] = {"participant_folders": len(participant_dirs_by_label.get(lab, [])),
                         "subjects_with_audio": len({r["subject_id"] for r in recs if r["subject_id"]}),
                         "recordings": len(recs),
                         "user_confirmed": cfg.get("user_confirmed_subject_counts", {}).get(lab)}
    per_subject = Counter(r["subject_id"] for r in rec_rows if r["subject_id"])
    sid_label = {p["subject_id"]: p["label"] for p in plist}
    rps_by_group = {lab: stats([n for s, n in per_subject.items() if sid_label.get(s) == lab]) for lab in labels_order}
    rps_hist = dict(sorted(Counter(per_subject.values()).items()))
    task_rows = []
    for code, tc in cfg["task_codes"].items():
        recs = [r for r in rec_rows if r["task_code"] == code]
        task_rows.append({"code": code, "name": tc["name"], "total": len(recs),
                          "by_label": {lab: sum(1 for r in recs if r["label"] == lab) for lab in labels_order},
                          "n_subjects": len({r["subject_id"] for r in recs if r["subject_id"]})})
    codes_per_subject = defaultdict(set)
    for r in rec_rows:
        if r["subject_id"] and r["task_code"]:
            codes_per_subject[r["subject_id"]].add(r["task_code"])
    cov = dict(sorted(Counter(len(v) for v in codes_per_subject.values()).items()))
    readable = [r for r in rec_rows if r["audio"]["readable"]]
    dur_by_task = {code: stats([r["audio"]["duration_sec"] for r in readable if r["task_code"] == code])
                   for code in cfg["task_codes"]}
    n_with_age = int(mdf["age_if_available"].notna().sum()) if len(mdf) else 0
    n_with_sex = int(mdf["sex_if_available"].notna().sum()) if len(mdf) else 0
    human = [
        "Open 3 or 4 participant folders at random and confirm each folder is ONE person (not a family folder or a mix of people).",
        "Confirm no person appears in two folders under different spellings (the script can only detect identical names / identical audio).",
        "Check the label-vs-metadata cross-tab in section 10 (if shown): do folder labels agree with the Excel group column?",
        "Confirm the Excel columns used for age/sex (section 10) really are age and sex, and that the sex coding is what you expect.",
        "Check the filename shapes (section 6) look like what you see in the file explorer; if many files failed task decoding, send me 5 name-REDACTED example filenames.",
        "Listen to 2 or 3 WAVs flagged by QC (if any) and to one WAV per task type to confirm the task code matches what is spoken.",
        "Confirm the sample-rate / channel summary matches the properties you saw earlier in your file explorer.",
        "Keep the private folder out of Git and out of any chat or upload.",
    ]
    report = {
        "meta": {"script_version": SCRIPT_VERSION, "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "git_commit": git_commit(), "python": sys.version.split()[0], "platform": platform.platform(),
                 "numpy": np.__version__, "pandas": pd.__version__, "soundfile": sf.__version__,
                 "libsndfile": sf.__libsndfile_version__, "config_sha256": cfg_sha, "data_root": "<not recorded>"},
        "all_hard_checks_passed": all(c["status"] == "PASS" for c in hard),
        "hard_checks": hard, "warnings": warnings,
        "counts": {"n_wav_files": len(rec_rows), "n_participant_folders": sum(len(v) for v in participant_dirs_by_label.values()),
                   "n_subjects_with_audio": len(per_subject), "n_other_audio_files": len(tree["other_audio"]),
                   "n_excel_files": len(tree["excel"]), "n_file_codes_files": len(tree["filecodes"]),
                   "other_extensions": dict(tree["other_ext"]), "by_label": by_label},
        "recordings_per_subject": {"by_group": rps_by_group, "histogram": rps_hist},
        "recordings_per_task": {"by_code": task_rows, "n_unparsed": n_badcode,
                                "parse_status_counts": dict(Counter(r["task_parse_status"] for r in rec_rows)),
                                "unknown_code_tokens": dict(Counter(r["task_parse_detail"] for r in rec_rows if r["task_parse_status"] == "unknown_code")),
                                "n_expected_codes": len(known_codes), "subjects_by_codes_covered": cov},
        "filename_shapes_top": Counter(shapes).most_common(15),
        "file_codes_check": fc_check,
        "audio": {"n_readable": len(readable), "n_unreadable": sum(1 for r in rec_rows if not r["audio"]["readable"]),
                  "n_empty": sum(1 for r in rec_rows if "empty" in r["audio"]["flags"]),
                  "sample_rate_counts": {str(k): v for k, v in sr_counts.items()},
                  "channel_counts": {str(k): v for k, v in ch_counts.items()},
                  "subtype_counts": dict(Counter(r["audio"]["subtype"] for r in readable)),
                  "flag_counts": dict(flag_counts),
                  "duration_overall": stats([r["audio"]["duration_sec"] for r in readable]),
                  "duration_by_task": dur_by_task},
        "duplicates": {"n_file_hash_groups": len(fdup), "n_files_in_file_hash_groups": sum(len(g) for g in fdup),
                       "n_audio_hash_groups": len(adup), "n_files_in_audio_hash_groups": sum(len(g) for g in adup),
                       "n_groups_spanning_subjects": len(spanning), "n_duplicate_subject_task_slots": len(dup_slots)},
        "metadata": {"files": [{**{k: v for k, v in er.items() if k != "sheets"},
                                "sheets": [{k: v for k, v in sh.items() if k != "_df"} for sh in er["sheets"]]}
                               for er in excel_results], "linkage": linkage},
        "manifest_summary": {"n_rows": int(len(mdf)), "columns": list(mdf.columns), "n_with_age": n_with_age,
                             "n_with_sex": n_with_sex,
                             "audio_path_mode": "REAL relative paths (may contain names)" if args.real_paths_in_manifest
                             else "pseudonymous logical paths (real paths only in private/path_lookup_PRIVATE.csv)"},
        "human_checks": human,
    }
    report = clean_nan(report)
    (report_dir / "dataset_report.json").write_text(json.dumps(report, indent=2, default=jsonable), encoding="utf-8")
    (report_dir / "dataset_report.md").write_text(render_markdown(report), encoding="utf-8")

    # ---- console summary ----
    print("[6/6] Done.\n")
    print("HARD-FAIL CHECKS")
    for c in hard:
        print(f"  {c['status']:4}  {c['id']}  (affected: {c['n_affected']})")
    print(f"\nWAV files: {len(rec_rows)} | subjects with audio: {len(per_subject)}")
    for lab, d in by_label.items():
        print(f"  {lab}: folders={d['participant_folders']} subjects={d['subjects_with_audio']} recordings={d['recordings']}")
    print(f"Metadata join status: {linkage['overall_status']}")
    print(f"Warnings: {len(warnings)}")
    print(f"\nWrote:\n  {args.manifest_out}\n  {args.excluded_out} (task-code exclusions)\n  {report_dir / 'dataset_report.json'}\n  {report_dir / 'dataset_report.md'}\n"
          f"  {private_dir}/ (PRIVATE: contains names; never commit or share)")
    ok = report["all_hard_checks_passed"]
    print("\nRESULT:", "ALL HARD CHECKS PASSED" if ok else "HARD CHECK FAILURE -> do not proceed; read dataset_report.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
