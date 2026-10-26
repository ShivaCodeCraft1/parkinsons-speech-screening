"""End-to-end tests of the verifier on SYNTHETIC data. Counts below are known by construction, not real."""
import json
import shutil
import subprocess
import sys

import pandas as pd

from src.data import verification as v
from tests.conftest import CONFIG, REPO
from tests.make_synthetic_dataset import FAKE_NAME_WORDS


def run(root, out, private):
    code = v.main(["--data-root", str(root), "--config", str(CONFIG), "--report-dir", str(out / "reports"),
                   "--manifest-out", str(out / "manifest.csv"), "--private-dir", str(private)])
    report = json.loads((out / "reports" / "dataset_report.json").read_text())
    return code, report, pd.read_csv(out / "manifest.csv")


def test_clean_dataset_passes_all_hard_checks(clean_root, tmp_path):
    code, rep, man = run(clean_root, tmp_path / "o", tmp_path / "p")
    assert code == 0 and rep["all_hard_checks_passed"]
    assert rep["counts"]["n_wav_files"] == 160 and rep["counts"]["n_subjects_with_audio"] == 10
    assert {k: d["subjects_with_audio"] for k, d in rep["counts"]["by_label"].items()} == {
        "young_healthy_control": 3, "elderly_healthy_control": 3, "parkinsons_disease": 4}
    assert len(man) == 160 and man["recording_id"].is_unique
    assert (man.groupby("subject_id")["label"].nunique() == 1).all()          # one label per subject
    assert (man.groupby("subject_id")["task_code"].nunique() == 16).all()


def test_manifest_has_required_columns(clean_root, tmp_path):
    _, _, man = run(clean_root, tmp_path / "o", tmp_path / "p")
    required = {"subject_id", "recording_id", "label", "age_if_available", "sex_if_available", "task_code",
                "task_name", "audio_path", "duration_sec", "sample_rate"}
    assert required <= set(man.columns)


def test_no_participant_names_in_reports_or_manifest(clean_root, tmp_path):
    run(clean_root, tmp_path / "o", tmp_path / "p")
    text = "\n".join(p.read_text(errors="ignore") for p in (tmp_path / "o").rglob("*") if p.is_file())
    for word in FAKE_NAME_WORDS:
        assert word not in text, f"name leaked: {word}"
    private = (tmp_path / "p" / "subject_key_PRIVATE.csv").read_text()
    assert "Bianchi" in private                                              # names live ONLY in the private folder


def test_faulty_dataset_fails_expected_checks(bad_root, tmp_path):
    code, rep, _ = run(bad_root, tmp_path / "o", tmp_path / "p")
    failed = {c["id"] for c in rep["hard_checks"] if c["status"] == "FAIL"}
    assert code == 1 and not rep["all_hard_checks_passed"]
    assert failed == {"H3_subject_leakage_risk", "H4_all_audio_readable_and_non_empty",
                      "H6_no_duplicate_recording_content"}
    # Unknown codes are quarantined, not guessed, and do not block otherwise-safe verification.
    excluded = pd.read_csv(tmp_path / "o" / "excluded_recordings.csv")
    assert (excluded["task_parse_status"] == "unknown_code").any()
    assert excluded["exclusion_reason"].notna().all()


def test_same_person_name_in_two_groups_fails_identity_and_leakage(clean_root, tmp_path):
    src = clean_root / "3 Young Healthy Control" / "Anna Bianchi"
    shutil.copytree(src, clean_root / "4 People with Parkinson's disease" / "Anna Bianchi")
    code, rep, _ = run(clean_root, tmp_path / "o", tmp_path / "p")
    failed = {c["id"] for c in rep["hard_checks"] if c["status"] == "FAIL"}
    assert code == 1 and {"H2_subject_identity_unambiguous", "H3_subject_leakage_risk"} <= failed


def test_subject_ids_stable_with_same_salt_and_change_with_new_salt(clean_root, tmp_path):
    _, _, m1 = run(clean_root, tmp_path / "o1", tmp_path / "p")
    _, _, m2 = run(clean_root, tmp_path / "o2", tmp_path / "p")
    _, _, m3 = run(clean_root, tmp_path / "o3", tmp_path / "p_other")
    assert set(m1.subject_id) == set(m2.subject_id)
    assert set(m1.subject_id).isdisjoint(set(m3.subject_id))


def test_missing_metadata_leaves_age_and_sex_empty_never_guessed(clean_root, tmp_path):
    next(clean_root.glob("*.xlsx")).unlink()
    code, rep, man = run(clean_root, tmp_path / "o", tmp_path / "p")
    assert man["age_if_available"].isna().all() and man["sex_if_available"].isna().all()
    assert rep["metadata"]["linkage"]["overall_status"] == "NOT_JOINABLE"
    assert any("No Excel metadata" in w for w in rep["warnings"])


def test_unlabelled_folder_fails_label_check(clean_root, tmp_path):
    (clean_root / "stray folder").mkdir()
    shutil.copy(next((clean_root / "3 Young Healthy Control" / "Anna Bianchi").glob("VA1*.wav")),
                clean_root / "stray folder" / "VA1 x.wav")
    code, rep, _ = run(clean_root, tmp_path / "o", tmp_path / "p")
    assert code == 1
    assert {c["id"]: c["status"] for c in rep["hard_checks"]}["H1_every_wav_has_exactly_one_label"] == "FAIL"


def test_cli_wrapper_runs_and_returns_exit_code(clean_root, tmp_path):
    cmd = [sys.executable, str(REPO / "scripts" / "verify_dataset.py"), "--data-root", str(clean_root),
           "--config", str(CONFIG), "--report-dir", str(tmp_path / "r"), "--manifest-out", str(tmp_path / "m.csv"),
           "--private-dir", str(tmp_path / "p")]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def test_nonexistent_data_root_exits_2(tmp_path):
    assert v.main(["--data-root", str(tmp_path / "nope"), "--config", str(CONFIG)]) == 2
