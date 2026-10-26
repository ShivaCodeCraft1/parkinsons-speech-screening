import re

import pytest

from src.data.verification import detect_group, extract_task_code, pseudo_id

KNOWN = {"B1", "B2", "D1", "D2", "FB1", "FB2", "VA1", "VA2", "VE1", "VE2", "VI1", "VI2", "VO1", "VO2", "VU1", "VU2"}
GROUPS = {
    "young_healthy_control": re.compile("young", re.I),
    "elderly_healthy_control": re.compile("elder", re.I),
    "parkinsons_disease": re.compile("people with parkinson|parkinson.?s? disease", re.I),
}


@pytest.mark.parametrize("stem,code", [
    ("VA1", "VA1"), ("VA1 Some Name", "VA1"), ("va2_x", "VA2"), ("FB1_x", "FB1"),   # FB1 must not be read as B1
    ("B1", "B1"), ("xyzFB2", "FB2"), ("D2-foo", "D2"),
])
def test_known_codes_decoded(stem, code):
    got, status, _ = extract_task_code(stem, KNOWN)
    assert (got, status) == (code, "ok")


def test_unknown_code_is_reported_not_guessed():
    code, status, detail = extract_task_code("ZZ9 name", KNOWN)
    assert code is None and status == "unknown_code" and "ZZ9" in detail
    code, status, _ = extract_task_code("VA10 name", KNOWN)       # VA10 must not be silently read as VA1
    assert code is None and status == "unknown_code"


def test_no_code_and_ambiguous():
    assert extract_task_code("recording", KNOWN)[1] == "no_code"
    assert extract_task_code("VA1 VE1", KNOWN)[1] == "ambiguous_code"


def test_group_detection_ignores_wrapper_folder():
    g = detect_group(("Italian Parkinson's Voice and Speech", "15 Young Healthy Control", "P"), GROUPS)
    assert (g["status"], g["label"], g["participant"]) == ("ok", "young_healthy_control", "P")


def test_group_detection_failure_modes():
    assert detect_group(("nothing",), GROUPS)["status"] == "no_label"
    assert detect_group(("28 People with Parkinson's disease",), GROUPS)["status"] == "no_participant"
    assert detect_group(("young", "elder p"), GROUPS)["status"] == "ambiguous_label"


def test_pseudo_id_is_stable_and_salt_dependent():
    assert pseudo_id("SUBJ", "salt", "g", "n") == pseudo_id("SUBJ", "salt", "g", "n")
    assert pseudo_id("SUBJ", "salt1", "g", "n") != pseudo_id("SUBJ", "salt2", "g", "n")


def test_pd_range_folder_resolves_real_participant_not_range():
    g = detect_group(("28 People with Parkinson's disease", "1-5", "Participant_03"), GROUPS)
    assert g["status"] == "ok"
    assert g["label"] == "parkinsons_disease"
    assert g["participant"] == "Participant_03"
    assert g["participant_idx"] == 2


def test_pd_range_folder_without_participant_is_not_a_subject():
    g = detect_group(("28 People with Parkinson's disease", "1-5"), GROUPS)
    assert g["status"] == "no_participant"
