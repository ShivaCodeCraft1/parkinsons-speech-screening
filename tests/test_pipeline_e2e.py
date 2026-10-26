"""End-to-end software tests on SYNTHETIC data (see conftest.e2e). They verify wiring, not medical performance."""
import numpy as np
import pandas as pd
import pytest

from src.data.splits import assert_no_subject_overlap
from src.evaluation.metrics import aggregate_by_subject, cluster_bootstrap, full_metrics, youden_threshold
from src.explainability.explain import explain_one, global_importance
from src.models.inference import analyze_audio_bytes, predict_from_analysis
from src.models.registry import ModelNotAvailable, load_artifacts
from src.models.training import score_with
from src.validation.features import features_to_row


def test_verifier_ran_and_manifest_has_expected_subjects(e2e):
    m = pd.read_csv(e2e["manifest"])
    assert m["subject_id"].nunique() == 24 and set(m["label"].dropna()) == {"young_healthy_control", "elderly_healthy_control", "parkinsons_disease"}
    assert m["task_family"].eq("sustained_vowel").all()


def test_training_split_is_subject_disjoint_and_threshold_comes_from_train_only(e2e):
    assert_no_subject_overlap(e2e["train"], e2e["test"])
    oof = e2e["res"]["oof_train"]
    assert set(oof.subject_id) == set(e2e["train"].subject_id)          # threshold derived from training subjects only
    assert not set(oof.subject_id) & set(e2e["test"].subject_id)
    assert np.isclose(e2e["res"]["threshold"], youden_threshold(oof.is_pd, oof.oof_score))


def test_saved_model_reloads_and_matches_in_memory_model(e2e):
    b = load_artifacts(e2e["model_dir"])
    names = b["schema"]["feature_names"]
    assert names == e2e["schema"]["feature_names"]
    X = e2e["test"][names].to_numpy(float)
    assert np.allclose(score_with(b["pipeline"], X), score_with(e2e["res"]["pipeline"], X))


def test_missing_model_artifacts_raise_clear_error(tmp_path):
    with pytest.raises(ModelNotAvailable) as e:
        load_artifacts(tmp_path / "nope")
    assert "scripts/train.py" in str(e.value)


def test_inference_features_equal_training_features(e2e):
    """Same audio through the API code path must reproduce the training-table features exactly (same code, same order)."""
    r = e2e["test"].iloc[0]
    lk = pd.read_csv(e2e["manifest"].parent / "private" / "path_lookup_PRIVATE.csv").set_index("recording_id")["relative_path"]
    path = e2e["root"] / lk[r.recording_id]
    a = analyze_audio_bytes(path.read_bytes())
    names = e2e["schema"]["feature_names"]
    assert list(a["features"]) == names
    np.testing.assert_allclose(features_to_row(a["features"], names)[0], r[names].to_numpy(float), equal_nan=True, rtol=1e-9)


def test_prediction_structure(e2e):
    b = load_artifacts(e2e["model_dir"])
    from tests.synthetic_pipeline_data import synth_voice, wav_bytes
    a = analyze_audio_bytes(wav_bytes(synth_voice(np.random.default_rng(9), 150, 0.01, 0.05, 0.02)))
    out = predict_from_analysis(a, b)
    assert out["prediction"] in {"above_threshold", "below_threshold"} and np.isfinite(out["score"])
    assert out["model_version"] == "test-synthetic-0" and any(w["code"] == "domain_shift" for w in out["warnings"])


def test_metrics_and_bootstrap_on_heldout(e2e):
    names, thr = e2e["schema"]["feature_names"], e2e["res"]["threshold"]
    te = e2e["test"].copy()
    te["score"] = score_with(e2e["res"]["pipeline"], te[names].to_numpy(float))
    m = full_metrics(te.is_pd, te.score, thr)
    assert 0 <= m["sensitivity"] <= 1 and 0 <= m["specificity"] <= 1 and m["tp"] + m["fn"] == int(te.is_pd.sum())
    subj = aggregate_by_subject(te)
    assert len(subj) == te.subject_id.nunique()
    ci = cluster_bootstrap(te, thr, 50, 0.95, 0)
    assert ci["auroc"]["low"] <= ci["auroc"]["high"]


def test_explainability_runs_for_selected_model(e2e):
    names = e2e["schema"]["feature_names"]
    te = e2e["test"]
    imp = global_importance(e2e["res"]["pipeline"], te, te.is_pd.to_numpy(), names, seed=0, max_rows=50)
    assert "permutation" in imp and len(imp["permutation"]) == len(names)
    one = explain_one(e2e["res"]["pipeline"], te[names].to_numpy(float)[:1], names)
    assert one is not None and len(one["top_features"]) > 0
