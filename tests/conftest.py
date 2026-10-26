from pathlib import Path

import pytest

from tests.make_synthetic_dataset import build_synthetic_dataset

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "dataset_italian_pvs.yaml"


@pytest.fixture
def clean_root(tmp_path):
    return build_synthetic_dataset(tmp_path / "clean", inject_faults=False)


@pytest.fixture
def bad_root(tmp_path):
    return build_synthetic_dataset(tmp_path / "bad", inject_faults=True)


# ---------------------------------------------------------------------------------------------
# End-to-end fixture on SYNTHETIC audio: verifier -> features -> grouped training -> saved artifacts.
# Proves the software wiring only; the synthetic voices carry no medical meaning.
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def e2e(tmp_path_factory):
    import copy

    import pandas as pd

    from src.config import FEATURES_CONFIG, TRAINING_CONFIG, load_yaml
    from src.data.dataset import resolve_paths, usable_recordings
    from src.data.splits import subject_holdout_split
    from src.data.verification import main as verify_main
    from src.features.extract import feature_schema
    from src.models.inference import analyze_audio_bytes
    from src.models.registry import save_artifacts
    from src.models.training import select_and_fit
    from tests.synthetic_pipeline_data import build_larger_synthetic_dataset

    work = tmp_path_factory.mktemp("e2e")
    root = build_larger_synthetic_dataset(work / "data", per_group=8)
    manifest, private = work / "manifest.csv", work / "private"
    code = verify_main(["--data-root", str(root), "--config", str(CONFIG), "--report-dir", str(work / "report"),
                        "--manifest-out", str(manifest), "--private-dir", str(private)])
    m = usable_recordings(pd.read_csv(manifest), ["sustained_vowel"])
    m = resolve_paths(m, root, private)
    fcfg, tcfg = load_yaml(FEATURES_CONFIG), copy.deepcopy(load_yaml(TRAINING_CONFIG))
    tcfg["models"] = {"logistic_regression": {"grid": {"clf__C": [0.1, 1.0]}},
                      "random_forest": {"grid": {"clf__n_estimators": [50], "clf__max_depth": [3]}}}
    tcfg["split"]["inner_cv_folds"] = 3
    rows = []
    for r in m.itertuples():
        a = analyze_audio_bytes(Path(r.abs_path).read_bytes(), fcfg, with_visuals=False)
        rows.append({"recording_id": r.recording_id, **a["features"]})
    cols = ["recording_id", "subject_id", "label", "is_pd", "task_family", "age_if_available", "sex_if_available", "task_code"]
    df = m[cols].merge(pd.DataFrame(rows), on="recording_id")
    schema = feature_schema(fcfg)
    train, test = subject_holdout_split(df, 0.25, tcfg["seed"])
    res = select_and_fit(train, schema["feature_names"], tcfg)
    model_dir = work / "model"
    meta = {"model_version": "test-synthetic-0", "model_name": res["model_name"], "score_type": res["score_type"], "threshold": res["threshold"],
            "threshold_method": "youden_oof", "positive_class": "parkinsons_disease", "task_families": ["sustained_vowel"],
            "evaluation_status": "SYNTHETIC TEST MODEL - no medical meaning", "training_data": {"n_subjects": int(train.subject_id.nunique())}}
    save_artifacts(model_dir, res["pipeline"], schema, meta)
    return {"root": root, "verify_code": code, "manifest": manifest, "df": df, "train": train, "test": test, "res": res,
            "model_dir": model_dir, "schema": schema, "tcfg": tcfg}
