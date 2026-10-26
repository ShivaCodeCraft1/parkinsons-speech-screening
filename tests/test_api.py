"""API tests (FastAPI TestClient). Infrastructure only: no private dataset needed; the model is the synthetic test model."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from tests.synthetic_pipeline_data import synth_voice, wav_bytes


@pytest.fixture
def client(e2e):
    return TestClient(create_app(e2e["model_dir"]))


@pytest.fixture
def no_model_client(tmp_path):
    return TestClient(create_app(tmp_path / "empty"))


def good_wav(seed=0, dur=2.0):
    return wav_bytes(synth_voice(np.random.default_rng(seed), 150, 0.01, 0.05, 0.02, dur=dur))


def post(c, path, data, name="a.wav", ctype="audio/wav"):
    return c.post(path, files={"file": (name, data, ctype)})


def test_health_with_and_without_model(client, no_model_client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] is True and h["model_version"] == "test-synthetic-0"
    h2 = no_model_client.get("/health").json()
    assert h2["model_loaded"] is False and "train" in h2["detail"].lower()


def test_model_info(client):
    j = client.get("/model-info").json()
    assert j["n_features"] == len(j["feature_names"]) == 50 and j["score_is_calibrated_probability"] is False
    assert "SYNTHETIC" in j["evaluation_status"] and "disclaimer" in j


def test_missing_model_is_an_error_not_a_prediction(no_model_client):
    for r in (no_model_client.get("/model-info"), post(no_model_client, "/predict", good_wav())):
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_not_available" and "result" not in r.json()


def test_predict_returns_real_structure(client):
    r = post(client, "/predict", good_wav())
    assert r.status_code == 200, r.text
    j = r.json()
    res = j["result"]
    assert res["prediction"] in {"above_threshold", "below_threshold"} and isinstance(res["score"], float)
    assert res["threshold"] == pytest.approx(client.get("/model-info").json()["threshold"])
    assert len(j["analysis"]["features"]) == 50 and j["analysis"]["waveform"]["min"] and j["analysis"]["spectrogram"]["db"]
    assert "not a medical device" in j["disclaimer"].lower()
    assert any(w["code"] == "domain_shift" for w in res["warnings"])


def test_predict_is_deterministic_and_depends_on_audio(client):
    a = post(client, "/predict", good_wav(0)).json()["result"]["score"]
    a2 = post(client, "/predict", good_wav(0)).json()["result"]["score"]
    other = wav_bytes(synth_voice(np.random.default_rng(5), 140, 0.03, 0.15, 0.08))
    b = post(client, "/predict", other).json()["result"]["score"]
    assert a == a2 and a != b


def test_predict_with_explanation(client):
    j = post(client, "/predict?explain=true", good_wav()).json()
    assert j["result"]["explanation"]["top_features"]


def test_analyze_works_without_model(no_model_client):
    r = post(no_model_client, "/analyze", good_wav())
    assert r.status_code == 200 and r.json()["validation"]["ok"] and len(r.json()["features"]) == 50


def test_invalid_uploads(client):
    assert post(client, "/predict", b"").status_code == 400
    r = post(client, "/predict", b"not audio at all" * 100, "x.txt", "text/plain")
    assert r.status_code in (415, 422) and r.json()["error"]["code"] in {"unsupported_format", "undecodable_audio"}
    short = post(client, "/predict", wav_bytes(synth_voice(np.random.default_rng(1), 150, 0.01, 0.05, 0.02, dur=0.3)))
    assert short.status_code == 422 and short.json()["error"]["code"] == "too_short"
    silent = post(client, "/predict", wav_bytes(np.zeros(32000, np.float32)))
    assert silent.status_code == 422 and silent.json()["error"]["code"] == "silent_or_constant"
    assert client.post("/predict").status_code == 422   # no file field


def test_oversized_upload_rejected(client):
    r = post(client, "/predict", b"\0" * 21_000_000)
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_demo_mode_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("PD_ENABLE_DEMO_MODE", raising=False)
    assert client.get("/demo/samples").status_code == 404


def test_metrics_endpoint_counts_requests(client):
    client.get("/health"); post(client, "/predict", b"")
    m = client.get("/metrics").json()
    assert m["request_count"] >= 2 and m["error_count"] >= 1 and "model_version" in m


def test_frontend_is_served_and_calls_documented_endpoints(client):
    r = client.get("/app/")
    assert r.status_code == 200 and "Research prototype" in r.text
    js = client.get("/app/app.js").text
    for path in ("/health", "/model-info", "/predict", "/analyze"):
        assert path in js
    assert client.get("/docs").status_code == 200 and client.get("/openapi.json").status_code == 200
