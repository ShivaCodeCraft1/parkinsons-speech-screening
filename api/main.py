"""FastAPI application. Run:  uvicorn api.main:app --reload     (docs at /docs, web UI at /app/)"""
from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    DISCLAIMER,
    AnalysisResponse,
    DemoPredictResponse,
    DemoSample,
    ErrorResponse,
    HealthResponse,
    PredictResponse,
)
from src.audio.loading import AudioError
from src.config import REPO_ROOT, get_model_dir, load_dotenv_if_present
from src.explainability.explain import explain_one
from src.models.inference import analyze_audio_bytes, load_feature_config, predict_from_analysis
from src.models.registry import ModelNotAvailable, load_artifacts
from src.utils.logging_utils import setup_logging
from src.utils.metrics import Metrics
from src.validation.features import features_to_row

log = logging.getLogger("api")
STATUS_FOR_CODE = {"unsupported_format": 415, "empty_file": 400, "file_too_large": 413}


def _clean(o):
    """NaN/Inf are not valid JSON -> None."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def create_app(model_dir: Path | None = None) -> FastAPI:
    load_dotenv_if_present()
    setup_logging()
    cfg = load_feature_config()
    max_bytes = int(cfg["audio"]["max_file_bytes"])
    mdir = Path(model_dir) if model_dir else get_model_dir()
    metrics = Metrics()
    state: dict = {"bundle": None}
    app = FastAPI(title="Parkinson's Speech Screening (research prototype)", version="1.0.0",
                  description=DISCLAIMER)

    def get_bundle() -> dict:
        if state["bundle"] is None:
            state["bundle"] = load_artifacts(mdir)   # raises ModelNotAvailable; retried on the next request
            log.info("Model loaded: %s", state["bundle"]["meta"].get("model_version"))
        return state["bundle"]

    def version() -> str | None:
        return state["bundle"]["meta"].get("model_version") if state["bundle"] else None

    @app.middleware("http")
    async def _timing(request: Request, call_next):
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            ms = (time.perf_counter() - t0) * 1000
            path = request.url.path if not request.url.path.startswith("/demo/predict/") else "/demo/predict/{id}"
            metrics.record(path, status, ms)
            log.info("%s %s -> %s (%.0f ms)", request.method, path, status, ms)

    def err(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})

    @app.exception_handler(AudioError)
    async def _audio_error(_, e: AudioError):
        return err(STATUS_FOR_CODE.get(e.code, 422), e.code, e.message)

    @app.exception_handler(ModelNotAvailable)
    async def _no_model(_, e: ModelNotAvailable):
        return err(503, "model_not_available", str(e))

    async def read_limited(file: UploadFile) -> bytes:
        buf = bytearray()
        while chunk := await file.read(1 << 20):
            buf += chunk
            if len(buf) > max_bytes:
                raise AudioError("file_too_large", f"File exceeds the {max_bytes // 1_000_000} MB limit.")
        return bytes(buf)

    def run_prediction(analysis: dict, explain: bool) -> dict:
        bundle = get_bundle()
        res = predict_from_analysis(analysis, bundle, cfg)
        res["prediction_label"] = ("Model score is AT OR ABOVE the research operating threshold" if res["prediction"] == "above_threshold"
                                   else "Model score is BELOW the research operating threshold")
        if explain:
            names = bundle["schema"]["feature_names"]
            res["explanation"] = explain_one(bundle["pipeline"], features_to_row(analysis["features"], names), names)
        return {"result": res, "analysis": analysis}

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health():
        try:
            get_bundle()
            return HealthResponse(status="ok", model_loaded=True, model_version=version())
        except ModelNotAvailable as e:
            return HealthResponse(status="ok", model_loaded=False, detail=str(e))

    @app.get("/model-info", tags=["system"], responses={503: {"model": ErrorResponse}})
    def model_info():
        b = get_bundle()
        m = b["meta"]
        return _clean({
            "model_version": m.get("model_version"), "model_name": m.get("model_name"), "score_type": m.get("score_type"),
            "score_is_calibrated_probability": False, "threshold": m.get("threshold"), "threshold_method": m.get("threshold_method"),
            "positive_class": m.get("positive_class"), "task_families": m.get("task_families"),
            "n_features": b["schema"]["n_features"], "feature_names": b["schema"]["feature_names"],
            "training_data": m.get("training_data"), "evaluation_status": m.get("evaluation_status"),
            "evaluation": m.get("evaluation"), "created_utc": m.get("created_utc"), "environment": m.get("environment"),
            "intended_use": m.get("intended_use"), "disclaimer": DISCLAIMER})

    @app.post("/analyze", response_model=AnalysisResponse, tags=["analysis"],
              responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
    async def analyze(file: UploadFile = File(...)):
        """Validate + extract features + waveform/spectrogram. Does not need a trained model."""
        data = await read_limited(file)
        return _clean(analyze_audio_bytes(data, cfg))

    @app.post("/predict", response_model=PredictResponse, tags=["prediction"],
              responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse},
                         422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    async def predict(file: UploadFile = File(...), explain: bool = Query(False, description="include per-prediction feature contributions")):
        """Run the saved model on an uploaded recording (WAV recommended). The audio is processed in memory and not stored."""
        get_bundle()   # fail fast with 503 before doing any work
        data = await read_limited(file)
        return _clean(run_prediction(analyze_audio_bytes(data, cfg), explain))

    # ---------------- local demo mode (never bundles recordings) ----------------
    def demo_tables():
        if os.environ.get("PD_ENABLE_DEMO_MODE", "0") != "1":
            raise HTTPException(404, "Demo mode is disabled. Set PD_ENABLE_DEMO_MODE=1 and PD_DATA_ROOT (see README).")
        import pandas as pd

        from src.config import get_data_root
        from src.data.dataset import DEFAULT_MANIFEST, DEFAULT_PRIVATE_DIR, load_manifest, resolve_paths, usable_recordings
        m = resolve_paths(usable_recordings(load_manifest(DEFAULT_MANIFEST), get_bundle()["meta"].get("task_families")), get_data_root(), DEFAULT_PRIVATE_DIR)
        sp = REPO_ROOT / "reports" / "training" / "split_assignment.csv"
        part = pd.read_csv(sp).set_index("recording_id")["partition"] if sp.is_file() else {}
        m["partition"] = m["recording_id"].map(part).fillna("unknown") if len(part) else "unknown"
        return m

    @app.get("/demo/samples", response_model=list[DemoSample], tags=["demo"])
    def demo_samples(partition: str = Query("test", pattern="^(test|train|all)$")):
        """Anonymous sample IDs from YOUR local dataset. Labels are deliberately not listed here."""
        m = demo_tables()
        if partition != "all":
            m = m[m["partition"] == partition]
        return [DemoSample(sample_id=r.recording_id, task_code=r.task_code, task_family=r.task_family, partition=r.partition) for r in m.itertuples()]

    @app.post("/demo/predict/{sample_id}", response_model=DemoPredictResponse, tags=["demo"])
    def demo_predict(sample_id: str, explain: bool = False):
        m = demo_tables()
        row = m[m["recording_id"] == sample_id]
        if row.empty or not isinstance(row.iloc[0]["abs_path"], str) or not Path(row.iloc[0]["abs_path"]).is_file():
            raise HTTPException(404, "Unknown sample id (or file not found on this machine).")
        r = row.iloc[0]
        out = run_prediction(analyze_audio_bytes(Path(r["abs_path"]).read_bytes(), cfg), explain)
        out["dataset_label_not_a_model_output"] = str(r["label"])
        out["partition"] = str(r["partition"])
        return _clean(out)

    @app.get("/metrics", tags=["system"])
    def get_metrics():
        return metrics.snapshot(version())

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/app/")

    frontend = REPO_ROOT / "frontend"
    if frontend.is_dir():
        app.mount("/app", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
