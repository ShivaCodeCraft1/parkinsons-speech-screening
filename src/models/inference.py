"""Inference on a new recording using EXACTLY the training feature code and the saved feature order."""
from __future__ import annotations

import time

import numpy as np

from src.audio.analysis import spectrogram_db, waveform_envelope
from src.audio.loading import load_audio_bytes
from src.config import FEATURES_CONFIG, load_yaml
from src.features.extract import extract_from_signal, feature_schema
from src.models.training import score_with
from src.preprocessing.preprocess import preprocess
from src.validation.audio import raise_if_invalid, validate_raw
from src.validation.features import features_to_row, validate_feature_dict


def load_feature_config() -> dict:
    return load_yaml(FEATURES_CONFIG)


def analyze_audio_bytes(data: bytes, cfg: dict | None = None, with_visuals: bool = True) -> dict:
    """Decode -> validate -> preprocess -> extract. No model needed. Raises AudioError for unusable audio."""
    cfg = cfg or load_feature_config()
    audio = load_audio_bytes(data)
    val = validate_raw(audio, cfg)
    raise_if_invalid(val)
    y, sr = preprocess(audio.y, audio.sr, cfg)
    feats = extract_from_signal(y, sr, cfg)
    warnings = list(val.warnings)
    vf = feats.get("voiced_fraction")
    if vf is not None and np.isfinite(vf) and vf < cfg["audio"]["min_voiced_fraction_warn"]:
        warnings.append({"code": "low_voiced_fraction", "message": f"Only {vf:.0%} of frames contain detectable voicing."})
    out = {"validation": {"ok": True, "info": val.info, "warnings": warnings}, "features": dict(feats), "sample_rate": sr}
    if with_visuals:
        out["waveform"] = waveform_envelope(y)
        out["spectrogram"] = spectrogram_db(y, sr)
    return out


def predict_from_analysis(analysis: dict, bundle: dict, cfg: dict | None = None) -> dict:
    """Run the saved pipeline on features from `analyze_audio_bytes`. Checks schema and config consistency."""
    cfg = cfg or load_feature_config()
    schema, meta, pipe = bundle["schema"], bundle["meta"], bundle["pipeline"]
    names = schema["feature_names"]
    warnings = list(analysis["validation"]["warnings"])
    cur = feature_schema(cfg)
    if cur["feature_names"] != names:
        raise RuntimeError("Feature schema in configs/features.yaml no longer matches the trained model. Re-extract and retrain.")
    if cur["feature_config_sha256"] != schema.get("feature_config_sha256"):
        warnings.append({"code": "feature_config_changed", "message": "configs/features.yaml differs from training time; results may be unreliable."})
    chk = validate_feature_dict(analysis["features"], names)
    if chk["undefined"]:
        warnings.append({"code": "undefined_features", "message": f"{len(chk['undefined'])} feature(s) could not be measured and were imputed with training medians: "
                                                                  + ", ".join(chk["undefined"][:8]) + ("..." if len(chk["undefined"]) > 8 else "")})
    row = features_to_row(analysis["features"], names)
    t0 = time.perf_counter()
    score = float(score_with(pipe, row)[0])
    latency_ms = (time.perf_counter() - t0) * 1000
    thr = float(meta["threshold"])
    task_fams = meta.get("task_families")
    warnings.append({"code": "domain_shift", "message": "The model was trained on controlled dataset recordings"
                     + (f" of the task(s) {task_fams}" if task_fams else "")
                     + ". Arbitrary microphone recordings may differ; treat the output as a research demonstration only."})
    return {"prediction": "above_threshold" if score >= thr else "below_threshold", "score": score, "threshold": thr,
            "score_type": meta.get("score_type"), "model_version": meta.get("model_version"), "model_name": meta.get("model_name"),
            "warnings": warnings, "inference_ms": round(latency_ms, 2)}
