"""Acoustic feature extraction: the SINGLE implementation used by training, evaluation, the API and the demo.

Design rules
  * Feature names and order come from `feature_names(cfg)` and are saved with the model (feature_schema.json).
  * A measurement that cannot be computed is returned as NaN. It is NOT replaced by an arbitrary value here;
    the imputer inside the trained sklearn Pipeline (fitted on training data only) handles NaN.
  * Units: f0 in Hz (semitone_std in semitones re 1 Hz-ratio), jitter/shimmer as fractions (0.01 = 1 %),
    HNR in dB, spectral centroid/bandwidth in Hz, RMS in dB relative to the peak-normalised signal,
    ZCR as crossings per sample, MFCCs unitless (coefficients 1..n, c0 excluded because it tracks energy).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from src.preprocessing.preprocess import preprocess

SCHEMA_VERSION = "1.0"


def feature_names(cfg: dict) -> list[str]:
    n_mfcc = int(cfg["spectral"]["n_mfcc"])
    names = ["f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz", "f0_std_semitones", "voiced_fraction",
             "jitter_local", "jitter_rap", "jitter_ppq5", "shimmer_local", "shimmer_apq3", "shimmer_apq5",
             "hnr_mean_db", "hnr_std_db",
             "spectral_centroid_mean_hz", "spectral_centroid_std_hz", "spectral_bandwidth_mean_hz",
             "spectral_bandwidth_std_hz", "spectral_flatness_mean", "spectral_flatness_std",
             "rms_mean_db", "rms_std_db", "zcr_mean", "zcr_std"]
    names += [f"mfcc{i}_mean" for i in range(1, n_mfcc + 1)] + [f"mfcc{i}_std" for i in range(1, n_mfcc + 1)]
    return names


def feature_schema(cfg: dict) -> dict:
    names = feature_names(cfg)
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    return {"schema_version": SCHEMA_VERSION, "n_features": len(names), "feature_names": names,
            "feature_config_sha256": cfg_hash, "target_sample_rate": int(cfg["audio"]["target_sample_rate"])}


def _nan() -> float:
    return float("nan")


def _safe(fn, *a, **k) -> float:
    try:
        v = fn(*a, **k)
        v = float(v)
        return v if math.isfinite(v) else _nan()
    except Exception:  # Praat raises on undefined measurements
        return _nan()


def _praat_features(y: np.ndarray, sr: int, cfg: dict) -> dict[str, float]:
    p, js, h = cfg["pitch"], cfg["jitter_shimmer"], cfg["hnr"]
    out = {k: _nan() for k in feature_names(cfg)[:14]}
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=float(sr))
    try:
        pitch = snd.to_pitch_ac(time_step=p["time_step_sec"], pitch_floor=p["floor_hz"], pitch_ceiling=p["ceiling_hz"])
    except parselmouth.PraatError:   # e.g. signal shorter than Praat's minimum analysis window -> everything undefined
        return out
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    out["voiced_fraction"] = float(voiced.size / f0.size) if f0.size else _nan()
    if voiced.size >= 2:
        out.update(f0_mean_hz=float(voiced.mean()), f0_std_hz=float(voiced.std(ddof=1)),
                   f0_min_hz=float(voiced.min()), f0_max_hz=float(voiced.max()),
                   f0_std_semitones=float(np.std(12 * np.log2(voiced / voiced.mean()), ddof=1)))
    # perturbation measures need enough voiced periods
    pp = None
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", p["floor_hz"], p["ceiling_hz"])
    except parselmouth.PraatError:
        pass
    n_pts = _safe(call, pp, "Get number of points") if pp is not None else _nan()
    if math.isfinite(n_pts) and n_pts >= cfg["min_voiced_points_for_perturbation"]:
        a = (0, 0, js["period_floor_s"], js["period_ceiling_s"], js["max_period_factor"])
        out["jitter_local"] = _safe(call, pp, "Get jitter (local)", *a)
        out["jitter_rap"] = _safe(call, pp, "Get jitter (rap)", *a)
        out["jitter_ppq5"] = _safe(call, pp, "Get jitter (ppq5)", *a)
        b = (0, 0, js["period_floor_s"], js["period_ceiling_s"], js["max_period_factor"], js["max_amplitude_factor"])
        out["shimmer_local"] = _safe(call, [snd, pp], "Get shimmer (local)", *b)
        out["shimmer_apq3"] = _safe(call, [snd, pp], "Get shimmer (apq3)", *b)
        out["shimmer_apq5"] = _safe(call, [snd, pp], "Get shimmer (apq5)", *b)
    try:
        harm = snd.to_harmonicity_cc(time_step=h["time_step_sec"], minimum_pitch=h["min_pitch_hz"],
                                     silence_threshold=h["silence_threshold"], periods_per_window=h["periods_per_window"])
    except parselmouth.PraatError:
        return out
    vals = harm.values[0]
    vals = vals[vals > -100]       # Praat marks unvoiced frames as -200 dB
    if vals.size >= 2:
        out["hnr_mean_db"], out["hnr_std_db"] = float(vals.mean()), float(vals.std(ddof=1))
    return out


def _spectral_features(y: np.ndarray, sr: int, cfg: dict) -> dict[str, float]:
    s = cfg["spectral"]
    n_fft, hop = int(s["n_fft"]), int(s["hop_length"])
    out: dict[str, float] = {}
    if y.size < n_fft:
        return {k: _nan() for k in feature_names(cfg)[14:]}
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    cen = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(S=np.sqrt(S), sr=sr)[0]
    flat = librosa.feature.spectral_flatness(S=np.sqrt(S))[0]
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-8))
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]
    n_mfcc = int(s["n_mfcc"])
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc + 1, n_fft=n_fft, hop_length=hop, n_mels=40)[1:]
    out.update(spectral_centroid_mean_hz=cen.mean(), spectral_centroid_std_hz=cen.std(ddof=1),
               spectral_bandwidth_mean_hz=bw.mean(), spectral_bandwidth_std_hz=bw.std(ddof=1),
               spectral_flatness_mean=flat.mean(), spectral_flatness_std=flat.std(ddof=1),
               rms_mean_db=rms_db.mean(), rms_std_db=rms_db.std(ddof=1),
               zcr_mean=zcr.mean(), zcr_std=zcr.std(ddof=1))
    for i in range(n_mfcc):
        out[f"mfcc{i + 1}_mean"] = mf[i].mean()
        out[f"mfcc{i + 1}_std"] = mf[i].std(ddof=1)
    return {k: float(v) for k, v in out.items()}


def extract_from_signal(y: np.ndarray, sr: int, cfg: dict) -> "OrderedDict[str, float]":
    """Extract features from an ALREADY preprocessed signal (mono float32 at the target sample rate)."""
    values = {**_praat_features(y, sr, cfg), **_spectral_features(y, sr, cfg)}
    return OrderedDict((n, float(values.get(n, _nan()))) for n in feature_names(cfg))


def extract_features(y_raw: np.ndarray, sr_raw: int, cfg: dict) -> "OrderedDict[str, float]":
    """Raw mono signal at its original sample rate -> ordered feature dict (the one public entry point)."""
    y, sr = preprocess(y_raw, sr_raw, cfg)
    return extract_from_signal(y, sr, cfg)
