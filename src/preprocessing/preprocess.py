"""Consistent preprocessing used by BOTH training-time extraction and the API."""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32, copy=False)
    g = gcd(int(sr), int(target_sr))
    return resample_poly(y, target_sr // g, sr // g).astype(np.float32)


def preprocess(y: np.ndarray, sr: int, cfg: dict) -> tuple[np.ndarray, int]:
    """mono float32 @ original sr  ->  cleaned float32 @ target sr.

    Steps (all configurable in configs/features.yaml): resample, remove DC offset, peak-normalise.
    No noise reduction and no silence trimming unless explicitly enabled.
    """
    target = int(cfg["audio"]["target_sample_rate"])
    pre = cfg["preprocessing"]
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y, target
    y = resample(y, sr, target)
    if pre.get("remove_dc_offset", True):
        y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if pre.get("peak_normalize_to") and peak > 0:
        y = y * (float(pre["peak_normalize_to"]) / peak)
    if pre.get("trim_silence", False):
        import librosa
        y, _ = librosa.effects.trim(y, top_db=40)
    return y.astype(np.float32), target
