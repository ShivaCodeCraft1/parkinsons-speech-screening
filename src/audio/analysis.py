"""Small, JSON-friendly visualisation data computed from the actual (preprocessed) signal."""
from __future__ import annotations

import librosa
import numpy as np


def waveform_envelope(y: np.ndarray, n_points: int = 400) -> dict:
    n = max(1, min(n_points, y.size))
    chunks = np.array_split(y, n)
    return {"min": [round(float(c.min()), 4) for c in chunks], "max": [round(float(c.max()), 4) for c in chunks]}


def spectrogram_db(y: np.ndarray, sr: int, max_frames: int = 160, n_freq: int = 64, fmax: float = 5000.0) -> dict:
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=max(160, y.size // max_frames))) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    keep = freqs <= fmax
    S, freqs = S[keep], freqs[keep]
    idx = np.array_split(np.arange(S.shape[0]), min(n_freq, S.shape[0]))
    S = np.stack([S[i].mean(axis=0) for i in idx])
    db = librosa.power_to_db(S, ref=np.max, top_db=80.0)
    return {"freq_hz": [round(float(freqs[i].mean()), 1) for i in idx], "n_frames": int(db.shape[1]),
            "duration_sec": round(y.size / sr, 3), "db": np.round(db, 1).tolist()}
