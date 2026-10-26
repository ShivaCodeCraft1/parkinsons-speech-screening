"""SYNTHETIC voices (harmonic source + random f0/amplitude perturbation) for testing the SOFTWARE pipeline end to end.
They are not speech from real people. Any score computed on them says nothing about medical accuracy."""
from __future__ import annotations

import io
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

WRAPPER = "Italian Parkinson's Voice and Speech"
GROUPS = {"3 Young Healthy Control": dict(f0=190, jit=0.004, shim=0.03, noise=0.01, age=(20, 30)),
          "3 Elderly Healthy Control": dict(f0=150, jit=0.008, shim=0.05, noise=0.02, age=(60, 75)),
          "4 People with Parkinson's disease": dict(f0=140, jit=0.025, shim=0.12, noise=0.06, age=(60, 78))}
CODES = ["VA1", "VA2", "VE1", "VE2"]


def synth_voice(rng, f0, jit, shim, noise, dur=1.6, sr=16000) -> np.ndarray:
    n = int(dur * sr)
    knots = np.arange(0, n + 160, 160)
    f0_t = np.interp(np.arange(n), knots, f0 * (1 + jit * rng.standard_normal(knots.size)))
    phase = 2 * np.pi * np.cumsum(f0_t) / sr
    amp = 1 + shim * np.interp(np.arange(n), knots, rng.standard_normal(knots.size))
    y = sum(np.sin(k * phase) / k for k in range(1, 6)) * amp
    y = y / np.max(np.abs(y)) * 0.5 + noise * rng.standard_normal(n)
    return y.astype(np.float32)


def wav_bytes(y: np.ndarray, sr: int = 16000, subtype: str = "PCM_16") -> bytes:
    b = io.BytesIO()
    sf.write(b, y, sr, format="WAV", subtype=subtype)
    return b.getvalue()


def build_larger_synthetic_dataset(out: Path, per_group: int = 8, seed: int = 0) -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    root = out / WRAPPER
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    rows = []
    for gname, p in GROUPS.items():
        for i in range(per_group):
            first, last = f"Fake{gname.split()[1][:2]}{i}", f"Person{i}x{gname.split()[0]}"
            d = root / gname / f"{first} {last}"
            d.mkdir(parents=True)
            subj = dict(f0=p["f0"] * rng.uniform(0.85, 1.15), jit=p["jit"] * rng.uniform(0.6, 1.4), shim=p["shim"], noise=p["noise"])
            for code in CODES:
                sf.write(d / f"{code} {first} {last}.wav", synth_voice(rng, **subj), 16000, subtype="PCM_16")
            rows.append({"Name": f"{last} {first}", "Age": int(rng.integers(*p["age"])), "Sex": rng.choice(["F", "M"]), "Group": gname.split(" ", 1)[1]})
    pd.DataFrame(rows).to_excel(root / "metadata.xlsx", index=False)
    (root / "FILE CODES.txt").write_text("VA1/VA2 - vowel a\nVE1/VE2 - vowel e\n", encoding="utf-8")
    return root
