#!/usr/bin/env python3
"""
SYNTHETIC toy dataset (fake names, sine-wave 'audio') for testing the verifier.
Nothing here is real data; numbers measured on it mean nothing scientifically.

  python tests/make_synthetic_dataset.py --out /tmp/synthetic_pvs [--inject-faults]
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

CODES = ["B1", "B2", "D1", "D2", "FB1", "FB2", "VA1", "VA2", "VE1", "VE2", "VI1", "VI2", "VO1", "VO2", "VU1", "VU2"]
FAKE = {
    "3 Young Healthy Control": [("Anna", "Bianchi", "F", 24), ("Luca", "Verdi", "M", 27), ("Sara", "Neri", "F", 22)],
    "3 Elderly Healthy Control": [("Paolo", "Gialli", "M", 68), ("Maria", "Rosa", "F", 71), ("Gino", "Blu", "M", 65)],
    "4 People with Parkinson's disease": [("Carlo", "Viola", "M", 70), ("Elena", "Arancio", "F", 66),
                                          ("Franco", "Grigi", "M", 74), ("Rita", "Marroni", "F", 62)],
}
FAKE_NAME_WORDS = sorted({w for people in FAKE.values() for p in people for w in p[:2]})
WRAPPER = "Italian Parkinson's Voice and Speech"


def build_synthetic_dataset(out: Path, inject_faults: bool = False) -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    root = out / WRAPPER
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    sr = 22050
    rows = []
    for gi, (grp, people) in enumerate(FAKE.items()):
        for pi, (first, last, sex, age) in enumerate(people):
            d = root / grp / f"{first} {last}"
            d.mkdir(parents=True)
            for ci, code in enumerate(CODES):
                t = np.arange(int(sr * 0.7)) / sr
                f0 = 110 + 15 * gi + 7 * pi + 3 * ci
                x = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.01 * rng.standard_normal(t.size)
                sf.write(d / f"{code} {first} {last}.wav", x.astype("float32"), sr, subtype="PCM_16")
            rows.append({"Name": f"{last} {first}", "Age": age, "Sex": sex, "Group": grp.split(" ", 1)[1],
                         "Jitter": float(rng.uniform(0.2, 1.0))})
    pd.DataFrame(rows).to_excel(root / "metadata.xlsx", index=False)
    (root / "FILE CODES.txt").write_text(
        "\n".join(["B1/B2 - balanced text reading", "D1/D2 - syllables 5 s", "FB1/FB2 - phrases/words",
                   "VA1/VA2 - vowel a", "VE1/VE2 - vowel e", "VI1/VI2 - vowel i", "VO1/VO2 - vowel o",
                   "VU1/VU2 - vowel u"]), encoding="utf-8")
    if inject_faults:
        pd_dir = root / "4 People with Parkinson's disease" / "Carlo Viola"
        (pd_dir / "VA1 Carlo Viola.wav").write_bytes(b"this is not a wav file")           # corrupt audio
        shutil.copy(pd_dir / "B1 Carlo Viola.wav", pd_dir / "ZZ9 Carlo Viola copy.wav")     # unknown code + duplicate bytes
        shutil.copy(root / "3 Young Healthy Control" / "Anna Bianchi" / "VE1 Anna Bianchi.wav",
                    root / "4 People with Parkinson's disease" / "Elena Arancio" / "VE1 dup.wav")  # cross-subject duplicate
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inject-faults", action="store_true")
    a = ap.parse_args()
    build_synthetic_dataset(Path(a.out), a.inject_faults)
    print(f"synthetic dataset written to {a.out}")
