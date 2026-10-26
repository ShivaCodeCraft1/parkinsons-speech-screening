"""Audio validation. Returns structured errors (reject) and warnings (analyse, but flag)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.audio.loading import AudioError, LoadedAudio


@dataclass
class AudioValidation:
    ok: bool
    errors: list[dict] = field(default_factory=list)      # [{"code","message"}]
    warnings: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _dbfs(x: float) -> float | None:
    return None if x <= 0 else float(20 * np.log10(x))


def validate_raw(audio: LoadedAudio, cfg: dict) -> AudioValidation:
    """Checks on the decoded (not yet preprocessed) signal."""
    a = cfg["audio"]
    v = AudioValidation(ok=True)
    y = audio.y
    dur = float(y.size / audio.sr) if audio.sr else 0.0
    v.info = {"duration_sec": round(dur, 3), "original_sample_rate": audio.sr, "source_channels": audio.n_channels,
              "decoder": audio.decoder, "n_samples": int(y.size)}
    if y.size == 0:
        v.errors.append({"code": "empty_signal", "message": "The recording contains no samples."})
    elif not np.all(np.isfinite(y)):
        v.errors.append({"code": "non_finite_samples", "message": "The recording contains NaN/Inf samples."})
    else:
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        v.info.update(peak_abs=round(peak, 5), rms_dbfs=None if _dbfs(rms) is None else round(_dbfs(rms), 2))
        if peak == 0.0 or float(np.std(y)) < 1e-9:
            v.errors.append({"code": "silent_or_constant", "message": "The recording is silent or constant."})
        else:
            clip = float(np.mean(np.abs(y) >= a["clip_sample_threshold"]))
            v.info["clip_fraction"] = round(clip, 6)
            if clip > a["clip_fraction_warn"]:
                v.warnings.append({"code": "clipping", "message": f"{clip:.2%} of samples are clipped; jitter/shimmer may be unreliable."})
            if v.info["rms_dbfs"] is not None and v.info["rms_dbfs"] < a["near_silent_rms_dbfs_warn"]:
                v.warnings.append({"code": "very_quiet", "message": "The recording is very quiet; move closer to the microphone."})
        if dur < a["min_duration_sec"]:
            v.errors.append({"code": "too_short", "message": f"Recording is {dur:.2f}s; at least {a['min_duration_sec']}s is required."})
        elif dur < a["warn_duration_sec_below"]:
            v.warnings.append({"code": "short_recording", "message": f"Recording is only {dur:.2f}s; features may be unstable."})
    if dur > a["max_duration_sec"]:
        v.errors.append({"code": "too_long", "message": f"Recording is {dur:.1f}s; the maximum is {a['max_duration_sec']}s."})
    if audio.sr < a["target_sample_rate"]:
        v.warnings.append({"code": "low_sample_rate",
                           "message": f"Original sample rate {audio.sr} Hz is below {a['target_sample_rate']} Hz; upsampling adds no information."})
    v.ok = not v.errors
    return v


def raise_if_invalid(v: AudioValidation) -> None:
    if not v.ok:
        first = v.errors[0]
        raise AudioError(first["code"], first["message"])
