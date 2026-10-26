"""Audio decoding. WAV/FLAC/OGG via libsndfile; other containers (e.g. browser WebM/MP4) via ffmpeg if installed.

All decoders return a mono float32 signal in [-1, 1] at the ORIGINAL sample rate; resampling happens in
src/preprocessing/preprocess.py so there is exactly one place that does it.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioError(Exception):
    """Raised for unusable audio. `code` is a stable machine-readable string used by the API."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class LoadedAudio:
    y: np.ndarray            # mono float32
    sr: int                  # original sample rate
    n_channels: int          # channels in the source file
    decoder: str             # "soundfile" | "ffmpeg"


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.float32, copy=False)
    return data.mean(axis=1).astype(np.float32)


def _decode_soundfile(data: bytes) -> LoadedAudio:
    with sf.SoundFile(io.BytesIO(data)) as f:
        sr, ch = f.samplerate, f.channels
        arr = f.read(dtype="float32", always_2d=True)
    return LoadedAudio(_to_mono(arr), int(sr), int(ch), "soundfile")


def _decode_ffmpeg(data: bytes) -> LoadedAudio:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioError("unsupported_format",
                         "Could not decode this audio and ffmpeg is not installed. Upload a WAV file "
                         "(the web interface converts microphone recordings to WAV automatically).")
    fd, tmp_in = tempfile.mkstemp(suffix=".bin")
    tmp_out = tmp_in + ".wav"
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        # keep the original sample rate and channels; just decode to PCM WAV
        proc = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", tmp_in, "-f", "wav", tmp_out],
                              capture_output=True, timeout=60)
        if proc.returncode != 0 or not Path(tmp_out).is_file():
            raise AudioError("undecodable_audio", "The file could not be decoded as audio.")
        decoded = _decode_soundfile(Path(tmp_out).read_bytes())
        return LoadedAudio(decoded.y, decoded.sr, decoded.n_channels, "ffmpeg")
    except subprocess.TimeoutExpired as e:
        raise AudioError("undecodable_audio", "Audio decoding timed out.") from e
    finally:
        for p in (tmp_in, tmp_out):
            try:
                os.remove(p)
            except OSError:
                pass


def load_audio_bytes(data: bytes) -> LoadedAudio:
    """Decode audio from raw bytes. Raises AudioError with a clear code on failure."""
    if not data:
        raise AudioError("empty_file", "The uploaded file is empty.")
    try:
        return _decode_soundfile(data)
    except (sf.LibsndfileError, RuntimeError, ValueError):
        return _decode_ffmpeg(data)


def load_audio_file(path: str | Path) -> LoadedAudio:
    p = Path(path)
    if not p.is_file():
        raise AudioError("file_not_found", f"File not found: {p.name}")
    return load_audio_bytes(p.read_bytes())
