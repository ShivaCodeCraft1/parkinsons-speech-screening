import numpy as np
import pytest

from src.audio.loading import AudioError, LoadedAudio, load_audio_bytes
from src.config import FEATURES_CONFIG, load_yaml
from src.preprocessing.preprocess import preprocess
from src.validation.audio import raise_if_invalid, validate_raw
from tests.synthetic_pipeline_data import synth_voice, wav_bytes

CFG = load_yaml(FEATURES_CONFIG)
RNG = np.random.default_rng(1)


def voice(dur=2.0, sr=16000):
    return synth_voice(RNG, 150, 0.005, 0.05, 0.01, dur=dur, sr=sr)


def test_load_valid_wav_mono():
    a = load_audio_bytes(wav_bytes(voice()))
    assert a.sr == 16000 and a.y.ndim == 1 and a.n_channels == 1 and a.decoder == "soundfile"


def test_stereo_is_downmixed_to_mono():
    y = voice()
    a = load_audio_bytes(wav_bytes(np.stack([y, y], axis=1)))
    assert a.n_channels == 2 and a.y.ndim == 1 and a.y.size == y.size


def test_empty_bytes_rejected():
    with pytest.raises(AudioError) as e:
        load_audio_bytes(b"")
    assert e.value.code == "empty_file"


def test_corrupt_bytes_rejected():
    with pytest.raises(AudioError) as e:
        load_audio_bytes(b"this is definitely not audio" * 50)
    assert e.value.code in {"unsupported_format", "undecodable_audio"}


def test_validation_rejects_empty_signal():
    v = validate_raw(LoadedAudio(np.zeros(0, np.float32), 16000, 1, "soundfile"), CFG)
    assert not v.ok and v.errors[0]["code"] == "empty_signal"


def test_validation_rejects_silence_and_nan():
    v = validate_raw(LoadedAudio(np.zeros(32000, np.float32), 16000, 1, "soundfile"), CFG)
    assert not v.ok and v.errors[0]["code"] == "silent_or_constant"
    bad = voice(); bad[10] = np.nan
    assert validate_raw(LoadedAudio(bad, 16000, 1, "soundfile"), CFG).errors[0]["code"] == "non_finite_samples"


def test_validation_rejects_short_and_warns_borderline():
    short = validate_raw(LoadedAudio(voice(0.4), 16000, 1, "soundfile"), CFG)
    assert not short.ok and short.errors[0]["code"] == "too_short"
    border = validate_raw(LoadedAudio(voice(1.5), 16000, 1, "soundfile"), CFG)
    assert border.ok and any(w["code"] == "short_recording" for w in border.warnings)
    with pytest.raises(AudioError):
        raise_if_invalid(short)


def test_validation_warns_on_clipping_and_low_sample_rate():
    y = np.clip(voice() * 4, -1, 1)
    v = validate_raw(LoadedAudio(y.astype(np.float32), 8000, 1, "soundfile"), CFG)
    codes = {w["code"] for w in v.warnings}
    assert {"clipping", "low_sample_rate"} <= codes


def test_preprocess_resamples_and_normalises():
    y = voice(2.0, sr=44100)
    out, sr = preprocess(y + 0.2, 44100, CFG)
    assert sr == 16000 and abs(out.size - 32000) <= 2
    assert abs(np.max(np.abs(out)) - CFG["preprocessing"]["peak_normalize_to"]) < 1e-3
    assert abs(float(np.mean(out))) < 1e-3
