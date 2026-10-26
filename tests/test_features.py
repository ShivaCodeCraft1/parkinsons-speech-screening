import math

import numpy as np
import pytest

from src.config import FEATURES_CONFIG, load_yaml
from src.features.extract import extract_features, feature_names, feature_schema
from src.validation.features import features_to_row, validate_feature_dict
from tests.synthetic_pipeline_data import synth_voice

CFG = load_yaml(FEATURES_CONFIG)
RNG = np.random.default_rng(2)


@pytest.fixture(scope="module")
def feats():
    y = synth_voice(RNG, 140, 0.005, 0.05, 0.01, dur=2.0)
    return extract_features(y, 16000, CFG)


def test_schema_matches_extraction_order_and_dimension(feats):
    names = feature_names(CFG)
    assert list(feats.keys()) == names
    assert len(names) == feature_schema(CFG)["n_features"] == len(set(names))
    assert validate_feature_dict(feats, names)["ok"]


def test_values_are_plausible_for_a_harmonic_voice(feats):
    assert 120 < feats["f0_mean_hz"] < 160
    assert feats["voiced_fraction"] > 0.8
    assert not any(math.isinf(v) for v in feats.values())
    assert 0 <= feats["jitter_local"] < 0.1


def test_extraction_is_deterministic(feats):
    y = synth_voice(np.random.default_rng(2), 140, 0.005, 0.05, 0.01, dur=2.0)
    a = extract_features(y, 16000, CFG)
    b = extract_features(y, 16000, CFG)
    assert all((x == y_ or (math.isnan(x) and math.isnan(y_))) for x, y_ in zip(a.values(), b.values()))


def test_unvoiced_noise_gives_undefined_not_fake_values():
    noise = (np.random.default_rng(3).standard_normal(32000) * 0.1).astype(np.float32)
    f = extract_features(noise, 16000, CFG)
    assert math.isnan(f["jitter_local"]) and math.isnan(f["shimmer_local"])   # undefined stays NaN
    assert not any(math.isinf(v) for v in f.values())
    assert len(f) == len(feature_names(CFG))


def test_very_short_signal_does_not_crash():
    f = extract_features((np.random.default_rng(4).standard_normal(100) * 0.1).astype(np.float32), 16000, CFG)
    assert list(f.keys()) == feature_names(CFG)


def test_features_to_row_enforces_schema(feats):
    names = feature_names(CFG)
    row = features_to_row(dict(feats), names)
    assert row.shape == (1, len(names))
    broken = dict(feats); broken.pop(names[3])
    with pytest.raises(ValueError):
        features_to_row(broken, names)
    shuffled = dict(reversed(list(feats.items())))
    assert not validate_feature_dict(shuffled, names)["order_ok"]
    assert features_to_row(shuffled, names)[0, 0] == feats[names[0]]    # row order always follows the schema, not dict order
