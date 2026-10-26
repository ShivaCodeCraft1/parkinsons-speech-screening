"""Validation of extracted feature vectors against the saved schema."""
from __future__ import annotations

import math

import numpy as np


def validate_feature_dict(features: dict, expected_names: list[str]) -> dict:
    """Returns {"ok", "missing_names", "extra_names", "order_ok", "undefined", "non_finite"}.
    NaN means 'undefined measurement' (allowed; reported). +/-Inf is never allowed."""
    names = list(features.keys())
    missing = [n for n in expected_names if n not in features]
    extra = [n for n in names if n not in expected_names]
    undefined = [n for n, v in features.items() if isinstance(v, float) and math.isnan(v)]
    infinite = [n for n, v in features.items() if isinstance(v, float) and math.isinf(v)]
    order_ok = names == expected_names
    return {"ok": not missing and not extra and order_ok and not infinite, "missing_names": missing,
            "extra_names": extra, "order_ok": order_ok, "undefined": undefined, "non_finite": infinite}


def features_to_row(features: dict, expected_names: list[str]) -> np.ndarray:
    """Dict -> 1 x n array in the EXACT schema order. Raises if the schema does not match."""
    chk = validate_feature_dict(features, expected_names)
    if chk["missing_names"] or chk["extra_names"] or chk["non_finite"]:
        raise ValueError(f"Feature schema mismatch: {chk}")
    return np.array([[features[n] for n in expected_names]], dtype=float)
