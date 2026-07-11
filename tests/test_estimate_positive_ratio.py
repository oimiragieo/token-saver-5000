"""#142 regression: the ingest estimate ratio must stay POSITIVE for high skeleton
ratios.

``compute_adaptive_ratio`` returns ~0.8 for small docs, but the old
``ratio_adjustment = 1.0 + (0.2 - skeleton_ratio) * 5`` went NEGATIVE for
skeleton_ratio > 0.4 (and exactly zero at 0.4 -> ZeroDivisionError in
``int(original / adjusted_ratio)``), producing a nonsensical negative
``estimated_ratio`` (a live dogfood on a 213-token doc returned -2.45). A
compression ratio is always positive; a value below 1.0 is honest expansion.
"""

from __future__ import annotations

import pytest

from src.compression_advisor import CompressionAdvisor

_TXT = "Semantic compression keeps the important nodes and drops the rest. " * 8


@pytest.mark.parametrize("skeleton_ratio", [0.4, 0.5, 0.8, 0.95, 1.0])
def test_estimate_ratio_stays_positive_for_high_skeleton_ratio(skeleton_ratio):
    est = CompressionAdvisor().estimate_compression(_TXT, skeleton_ratio=skeleton_ratio)
    assert est.compression_ratio > 0  # never negative, never zero (div-by-zero)
    assert est.estimated_compressed > 0


def test_estimate_designed_band_boost_preserved():
    a = CompressionAdvisor()
    # The clamp must not perturb the designed [0.05, 0.2] band: a lower skeleton
    # ratio still boosts the estimated ratio more than the 0.2 design center.
    r_low = a.estimate_compression(_TXT, skeleton_ratio=0.05).compression_ratio
    r_center = a.estimate_compression(_TXT, skeleton_ratio=0.2).compression_ratio
    assert r_low > r_center > 0
