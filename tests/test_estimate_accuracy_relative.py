"""`estimate_accuracy` must grade RELATIVE error, not absolute ratio points.

Dogfood-found on the live prod MCP, 2026-07-25. Two real
`ingest_context(file_url=...)` calls:

    pgvector README      estimated 2.177  actual 2.194  -> "excellent"  (right)
    fastapi oauth2-jwt   estimated 1.253  actual 2.810  -> "excellent"  (WRONG)

The second is a 2.24x under-prediction -- 124% relative error -- reported to the
calling agent as an excellent estimate. The cause was an ABSOLUTE threshold:

    "excellent" if abs(actual_ratio - estimate.compression_ratio) < 2

Absolute ratio points do not mean the same thing at different scales. A gap of
1.5 is catastrophic near ratio 1.0 and negligible near ratio 16. Under that
rule ANY miss smaller than 2.0 ratio-points grades excellent regardless of how
wrong it is proportionally.

This matters because it sits on the activation surface. #277 fixed the estimate
itself under-predicting; this is the same harm resurfacing in the LABEL. An
agent reading accuracy="excellent" will trust a materially wrong number and may
skip compression that would have paid for itself. On the failing doc the
`reasoning` string even hedged correctly ("Highly redundant prose can land far
above this estimate") while the label flatly contradicted it.
"""

from __future__ import annotations

import pytest

from src.handlers.compression_handlers import classify_estimate_accuracy


class TestLiveDogfoodCases:
    """The two measured production cases that exposed the bug."""

    def test_close_estimate_on_a_large_doc_is_excellent(self):
        # pgvector README: 0.8% relative error.
        assert classify_estimate_accuracy(actual=2.194, estimated=2.177) == "excellent"

    def test_the_124_percent_under_prediction_is_not_excellent(self):
        # fastapi oauth2-jwt: 55.4% relative error. Graded "excellent" before
        # this fix purely because the absolute gap (1.557) was under 2.0.
        assert classify_estimate_accuracy(actual=2.810, estimated=1.253) != "excellent"


class TestScaleInvariance:
    """The property the absolute threshold lacked: same proportional miss,
    same grade, wherever it lands on the ratio scale."""

    @pytest.mark.parametrize(
        "actual,estimated",
        [
            (2.0, 1.9),  # 5% off, low ratio
            (10.0, 9.5),  # 5% off, mid ratio
            (20.0, 19.0),  # 5% off, high ratio
        ],
    )
    def test_a_small_proportional_miss_grades_the_same_at_any_scale(self, actual, estimated):
        assert classify_estimate_accuracy(actual=actual, estimated=estimated) == "excellent"

    @pytest.mark.parametrize(
        "actual,estimated",
        [
            (2.0, 1.0),  # 50% off, low ratio
            (10.0, 5.0),  # 50% off, mid ratio
            (20.0, 10.0),  # 50% off, high ratio
        ],
    )
    def test_a_large_proportional_miss_grades_the_same_at_any_scale(self, actual, estimated):
        # The old absolute rule called the first of these "excellent" (gap 1.0)
        # and the last "fair" (gap 10.0) -- for identical proportional error.
        assert classify_estimate_accuracy(actual=actual, estimated=estimated) == "fair"


class TestBandsAreOrdered:
    def test_grades_degrade_monotonically_as_error_grows(self):
        grades = [classify_estimate_accuracy(actual=10.0, estimated=est) for est in (9.8, 7.5, 3.0)]
        assert grades == ["excellent", "good", "fair"]


class TestDegenerateInputs:
    """A grader that raises on the compression path is worse than a wrong grade."""

    def test_zero_actual_ratio_does_not_divide_by_zero(self):
        assert classify_estimate_accuracy(actual=0.0, estimated=1.0) == "fair"

    def test_both_zero_is_not_an_error(self):
        assert isinstance(classify_estimate_accuracy(actual=0.0, estimated=0.0), str)

    def test_negative_or_nonsense_input_still_returns_a_grade(self):
        assert isinstance(classify_estimate_accuracy(actual=-1.0, estimated=2.0), str)

    def test_over_prediction_is_graded_the_same_as_under_prediction(self):
        # Symmetry: being 50% high is as wrong as being 50% low. Measured
        # against `actual` in both directions.
        under = classify_estimate_accuracy(actual=10.0, estimated=5.0)
        over = classify_estimate_accuracy(actual=10.0, estimated=15.0)
        assert under == over == "fair"


class TestCodexReviewFindings:
    """Two defects the narrow codex review found, both verified empirically
    before fixing rather than taken on faith."""

    def test_huge_int_does_not_raise_out_of_the_compression_path(self):
        # math.isfinite(10**400) raises OverflowError ("int too large to
        # convert to float"), which the original guard did not catch. A
        # cosmetic grade must never crash an ingest.
        assert classify_estimate_accuracy(actual=10**400, estimated=1.0) == "fair"
        assert classify_estimate_accuracy(actual=2.0, estimated=10**400) == "fair"

    def test_an_exact_band_boundary_is_not_demoted_by_float_dust(self):
        # abs(3.0-4.2)/3.0 == 0.4000000000000001, so a bare `<=` graded an
        # exact 40% miss "fair" instead of "good".
        assert classify_estimate_accuracy(actual=3.0, estimated=4.2) == "good"

    def test_the_epsilon_does_not_widen_the_band_meaningfully(self):
        # A genuinely worse-than-40% miss must still be "fair" -- the epsilon
        # absorbs representation error, not real error.
        assert classify_estimate_accuracy(actual=3.0, estimated=4.25) == "fair"
