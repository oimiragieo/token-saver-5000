"""#92 (2026-06-12): estimated_ratio honesty locks.

The advisor's size bands must mirror the engine's REAL adaptive curve
(semantic_compressor.compute_adaptive_ratio: keep 80% of nodes <8K tokens,
50% to 32K, 20% to 100K, 10% above), anchored on the conservative end.
The legacy bands promised 7-9x on <500-token docs while the engine delivered
1.17x on the same document — a 6x overclaim inside a single live dogfood
response (estimated_ratio=7.08 vs compression_ratio=1.17).
"""

from src.compression_advisor import CompressionAdvisor


def _prose(sentences: int) -> str:
    """Mildly varied filler prose (~10 tokens per sentence)."""
    return " ".join(
        f"The deployment pipeline builds artifact number {i} for the staging cluster."
        for i in range(sentences)
    )


class TestEstimateMirrorsAdaptiveCurve:
    def setup_method(self):
        self.advisor = CompressionAdvisor()

    def test_tiny_doc_estimates_no_compression(self):
        """<200 tokens: skeleton overhead cancels savings — say so, estimate ~1x."""
        est = self.advisor.estimate_compression(_prose(8))  # ~90 tokens
        assert est.compression_ratio <= 1.5
        assert "as-is" in est.reasoning

    def test_sub_8k_doc_estimate_is_conservative_not_7x(self):
        """Locks the dogfood regression: the engine keeps ~80% of nodes below
        8K tokens, so estimates there must be modest (measured 1.17x), never
        the legacy 7-9x."""
        est = self.advisor.estimate_compression(_prose(50))  # ~600 tokens
        assert est.compression_ratio < 2.5, (
            f"sub-8K estimate overclaims again: {est.compression_ratio:.2f}x "
            "(engine keeps ~80% of nodes at this size)"
        )
        assert est.confidence == "low"
        assert "80%" in est.reasoning

    def test_band_estimates_are_monotonic_nondecreasing(self):
        """Bigger docs never get a SMALLER estimate — the curve must be sane."""
        sizes = [50, 800, 3_200, 9_500]  # ~600 / ~9.6K / ~38K / ~114K tokens
        ratios = [self.advisor.estimate_compression(_prose(n)).compression_ratio for n in sizes]
        assert ratios == sorted(ratios), f"non-monotonic estimate curve: {ratios}"

    def test_largest_band_stays_below_measured_ceiling_plus_margin(self):
        """100K+ band estimates ~6-10x — within sight of the 9.54x benchmark,
        never the legacy 15-20x."""
        est = self.advisor.estimate_compression(_prose(9_500))  # ~114K tokens
        assert 5.0 <= est.compression_ratio <= 12.0
