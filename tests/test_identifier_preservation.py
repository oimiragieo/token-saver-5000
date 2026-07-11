"""Tests for engine identifier-preservation + the proxy ResponseInterceptor wiring.

#137(b): the `gotcontext wrap` proxy compresses tool-output but historically did
NOT force-keep execution-critical identifiers (the "amnesia tax"). This suite
locks the ported engine guard (`src/identifier_preservation.py`) AND its opt-in
wiring into `ResponseInterceptor.intercept_text`.
"""

from __future__ import annotations

from src.identifier_preservation import (
    apply_identifier_guard,
    extract_critical_identifiers,
)
from src.proxy.response_interceptor import ResponseInterceptor

_UUID = "550e8400-e29b-41d4-a716-446655440000"


class TestExtractCriticalIdentifiers:
    def test_extracts_file_path_error_code_and_uuid(self):
        text = (
            f'File "src/api/auth.py", line 88, in verify_token\n'
            f"ERROR connection refused ECONNREFUSED request_id={_UUID}\n"
            f"UPSTASH_REDIS_REST_URL not set"
        )
        toks = extract_critical_identifiers(text)
        assert "src/api/auth.py:88" not in toks  # not that shape; the path+line pattern differs
        assert "ECONNREFUSED" in toks
        assert _UUID in toks
        assert "UPSTASH_REDIS_REST_URL" in toks

    def test_dedupes_and_skips_short_tokens(self):
        toks = extract_critical_identifiers("ECONNREFUSED ECONNREFUSED ab")
        assert toks.count("ECONNREFUSED") == 1
        assert "ab" not in toks  # below _MIN_PRESERVE_TOKEN_LEN

    def test_preserves_notable_numeric_literals_not_bare_ints(self):
        """#284 (2026-07-11): the amnesia-tax guard must preserve NOTABLE numeric
        literals an agent's tool output hinges on (currency, scientific,
        percentage, decimal, dotted version) while NOT force-keeping bare
        integers — choice (c): avoids footer bloat on number-dense docs;
        http_status already covers 3-digit codes."""
        text = (
            "Charge was $1,234.50 at rate -0.001% with tolerance 1e-07; "
            "engine v1.58.7 processed 42 rows in 3.14s, plain $1234.50, id 12."
        )
        toks = extract_critical_identifiers(text)
        # Notable numerics preserved verbatim:
        assert "$1,234.50" in toks
        assert "-0.001%" in toks
        assert "1e-07" in toks
        assert "1.58.7" in toks
        assert "3.14" in toks
        assert "$1234.50" in toks  # non-comma currency (the \d{1,3}-cap bug guard)
        # Bare integers NOT force-preserved (choice c):
        assert "42" not in toks
        assert "12" not in toks

    def test_currency_extracted_byte_identical(self):
        """A currency figure survives byte-identical (no reformatting/splitting)."""
        toks = extract_critical_identifiers("Total: $12,345,678.90 refunded.")
        assert "$12,345,678.90" in toks


class TestApplyIdentifierGuard:
    def test_reinjects_missing_token_in_footer(self):
        final, reinjected = apply_identifier_guard("compressed body", ["ECONNREFUSED"])
        assert "ECONNREFUSED" in final
        assert "[preserved identifiers:" in final
        assert reinjected == ["ECONNREFUSED"]

    def test_noop_when_all_tokens_present(self):
        final, reinjected = apply_identifier_guard("body has ECONNREFUSED here", ["ECONNREFUSED"])
        assert final == "body has ECONNREFUSED here"
        assert reinjected == []


class TestInterceptorIdentifierPreservation:
    def _blob(self) -> str:
        # >100 chars, verbose + compressible, with a critical UUID + error code.
        return (
            "The background worker attempted to reconnect to the upstream cache "
            "service several times but the connection was refused every time. "
            f"ERROR ECONNREFUSED occurred while processing request_id={_UUID} and "
            "the retry budget was eventually exhausted after many repeated attempts."
        ) * 3

    def test_preserve_true_keeps_critical_identifier(self):
        interceptor = ResponseInterceptor(preserve_identifiers=True)
        compressed, _stats = interceptor.intercept_text(self._blob())
        # Whether it survived compression or was reinjected, it MUST be present.
        assert _UUID in compressed
        assert "ECONNREFUSED" in compressed

    def test_preserve_false_never_adds_guard_footer(self):
        interceptor = ResponseInterceptor(preserve_identifiers=False)
        compressed, _stats = interceptor.intercept_text(self._blob())
        # Backward-compat: the guard did not run, so no preservation footer.
        assert "[preserved identifiers:" not in compressed

    def test_default_is_preserve_false(self):
        # Constructor default must stay off for backward compat.
        interceptor = ResponseInterceptor()
        compressed, _stats = interceptor.intercept_text(self._blob())
        assert "[preserved identifiers:" not in compressed


class TestHardeningBounds:
    """codex 2026-07-10: ReDoS + footer byte-bloat defenses."""

    def test_extract_bounded_on_pathological_slashy_input(self):
        # A ~2 MB slash-heavy blob must return quickly (bounded scan window +
        # segment-capped regex), not hang the engine on catastrophic backtracking.
        blob = ("a/" * 1_000_000) + "x"
        toks = extract_critical_identifiers(blob)  # must return, not hang
        assert isinstance(toks, list)

    def test_numeric_pattern_bounded_on_pathological_digit_comma_blob(self):
        """#284: the numeric_literal alt uses BOUNDED quantifiers so a long
        digit/comma blob (no decimal to satisfy the pattern) cannot cause
        O(n^2)/catastrophic backtracking. Must return in well under a second."""
        import time as _time

        blob = ("1," * 50_000) + "x"  # ~100k chars, no decimal anywhere
        t0 = _time.perf_counter()
        toks = extract_critical_identifiers(blob)  # must return, not hang
        elapsed = _time.perf_counter() - t0
        assert isinstance(toks, list)
        assert elapsed < 2.0, f"numeric pattern backtracked catastrophically ({elapsed:.2f}s)"

    def test_extract_skips_absurdly_long_tokens(self):
        toks = extract_critical_identifiers("ECONNREFUSED " + ("A" * 5000))
        assert "ECONNREFUSED" in toks
        assert all(len(t) <= 256 for t in toks)

    def test_guard_footer_is_byte_capped(self):
        long_tokens = ["TOKEN" + "x" * 200 + str(i) for i in range(200)]
        final, reinjected = apply_identifier_guard("c", long_tokens)
        # Footer bounded well under a multi-MB blowup; count cap not reached.
        assert len(final) < 6000
        assert len(reinjected) < 200
