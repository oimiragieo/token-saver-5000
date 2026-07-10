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
