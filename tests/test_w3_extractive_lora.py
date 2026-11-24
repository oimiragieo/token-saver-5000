"""Tests for Wave 3 local loopback Extractive LoRA tier."""

from __future__ import annotations

import pytest

from src.cli_output_optimizer import CLIOutputOptimizer
from src.extractive_lora_adapter import (
    ExtractiveLoRAAdapter,
    ExtractivePrunerConfig,
    _intent_budget,
    check_extractiveness,
    validate_loopback_url,
)


class TestLoopbackUrlValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:11434",
            "http://localhost:8080",
            "http://[::1]:11434",
            "http://LOCALHOST:9000/",
        ],
    )
    def test_accepts_valid_loopback_urls(self, url):
        assert validate_loopback_url(url).startswith("http://")

    @pytest.mark.parametrize(
        "url",
        [
            "https://127.0.0.1:11434",
            "http://user:pass@127.0.0.1:11434",
            "http://192.168.1.10:11434",
            "http://0.0.0.0:11434",
            "http://example.com:11434",
            "http://127.0.0.1:99999",
            "http://127.0.0.1:11434?leak=1",
            "",
        ],
    )
    def test_rejects_non_loopback_or_unsafe_urls(self, url):
        with pytest.raises(ValueError):
            ExtractivePrunerConfig(base_url=url)


class TestIntentBudgetAndExtractiveness:
    def test_intent_budget_prevents_num_ctx_overflow(self):
        budgeted_content, num_predict = _intent_budget(
            system_tokens=200,
            content_tokens=50_000,
            num_ctx=4096,
            min_num_predict=256,
            ctx_safety_margin=128,
        )
        assert num_predict >= 256
        assert 200 + budgeted_content + num_predict + 128 <= 4096

    def test_intent_budget_raises_when_ctx_impossible(self):
        with pytest.raises(ValueError):
            _intent_budget(
                system_tokens=500,
                content_tokens=100,
                num_ctx=600,
                min_num_predict=256,
                ctx_safety_margin=128,
            )

    def test_extractiveness_check_passes_and_fails(self):
        src = "def verify_signature(payload_bytes, public_key): return True"
        ok, ratio = check_extractiveness(src, "def verify_signature(payload_bytes, public_key):")
        assert ok is True
        assert ratio == 1.0

        bad, bad_ratio = check_extractiveness(src, "hallucinated_symbol invented_func other_var")
        assert bad is False
        assert bad_ratio < 0.95


class TestExtractiveLoRAAdapter:
    def test_caches_segment_and_returns_lora_output(self):
        calls = 0

        def fake_transport(url: str, payload: dict, timeout: float) -> str:
            nonlocal calls
            calls += 1
            return "ERROR auth_handler failed at verify_token"

        adapter = ExtractiveLoRAAdapter(transport=fake_transport)
        text = "INFO boot\nERROR auth_handler failed at verify_token\nINFO done\n" * 3

        res1 = adapter.prune(text)
        assert res1.used_lora is True
        assert res1.cache_hit is False
        assert res1.extractiveness_ratio >= 0.95
        assert calls == 1

        res2 = adapter.prune(text)
        assert res2.used_lora is True
        assert res2.cache_hit is True
        assert calls == 1

    def test_falls_back_on_timeout_or_low_extractiveness(self):
        def timeout_transport(url: str, payload: dict, timeout: float) -> str:
            raise TimeoutError("loopback timed out")

        adapter_timeout = ExtractiveLoRAAdapter(transport=timeout_transport)
        sample = "alpha_service started\n" * 10
        res_timeout = adapter_timeout.prune(sample)
        assert res_timeout.used_lora is False
        assert "transport_error" in (res_timeout.fallback_reason or "")

        def hallucinating_transport(url: str, payload: dict, timeout: float) -> str:
            return "completely_invented_fn fabricated_var"

        adapter_halluc = ExtractiveLoRAAdapter(transport=hallucinating_transport)
        res_halluc = adapter_halluc.prune(sample)
        assert res_halluc.used_lora is False
        assert "extractiveness_below_threshold" in (res_halluc.fallback_reason or "")

    def test_cli_output_optimizer_uses_optional_extractive_adapter(self):
        def fake_transport(url: str, payload: dict, timeout: float) -> str:
            return "important_symbol_alpha"

        adapter = ExtractiveLoRAAdapter(transport=fake_transport)
        opt = CLIOutputOptimizer(extractive_adapter=adapter)
        raw = "important_symbol_alpha\n" + "\n".join(
            f"unclassified_record_{i} value_{i}" for i in range(12)
        )
        result = opt.filter(raw)
        assert result.command_detected == "unknown"
        assert result.strategy_applied == "extractive_lora"
        assert result.filtered_text == "important_symbol_alpha"


class TestExtractiveLoraSecurityAndTokenEstimator:
    def test_rejects_http_302_redirect_and_falls_back(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
                self.end_headers()

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            cfg = ExtractivePrunerConfig(base_url=f"http://127.0.0.1:{server.server_port}")
            adapter = ExtractiveLoRAAdapter(config=cfg)
            res = adapter.prune("alpha_symbol_one\nbeta_symbol_two\n")
            assert res.used_lora is False
            assert "transport_error" in (res.fallback_reason or "")
        finally:
            server.shutdown()
            server.server_close()

    def test_rejects_oversized_response_body(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading

        class BigHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = b'{"response":"' + (b"A" * 1_050_000) + b'"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), BigHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            cfg = ExtractivePrunerConfig(base_url=f"http://127.0.0.1:{server.server_port}")
            adapter = ExtractiveLoRAAdapter(config=cfg)
            res = adapter.prune("alpha_symbol_one\n")
            assert res.used_lora is False
            assert "exceeds" in (res.fallback_reason or "")
        finally:
            server.shutdown()
            server.server_close()

    def test_uses_token_estimator_for_intent_budget(self):
        class CountingEstimator:
            def __init__(self):
                self.calls = 0

            def count_tokens(self, text: str) -> int:
                self.calls += 1
                return max(1, len(text.split()))

        est = CountingEstimator()
        captured_payload = {}

        def fake_transport(url: str, payload: dict, timeout: float) -> str:
            captured_payload.update(payload)
            return "verify_token"

        adapter = ExtractiveLoRAAdapter(
            config=ExtractivePrunerConfig(num_ctx=1024, min_num_predict=256, ctx_safety_margin=128),
            transport=fake_transport,
            token_estimator=est,
        )
        res = adapter.prune("verify_token line one", system_prompt="Extract verbatim")
        assert res.used_lora is True
        assert est.calls >= 2
        assert captured_payload["options"]["num_predict"] >= 256

    def test_zero_available_content_tokens_falls_back_without_calling_endpoint(self):
        calls = 0

        def fail_if_called(url: str, payload: dict, timeout: float) -> str:
            nonlocal calls
            calls += 1
            return "should_not_be_called"

        # num_ctx=400, min_num_predict=256, ctx_safety_margin=128 => max 16 tokens for sys+content.
        # _PROMPT_OVERHEAD_TOKENS=4, so count_tokens("sys")=12 => sys_tokens=16 => budgeted_tokens=0.
        class ZeroBudgetEstimator:
            def count_tokens(self, text: str) -> int:
                if text == "sys":
                    return 12
                return 10

        adapter = ExtractiveLoRAAdapter(
            config=ExtractivePrunerConfig(num_ctx=400, min_num_predict=256, ctx_safety_margin=128),
            transport=fail_if_called,
            token_estimator=ZeroBudgetEstimator(),
        )
        res = adapter.prune("important_symbol_alpha\n", system_prompt="sys")
        assert res.used_lora is False
        assert calls == 0
        assert "available_content_tokens <= 0" in (res.fallback_reason or "")

        # Even with 1 token budget, never force 1 char if count_tokens(char) > budgeted_tokens.
        # _PROMPT_OVERHEAD_TOKENS=4, count_tokens("sys")=11 => sys_tokens=15 => budgeted_tokens=1.
        class OneBudgetEstimator:
            def count_tokens(self, text: str) -> int:
                if text == "sys":
                    return 11
                return 5 if text else 0

        adapter_one = ExtractiveLoRAAdapter(
            config=ExtractivePrunerConfig(num_ctx=400, min_num_predict=256, ctx_safety_margin=128),
            transport=fail_if_called,
            token_estimator=OneBudgetEstimator(),
        )
        res_one = adapter_one.prune("abcde", system_prompt="sys")
        assert res_one.used_lora is False
        assert calls == 0
        assert "exceeds available_content_tokens" in (res_one.fallback_reason or "")
