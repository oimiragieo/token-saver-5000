"""Optional local loopback extractive LoRA pruner adapter (Wave 3).

Enforces strict loopback-only URL validation, intent-budget protection against
num_ctx overflow (arXiv:2608.24188), segment-level SHA-256 caching, and
verbatim-identifier extractiveness verification with automatic fallback.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .token_estimation import TokenEstimator

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_MAX_RESPONSE_BYTES = 1_048_576
_PROMPT_OVERHEAD_TOKENS = 4


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject all HTTP 3xx redirects to prevent SSRF redirect escapes."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            f"HTTP redirects are forbidden ({code})",
            headers,
            fp,
        )


def _verify_resolved_loopback(url: str) -> None:
    """Resolve hostname via socket.getaddrinfo and verify all IPs are loopback."""
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    port = parsed.port or 80
    infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    if not infos:
        raise ValueError(f"Could not resolve loopback host {hostname!r}")
    for _, _, _, _, sockaddr in infos:
        ip_str = str(sockaddr[0]).split("%")[0]
        ip_obj = ipaddress.ip_address(ip_str)
        if not ip_obj.is_loopback:
            raise ValueError(f"Resolved non-loopback IP {ip_str!r} for host {hostname!r}")


def validate_loopback_url(url: str) -> str:
    """Validate that *url* is a clean HTTP loopback endpoint.

    Allows only ``http`` scheme and ``127.0.0.1``, ``::1``, or ``localhost``.
    Rejects credentials, non-loopback hosts, invalid ports, and query/fragments.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("base_url must be a non-empty string")

    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme != "http":
        raise ValueError(
            f"Only 'http' scheme is allowed for loopback endpoints, got {parsed.scheme!r}"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials in base_url are forbidden")
    if parsed.query or parsed.fragment:
        raise ValueError("Query parameters and fragments are not allowed in base_url")

    hostname = (parsed.hostname or "").lower()
    if hostname not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"Non-loopback host {hostname!r} is forbidden; allowed: {sorted(_LOOPBACK_HOSTS)}"
        )

    # Accessing parsed.port raises ValueError on malformed/out-of-range ports
    port = parsed.port
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"Port out of range: {port}")

    return url.strip().rstrip("/")


@dataclass
class ExtractivePrunerConfig:
    """Configuration for the local loopback extractive LoRA pruner."""

    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3-4b-extractive"
    num_ctx: int = 8192
    min_num_predict: int = 256
    ctx_safety_margin: int = 128
    timeout_s: float = 5.0

    def __post_init__(self) -> None:
        self.base_url = validate_loopback_url(self.base_url)
        if self.num_ctx <= 0:
            raise ValueError("num_ctx must be positive")
        if self.min_num_predict <= 0:
            raise ValueError("min_num_predict must be positive")
        if self.ctx_safety_margin < 0:
            raise ValueError("ctx_safety_margin must be non-negative")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")


def _intent_budget(
    system_tokens: int,
    content_tokens: int,
    num_ctx: int,
    min_num_predict: int,
    ctx_safety_margin: int,
) -> Tuple[int, int]:
    """Compute safe (budgeted_content_tokens, num_predict) within *num_ctx*.

    Guarantees:
        system_tokens + budgeted_content_tokens + num_predict + ctx_safety_margin <= num_ctx
        num_predict >= min_num_predict
    """
    if system_tokens < 0 or content_tokens < 0 or ctx_safety_margin < 0:
        raise ValueError("Token counts and safety margin must be non-negative")
    if num_ctx <= 0 or min_num_predict <= 0:
        raise ValueError("num_ctx and min_num_predict must be positive")

    max_content_tokens = num_ctx - system_tokens - min_num_predict - ctx_safety_margin
    if max_content_tokens < 0:
        raise ValueError(
            "num_ctx is too small for system_tokens + min_num_predict + ctx_safety_margin"
        )

    budgeted_content = min(content_tokens, max_content_tokens)
    num_predict = max(
        min_num_predict,
        num_ctx - system_tokens - budgeted_content - ctx_safety_margin,
    )
    return budgeted_content, num_predict


def check_extractiveness(
    source_text: str,
    pruned_text: str,
    min_ratio: float = 0.95,
) -> Tuple[bool, float]:
    """Verify >= *min_ratio* of emitted identifiers appear verbatim in *source_text*."""
    if not pruned_text.strip():
        return (not source_text.strip(), 1.0 if not source_text.strip() else 0.0)

    emitted = _IDENT_RE.findall(pruned_text)
    if not emitted:
        verbatim = pruned_text.strip() in source_text
        return (verbatim, 1.0 if verbatim else 0.0)

    source_idents = set(_IDENT_RE.findall(source_text))
    matched = sum(1 for tok in emitted if tok in source_idents)
    ratio = matched / len(emitted)
    return (ratio >= min_ratio, ratio)


@dataclass
class ExtractivePruneResult:
    """Result of an extractive pruning attempt."""

    text: str
    used_lora: bool
    cache_hit: bool
    extractiveness_ratio: float
    fallback_reason: Optional[str] = None


class ExtractiveLoRAAdapter:
    """Optional loopback extractive LoRA adapter with caching and safe fallback."""

    def __init__(
        self,
        config: Optional[ExtractivePrunerConfig] = None,
        min_extractiveness: float = 0.95,
        transport: Optional[Callable[[str, dict[str, Any], float], str]] = None,
        fallback_fn: Optional[Callable[[str], str]] = None,
        token_estimator: Optional[TokenEstimator] = None,
    ) -> None:
        self.config = config or ExtractivePrunerConfig()
        self.min_extractiveness = min_extractiveness
        self._transport = transport
        self._fallback_fn = fallback_fn
        self._estimator = token_estimator or TokenEstimator()
        self._cache: Dict[str, str] = {}

    @staticmethod
    def _segment_key(model: str, segment: str) -> str:
        return hashlib.sha256(f"{model}:{segment}".encode("utf-8")).hexdigest()

    def _default_fallback(self, text: str) -> str:
        if self._fallback_fn is not None:
            return self._fallback_fn(text)
        from .cli_output_optimizer import CLIOutputOptimizer

        return CLIOutputOptimizer().filter(text, fallback_strategy="generic").filtered_text

    def _call_loopback(self, prompt: str, num_predict: int) -> str:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": self.config.num_ctx,
                "num_predict": num_predict,
                "temperature": 0.0,
            },
        }
        if self._transport is not None:
            return self._transport(
                f"{self.config.base_url}/api/generate", payload, self.config.timeout_s
            )

        _verify_resolved_loopback(self.config.base_url)
        req = urllib.request.Request(
            url=f"{self.config.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )
        with opener.open(req, timeout=self.config.timeout_s) as resp:
            raw_bytes = resp.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw_bytes) > _MAX_RESPONSE_BYTES:
                raise ValueError(f"Response body exceeds {_MAX_RESPONSE_BYTES} bytes cap")
            body = raw_bytes.decode("utf-8")
            parsed = json.loads(body)
            return str(parsed.get("response", ""))

    def prune(
        self, text: str, system_prompt: str = "Extract key lines verbatim."
    ) -> ExtractivePruneResult:
        """Prune *text* via local loopback LoRA, falling back cleanly on any failure."""
        if not text or not text.strip():
            return ExtractivePruneResult(
                text=text, used_lora=False, cache_hit=False, extractiveness_ratio=1.0
            )

        cache_key = self._segment_key(self.config.model, text)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            ok, ratio = check_extractiveness(text, cached, self.min_extractiveness)
            if ok:
                return ExtractivePruneResult(
                    text=cached, used_lora=True, cache_hit=True, extractiveness_ratio=ratio
                )

        # Use TokenEstimator + explicit prompt framing overhead to enforce _intent_budget
        sys_tokens = max(1, self._estimator.count_tokens(system_prompt) + _PROMPT_OVERHEAD_TOKENS)
        content_tokens = max(1, self._estimator.count_tokens(text))
        try:
            budgeted_tokens, num_predict = _intent_budget(
                system_tokens=sys_tokens,
                content_tokens=content_tokens,
                num_ctx=self.config.num_ctx,
                min_num_predict=self.config.min_num_predict,
                ctx_safety_margin=self.config.ctx_safety_margin,
            )
        except ValueError as exc:
            return ExtractivePruneResult(
                text=self._default_fallback(text),
                used_lora=False,
                cache_hit=False,
                extractiveness_ratio=1.0,
                fallback_reason=f"budget_error: {exc}",
            )

        if budgeted_tokens <= 0:
            return ExtractivePruneResult(
                text=self._default_fallback(text),
                used_lora=False,
                cache_hit=False,
                extractiveness_ratio=1.0,
                fallback_reason="budget_error: available_content_tokens <= 0",
            )

        if content_tokens > budgeted_tokens:
            max_chars = max(1, int(len(text) * (budgeted_tokens / content_tokens)))
            bounded_text = text[:max_chars]
            while bounded_text and self._estimator.count_tokens(bounded_text) > budgeted_tokens:
                next_len = int(len(bounded_text) * 0.9)
                if next_len >= len(bounded_text):
                    next_len = len(bounded_text) - 1
                bounded_text = bounded_text[:next_len]
            if not bounded_text:
                return ExtractivePruneResult(
                    text=self._default_fallback(text),
                    used_lora=False,
                    cache_hit=False,
                    extractiveness_ratio=1.0,
                    fallback_reason="budget_error: truncated content exceeds available_content_tokens",
                )
        else:
            bounded_text = text
        prompt = f"{system_prompt}\n\n{bounded_text}"

        try:
            candidate = self._call_loopback(prompt, num_predict=num_predict)
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            return ExtractivePruneResult(
                text=self._default_fallback(text),
                used_lora=False,
                cache_hit=False,
                extractiveness_ratio=1.0,
                fallback_reason=f"transport_error: {exc}",
            )

        ok, ratio = check_extractiveness(bounded_text, candidate, self.min_extractiveness)
        if not ok:
            return ExtractivePruneResult(
                text=self._default_fallback(text),
                used_lora=False,
                cache_hit=False,
                extractiveness_ratio=ratio,
                fallback_reason=f"extractiveness_below_threshold:{ratio:.3f}",
            )

        self._cache[cache_key] = candidate
        return ExtractivePruneResult(
            text=candidate,
            used_lora=True,
            cache_hit=False,
            extractiveness_ratio=ratio,
        )
