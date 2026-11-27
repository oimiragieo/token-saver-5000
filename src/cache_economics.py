"""Prompt-cache economics contracts and CAPC (Cache-Aware Prompt Compression) math.

Models provider-specific prompt-cache pricing (write multiplier ``alpha``,
read multiplier ``beta``, and persistent-tier minimum token threshold ``T``),
crossover hit rates, tier-preserving max compression ratios, and turn-by-turn
longest-common-prefix billing.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import threading
from typing import Any
from weakref import WeakKeyDictionary


@dataclass(frozen=True)
class ProviderCacheProfile:
    """Prompt-cache pricing and tier profile for a model provider."""

    name: str
    input_rate_per_1m: float
    cache_write_mult: float
    cache_read_mult: float
    persistent_tier_tokens: int


ANTHROPIC_SONNET_4_6 = ProviderCacheProfile(
    name="anthropic_sonnet_4_6",
    input_rate_per_1m=3.0,
    cache_write_mult=1.25,
    cache_read_mult=0.10,
    persistent_tier_tokens=3500,
)

OPENAI_GPT5 = ProviderCacheProfile(
    name="openai_gpt5",
    input_rate_per_1m=2.50,
    cache_write_mult=1.0,
    cache_read_mult=0.25,
    persistent_tier_tokens=1024,
)

DEEPSEEK_V3 = ProviderCacheProfile(
    name="deepseek_v3",
    input_rate_per_1m=0.27,
    cache_write_mult=1.0,
    cache_read_mult=0.10,
    persistent_tier_tokens=64,
)

GEMINI_FLASH = ProviderCacheProfile(
    name="gemini_flash",
    input_rate_per_1m=0.15,
    cache_write_mult=1.0,
    cache_read_mult=0.25,
    persistent_tier_tokens=1024,
)

PROVIDER_CACHE_PROFILES: dict[str, ProviderCacheProfile] = {
    "anthropic_sonnet_4_6": ANTHROPIC_SONNET_4_6,
    "openai_gpt5": OPENAI_GPT5,
    "deepseek_v3": DEEPSEEK_V3,
    "gemini_flash": GEMINI_FLASH,
}


def get_provider_cache_profile(name: str) -> ProviderCacheProfile:
    """Resolve a preset or model ID to a ``ProviderCacheProfile``."""
    if name in PROVIDER_CACHE_PROFILES:
        return PROVIDER_CACHE_PROFILES[name]

    lower = (name or "").strip().lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4")) or "openai" in lower or "codex" in lower:
        return OPENAI_GPT5
    if lower.startswith("deepseek") or "deepseek" in lower:
        return DEEPSEEK_V3
    if lower.startswith("gemini") or "gemini" in lower:
        return GEMINI_FLASH
    if lower.startswith(("claude", "sonnet", "opus", "haiku")) or "anthropic" in lower:
        return ANTHROPIC_SONNET_4_6
    return ANTHROPIC_SONNET_4_6


@dataclass(frozen=True)
class ExplainAnalyzeTrace:
    """Structured explanation of a cache-aware compression decision."""

    decision: str
    reason: str
    recommended_ratio: float
    crossover_hit_rate: float
    tier_preserving_max_ratio: int
    estimated_net_savings_pct: float


TOKENIZER_NAME = "tiktoken:cl100k_base"
_SEGMENT_CACHE_MAX_CHARS = 65_536
_SEGMENT_CACHE_MAX_SIZE = 256

_SEGMENT_CACHE_LOCK = threading.Lock()
_SEGMENT_COUNT_CACHES: WeakKeyDictionary[Any, OrderedDict[tuple[bytes, int], int]] = (
    WeakKeyDictionary()
)


@lru_cache(maxsize=1)
def _get_encoder() -> Any:
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError("tiktoken is required for cache_economics token/cost accounting") from exc
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        raise RuntimeError(f"Failed to load {TOKENIZER_NAME} encoder for cache_economics") from exc


def _segment_cache_key(seg: str) -> tuple[bytes, int]:
    digest = hashlib.blake2b(seg.encode("utf-8", errors="surrogatepass"), digest_size=32).digest()
    return (digest, len(seg))


def _count_segment_tokens(enc: Any, seg: str) -> int:
    if not seg:
        return 0
    if len(seg) > _SEGMENT_CACHE_MAX_CHARS:
        return len(enc.encode(seg))

    key = _segment_cache_key(seg)
    with _SEGMENT_CACHE_LOCK:
        try:
            cache = _SEGMENT_COUNT_CACHES.get(enc)
            if cache is None:
                cache = OrderedDict()
                _SEGMENT_COUNT_CACHES[enc] = cache
        except TypeError:
            cache = None
        if cache is not None:
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                return cached

    count = len(enc.encode(seg))
    with _SEGMENT_CACHE_LOCK:
        try:
            cache = _SEGMENT_COUNT_CACHES.get(enc)
            if cache is None:
                cache = OrderedDict()
                _SEGMENT_COUNT_CACHES[enc] = cache
        except TypeError:
            cache = None
        if cache is not None:
            if key in cache:
                cache.move_to_end(key)
            else:
                cache[key] = count
                while len(cache) > _SEGMENT_CACHE_MAX_SIZE:
                    cache.popitem(last=False)
    return count


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return _count_segment_tokens(_get_encoder(), text)


def compute_crossover_hit_rate(r: float, alpha: float, beta: float) -> float:
    """Return the CAPC crossover cache-hit rate ``h*``.

    Formula: ``(alpha - 1.0 / r) / (alpha - beta)``, clamped to ``[0.0, 1.0]``.
    Returns ``1.0`` when ``r <= 1.0`` or ``alpha <= beta``.
    """
    if r <= 1.0 or alpha <= beta:
        return 1.0
    raw = (alpha - 1.0 / r) / (alpha - beta)
    return max(0.0, min(1.0, raw))


def compute_tier_preserving_max_ratio(doc_tokens: int, tier_tokens: int = 3500) -> int:
    """Return max integer compression ratio that keeps ``doc_tokens / r >= tier_tokens``."""
    if tier_tokens <= 0 or doc_tokens < tier_tokens:
        return 1
    return max(1, doc_tokens // tier_tokens)


def bill_prefix_turn(
    prev_segments: list[str],
    curr_segments: list[str],
    profile: ProviderCacheProfile,
) -> dict[str, float | int]:
    """Bill a turn using longest-common-token-prefix cache reads vs fresh writes."""
    prefix_segments = 0
    for prev_seg, curr_seg in zip(prev_segments, curr_segments):
        if prev_seg == curr_seg:
            prefix_segments += 1
        else:
            break

    enc = _get_encoder()
    prev_ne = [seg for seg in prev_segments if seg]
    curr_ne = [seg for seg in curr_segments if seg]

    k = 0
    lcp_tokens = 0
    total_tokens = 0
    limit = min(len(prev_ne), len(curr_ne))
    while k < limit and prev_ne[k] == curr_ne[k]:
        seg_len = _count_segment_tokens(enc, prev_ne[k])
        lcp_tokens += seg_len
        total_tokens += seg_len
        k += 1

    prev_rem: list[int] = []
    for seg in prev_ne[k:]:
        prev_rem.extend(enc.encode(seg))
    curr_rem: list[int] = []
    for seg in curr_ne[k:]:
        curr_rem.extend(enc.encode(seg))

    total_tokens += len(curr_rem)
    for p_tok, c_tok in zip(prev_rem, curr_rem):
        if p_tok == c_tok:
            lcp_tokens += 1
        else:
            break
    cached_tokens = lcp_tokens if lcp_tokens >= profile.persistent_tier_tokens else 0
    fresh_tokens = total_tokens - cached_tokens

    rate_per_tok = profile.input_rate_per_1m / 1_000_000.0
    cache_read_dollars = cached_tokens * rate_per_tok * profile.cache_read_mult
    cache_write_dollars = fresh_tokens * rate_per_tok * profile.cache_write_mult
    net_billed_dollars = cache_read_dollars + cache_write_dollars
    uncached_dollars = total_tokens * rate_per_tok
    dollars_saved_vs_uncached = uncached_dollars - net_billed_dollars

    return {
        "cached_prefix_segments": prefix_segments,
        "lcp_tokens": lcp_tokens,
        "cached_tokens": cached_tokens,
        "fresh_tokens": fresh_tokens,
        "total_tokens": total_tokens,
        "cache_read_dollars": round(cache_read_dollars, 8),
        "cache_write_dollars": round(cache_write_dollars, 8),
        "net_billed_dollars": round(net_billed_dollars, 8),
        "dollars_saved_vs_uncached": round(dollars_saved_vs_uncached, 8),
    }
