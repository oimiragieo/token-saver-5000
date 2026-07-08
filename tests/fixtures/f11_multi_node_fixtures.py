"""Sealed multi-node fixture corpus for the F11 ranker decision (#250).

WHY THIS FILE EXISTS (the gap it closes): the 5 existing sealed fixtures in
``tests/quality_gate_fixtures.py`` (qg_prose/code/mixed/json/qa) each
collapse to ONE real chunk after ingestion -- see that file's module
docstring -- so ``F11_RANKER_PATH`` (which only matters with >=2 ranking
candidates; the COMI coarse-filter needs >3 nodes) has ZERO test coverage
from that corpus. A prior measurement of Path A (dense cosine) vs Path C
(BM25+RRF) was a 1-win/1-loss/6-tie wash on only 8 adversarial queries --
no statistical power, and the aggregate hid two opposite-signed per-class
effects (Simpson's paradox -- see
``docs/audits/2026-07-08-f11-retrieval-fusion-ideas.md`` section 3).

This module formalizes 3 sealed, hand-labeled, MULTI-NODE documents (6
sections each, verified below and by ``tests/test_f11_fixture_corpus.py``
to chunk into 6 real nodes with the real engine) plus a stratified query
set covering the 7 query classes the design memo specifies:

    bare_numeric | identifier | quoted_phrase | keyword_nl |
    pure_paraphrase | lexical_trap | multi_hop

This is a FIRST INCREMENT, not the full spec (12-16 docs / ~240 queries).
3 fixtures land now with real per-class engagement; expanding to the full
corpus is future work -- see the TODO at the bottom of this file.

Ground truth is FIXED LABELS authored by hand here -- never computed by
``search_semantic_with_scores`` or any other ranker/embedder under test
(same independent-oracle discipline as ``tests/fixtures/quality_gate_fixtures.py``).

Two of the three documents reuse text VERBATIM from already-proven fixtures
rather than inventing new prose from scratch:

  * ``F11_ARCH_FIXTURE``'s first 5 sections are ``_SOURCE_ORDER_DOC`` from
    ``tests/test_quality_gate.py`` (``TestRealCompressorSourceOrderEndToEnd``)
    -- already proven to chunk into 5 real nodes with the real engine.
  * ``F11_CODE_FIXTURE`` / ``F11_JSON_FIXTURE``'s first 5 sections reuse the
    prior subagent's proven ``DOC_CODE`` / ``DOC_JSON`` scratchpad prototypes
    (same 5-node shape, same real-engine verification approach).

Each fixture adds ONE new "ZETA_*" section beyond the proven 5, both to
push the node count comfortably above the >3-node F11 engagement floor
(all three fixtures chunk into 6 real nodes -- see
``tests/test_f11_fixture_corpus.py::TestChunkCountEngagesF11``) and to host
a deliberately constructed ``lexical_trap`` query: a term that the design
memo's gated-fusion idea (#1 in the memo) must learn to discount, because
it occurs MORE often in the wrong (decoy) section than in the true answer
section. That numeric assertion is checked, model-free, in
``TestLexicalTrapConstructionIsValid``.

The ``pure_paraphrase`` queries are checked, model-free, in
``TestParaphraseQueriesHaveZeroContentOverlap`` -- each one is verified to
share ZERO non-stopword tokens with its gold section's raw text (the
design memo's explicit requirement: "construct by synonym-rewriting a gold
section, then programmatically verify zero non-stopword token overlap").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional

# ===========================================================================
# Query-class vocabulary (design memo section 3, "Query classes (stratified)")
# ===========================================================================

QUERY_CLASS_BARE_NUMERIC = "bare_numeric"
QUERY_CLASS_IDENTIFIER = "identifier"
QUERY_CLASS_QUOTED_PHRASE = "quoted_phrase"
QUERY_CLASS_KEYWORD_NL = "keyword_nl"
QUERY_CLASS_PURE_PARAPHRASE = "pure_paraphrase"
QUERY_CLASS_LEXICAL_TRAP = "lexical_trap"
QUERY_CLASS_MULTI_HOP = "multi_hop"

ALL_QUERY_CLASSES = (
    QUERY_CLASS_BARE_NUMERIC,
    QUERY_CLASS_IDENTIFIER,
    QUERY_CLASS_QUOTED_PHRASE,
    QUERY_CLASS_KEYWORD_NL,
    QUERY_CLASS_PURE_PARAPHRASE,
    QUERY_CLASS_LEXICAL_TRAP,
    QUERY_CLASS_MULTI_HOP,
)


# ===========================================================================
# Content-word / stopword helpers -- used to PROVE (not assert-by-eye) that
# a "pure_paraphrase" query shares zero content-word overlap with its gold
# section, per the design memo's explicit requirement.
# ===========================================================================

_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "from",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
        "do",
        "does",
        "did",
        "doing",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "as",
        "belong",
        "means",
    }
)


def content_words(text: str) -> FrozenSet[str]:
    """Lowercase, punctuation-stripped, stopword-filtered token set."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return frozenset(t for t in tokens if t not in _STOPWORDS and len(t) > 1)


def term_count(text: str, term: str) -> int:
    """Case-insensitive substring count -- used to prove a lexical-trap
    decoy section literally contains the trap term MORE often than the
    true gold section (the construction-level claim, independent of what
    any ranker does with it)."""
    return text.lower().count(term.lower())


# ===========================================================================
# Fixture dataclasses
# ===========================================================================


@dataclass(frozen=True)
class F11Query:
    """One stratified query against a multi-node fixture."""

    query_id: str
    query_class: str
    query_text: str
    # Section marker(s) that hold the true answer. Exactly 1 for every
    # class except QUERY_CLASS_MULTI_HOP, which names 2.
    gold_markers: List[str]
    answer_spans: List[str]
    load_bearing_tokens: List[str] = field(default_factory=list)
    # lexical_trap only: the term whose raw-text frequency is HIGHER in
    # decoy_marker's section than in gold_markers[0]'s section.
    trap_term: Optional[str] = None
    decoy_marker: Optional[str] = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.query_class not in ALL_QUERY_CLASSES:
            raise ValueError(f"unknown query_class: {self.query_class!r}")
        expected_gold = 2 if self.query_class == QUERY_CLASS_MULTI_HOP else 1
        if len(self.gold_markers) != expected_gold:
            raise ValueError(
                f"{self.query_id}: query_class={self.query_class!r} expects "
                f"{expected_gold} gold_markers, got {len(self.gold_markers)}"
            )
        if self.query_class == QUERY_CLASS_LEXICAL_TRAP and (
            not self.trap_term or not self.decoy_marker
        ):
            raise ValueError(
                f"{self.query_id}: lexical_trap queries require trap_term + decoy_marker"
            )


@dataclass(frozen=True)
class F11MultiNodeFixture:
    """A sealed, multi-section document + its stratified query set."""

    fixture_id: str
    doc_type: str  # "prose" | "code" | "config"
    # marker -> raw section text (WITHOUT the "## MARKER\n\n" prefix),
    # in original document order.
    sections: "dict[str, str]"
    queries: List[F11Query]

    @property
    def section_order(self) -> List[str]:
        return list(self.sections.keys())

    @property
    def source_text(self) -> str:
        """Header-aware doc text: '## MARKER\\n\\n<section text>' per
        section, blank-line separated -- the shape proven (by the real
        engine, in ``tests/test_quality_gate.py::_SOURCE_ORDER_DOC``) to
        chunk one node per section, unlike the colon-prefixed single-node
        shape used by ``tests/fixtures/quality_gate_fixtures.py``."""
        return "\n\n".join(f"## {marker}\n\n{text}" for marker, text in self.sections.items())

    def section_text(self, marker: str) -> str:
        return self.sections[marker]


# ===========================================================================
# Fixture 1 -- f11_arch_doc (prose / architecture doc).
# Sections ALPHA..EPSILON reused VERBATIM from
# tests/test_quality_gate.py::_SOURCE_ORDER_DOC. ZETA is new.
# ===========================================================================

_ARCH_SECTIONS = {
    "ALPHA_AUTH_SECTION": (
        "Authentication in gotcontext.ai resolves an inbound request to a "
        "(user_id, plan) tuple using one of three mechanisms: a Clerk-issued "
        "session JWT verified against the JWKS endpoint, a gc_ prefixed API key "
        "verified via HMAC signature, or a self-hosted Ed25519 license token. "
        "Each mechanism populates request.state with the resolved plan so "
        "downstream middleware can apply plan-gating without a second lookup."
    ),
    "BETA_BILLING_SECTION": (
        "Billing is handled entirely by Polar as the merchant of record. Webhook "
        "events for subscription.created, subscription.updated, and "
        "subscription.canceled are verified with Svix signatures before the "
        "handler mutates the local subscriptions table. A daily reconciliation "
        "cron treats the Polar API as the source of truth and never downgrades "
        "a user's plan on an empty or unparseable provider response."
    ),
    "GAMMA_CACHE_SECTION": (
        "The semantic cache stores compressed skeleton output keyed by a hash "
        "of the source document plus the requested fidelity level. Cache "
        "entries live in Upstash Redis with a five minute TTL for the plan "
        "cache and a longer TTL for the semantic cache proper, using pgvector "
        "HNSW indexing for approximate nearest-neighbor lookups."
    ),
    "DELTA_WEBHOOK_SECTION": (
        "Outbound webhook delivery is best effort with a durable retry queue "
        "backed by a webhook_deliveries table. A drain cron runs hourly and "
        "attempts redelivery for any row still marked pending, applying "
        "exponential backoff between attempts and giving up after a bounded "
        "number of retries so a permanently dead endpoint cannot loop forever."
    ),
    "EPSILON_RECAP_SECTION": (
        "To recap: authentication resolves a JWT or API key to a plan, billing "
        "runs through Polar webhooks reconciled daily against the provider, the "
        "semantic cache uses Redis and pgvector HNSW for fast lookups, and "
        "outbound webhook delivery retries through a durable queue with "
        "exponential backoff until a bounded retry ceiling is reached."
    ),
    "ZETA_MONITORING_SECTION": (
        "The dashboard's Queue Monitor widget polls the webhook backlog every "
        "few seconds and displays a live webhook retry counter; when the "
        "underlying webhook drain cron itself times out, the widget logs a "
        "retry-exhaustion warning without ever touching the actual delivery "
        "queue rows."
    ),
}

F11_ARCH_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_arch_doc",
    doc_type="prose",
    sections=_ARCH_SECTIONS,
    queries=[
        F11Query(
            query_id="arch_keyword_nl",
            query_class=QUERY_CLASS_KEYWORD_NL,
            query_text="What TTL and index does the semantic cache use?",
            gold_markers=["GAMMA_CACHE_SECTION"],
            answer_spans=["using pgvector HNSW indexing for approximate nearest-neighbor lookups"],
            load_bearing_tokens=["HNSW"],
        ),
        F11Query(
            query_id="arch_pure_paraphrase",
            query_class=QUERY_CLASS_PURE_PARAPHRASE,
            query_text=(
                "By what means does the software figure out who is calling and "
                "what tier they belong to?"
            ),
            gold_markers=["ALPHA_AUTH_SECTION"],
            answer_spans=["a gc_ prefixed API key verified via HMAC signature"],
            load_bearing_tokens=["JWKS"],
            note="Zero non-stopword overlap with ALPHA_AUTH_SECTION verified by test.",
        ),
        F11Query(
            query_id="arch_multi_hop",
            query_class=QUERY_CLASS_MULTI_HOP,
            query_text=(
                "Summarize how authentication and billing both rely on "
                "cryptographic verification before trusting an inbound event."
            ),
            gold_markers=["ALPHA_AUTH_SECTION", "BETA_BILLING_SECTION"],
            answer_spans=[
                "a gc_ prefixed API key verified via HMAC signature",
                "verified with Svix signatures before the handler mutates the local subscriptions table",
            ],
            load_bearing_tokens=["JWKS", "Svix"],
        ),
        F11Query(
            query_id="arch_lexical_trap",
            query_class=QUERY_CLASS_LEXICAL_TRAP,
            query_text="How is a webhook event handled and retried if it fails?",
            gold_markers=["DELTA_WEBHOOK_SECTION"],
            answer_spans=[
                "giving up after a bounded number of retries so a permanently "
                "dead endpoint cannot loop forever"
            ],
            load_bearing_tokens=["webhook_deliveries"],
            trap_term="webhook",
            decoy_marker="ZETA_MONITORING_SECTION",
            note="ZETA mentions 'webhook' 3x / 'retry' 2x vs DELTA's 2x / 1x -- verified by test.",
        ),
        F11Query(
            query_id="arch_identifier",
            query_class=QUERY_CLASS_IDENTIFIER,
            query_text="JWKS",
            gold_markers=["ALPHA_AUTH_SECTION"],
            answer_spans=["verified against the JWKS endpoint"],
            load_bearing_tokens=["JWKS"],
        ),
        F11Query(
            query_id="arch_quoted_phrase",
            query_class=QUERY_CLASS_QUOTED_PHRASE,
            query_text='"never downgrades a user\'s plan on an empty or unparseable provider response"',
            gold_markers=["BETA_BILLING_SECTION"],
            answer_spans=[
                "never downgrades a user's plan on an empty or unparseable provider response"
            ],
            load_bearing_tokens=["Svix"],
        ),
    ],
)


# ===========================================================================
# Fixture 2 -- f11_code_doc (code-flavored). Sections ALPHA..EPSILON reused
# verbatim from the prior subagent's f11_pathc_quality_compare.py DOC_CODE.
# ZETA is new.
# ===========================================================================

_CODE_SECTIONS = {
    "ALPHA_AUTH_SECTION": (
        "def verify_api_key(key: str) -> bool: the function first checks that "
        "the key starts with the gc_ prefix and is exactly 40 characters long, "
        "then recomputes an HMAC-SHA256 digest over the key body using "
        "API_KEY_HMAC_SECRET and compares it in constant time against the "
        "stored signature suffix before returning True or False."
    ),
    "BETA_RATELIMIT_SECTION": (
        "RATE_LIMIT_PER_MINUTE is set to 120 requests per API key per rolling "
        "sixty second window. The function check_rate_limit increments a "
        "Redis counter with a sixty second TTL and raises "
        "RateLimitExceededError with a Retry-After header once the counter "
        "exceeds the configured ceiling for that key."
    ),
    "GAMMA_CACHE_SECTION": (
        "CACHE_TTL_SECONDS is configured to 300 for the embedding LRU cache. "
        "The function get_cached_value looks up the hashed document key in "
        "the in-process OrderedDict and returns None on a cache miss, moving "
        "the entry to the end of the dict on a hit to mark it as recently "
        "used for eviction ordering."
    ),
    "DELTA_RETRY_SECTION": (
        "The retry policy caps max_attempts at 4 and uses an exponential "
        "backoff starting at 250 milliseconds, doubling on each subsequent "
        "attempt, capped at a maximum delay of 8000 milliseconds before the "
        "call is finally abandoned and an error is propagated to the caller."
    ),
    "EPSILON_SUMMARY_SECTION": (
        "In summary, verify_api_key gates every request, check_rate_limit "
        "enforces 120 requests per minute, get_cached_value serves the "
        "300-second embedding cache, and the retry policy backs off "
        "exponentially over 4 attempts before giving up."
    ),
    "ZETA_METRICS_SECTION": (
        "The internal metrics exporter listens on port 8000 for Prometheus "
        "scraping, and a separate sidecar also binds 8000 on the loopback "
        "interface for local debugging, while the retry ceiling for that "
        "sidecar's own health probe is capped at 8000 milliseconds as well."
    ),
}

F11_CODE_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_code_doc",
    doc_type="code",
    sections=_CODE_SECTIONS,
    queries=[
        F11Query(
            query_id="code_bare_numeric",
            query_class=QUERY_CLASS_BARE_NUMERIC,
            query_text="250",
            gold_markers=["DELTA_RETRY_SECTION"],
            answer_spans=["starting at 250 milliseconds"],
            load_bearing_tokens=["250"],
        ),
        F11Query(
            query_id="code_lexical_trap",
            query_class=QUERY_CLASS_LEXICAL_TRAP,
            query_text="8000",
            gold_markers=["DELTA_RETRY_SECTION"],
            answer_spans=[
                "capped at a maximum delay of 8000 milliseconds before the call "
                "is finally abandoned"
            ],
            load_bearing_tokens=["8000"],
            trap_term="8000",
            decoy_marker="ZETA_METRICS_SECTION",
            note="ZETA mentions literal '8000' 3x vs DELTA's 1x -- verified by test.",
        ),
        F11Query(
            query_id="code_identifier",
            query_class=QUERY_CLASS_IDENTIFIER,
            query_text="API_KEY_HMAC_SECRET",
            gold_markers=["ALPHA_AUTH_SECTION"],
            answer_spans=[
                "recomputes an HMAC-SHA256 digest over the key body using API_KEY_HMAC_SECRET"
            ],
            load_bearing_tokens=["API_KEY_HMAC_SECRET"],
        ),
        F11Query(
            query_id="code_quoted_phrase",
            query_class=QUERY_CLASS_QUOTED_PHRASE,
            query_text='"raises RateLimitExceededError with a Retry-After header"',
            gold_markers=["BETA_RATELIMIT_SECTION"],
            answer_spans=["raises RateLimitExceededError with a Retry-After header"],
            load_bearing_tokens=["RateLimitExceededError"],
        ),
        F11Query(
            query_id="code_keyword_nl",
            query_class=QUERY_CLASS_KEYWORD_NL,
            query_text="What does get_cached_value return on a cache miss?",
            gold_markers=["GAMMA_CACHE_SECTION"],
            answer_spans=["returns None on a cache miss"],
            load_bearing_tokens=["CACHE_TTL_SECONDS"],
        ),
        F11Query(
            query_id="code_pure_paraphrase",
            query_class=QUERY_CLASS_PURE_PARAPHRASE,
            query_text="How does the system stop someone from calling too often?",
            gold_markers=["BETA_RATELIMIT_SECTION"],
            answer_spans=["raises RateLimitExceededError with a Retry-After header"],
            load_bearing_tokens=["RateLimitExceededError"],
            note=(
                "Zero non-stopword overlap with BETA_RATELIMIT_SECTION verified by "
                "test. This exact phrasing is the documented LOSS case from the "
                "n=8 adversarial run in the 2026-07-08 design memo."
            ),
        ),
        F11Query(
            query_id="code_multi_hop",
            query_class=QUERY_CLASS_MULTI_HOP,
            query_text=(
                "Explain how both the auth check and the rate limiter reject a "
                "bad or over-quota request."
            ),
            gold_markers=["ALPHA_AUTH_SECTION", "BETA_RATELIMIT_SECTION"],
            answer_spans=[
                "compares it in constant time against the stored signature suffix "
                "before returning True or False",
                "raises RateLimitExceededError with a Retry-After header",
            ],
            load_bearing_tokens=["API_KEY_HMAC_SECRET", "RateLimitExceededError"],
        ),
    ],
)


# ===========================================================================
# Fixture 3 -- f11_json_doc (config/table-heavy). Sections ALPHA..EPSILON
# reused verbatim from the prior subagent's f11_pathc_quality_compare.py
# DOC_JSON. ZETA is new.
# ===========================================================================

_JSON_SECTIONS = {
    "ALPHA_SERVICE_SECTION": (
        'The default service block is { "service": "gotcontext-api", '
        '"timeout_ms": 3000, "region": "iad" }. Every outbound HTTP call to a '
        "third-party provider inherits this timeout unless a route-specific "
        "override is configured in the provider profile registry."
    ),
    "BETA_RETRY_SECTION": (
        'The retry block is { "retry": { "max_attempts": 4, "backoff_ms": 250 '
        "} } and controls webhook redelivery for Polar and GitHub inbound "
        "events, applying the same exponential doubling used by the outbound "
        "webhook drain cron."
    ),
    "GAMMA_LIMIT_SECTION": (
        'The plan-gating record is { "tier": "pro", "monthly_limit": 2000000 '
        "} and is read by the middleware on every request to decide whether "
        "the caller has exceeded their monthly compression quota before the "
        "handler runs."
    ),
    "DELTA_CACHE_SECTION": (
        'The cache block is { "ttl_seconds": 300, "max_entries": 5000 } and '
        "governs the semantic cache's Redis-backed entry lifetime and the "
        "in-process LRU eviction ceiling before entries are proactively "
        "evicted."
    ),
    "EPSILON_RECAP_SECTION": (
        "Taken together, the service timeout, the retry backoff, the "
        "monthly plan limit, and the cache TTL are the four config blocks "
        "that govern request handling end to end for every gotcontext.ai "
        "API call."
    ),
    "ZETA_AUDIT_SECTION": (
        "The nightly usage-audit report cross-checks every project's "
        "monthly_limit against monthly_limit history snapshots taken at "
        "2000000-row batch boundaries, re-deriving a synthetic monthly_limit "
        "baseline of 2000000 purely for drift detection, never for live "
        "plan-gating decisions."
    ),
}

F11_JSON_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_json_doc",
    doc_type="config",
    sections=_JSON_SECTIONS,
    queries=[
        F11Query(
            query_id="json_bare_numeric",
            query_class=QUERY_CLASS_BARE_NUMERIC,
            query_text="3000",
            gold_markers=["ALPHA_SERVICE_SECTION"],
            answer_spans=['"timeout_ms": 3000, "region": "iad"'],
            load_bearing_tokens=["3000"],
        ),
        F11Query(
            query_id="json_lexical_trap",
            query_class=QUERY_CLASS_LEXICAL_TRAP,
            query_text="2000000",
            gold_markers=["GAMMA_LIMIT_SECTION"],
            answer_spans=['"tier": "pro", "monthly_limit": 2000000'],
            load_bearing_tokens=["2000000"],
            trap_term="monthly_limit",
            decoy_marker="ZETA_AUDIT_SECTION",
            note=(
                "ZETA mentions 'monthly_limit' 3x / '2000000' 2x vs GAMMA's 1x / "
                "1x -- verified by test."
            ),
        ),
        F11Query(
            query_id="json_identifier",
            query_class=QUERY_CLASS_IDENTIFIER,
            query_text="backoff_ms",
            gold_markers=["BETA_RETRY_SECTION"],
            answer_spans=['"max_attempts": 4, "backoff_ms": 250'],
            load_bearing_tokens=["backoff_ms"],
        ),
        F11Query(
            query_id="json_quoted_phrase",
            query_class=QUERY_CLASS_QUOTED_PHRASE,
            query_text='"ttl_seconds": 300, "max_entries": 5000',
            gold_markers=["DELTA_CACHE_SECTION"],
            answer_spans=['"ttl_seconds": 300, "max_entries": 5000'],
            load_bearing_tokens=["5000"],
        ),
        F11Query(
            query_id="json_keyword_nl",
            query_class=QUERY_CLASS_KEYWORD_NL,
            query_text="What is the monthly compression quota for a pro tier key?",
            gold_markers=["GAMMA_LIMIT_SECTION"],
            answer_spans=['"tier": "pro", "monthly_limit": 2000000'],
            load_bearing_tokens=["monthly_limit"],
        ),
        F11Query(
            query_id="json_pure_paraphrase",
            query_class=QUERY_CLASS_PURE_PARAPHRASE,
            query_text=(
                "How long do stored records stick around before the system "
                "clears out old ones automatically?"
            ),
            gold_markers=["DELTA_CACHE_SECTION"],
            answer_spans=['"ttl_seconds": 300, "max_entries": 5000'],
            load_bearing_tokens=["ttl_seconds"],
            note="Zero non-stopword overlap with DELTA_CACHE_SECTION verified by test.",
        ),
        F11Query(
            query_id="json_multi_hop",
            query_class=QUERY_CLASS_MULTI_HOP,
            query_text=(
                "Summarize both the request timeout and the retry backoff "
                "settings used for outbound calls."
            ),
            gold_markers=["ALPHA_SERVICE_SECTION", "BETA_RETRY_SECTION"],
            answer_spans=[
                '"timeout_ms": 3000, "region": "iad"',
                '"max_attempts": 4, "backoff_ms": 250',
            ],
            load_bearing_tokens=["timeout_ms", "backoff_ms"],
        ),
    ],
)


ALL_F11_FIXTURES: List[F11MultiNodeFixture] = [
    F11_ARCH_FIXTURE,
    F11_CODE_FIXTURE,
    F11_JSON_FIXTURE,
]

# TODO(#250 full spec): expand toward the design memo's 12-16 doc / ~240
# query target -- 1 more prose/tutorial doc, 3 more API-reference docs, 2
# more mixed code+prose docs, 2 more config/table-heavy docs; ~40 queries
# per class instead of the ~3 landed here per class. Keep the per-class
# reporting discipline (tests/f11_fixture_harness.py) unchanged when doing
# so -- it is what prevents the aggregate-wash failure mode this file
# exists to fix.
