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


# =========================================================================
# Corpus scale-up (#266, 2026-07-10): 12 prose docs w/ semantically-DISTINCT
# (non-parallel) sections, 7 query classes each. model-free + real-engine
# 1-node-per-section validated. Uniform/parallel-structure docs (schema,
# pricing-tiers, cli-subcommands, changelog) were dropped -- they don't
# chunk 1:1. Corpus: 15 fixtures / ~104 queries (was 3 / ~20).
# =========================================================================

_F11_API_REF_SECTIONS = {
    "RATELIMIT_SECTION": (
        "Each API key is throttled to 300 requests per minute across every endpoint. When a client exceeds this ceiling, the server responds with a 429 status code and a Wait-Seconds header naming how long to pause before sending another request. The counter operates on a rolling window rather than resetting at a fixed clock boundary. Repeated violations beyond the ceiling can trigger a temporary suspension of the offending key."
    ),
    "PAGINATION_SECTION": (
        "List endpoints return results using an opaque cursor rather than an offset number. Pass the cursor value from the previous response in the next_cursor parameter to fetch the following batch. Clients may set the limit parameter to control page size, with a maximum of 100 items per page. Omitting the cursor always starts iteration from the beginning of the result set."
    ),
    "AUTH_HEADERS_SECTION": (
        "Every request must include an Authorization header using the Bearer scheme followed by a key prefixed with gc_. The gateway derives the request signature using the secret referenced by the API_KEY_HMAC_SECRET constant. Requests missing the Authorization header or presenting a malformed gc_ key are rejected before reaching any business logic. Rotating the underlying secret invalidates every previously issued key."
    ),
    "ERROR_CODES_SECTION": (
        "Every failed request returns a JSON body containing an error field, a numeric code field, and a human-readable message field. A validation failure on the request body always surfaces error code 40010 alongside a list of the specific fields that failed. Authentication failures use a distinct code range starting at 40100. Clients should treat any unrecognized code as a transient server condition rather than a permanent failure."
    ),
    "WEBHOOK_EVENTS_SECTION": (
        "Webhook events are delivered as an HTTP POST to the configured endpoint, and each delivery carries a signature header for verification. If the endpoint fails to return a success status, the dispatcher will retry the delivery using exponential backoff. Every retry increases the delay before the following retry, and the dispatcher stops after a bounded number of retry attempts. Verifying the signature on every retry prevents a spoofed sender from injecting a fake event during a retry storm."
    ),
    "IDEMPOTENCY_SECTION": (
        "Clients may attach an idempotency key to any write request so that the operation is only applied once. The server stores each idempotency key for 24 hours before it expires and can be reused. If a client must retry a request after a timeout, reusing the same idempotency key guarantees the original result is returned instead of creating a duplicate. Keys are scoped per API credential, so two different clients may safely reuse the same key value."
    ),
}

F11_API_REF_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_api_ref",
    doc_type="prose",
    sections=_F11_API_REF_SECTIONS,
    queries=[
        F11Query(
            query_id="apiref_bare_numeric",
            query_class="bare_numeric",
            query_text="300",
            gold_markers=["RATELIMIT_SECTION"],
            answer_spans=[
                "Each API key is throttled to 300 requests per minute across every endpoint."
            ],
        ),
        F11Query(
            query_id="apiref_identifier",
            query_class="identifier",
            query_text="API_KEY_HMAC_SECRET",
            gold_markers=["AUTH_HEADERS_SECTION"],
            answer_spans=[
                "The gateway derives the request signature using the secret referenced by the API_KEY_HMAC_SECRET constant."
            ],
        ),
        F11Query(
            query_id="apiref_quoted_phrase",
            query_class="quoted_phrase",
            query_text='What does "an opaque cursor rather than an offset number" mean for list endpoints?',
            gold_markers=["PAGINATION_SECTION"],
            answer_spans=[
                "List endpoints return results using an opaque cursor rather than an offset number."
            ],
        ),
        F11Query(
            query_id="apiref_keyword_nl",
            query_class="keyword_nl",
            query_text="What numeric error code is returned when request body validation fails?",
            gold_markers=["ERROR_CODES_SECTION"],
            answer_spans=[
                "A validation failure on the request body always surfaces error code 40010 alongside a list of the specific fields that failed."
            ],
        ),
        F11Query(
            query_id="apiref_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="How does the platform block a bad actor from forging a fabricated occurrence sent to a listener?",
            gold_markers=["WEBHOOK_EVENTS_SECTION"],
            answer_spans=[
                "Verifying the signature on every retry prevents a spoofed sender from injecting a fake event during a retry storm."
            ],
        ),
        F11Query(
            query_id="apiref_lexical_trap",
            query_class="lexical_trap",
            query_text="If a client needs to retry a write request, how does the API make sure a duplicate is not created?",
            gold_markers=["IDEMPOTENCY_SECTION"],
            answer_spans=[
                "If a client must retry a request after a timeout, reusing the same idempotency key guarantees the original result is returned instead of creating a duplicate."
            ],
            trap_term="retry",
            decoy_marker="WEBHOOK_EVENTS_SECTION",
        ),
        F11Query(
            query_id="apiref_multi_hop",
            query_class="multi_hop",
            query_text="When a client is rate limited, what HTTP status is returned and what fields appear in the resulting error body?",
            gold_markers=["RATELIMIT_SECTION", "ERROR_CODES_SECTION"],
            answer_spans=[
                "When a client exceeds this ceiling, the server responds with a 429 status code and a Wait-Seconds header naming how long to pause before sending another request.",
                "Every failed request returns a JSON body containing an error field, a numeric code field, and a human-readable message field.",
            ],
        ),
    ],
)

_F11_INFRA_RUNBOOK_SECTIONS = {
    "DEPLOY_ROLLBACK_SECTION": (
        "When a deploy introduces a regression, operators should immediately trigger a rollback to the previous known-good release rather than attempting a forward fix under pressure. The release command is configured with a 90 second timeout, after which the deployment is marked as failed and the platform automatically reverts to the prior machine image. Rolling back requires re-running the deploy pipeline with the last stable git tag rather than the current HEAD. Once the rollback completes, confirm the application version string matches the expected prior release before closing the incident."
    ),
    "SCALING_SECTION": (
        "Horizontal scaling adds additional machine instances behind the load balancer whenever sustained traffic approaches capacity. Each machine is configured with a soft concurrency limit of 64 simultaneous requests and a hard concurrency limit of 128, beyond which new connections are queued rather than dropped. The soft limit is controlled by the environment variable SOFT_CONCURRENCY_LIMIT, which the autoscaler reads every thirty seconds to decide whether to provision another machine. Operators should avoid manually overriding this variable in production without first notifying the on-call engineer."
    ),
    "SECRET_ROTATION_SECTION": (
        "Rotating a shared secret safely requires updating the value on the receiving platform before the sending platform, so that the new credential is already accepted when the old one stops being sent. Operators first set the new secret on the internal platform and confirm it is live, then update the corresponding secret on the external platform that triggers requests. Only after both platforms report the updated value should the old secret be revoked, since revoking early causes every in-flight request to fail authentication. This order of operations prevents a window where requests are signed with a secret neither side recognizes."
    ),
    "HEALTHCHECK_SECTION": (
        "A liveness probe answers whether the process is running at all, while a readiness probe answers whether the process is currently able to serve traffic, and the two failing for different reasons should never trigger the same remediation. The readiness probe is granted a 45 second grace period after machine startup before failures count against it, so that dependencies like the database connection pool have time to warm up. A liveness probe that keeps failing after the grace period causes the orchestrator to restart the container, whereas a failing readiness probe simply removes the machine from the load balancer rotation. Confusing the two probes has previously caused healthy machines to be restarted unnecessarily during slow cold starts."
    ),
    "INCIDENT_ESCALATION_SECTION": (
        "Incidents are classified into three severity tiers, where a tier one incident pages the on-call engineer immediately and a tier three incident is only logged for the next business day review. Every tier one alert also triggers a secondary alert to the engineering manager if the page is not acknowledged within five minutes. A tier two alert pages only the primary on-call engineer and escalates to a tier one alert if it remains unresolved for thirty minutes. The alerting system deliberately favors sending an extra alert over staying silent during a customer-impacting outage."
    ),
    "BACKUP_RESTORE_SECTION": (
        "The primary database is backed up automatically every six hours, with each snapshot retained for thirty days before being rotated out of storage. Restoring from a backup requires provisioning a new database instance, loading the chosen snapshot, and then replaying the write-ahead log up to the desired point in time. Before a restore is considered complete, an automated verification query checks that row counts match the expected pre-incident baseline. If the verification query fails, the restore process raises a single alert to the database team rather than silently marking the restore as successful."
    ),
}

F11_INFRA_RUNBOOK_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_infra_runbook",
    doc_type="prose",
    sections=_F11_INFRA_RUNBOOK_SECTIONS,
    queries=[
        F11Query(
            query_id="infra_bare_numeric",
            query_class="bare_numeric",
            query_text="45",
            gold_markers=["HEALTHCHECK_SECTION"],
            answer_spans=[
                "The readiness probe is granted a 45 second grace period after machine startup before failures count against it, so that dependencies like the database connection pool have time to warm up."
            ],
        ),
        F11Query(
            query_id="infra_identifier",
            query_class="identifier",
            query_text="SOFT_CONCURRENCY_LIMIT",
            gold_markers=["SCALING_SECTION"],
            answer_spans=[
                "The soft limit is controlled by the environment variable SOFT_CONCURRENCY_LIMIT, which the autoscaler reads every thirty seconds to decide whether to provision another machine."
            ],
        ),
        F11Query(
            query_id="infra_quoted_phrase",
            query_class="quoted_phrase",
            query_text='"immediately trigger a rollback to the previous known-good release"',
            gold_markers=["DEPLOY_ROLLBACK_SECTION"],
            answer_spans=[
                "When a deploy introduces a regression, operators should immediately trigger a rollback to the previous known-good release rather than attempting a forward fix under pressure."
            ],
        ),
        F11Query(
            query_id="infra_keyword_nl",
            query_class="keyword_nl",
            query_text="What is the correct order for rotating a secret across two platforms?",
            gold_markers=["SECRET_ROTATION_SECTION"],
            answer_spans=[
                "Rotating a shared secret safely requires updating the value on the receiving platform before the sending platform, so that the new credential is already accepted when the old one stops being sent."
            ],
        ),
        F11Query(
            query_id="infra_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="How often does the system generate a fresh copy of its records, and for how long does it keep an individual copy before discarding it?",
            gold_markers=["BACKUP_RESTORE_SECTION"],
            answer_spans=[
                "The primary database is backed up automatically every six hours, with each snapshot retained for thirty days before being rotated out of storage."
            ],
        ),
        F11Query(
            query_id="infra_lexical_trap",
            query_class="lexical_trap",
            query_text="Which alert fires when a database restore fails its verification check?",
            gold_markers=["BACKUP_RESTORE_SECTION"],
            answer_spans=[
                "If the verification query fails, the restore process raises a single alert to the database team rather than silently marking the restore as successful."
            ],
            trap_term="alert",
            decoy_marker="INCIDENT_ESCALATION_SECTION",
        ),
        F11Query(
            query_id="infra_multi_hop",
            query_class="multi_hop",
            query_text="During a rollback, how long can the release command run before it's marked failed, and how long is the readiness grace period after a machine starts?",
            gold_markers=["DEPLOY_ROLLBACK_SECTION", "HEALTHCHECK_SECTION"],
            answer_spans=[
                "The release command is configured with a 90 second timeout, after which the deployment is marked as failed and the platform automatically reverts to the prior machine image.",
                "The readiness probe is granted a 45 second grace period after machine startup before failures count against it, so that dependencies like the database connection pool have time to warm up.",
            ],
        ),
    ],
)

_F11_BILLING_POLICY_SECTIONS = {
    "TRIAL_TERMS_SECTION": (
        "New customers can start a 14-day free trial by entering a valid payment card at signup. If the trial is not cancelled before it ends, the account automatically converts to a paid Pro subscription and the card on file is charged the full monthly rate. Trial cancellations can be submitted at any time from the billing dashboard without penalty. No charge is made during the trial period itself."
    ),
    "PRORATION_SECTION": (
        "When a customer upgrades or downgrades their plan in the middle of a billing cycle, the system calculates a prorated credit or charge based on the number of days remaining. This behavior is controlled internally by the proration_behavior configuration flag, which determines whether unused time on the old plan is credited toward the new plan. Downgrades take effect immediately, while the prorated credit is applied to the next invoice. Customers can view the itemized proration breakdown before confirming a plan change."
    ),
    "REFUND_WINDOW_SECTION": (
        "Customers who are unsatisfied with their subscription may request a full refund within 30 days of the original purchase date. Refund eligibility requires that the account has not exceeded twice the plan's included monthly quota during the billing period. Refunds are processed back to the original payment method within five to seven business days. Annual plans purchased through a reseller are not eligible for this refund window."
    ),
    "OVERAGE_METERING_SECTION": (
        "Once a project's monthly usage exceeds the plan's included quota, additional usage is billed as metered overage at a per-unit price of $0.002 per extra compression call. Overage usage is tracked in real time and summarized on the next invoice as a separate line item. Customers can set a hard usage cap to prevent unexpected overage charges, or leave the cap unset to allow usage to scale freely."
    ),
    "DUNNING_SECTION": (
        "When a subscription payment fails, the billing system automatically retries the charge up to three times over the following seven days before the subscription is marked past due. Each retry attempt sends an email notification to the account owner with a link to update their payment method. If a project's usage would otherwise be interrupted by a lapsed subscription, it is throttled to the free tier limits rather than immediately disabled. Accounts that remain unpaid after the final retry are suspended until a valid payment method is added."
    ),
    "TAX_HANDLING_SECTION": (
        "As the merchant of record, the billing provider is responsible for calculating, collecting, and remitting applicable sales tax, VAT, and GST on every transaction. Tax rates are determined automatically based on the billing address provided at checkout and displayed as a separate line item before payment is confirmed. Each successful transaction generates a downloadable invoice showing the tax breakdown, which is emailed to the customer and stored in the billing history. Customers in tax-exempt regions or organizations can submit an exemption certificate to have future invoices adjusted accordingly."
    ),
}

F11_BILLING_POLICY_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_billing_policy",
    doc_type="prose",
    sections=_F11_BILLING_POLICY_SECTIONS,
    queries=[
        F11Query(
            query_id="billing_bare_numeric",
            query_class="bare_numeric",
            query_text="14",
            gold_markers=["TRIAL_TERMS_SECTION"],
            answer_spans=[
                "New customers can start a 14-day free trial by entering a valid payment card at signup."
            ],
        ),
        F11Query(
            query_id="billing_identifier",
            query_class="identifier",
            query_text="proration_behavior",
            gold_markers=["PRORATION_SECTION"],
            answer_spans=[
                "This behavior is controlled internally by the proration_behavior configuration flag, which determines whether unused time on the old plan is credited toward the new plan."
            ],
        ),
        F11Query(
            query_id="billing_quoted_phrase",
            query_class="quoted_phrase",
            query_text='"within 30 days of the original purchase date"',
            gold_markers=["REFUND_WINDOW_SECTION"],
            answer_spans=[
                "Customers who are unsatisfied with their subscription may request a full refund within 30 days of the original purchase date."
            ],
        ),
        F11Query(
            query_id="billing_keyword_nl",
            query_class="keyword_nl",
            query_text="Who is responsible for collecting and remitting sales tax on transactions?",
            gold_markers=["TAX_HANDLING_SECTION"],
            answer_spans=[
                "As the merchant of record, the billing provider is responsible for calculating, collecting, and remitting applicable sales tax, VAT, and GST on every transaction."
            ],
        ),
        F11Query(
            query_id="billing_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="What happens financially when a client's consumption climbs past their subscription's built-in allotment?",
            gold_markers=["OVERAGE_METERING_SECTION"],
            answer_spans=[
                "Once a project's monthly usage exceeds the plan's included quota, additional usage is billed as metered overage at a per-unit price of $0.002 per extra compression call."
            ],
        ),
        F11Query(
            query_id="billing_lexical_trap",
            query_class="lexical_trap",
            query_text="If a customer's usage is at risk of being cut off because a payment failed, what happens to their account?",
            gold_markers=["DUNNING_SECTION"],
            answer_spans=[
                "If a project's usage would otherwise be interrupted by a lapsed subscription, it is throttled to the free tier limits rather than immediately disabled."
            ],
            trap_term="usage",
            decoy_marker="OVERAGE_METERING_SECTION",
        ),
        F11Query(
            query_id="billing_multi_hop",
            query_class="multi_hop",
            query_text="After a free trial automatically converts into a paid subscription, how many days does the customer have to request a refund if they are unhappy with the charge?",
            gold_markers=["TRIAL_TERMS_SECTION", "REFUND_WINDOW_SECTION"],
            answer_spans=[
                "If the trial is not cancelled before it ends, the account automatically converts to a paid Pro subscription and the card on file is charged the full monthly rate.",
                "Customers who are unsatisfied with their subscription may request a full refund within 30 days of the original purchase date.",
            ],
        ),
    ],
)

_F11_SECURITY_AUDIT_SECTIONS = {
    "AUDIT_SUMMARY_SECTION": (
        "The security audit of the payment-processing service concluded with an overall verdict of FAIL, and the release must remain blocked until the critical issues are resolved. Auditors catalogued a total of 14 findings spanning authentication, injection, and rate-limiting weaknesses. Three of the 14 findings were rated critical and demand remediation before the next deployment window opens. The engagement ran across six business days and covered the public REST API alongside the internal admin console."
    ),
    "AUTH_BYPASS_SECTION": (
        "Auditors discovered that the administrative login flow calls a helper named verify_session_token but silently skips it whenever the request originates from an internal IP range. This authentication-bypass path lets an attacker who spoofs the X-Forwarded-For header reach the admin dashboard without ever supplying valid credentials. The flaw was reproduced by sending a crafted header value pointing at an internal address to the staging admin console. Because the skipped check is the only gate protecting privileged account settings, the bypass grants full administrative control to anyone who can forge the header."
    ),
    "INJECTION_RISK_SECTION": (
        "The reporting module builds its SQL query by concatenating the raw sort_by query parameter directly into the ORDER BY clause without any allowlist or parameterization. An attacker can append a crafted payload such as a UNION-based subquery to exfiltrate customer email addresses from an unrelated table. The same handler also shells out to a system command using the unsanitized filename field, creating a secondary command-injection vector. Because both paths trust client-supplied strings, a single malicious request could chain the SQL injection into a command injection for full database and host compromise."
    ),
    "RATELIMIT_GAP_SECTION": (
        "The password-reset endpoint enforces a rate limit of five attempts per hour, but the limit is keyed only on the authenticated session cookie rather than on the account or source IP. An attacker can bypass this limit simply by omitting the cookie, since unauthenticated requests are never counted against any limit at all. Auditors confirmed that hundreds of reset attempts succeeded within a single minute once the cookie was stripped from the request, showing the limit provides no real protection. This rate-limit gap effectively leaves the password-reset flow completely unthrottled for any attacker willing to skip the cookie."
    ),
    "REMEDIATION_PLAN_SECTION": (
        "To close the password-reset gap, engineering will key throttling on the account identifier and source IP instead of the session cookie, so unauthenticated requests can no longer dodge the cap. The rollout proceeds in three stages: first the authentication-bypass fix ships to staging, then the injection sanitization patch follows once the staging soak test passes, and finally the throttling change deploys to production during the following maintenance window. Each stage requires a clean regression run before the next stage may begin. The team will confirm the account-keyed limit is effective by replaying the earlier attack traffic against staging."
    ),
    "RISK_SCORING_SECTION": (
        "Using CVSS v3.1 scoring, the authentication-bypass finding receives a base score of 9.1, reflecting critical severity with network-based attack vector and no privileges required. The SQL injection finding scores 8.6 due to the confidentiality impact on customer data, while the rate-limiting gap is rated 5.3 as a moderate availability concern. Combined, the report assigns the engagement an aggregate risk rating of critical, driven primarily by the 9.1 authentication-bypass score. The scoring model weights exploitability and impact equally when calculating each base score."
    ),
}

F11_SECURITY_AUDIT_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_security_audit",
    doc_type="prose",
    sections=_F11_SECURITY_AUDIT_SECTIONS,
    queries=[
        F11Query(
            query_id="secaudit_bare_numeric",
            query_class="bare_numeric",
            query_text="14",
            gold_markers=["AUDIT_SUMMARY_SECTION"],
            answer_spans=[
                "Auditors catalogued a total of 14 findings spanning authentication, injection, and rate-limiting weaknesses."
            ],
        ),
        F11Query(
            query_id="secaudit_identifier",
            query_class="identifier",
            query_text="verify_session_token",
            gold_markers=["AUTH_BYPASS_SECTION"],
            answer_spans=[
                "Auditors discovered that the administrative login flow calls a helper named verify_session_token but silently skips it whenever the request originates from an internal IP range."
            ],
        ),
        F11Query(
            query_id="secaudit_quoted_phrase",
            query_class="quoted_phrase",
            query_text='"creating a secondary command-injection vector"',
            gold_markers=["INJECTION_RISK_SECTION"],
            answer_spans=[
                "The same handler also shells out to a system command using the unsanitized filename field, creating a secondary command-injection vector."
            ],
        ),
        F11Query(
            query_id="secaudit_keyword_nl",
            query_class="keyword_nl",
            query_text="What CVSS score was given to the SQL injection finding?",
            gold_markers=["RISK_SCORING_SECTION"],
            answer_spans=[
                "The SQL injection finding scores 8.6 due to the confidentiality impact on customer data, while the rate-limiting gap is rated 5.3 as a moderate availability concern."
            ],
        ),
        F11Query(
            query_id="secaudit_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="Which private shortcut permits someone to enter restricted management screens while providing no genuine identity proof at all?",
            gold_markers=["AUTH_BYPASS_SECTION"],
            answer_spans=[
                "This authentication-bypass path lets an attacker who spoofs the X-Forwarded-For header reach the admin dashboard without ever supplying valid credentials."
            ],
        ),
        F11Query(
            query_id="secaudit_lexical_trap",
            query_class="lexical_trap",
            query_text="What is the rollout order for the fix to the rate limit gap?",
            gold_markers=["REMEDIATION_PLAN_SECTION"],
            answer_spans=[
                "The rollout proceeds in three stages: first the authentication-bypass fix ships to staging, then the injection sanitization patch follows once the staging soak test passes, and finally the throttling change deploys to production during the following maintenance window."
            ],
            trap_term="limit",
            decoy_marker="RATELIMIT_GAP_SECTION",
        ),
        F11Query(
            query_id="secaudit_multi_hop",
            query_class="multi_hop",
            query_text="Which function was skipped in the authentication bypass, and what CVSS base score did that finding receive?",
            gold_markers=["AUTH_BYPASS_SECTION", "RISK_SCORING_SECTION"],
            answer_spans=[
                "Auditors discovered that the administrative login flow calls a helper named verify_session_token but silently skips it whenever the request originates from an internal IP range.",
                "Using CVSS v3.1 scoring, the authentication-bypass finding receives a base score of 9.1, reflecting critical severity with network-based attack vector and no privileges required.",
            ],
        ),
    ],
)

_F11_ML_PAPER_SECTIONS = {
    "MOTIVATION_SECTION": (
        "Long-context dialogue agents frequently lose track of user preferences stated early in a conversation once the context window is truncated. Prior approaches rely on fixed-window summarization, which silently discards facts that later turns depend on, forcing users to repeat themselves. These heuristics are insufficient because they cannot distinguish transient chit-chat from durable commitments the user expects the agent to remember."
    ),
    "METHOD_SECTION": (
        "We introduce SIEVE-7B, a retrieval-augmented memory module built around a lightweight component called SalienceGate, which scores each utterance for long-term importance before deciding whether to retain or discard it. Unlike fixed-window summarization, SalienceGate learns its scoring function jointly with the base transformer, allowing the system to keep sparse but critical facts across arbitrarily long dialogues. The gating scores are computed from a classifier head attached to the final decoder layer, adding negligible inference overhead."
    ),
    "DATASET_SECTION": (
        "SIEVE-7B is trained on a curated corpus of 842,000 multi-turn conversations collected from customer-support transcripts and crowd-sourced roleplay sessions. Each transcript is annotated with salience labels marking which utterances a downstream turn later references. The corpus spans an average of 34 turns per conversation, substantially longer than prior long-context benchmarks."
    ),
    "RESULTS_SECTION": (
        "On the held-out long-dialogue benchmark, SIEVE-7B improves fact-retention accuracy to 94.3%, compared to 71.2% for the strongest fixed-window baseline. The gain is most pronounced in conversations exceeding 50 turns, where baseline systems drop below random guessing. Latency overhead from the gating classifier remains under 8 milliseconds per turn."
    ),
    "ABLATION_SECTION": (
        "We ablate each component of SIEVE-7B to isolate its individual contribution. Removing the salience-classifier component causes retention accuracy to collapse by more than fifteen points, confirming that this component drives most of the overall gain. The retrieval component contributes a smaller but still measurable improvement, while the positional-encoding component has almost no effect in isolation. Only when every component is combined does the full system reach its best reported performance."
    ),
    "LIMITATIONS_SECTION": (
        "Despite these gains, SIEVE-7B's memory component was trained exclusively on English-language conversations and has not been validated on multilingual dialogue. The approach also assumes a single active user per conversation and may misattribute salience when multiple speakers interleave rapidly. Because the gating classifier was tuned on customer-support and roleplay data, its behavior on adversarial or safety-critical conversations remains untested."
    ),
}

F11_ML_PAPER_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_ml_paper",
    doc_type="prose",
    sections=_F11_ML_PAPER_SECTIONS,
    queries=[
        F11Query(
            query_id="mlpaper_bare_numeric",
            query_class="bare_numeric",
            query_text="34",
            gold_markers=["DATASET_SECTION"],
            answer_spans=[
                "The corpus spans an average of 34 turns per conversation, substantially longer than prior long-context benchmarks."
            ],
        ),
        F11Query(
            query_id="mlpaper_identifier",
            query_class="identifier",
            query_text="SalienceGate",
            gold_markers=["METHOD_SECTION"],
            answer_spans=[
                "We introduce SIEVE-7B, a retrieval-augmented memory module built around a lightweight component called SalienceGate, which scores each utterance for long-term importance before deciding whether to retain or discard it."
            ],
        ),
        F11Query(
            query_id="mlpaper_quoted_phrase",
            query_class="quoted_phrase",
            query_text='"forcing users to repeat themselves"',
            gold_markers=["MOTIVATION_SECTION"],
            answer_spans=[
                "Prior approaches rely on fixed-window summarization, which silently discards facts that later turns depend on, forcing users to repeat themselves."
            ],
        ),
        F11Query(
            query_id="mlpaper_keyword_nl",
            query_class="keyword_nl",
            query_text="What accuracy did the fact-retention benchmark show for the proposed method?",
            gold_markers=["RESULTS_SECTION"],
            answer_spans=[
                "On the held-out long-dialogue benchmark, SIEVE-7B improves fact-retention accuracy to 94.3%, compared to 71.2% for the strongest fixed-window baseline."
            ],
        ),
        F11Query(
            query_id="mlpaper_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="What helps the new setup determine which parts of a chat log to preserve instead of erasing them?",
            gold_markers=["METHOD_SECTION"],
            answer_spans=[
                "Unlike fixed-window summarization, SalienceGate learns its scoring function jointly with the base transformer, allowing the system to keep sparse but critical facts across arbitrarily long dialogues."
            ],
        ),
        F11Query(
            query_id="mlpaper_lexical_trap",
            query_class="lexical_trap",
            query_text="Which component of the system has not been validated for broader real-world use?",
            gold_markers=["LIMITATIONS_SECTION"],
            answer_spans=[
                "SIEVE-7B's memory component was trained exclusively on English-language conversations and has not been validated on multilingual dialogue."
            ],
            trap_term="component",
            decoy_marker="ABLATION_SECTION",
        ),
        F11Query(
            query_id="mlpaper_multi_hop",
            query_class="multi_hop",
            query_text="What model introduced with a salience-gating mechanism achieves 94.3% fact-retention accuracy?",
            gold_markers=["METHOD_SECTION", "RESULTS_SECTION"],
            answer_spans=[
                "We introduce SIEVE-7B, a retrieval-augmented memory module built around a lightweight component called SalienceGate, which scores each utterance for long-term importance before deciding whether to retain or discard it.",
                "On the held-out long-dialogue benchmark, SIEVE-7B improves fact-retention accuracy to 94.3%, compared to 71.2% for the strongest fixed-window baseline.",
            ],
        ),
    ],
)

_F11_LEGAL_TOS_SECTIONS = {
    "ACCEPTANCE_TERMS_SECTION": (
        "By creating an account or clicking the signup button, you agree to be bound by these Terms of Service. You must be at least 18 years old to use this platform. If you are accessing the service on behalf of an organization, you represent that you have authority to bind that organization to these terms. Continued use of the service after any modification constitutes acceptance of the revised terms."
    ),
    "LICENSE_GRANT_SECTION": (
        "Subject to your compliance with these terms, the company grants you a limited, non-exclusive, non-transferable, revocable license to access and use the software for internal business purposes only. This grant is designated internally by the identifier LICENSE_TYPE_B2B_REVOCABLE. The license does not include any right to sublicense, resell, or distribute the software to third parties. All rights not expressly granted herein remain reserved by the company."
    ),
    "DATA_PRIVACY_SECTION": (
        "We collect and process personal data solely to provide and improve the service, including your name, email address, and usage logs. Personal data is retained for 90 days after account closure before being permanently deleted from our systems. We do not sell personal data to third parties and only share it with subprocessors bound by confidentiality obligations. You may request a copy of your personal data or ask that it be deleted at any time by contacting our privacy team."
    ),
    "LIABILITY_LIMIT_SECTION": (
        "To the maximum extent permitted by law, the company shall not be liable for any indirect, incidental, special, consequential, or punitive damages arising out of your use of the service. In no event shall the company's aggregate liability for direct damages exceed the amount you paid in the twelve months preceding the claim. This limitation applies even if the company has been advised of the possibility of such damages. Some jurisdictions do not allow the exclusion of certain damages, so portions of this limitation may not apply to you."
    ),
    "TERMINATION_CLAUSE_SECTION": (
        "Either party may terminate this agreement at any time by providing thirty days written notice to the other party. The company may also terminate immediately if you breach any material provision of these terms or fail to cure a violation after notice. Upon termination, your right to access the service ends immediately, and any outstanding fees become due, though this section does not itself create a new claim for damages. All provisions that by their nature should survive termination, including payment obligations and confidentiality, will remain in effect."
    ),
    "GOVERNING_LAW_SECTION": (
        "These terms are governed by the laws of the State of Delaware, without regard to its conflict of law principles. Any dispute arising under this agreement shall be resolved exclusively in the state or federal courts located in Wilmington, Delaware. You waive any objection to personal jurisdiction or venue in those courts. Nothing in this section prevents either party from seeking injunctive relief in any court of competent jurisdiction."
    ),
}

F11_LEGAL_TOS_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_legal_tos",
    doc_type="prose",
    sections=_F11_LEGAL_TOS_SECTIONS,
    queries=[
        F11Query(
            query_id="legal_bare_numeric",
            query_class="bare_numeric",
            query_text="90",
            gold_markers=["DATA_PRIVACY_SECTION"],
            answer_spans=[
                "Personal data is retained for 90 days after account closure before being permanently deleted from our systems."
            ],
        ),
        F11Query(
            query_id="legal_identifier",
            query_class="identifier",
            query_text="LICENSE_TYPE_B2B_REVOCABLE",
            gold_markers=["LICENSE_GRANT_SECTION"],
            answer_spans=[
                "This grant is designated internally by the identifier LICENSE_TYPE_B2B_REVOCABLE."
            ],
        ),
        F11Query(
            query_id="legal_quoted_phrase",
            query_class="quoted_phrase",
            query_text='"without regard to its conflict of law principles"',
            gold_markers=["GOVERNING_LAW_SECTION"],
            answer_spans=[
                "These terms are governed by the laws of the State of Delaware, without regard to its conflict of law principles."
            ],
        ),
        F11Query(
            query_id="legal_keyword_nl",
            query_class="keyword_nl",
            query_text="What kinds of damages is the company not liable for under this agreement?",
            gold_markers=["LIABILITY_LIMIT_SECTION"],
            answer_spans=[
                "To the maximum extent permitted by law, the company shall not be liable for any indirect, incidental, special, consequential, or punitive damages arising out of your use of the service."
            ],
        ),
        F11Query(
            query_id="legal_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="What is the youngest a person can be to register as a subscriber on this system?",
            gold_markers=["ACCEPTANCE_TERMS_SECTION"],
            answer_spans=["You must be at least 18 years old to use this platform."],
        ),
        F11Query(
            query_id="legal_lexical_trap",
            query_class="lexical_trap",
            query_text="How does this agreement come to an end, and what happens to any related damages claims afterward?",
            gold_markers=["TERMINATION_CLAUSE_SECTION"],
            answer_spans=[
                "Upon termination, your right to access the service ends immediately, and any outstanding fees become due, though this section does not itself create a new claim for damages."
            ],
            trap_term="damages",
            decoy_marker="LIABILITY_LIMIT_SECTION",
        ),
        F11Query(
            query_id="legal_multi_hop",
            query_class="multi_hop",
            query_text="What type of license is granted for using the software, and in which court would a resulting dispute over it be resolved?",
            gold_markers=["LICENSE_GRANT_SECTION", "GOVERNING_LAW_SECTION"],
            answer_spans=[
                "This grant is designated internally by the identifier LICENSE_TYPE_B2B_REVOCABLE.",
                "Any dispute arising under this agreement shall be resolved exclusively in the state or federal courts located in Wilmington, Delaware.",
            ],
        ),
    ],
)

_F11_POSTMORTEM_A_SECTIONS = {
    "INCIDENT_TIMELINE_SECTION": (
        "The incident began at 02:14 UTC when the deployment pipeline pushed a configuration change to the production payment cluster. Within minutes, error rates on the checkout service climbed sharply as new pods rolled out across every availability zone. The on-call engineer was paged shortly after the rollout completed and traffic began routing to the unhealthy pods. The outage lasted 47 minutes before the previous configuration was restored and error rates returned to baseline."
    ),
    "ROOT_CAUSE_SECTION": (
        "Root cause analysis traced the outage to a misconfigured environment variable, CONNECTION_POOL_MAX_SIZE, which had been silently reduced from 200 to 20 during an unrelated infrastructure migration the previous week. With only twenty available connections, the checkout service exhausted its database connection pool within seconds of receiving production traffic. Every subsequent request queued behind the pool limit and eventually timed out, producing a cascade of failures across dependent services. The change had passed code review because the diff was buried inside a larger, seemingly unrelated pull request."
    ),
    "IMPACT_ASSESSMENT_SECTION": (
        "Approximately 63% of checkout requests failed during the incident window, preventing an estimated 40,000 customers from completing purchases across the web and mobile apps. The payments team, the customer support queue, and the finance reporting pipeline were all directly affected, with support tickets spiking nearly tenfold within the first fifteen minutes. Several enterprise customers escalated directly to their account managers after repeated checkout failures. Finance later confirmed a measurable dip in that day's processed transaction volume."
    ),
    "DETECTION_GAP_SECTION": (
        "Monitoring failed to surface the incident quickly because the primary alert for checkout error rate had been muted during the previous week's migration and was never re-enabled. A secondary alert on database connection saturation existed but routed to a Slack channel that no one on the on-call rotation had joined. The alerting dashboard showed a spike in connection pool exhaustion, but no alert threshold had been configured for that specific metric. Engineers later found three separate alert rules that could have caught the issue sooner had any of them been active."
    ),
    "REMEDIATION_STEPS_SECTION": (
        "Once the faulty configuration was identified, the on-call engineer restored the connection pool setting to its previous value and redeployed the checkout service across all availability zones. Eighteen pods were manually restarted to clear queued connections and drop stuck requests still waiting on the exhausted pool. The team also triggered a single manual alert to the incident channel to confirm the fix was holding before standing down. Error rates returned to baseline within minutes of the redeploy completing."
    ),
    "PREVENTION_FOLLOWUP_SECTION": (
        "To prevent a recurrence, the team added automated alerting on connection pool saturation with a hard threshold and paging escalation. Configuration changes to core infrastructure now require a dedicated review checklist separate from routine pull requests. The engineering organization also scheduled a quarterly game-day exercise to rehearse checkout-service failure scenarios. A follow-up ticket was filed to add pool-exhaustion metrics to the primary on-call dashboard."
    ),
}

F11_POSTMORTEM_A_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_postmortem_a",
    doc_type="prose",
    sections=_F11_POSTMORTEM_A_SECTIONS,
    queries=[
        F11Query(
            query_id="pma_bare_numeric",
            query_class="bare_numeric",
            query_text="47",
            gold_markers=["INCIDENT_TIMELINE_SECTION"],
            answer_spans=[
                "The outage lasted 47 minutes before the previous configuration was restored and error rates returned to baseline."
            ],
        ),
        F11Query(
            query_id="pma_identifier",
            query_class="identifier",
            query_text="CONNECTION_POOL_MAX_SIZE",
            gold_markers=["ROOT_CAUSE_SECTION"],
            answer_spans=[
                "Root cause analysis traced the outage to a misconfigured environment variable, CONNECTION_POOL_MAX_SIZE, which had been silently reduced from 200 to 20 during an unrelated infrastructure migration the previous week."
            ],
        ),
        F11Query(
            query_id="pma_quoted_phrase",
            query_class="quoted_phrase",
            query_text='"completing purchases across the web and mobile apps"',
            gold_markers=["IMPACT_ASSESSMENT_SECTION"],
            answer_spans=[
                "Approximately 63% of checkout requests failed during the incident window, preventing an estimated 40,000 customers from completing purchases across the web and mobile apps."
            ],
        ),
        F11Query(
            query_id="pma_keyword_nl",
            query_class="keyword_nl",
            query_text="Why did monitoring fail to catch the incident quickly?",
            gold_markers=["DETECTION_GAP_SECTION"],
            answer_spans=[
                "Monitoring failed to surface the incident quickly because the primary alert for checkout error rate had been muted during the previous week's migration and was never re-enabled."
            ],
        ),
        F11Query(
            query_id="pma_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="What ongoing safeguards are being put in place so this kind of outage will not resurface?",
            gold_markers=["PREVENTION_FOLLOWUP_SECTION"],
            answer_spans=[
                "To prevent a recurrence, the team added automated alerting on connection pool saturation with a hard threshold and paging escalation."
            ],
        ),
        F11Query(
            query_id="pma_lexical_trap",
            query_class="lexical_trap",
            query_text="What alert did the team use to confirm the incident was resolved?",
            gold_markers=["REMEDIATION_STEPS_SECTION"],
            answer_spans=[
                "The team also triggered a single manual alert to the incident channel to confirm the fix was holding before standing down."
            ],
            trap_term="alert",
            decoy_marker="DETECTION_GAP_SECTION",
        ),
        F11Query(
            query_id="pma_multi_hop",
            query_class="multi_hop",
            query_text="What configuration mistake caused the outage, and what percentage of checkout requests failed as a result?",
            gold_markers=["ROOT_CAUSE_SECTION", "IMPACT_ASSESSMENT_SECTION"],
            answer_spans=[
                "Root cause analysis traced the outage to a misconfigured environment variable, CONNECTION_POOL_MAX_SIZE, which had been silently reduced from 200 to 20 during an unrelated infrastructure migration the previous week.",
                "Approximately 63% of checkout requests failed during the incident window, preventing an estimated 40,000 customers from completing purchases across the web and mobile apps.",
            ],
        ),
    ],
)

_F11_POSTMORTEM_B_SECTIONS = {
    "OUTAGE_SUMMARY_SECTION": (
        "On June 14th, a billing webhook processing failure caused 342 customers to be charged twice for their monthly subscription. The incident lasted roughly three hours before the on-call engineer identified the root cause. Customer support began receiving complaints within twenty minutes of the first duplicate charge appearing in the dashboard."
    ),
    "TRIGGER_EVENT_SECTION": (
        "The outage was triggered by a malformed subscription.updated webhook payload sent by the payment processor after a routine price migration. A null value in the proration field caused the event handler to throw an unhandled exception mid-transaction, leaving the database session open. Because the handler retried automatically without checking idempotency, each retry reprocessed the same event and issued a fresh charge. The event identifier evt_8834ff2201 was the first of nine duplicate deliveries recorded in the webhook log."
    ),
    "BLAST_RADIUS_SECTION": (
        "Total erroneous charges across the affected accounts came to $18,470, spread unevenly across Pro and Team tier subscribers. Enterprise accounts were unaffected because their billing runs through a separate manual invoicing pipeline. Finance flagged the discrepancy during the nightly reconciliation report, which is what first surfaced the incident internally. The largest single overcharge hit one Team plan customer for $612 across four duplicate line items."
    ),
    "MITIGATION_SECTION": (
        "The on-call team immediately paused the webhook consumer to stop further duplicate charges from processing. Within the hour, engineers issued a refund to every customer identified in the duplicate-charge log, prioritizing the largest refund amounts first. A second wave of refunds went out the next morning after finance manually cross-checked the ledger for any accounts the automated refund script had missed. Support proactively emailed each customer to confirm their refund had been issued and to apologize for the inconvenience."
    ),
    "RECOVERY_STEPS_SECTION": (
        "Once the duplicate charges were confirmed, the finance team reconciled every affected subscription against the payment processor's ledger to ensure the internal database matched the true billing state. Each subscription record was rolled back to its pre-incident renewal date so the next billing cycle would trigger correctly. A single manual refund was issued for the one enterprise-adjacent account that had been mistakenly swept into the batch despite being on the manual pipeline. Engineers also replayed the corrected webhook event through a sandboxed consumer to confirm the fix produced the expected subscription state before the affected accounts were marked resolved."
    ),
    "FOLLOWUP_ACTIONS_SECTION": (
        "To prevent a recurrence, the team added strict idempotency keys to every webhook handler so a retried event can never be processed twice. Engineers also introduced a circuit breaker that halts the consumer after three consecutive handler exceptions instead of retrying indefinitely. A new alert now fires the moment the reconciliation job detects any mismatch between processor totals and internal ledger totals, rather than waiting for the nightly batch report. The postmortem review board approved a quarterly webhook-resilience audit as a standing agenda item."
    ),
}

F11_POSTMORTEM_B_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_postmortem_b",
    doc_type="prose",
    sections=_F11_POSTMORTEM_B_SECTIONS,
    queries=[
        F11Query(
            query_id="pmb_bare_numeric",
            query_class="bare_numeric",
            query_text="342",
            gold_markers=["OUTAGE_SUMMARY_SECTION"],
            answer_spans=[
                "On June 14th, a billing webhook processing failure caused 342 customers to be charged twice for their monthly subscription."
            ],
        ),
        F11Query(
            query_id="pmb_identifier",
            query_class="identifier",
            query_text="evt_8834ff2201",
            gold_markers=["TRIGGER_EVENT_SECTION"],
            answer_spans=[
                "The event identifier evt_8834ff2201 was the first of nine duplicate deliveries recorded in the webhook log."
            ],
        ),
        F11Query(
            query_id="pmb_quoted_phrase",
            query_class="quoted_phrase",
            query_text='"spread unevenly across Pro and Team tier subscribers"',
            gold_markers=["BLAST_RADIUS_SECTION"],
            answer_spans=[
                "Total erroneous charges across the affected accounts came to $18,470, spread unevenly across Pro and Team tier subscribers."
            ],
        ),
        F11Query(
            query_id="pmb_keyword_nl",
            query_class="keyword_nl",
            query_text="What process changes were made to prevent the webhook failure from happening again?",
            gold_markers=["FOLLOWUP_ACTIONS_SECTION"],
            answer_spans=[
                "To prevent a recurrence, the team added strict idempotency keys to every webhook handler so a retried event can never be processed twice."
            ],
        ),
        F11Query(
            query_id="pmb_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="Once the problem was discovered, what did the responders do to halt the bug and make affected users whole again?",
            gold_markers=["MITIGATION_SECTION"],
            answer_spans=[
                "The on-call team immediately paused the webhook consumer to stop further duplicate charges from processing."
            ],
        ),
        F11Query(
            query_id="pmb_lexical_trap",
            query_class="lexical_trap",
            query_text="How were customer refunds reconciled against the billing state?",
            gold_markers=["RECOVERY_STEPS_SECTION"],
            answer_spans=[
                "Once the duplicate charges were confirmed, the finance team reconciled every affected subscription against the payment processor's ledger to ensure the internal database matched the true billing state."
            ],
            trap_term="refund",
            decoy_marker="MITIGATION_SECTION",
        ),
        F11Query(
            query_id="pmb_multi_hop",
            query_class="multi_hop",
            query_text="What caused the outage and how was the system state ultimately reconciled afterward?",
            gold_markers=["TRIGGER_EVENT_SECTION", "RECOVERY_STEPS_SECTION"],
            answer_spans=[
                "The outage was triggered by a malformed subscription.updated webhook payload sent by the payment processor after a routine price migration.",
                "Once the duplicate charges were confirmed, the finance team reconciled every affected subscription against the payment processor's ledger to ensure the internal database matched the true billing state.",
            ],
        ),
    ],
)

_F11_ONBOARDING_SECTIONS = {
    "PREREQUISITES_SECTION": (
        "Before you begin, install Node.js on your development machine, since both the command-line tool and the client library depend on its runtime. You need at least version 20.11.0, because earlier releases lack the streaming support the tool requires. A stable internet connection is also necessary so the installer can reach the package registry. Set aside about five minutes for this initial setup."
    ),
    "INSTALLATION_STEPS_SECTION": (
        "Install the command-line tool by running npm install -g acme-cli from any terminal. The installer places an ac binary on your PATH so you can invoke commands globally. Once installation finishes, run ac --version to confirm the tool responds correctly. If npm reports a permission error, rerun the command with a user-level prefix instead of using sudo."
    ),
    "FIRST_REQUEST_SECTION": (
        "Once installed, send a first request to confirm the whole stack works end to end. The local development server listens on port 8080, so point your client at http://localhost:8080/v1/ping. A successful call returns a small JSON payload confirming the service is reachable. This quick check is the fastest way to verify your setup before writing real integration code."
    ),
    "AUTH_SETUP_SECTION": (
        "Before making authenticated calls, you need to generate an access token from your account dashboard. Copy the token immediately after creation, since the dashboard only displays the full token once. Store the token in an environment variable rather than hardcoding it into your source files. Every request must include this token in the request headers, or the server will reject the call."
    ),
    "TROUBLESHOOTING_SECTION": (
        "If a request fails immediately after installation, first check that the CLI was updated to the latest release, since older builds silently drop malformed responses. A connection refused error usually means the local server was never started, so restart it and retry the same command. An invalid token error typically means the credential was copied with extra whitespace, so regenerate it and paste it again without trailing spaces. Persistent failures are often resolved by clearing the local cache directory and re-running the setup command."
    ),
    "NEXT_STEPS_SECTION": (
        "With the basics working, explore the reference documentation to see every available endpoint and parameter. The examples repository shows complete sample projects you can copy and adapt for your own use case. Community forums are a good place to ask questions when you get stuck on something unusual. Consider also subscribing to the changelog so you hear about new features as soon as they ship."
    ),
}

F11_ONBOARDING_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_onboarding",
    doc_type="prose",
    sections=_F11_ONBOARDING_SECTIONS,
    queries=[
        F11Query(
            query_id="onb_bare_numeric",
            query_class="bare_numeric",
            query_text="8080",
            gold_markers=["FIRST_REQUEST_SECTION"],
            answer_spans=[
                "The local development server listens on port 8080, so point your client at http://localhost:8080/v1/ping."
            ],
        ),
        F11Query(
            query_id="onb_identifier",
            query_class="identifier",
            query_text="acme-cli",
            gold_markers=["INSTALLATION_STEPS_SECTION"],
            answer_spans=[
                "Install the command-line tool by running npm install -g acme-cli from any terminal."
            ],
        ),
        F11Query(
            query_id="onb_quoted_phrase",
            query_class="quoted_phrase",
            query_text='"the streaming support the tool requires"',
            gold_markers=["PREREQUISITES_SECTION"],
            answer_spans=[
                "You need at least version 20.11.0, because earlier releases lack the streaming support the tool requires."
            ],
        ),
        F11Query(
            query_id="onb_keyword_nl",
            query_class="keyword_nl",
            query_text="How do I set up an access token for authenticating my requests?",
            gold_markers=["AUTH_SETUP_SECTION"],
            answer_spans=[
                "Before making authenticated calls, you need to generate an access token from your account dashboard."
            ],
        ),
        F11Query(
            query_id="onb_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="After finishing initial setup, where can a developer look for deeper guidance or extra learning material?",
            gold_markers=["NEXT_STEPS_SECTION"],
            answer_spans=[
                "With the basics working, explore the reference documentation to see every available endpoint and parameter."
            ],
        ),
        F11Query(
            query_id="onb_lexical_trap",
            query_class="lexical_trap",
            query_text="My API calls suddenly stop working with an authentication failure, how do I fix it?",
            gold_markers=["TROUBLESHOOTING_SECTION"],
            answer_spans=[
                "An invalid token error typically means the credential was copied with extra whitespace, so regenerate it and paste it again without trailing spaces."
            ],
            trap_term="token",
            decoy_marker="AUTH_SETUP_SECTION",
        ),
        F11Query(
            query_id="onb_multi_hop",
            query_class="multi_hop",
            query_text="Which credential must you attach to a call, and what port does the local server run on for testing it?",
            gold_markers=["AUTH_SETUP_SECTION", "FIRST_REQUEST_SECTION"],
            answer_spans=[
                "Every request must include this token in the request headers, or the server will reject the call.",
                "The local development server listens on port 8080, so point your client at http://localhost:8080/v1/ping.",
            ],
        ),
    ],
)

_F11_PRODUCT_FAQ_SECTIONS = {
    "WHAT_IS_SECTION": (
        "Gotcontext is a semantic compression platform that shrinks documents before they are sent to a large language model, preserving meaning while cutting token counts. Engineering teams use it to lower inference costs and to fit larger documents inside a limited context window. The platform is available as a REST API, a command-line tool, and a Model Context Protocol server."
    ),
    "HOW_WORKS_SECTION": (
        "Compression works by building a semantic skeleton of the document and ranking each node by relevance before deciding what to keep or hide. The core routine, called adaptive_skeleton_pruning, scores every paragraph against the surrounding context and collapses low-relevance nodes into placeholders. A query-aware pass can then pull specific hidden regions back into view when an agent needs more detail."
    ),
    "SUPPORTED_MODELS_SECTION": (
        "Gotcontext works with any large language model, since compression happens before the text ever reaches the model. The dashboard currently lists integration guides for 12 popular models, including GPT-5, Claude, and Gemini families. New model guides are added whenever a customer requests one, but the compressed output itself is never tied to a specific vendor."
    ),
    "DATA_RETENTION_SECTION": (
        "Uploaded documents and their compressed skeletons are stored for 90 days after the last time they were accessed, after which they are automatically deleted from the database. Customers on paid plans can request permanent deletion at any time through the dashboard. Audit logs of who accessed a document are kept separately and are never purged automatically."
    ),
    "SELF_HOSTING_SECTION": (
        "For teams that cannot send data to a hosted API, gotcontext ships a self-hosted Docker image that runs entirely inside your own infrastructure. Self-hosting requires a valid license key, and the license must be renewed annually to keep receiving model updates. Enterprise customers can also request a perpetual license for air-gapped deployments where no network calls are permitted. The license key is generated per organization and is tied to the number of machines running the compression engine."
    ),
    "SUPPORT_CHANNELS_SECTION": (
        "General support questions can be sent to the help desk through the in-app chat widget or by emailing the support address listed on the contact page. Most tickets receive a first response within one business day, and urgent production outages can be escalated through the priority queue. License questions are the only category redirected elsewhere, since billing and renewal matters go straight to the sales team instead of support."
    ),
}

F11_PRODUCT_FAQ_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_product_faq",
    doc_type="prose",
    sections=_F11_PRODUCT_FAQ_SECTIONS,
    queries=[
        F11Query(
            query_id="faq_bare_numeric",
            query_class="bare_numeric",
            query_text="90",
            gold_markers=["DATA_RETENTION_SECTION"],
            answer_spans=[
                "Uploaded documents and their compressed skeletons are stored for 90 days after the last time they were accessed, after which they are automatically deleted from the database."
            ],
        ),
        F11Query(
            query_id="faq_identifier",
            query_class="identifier",
            query_text="adaptive_skeleton_pruning",
            gold_markers=["HOW_WORKS_SECTION"],
            answer_spans=[
                "The core routine, called adaptive_skeleton_pruning, scores every paragraph against the surrounding context and collapses low-relevance nodes into placeholders."
            ],
        ),
        F11Query(
            query_id="faq_quoted_phrase",
            query_class="quoted_phrase",
            query_text='What does gotcontext mean by "preserving meaning while cutting token counts"?',
            gold_markers=["WHAT_IS_SECTION"],
            answer_spans=[
                "Gotcontext is a semantic compression platform that shrinks documents before they are sent to a large language model, preserving meaning while cutting token counts."
            ],
        ),
        F11Query(
            query_id="faq_keyword_nl",
            query_class="keyword_nl",
            query_text="How many models does the dashboard provide integration guides for?",
            gold_markers=["SUPPORTED_MODELS_SECTION"],
            answer_spans=[
                "The dashboard currently lists integration guides for 12 popular models, including GPT-5, Claude, and Gemini families."
            ],
        ),
        F11Query(
            query_id="faq_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="Can the software operate behind our internal firewall with a paid activation code, avoiding any outbound connections to the vendor?",
            gold_markers=["SELF_HOSTING_SECTION"],
            answer_spans=[
                "For teams that cannot send data to a hosted API, gotcontext ships a self-hosted Docker image that runs entirely inside your own infrastructure."
            ],
        ),
        F11Query(
            query_id="faq_lexical_trap",
            query_class="lexical_trap",
            query_text="Where should I direct a license question?",
            gold_markers=["SUPPORT_CHANNELS_SECTION"],
            answer_spans=[
                "License questions are the only category redirected elsewhere, since billing and renewal matters go straight to the sales team instead of support."
            ],
            trap_term="license",
            decoy_marker="SELF_HOSTING_SECTION",
        ),
        F11Query(
            query_id="faq_multi_hop",
            query_class="multi_hop",
            query_text="How many models does gotcontext provide integration guides for, and how many days does it retain uploaded documents before deleting them?",
            gold_markers=["SUPPORTED_MODELS_SECTION", "DATA_RETENTION_SECTION"],
            answer_spans=[
                "The dashboard currently lists integration guides for 12 popular models, including GPT-5, Claude, and Gemini families.",
                "Uploaded documents and their compressed skeletons are stored for 90 days after the last time they were accessed, after which they are automatically deleted from the database.",
            ],
        ),
    ],
)

_F11_ARCHITECTURE_SECTIONS = {
    "REQUEST_FLOW_SECTION": (
        "A client request first lands on the edge network, where TLS termination and basic bot filtering happen before anything reaches our infrastructure. From the edge, traffic is forwarded to the reverse proxy, then to the load balancer, and finally into the application container itself. In total a single request crosses 4 hops between the edge and the point where application code starts executing. Each additional hop is instrumented so operators can see where latency accumulates along the path."
    ),
    "AUTH_LAYER_SECTION": (
        "Every inbound call passes through the authentication middleware before any route handler runs. The middleware validates the bearer token and, on success, populates request.state.user_id so downstream handlers can read the caller's identity without re-parsing the token. If validation fails, the middleware short-circuits the chain and returns an unauthorized response immediately. This pattern keeps identity resolution in one place instead of scattering token parsing across every endpoint."
    ),
    "COMPRESSION_ENGINE_SECTION": (
        "The semantic compression engine rewrites long documents into a compact skeleton before they are sent to a downstream model. On typical technical documentation, it achieves roughly a 9x reduction in token count while preserving the sections a reader would actually need. The engine ranks each section by relevance and hides low-value boilerplate behind placeholder markers. Operators can always drill back into a hidden section on demand without re-running the whole pipeline."
    ),
    "PERSISTENCE_LAYER_SECTION": (
        "All durable state lives in a managed Postgres database, while a separate in-memory cache absorbs the read traffic that would otherwise hit the database directly. The cache stores hot lookups such as plan status and recent usage counters so the database is spared repetitive queries. When the cache expires an entry, the next read falls through to the database and repopulates the cache automatically. Operators can flush the cache manually during an incident without touching the underlying database rows."
    ),
    "OBSERVABILITY_SECTION": (
        "Every request is wrapped in a distributed trace so operators can follow it across the proxy, the application, and any downstream service it touches. Metrics such as request latency, error rate, and queue depth are exported continuously and rendered on a shared dashboard. A single cache-hit-ratio gauge is also published here, giving on-call engineers a quick signal without paging through raw logs. When a trace shows an unusual delay, the corresponding metrics panel usually explains why within seconds."
    ),
    "RESILIENCE_SECTION": (
        "When a downstream dependency starts failing repeatedly, a circuit breaker trips and stops sending it new traffic for a cooldown period. While the breaker is open, the system falls back to a degraded mode that serves slightly stale or simplified responses rather than failing outright. Health checks continue to probe the dependency in the background, and the breaker closes again once several consecutive checks succeed. This approach keeps a single failing dependency from cascading into a full outage."
    ),
}

F11_ARCHITECTURE_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_architecture",
    doc_type="prose",
    sections=_F11_ARCHITECTURE_SECTIONS,
    queries=[
        F11Query(
            query_id="arch2_bare_numeric",
            query_class="bare_numeric",
            query_text="4",
            gold_markers=["REQUEST_FLOW_SECTION"],
            answer_spans=[
                "In total a single request crosses 4 hops between the edge and the point where application code starts executing."
            ],
        ),
        F11Query(
            query_id="arch2_identifier",
            query_class="identifier",
            query_text="request.state.user_id",
            gold_markers=["AUTH_LAYER_SECTION"],
            answer_spans=[
                "The middleware validates the bearer token and, on success, populates request.state.user_id so downstream handlers can read the caller's identity without re-parsing the token."
            ],
        ),
        F11Query(
            query_id="arch2_quoted_phrase",
            query_class="quoted_phrase",
            query_text='"rewrites long documents into a compact skeleton"',
            gold_markers=["COMPRESSION_ENGINE_SECTION"],
            answer_spans=[
                "The semantic compression engine rewrites long documents into a compact skeleton before they are sent to a downstream model."
            ],
        ),
        F11Query(
            query_id="arch2_keyword_nl",
            query_class="keyword_nl",
            query_text="What does a circuit breaker do to traffic from a failing dependency?",
            gold_markers=["RESILIENCE_SECTION"],
            answer_spans=[
                "When a downstream dependency starts failing repeatedly, a circuit breaker trips and stops sending it new traffic for a cooldown period."
            ],
        ),
        F11Query(
            query_id="arch2_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="How does the platform keep popular records within quick reach rather than fetching them from slower long-term storage each time?",
            gold_markers=["PERSISTENCE_LAYER_SECTION"],
            answer_spans=[
                "The cache stores hot lookups such as plan status and recent usage counters so the database is spared repetitive queries."
            ],
        ),
        F11Query(
            query_id="arch2_lexical_trap",
            query_class="lexical_trap",
            query_text="What signal shows cache performance alongside request tracing for on-call engineers?",
            gold_markers=["OBSERVABILITY_SECTION"],
            answer_spans=[
                "A single cache-hit-ratio gauge is also published here, giving on-call engineers a quick signal without paging through raw logs."
            ],
            trap_term="cache",
            decoy_marker="PERSISTENCE_LAYER_SECTION",
        ),
        F11Query(
            query_id="arch2_multi_hop",
            query_class="multi_hop",
            query_text="After a request passes through the auth middleware, how many total hops does it take before reaching application code, and what identifier does the middleware set for downstream handlers?",
            gold_markers=["REQUEST_FLOW_SECTION", "AUTH_LAYER_SECTION"],
            answer_spans=[
                "In total a single request crosses 4 hops between the edge and the point where application code starts executing.",
                "The middleware validates the bearer token and, on success, populates request.state.user_id so downstream handlers can read the caller's identity without re-parsing the token.",
            ],
        ),
    ],
)

_F11_RELEASE_RUNBOOK_SECTIONS = {
    "PREFLIGHT_CHECKS_SECTION": (
        "Every release must pass four required gates before it is allowed to proceed to deployment. The gates cover automated unit tests, static lint checks, a dependency security scan, and a completed changelog entry. Release engineers confirm the staging environment configuration matches production before signing off. Any skipped gate requires written approval from the on-call release lead."
    ),
    "VERSION_BUMP_SECTION": (
        "Bumping the version requires editing the release_metadata.json file that stores the current semantic version string. The engineer increments the major, minor, or patch segment according to the scope of the change. Once the file is saved, a version-bump commit is created separately from any feature commits. The commit message must reference the ticket number driving the release."
    ),
    "DEPLOY_STEPS_SECTION": (
        "The deploy sequence starts with tagging the release commit, then triggers the build pipeline, and finally promotes the build to production traffic. A release is automatically aborted if the pipeline has not completed within 40 minutes. The deployment coordinator watches the pipeline dashboard until the promotion step reports success. Traffic is shifted gradually rather than all at once to limit blast radius."
    ),
    "SMOKE_VERIFICATION_SECTION": (
        "After deployment finishes, the on-call engineer must verify that the health endpoint returns a healthy status. The team also runs automated scripts to verify that key API routes respond within acceptable latency. Standard practice is to verify database connectivity and to verify that background workers have restarted cleanly. Only after every check passes does the release get marked complete in the tracking system."
    ),
    "ROLLBACK_PROCEDURE_SECTION": (
        "If a release causes elevated error rates, the team immediately triggers the rollback procedure to restore the previous stable version. The coordinator first reverts the production traffic pointer to the last known-good build, then re-runs the deployment pipeline against that earlier tag. Once traffic is restored to the prior version, the team will verify that error rates return to baseline before closing the incident. The affected release is marked as rolled back in the release tracker so it is never redeployed without a fix."
    ),
    "POST_DEPLOY_SECTION": (
        "Once a release ships, the on-call rotation keeps a close watch on error-rate dashboards for the next 24 hours. Latency graphs and background-job queue depth are also tracked to catch slow regressions that smoke checks might miss. Any anomaly gets logged in the incident channel even if it does not yet require a rollback. The release is only considered fully stable after the extended monitoring window closes without incident."
    ),
}

F11_RELEASE_RUNBOOK_FIXTURE = F11MultiNodeFixture(
    fixture_id="f11_release_runbook",
    doc_type="prose",
    sections=_F11_RELEASE_RUNBOOK_SECTIONS,
    queries=[
        F11Query(
            query_id="rel_bare_numeric",
            query_class="bare_numeric",
            query_text="40",
            gold_markers=["DEPLOY_STEPS_SECTION"],
            answer_spans=[
                "A release is automatically aborted if the pipeline has not completed within 40 minutes."
            ],
        ),
        F11Query(
            query_id="rel_identifier",
            query_class="identifier",
            query_text="release_metadata.json",
            gold_markers=["VERSION_BUMP_SECTION"],
            answer_spans=[
                "Bumping the version requires editing the release_metadata.json file that stores the current semantic version string."
            ],
        ),
        F11Query(
            query_id="rel_quoted_phrase",
            query_class="quoted_phrase",
            query_text='"written approval from the on-call release lead"',
            gold_markers=["PREFLIGHT_CHECKS_SECTION"],
            answer_spans=[
                "Any skipped gate requires written approval from the on-call release lead."
            ],
        ),
        F11Query(
            query_id="rel_keyword_nl",
            query_class="keyword_nl",
            query_text="How is traffic shifted during deployment to limit blast radius?",
            gold_markers=["DEPLOY_STEPS_SECTION"],
            answer_spans=[
                "Traffic is shifted gradually rather than all at once to limit blast radius."
            ],
        ),
        F11Query(
            query_id="rel_pure_paraphrase",
            query_class="pure_paraphrase",
            query_text="What is the duration the crew spends observing performance metrics before a new build is called settled?",
            gold_markers=["POST_DEPLOY_SECTION"],
            answer_spans=[
                "The release is only considered fully stable after the extended monitoring window closes without incident."
            ],
        ),
        F11Query(
            query_id="rel_lexical_trap",
            query_class="lexical_trap",
            query_text="What must be verified once a release has been rolled back to the last stable version?",
            gold_markers=["ROLLBACK_PROCEDURE_SECTION"],
            answer_spans=[
                "Once traffic is restored to the prior version, the team will verify that error rates return to baseline before closing the incident."
            ],
            trap_term="verify",
            decoy_marker="SMOKE_VERIFICATION_SECTION",
        ),
        F11Query(
            query_id="rel_multi_hop",
            query_class="multi_hop",
            query_text="Which file gets edited to bump the version, and what are the three steps of the deploy sequence that follow?",
            gold_markers=["VERSION_BUMP_SECTION", "DEPLOY_STEPS_SECTION"],
            answer_spans=[
                "Bumping the version requires editing the release_metadata.json file that stores the current semantic version string.",
                "The deploy sequence starts with tagging the release commit, then triggers the build pipeline, and finally promotes the build to production traffic.",
            ],
        ),
    ],
)


ALL_F11_FIXTURES: List[F11MultiNodeFixture] = [
    F11_ARCH_FIXTURE,
    F11_CODE_FIXTURE,
    F11_JSON_FIXTURE,
    F11_API_REF_FIXTURE,
    F11_INFRA_RUNBOOK_FIXTURE,
    F11_BILLING_POLICY_FIXTURE,
    F11_SECURITY_AUDIT_FIXTURE,
    F11_ML_PAPER_FIXTURE,
    F11_LEGAL_TOS_FIXTURE,
    F11_POSTMORTEM_A_FIXTURE,
    F11_POSTMORTEM_B_FIXTURE,
    F11_ONBOARDING_FIXTURE,
    F11_PRODUCT_FAQ_FIXTURE,
    F11_ARCHITECTURE_FIXTURE,
    F11_RELEASE_RUNBOOK_FIXTURE,
]

# TODO(#250 full spec): expand toward the design memo's 12-16 doc / ~240
# query target -- 1 more prose/tutorial doc, 3 more API-reference docs, 2
# more mixed code+prose docs, 2 more config/table-heavy docs; ~40 queries
# per class instead of the ~3 landed here per class. Keep the per-class
# reporting discipline (tests/f11_fixture_harness.py) unchanged when doing
# so -- it is what prevents the aggregate-wash failure mode this file
# exists to fix.
