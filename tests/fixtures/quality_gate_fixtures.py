"""Sealed ground-truth fixtures for the compression quality gate.

These fixtures are an INDEPENDENT oracle (compression-quality-eval skill,
codex MF1 constraint). The relevant sections, load-bearing literals, and
expected order are FIXED LABELS authored by hand right here -- never
computed by ``search_semantic_with_scores`` or any other ranker/embedder
under test. Grading a ranker change against labels the ranker itself
produced would be circular; these labels are independent of the engine.

Five sealed docs cover the task-sensitivity dimension the quality-eval skill
calls out (P3) across the content classes MF1 requires (prose / code /
JSON / QA): prose is robust, code and structured/mixed/JSON content are
fragile, and the QA fixture proves a specific question's answer survives
verbatim. A sixth, smaller fixture (``INTEGRATION_FIXTURE``) is used only by
the real-compressor integration test -- see the module docstring in
``tests/test_quality_gate.py`` for why it is deliberately single-paragraph.

Every non-integration fixture is authored as exactly three ``\n\n``-separated
sections with one ground-truth answer span / load-bearing token drawn from
each section. This shape is deliberate: ``src/quality_gate.py``'s
``first_paragraph_compressor`` reference compressor (used to prove the
oracle discriminates PARTIAL quality loss, not just the pass/fail extremes)
keeps only the first section, so it lands at roughly 1/3 recall on any
fixture built this way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class QualityGateFixture:
    """A sealed, hand-labeled document used to grade a compressed skeleton."""

    fixture_id: str
    doc_type: str  # "prose" | "code" | "mixed" | "json" | "qa"
    source_text: str
    # Answer spans: natural-language content that must survive somewhere in
    # the skeleton for the fixture's implied questions to remain answerable
    # (answerability / recall grader).
    answer_spans: List[str] = field(default_factory=list)
    # Load-bearing tokens: numbers / identifiers / hashes / code tokens that
    # must appear BYTE-IDENTICAL in the skeleton (extractive compression's
    # weakest dimension per the compression-quality-eval skill).
    load_bearing_tokens: List[str] = field(default_factory=list)
    # Order markers: unique tokens whose first-occurrence order in the
    # skeleton must match this list's order (source-order preservation).
    order_markers: List[str] = field(default_factory=list)
    # Optional: the natural-language question this fixture's answer_spans
    # answer (QA-style fixtures). Documentation only -- graders never read
    # this field, they only ever check `answer_spans`/`load_bearing_tokens`
    # survive verbatim. Kept optional + default None so it stays additive
    # and does not disturb the existing prose/code/mixed fixtures.
    query: Optional[str] = None


PROSE_FIXTURE = QualityGateFixture(
    fixture_id="qg_prose",
    doc_type="prose",
    source_text=(
        "ALPHA_PRICING_SECTION: The Pro plan costs $49 per month for up to "
        "500000 tokens of monthly compression.\n\n"
        "BETA_GATEWAY_SECTION: The MCP endpoint requires a gc_9f3a2b1c bearer "
        "key for authentication.\n\n"
        "GAMMA_ENTERPRISE_SECTION: Enterprise customers pay 499 dollars "
        "monthly for a dedicated Fly machine."
    ),
    answer_spans=[
        "Pro plan costs $49 per month",
        "gc_9f3a2b1c bearer key",
        "pay 499 dollars monthly",
    ],
    load_bearing_tokens=["$49", "500000", "gc_9f3a2b1c", "499"],
    order_markers=[
        "ALPHA_PRICING_SECTION",
        "BETA_GATEWAY_SECTION",
        "GAMMA_ENTERPRISE_SECTION",
    ],
)


CODE_FIXTURE = QualityGateFixture(
    fixture_id="qg_code",
    doc_type="code",
    source_text=(
        "ALPHA_AUTH_SECTION: def verify_api_key(key: str) -> bool: return "
        "key.startswith('gc_') and len(key) == 40.\n\n"
        "BETA_RATELIMIT_SECTION: RATE_LIMIT_PER_MINUTE = 120 and the function "
        "check_rate_limit raises RateLimitExceededError when exceeded.\n\n"
        "GAMMA_CACHE_SECTION: CACHE_TTL_SECONDS = 300 and get_cached_value "
        "returns None on a cache miss."
    ),
    answer_spans=[
        "def verify_api_key(key: str) -> bool",
        "raises RateLimitExceededError",
        "returns None on a cache miss",
    ],
    load_bearing_tokens=[
        "verify_api_key",
        "RATE_LIMIT_PER_MINUTE = 120",
        "CACHE_TTL_SECONDS = 300",
    ],
    order_markers=[
        "ALPHA_AUTH_SECTION",
        "BETA_RATELIMIT_SECTION",
        "GAMMA_CACHE_SECTION",
    ],
)


MIXED_FIXTURE = QualityGateFixture(
    fixture_id="qg_mixed",
    doc_type="mixed",
    source_text=(
        "ALPHA_TABLE_SECTION: The compression benchmark table shows ratio "
        "9.54x for large documents versus 7.78x for the prior baseline.\n\n"
        'BETA_CONFIG_SECTION: { "tier": "pro", "monthly_limit": 2000000, '
        '"embedding_model": "bge-small-en-v1.5" } is the default config block.\n\n'
        "GAMMA_NOTES_SECTION: The HNSW index swap happened in v1.3.1 and the "
        "pgvector extension version is 0.7.0."
    ),
    answer_spans=[
        "ratio 9.54x for large documents",
        '"monthly_limit": 2000000',
        "pgvector extension version is 0.7.0",
    ],
    load_bearing_tokens=["9.54x", "2000000", "bge-small-en-v1.5", "0.7.0"],
    order_markers=[
        "ALPHA_TABLE_SECTION",
        "BETA_CONFIG_SECTION",
        "GAMMA_NOTES_SECTION",
    ],
)


JSON_FIXTURE = QualityGateFixture(
    fixture_id="qg_json",
    doc_type="json",
    source_text=(
        'ALPHA_SERVICE_SECTION: { "service": "gotcontext-api", "timeout_ms": '
        '3000, "region": "iad" } is the default service block.\n\n'
        'BETA_RETRY_SECTION: { "retry": { "max_attempts": 4, "backoff_ms": 250 '
        "} } controls webhook redelivery.\n\n"
        'GAMMA_LIMIT_SECTION: { "tier": "pro", "monthly_limit": 2000000 } is '
        "the plan-gating record for a Pro key."
    ),
    answer_spans=[
        '"timeout_ms": 3000',
        '"max_attempts": 4',
        '"monthly_limit": 2000000',
    ],
    load_bearing_tokens=["3000", "iad", "250", "2000000"],
    order_markers=[
        "ALPHA_SERVICE_SECTION",
        "BETA_RETRY_SECTION",
        "GAMMA_LIMIT_SECTION",
    ],
)


QA_FIXTURE = QualityGateFixture(
    fixture_id="qg_qa",
    doc_type="qa",
    source_text=(
        "ALPHA_BUDGET_SECTION: The hard monthly spend ceiling for the shared "
        "compute cluster is $12,400, above which provisioning requests queue "
        "for manual finance approval.\n\n"
        "BETA_LICENSE_SECTION: A self-hosted license token is valid for 395 "
        "days from issuance before POST /v1/licenses/generate must be called "
        "for renewal.\n\n"
        "GAMMA_SLA_SECTION: The finance approval SLA target is 4 business "
        "hours from the moment a request queues."
    ),
    query="What is the hard monthly spend ceiling before finance approval is required?",
    answer_spans=[
        "hard monthly spend ceiling for the shared compute cluster is $12,400",
        "valid for 395 days from issuance",
        "approval SLA target is 4 business hours",
    ],
    load_bearing_tokens=[
        "$12,400",
        "395 days",
        "POST /v1/licenses/generate",
        "4 business hours",
    ],
    order_markers=[
        "ALPHA_BUDGET_SECTION",
        "BETA_LICENSE_SECTION",
        "GAMMA_SLA_SECTION",
    ],
)


# Small single-topic doc for the real-compressor integration test. Kept
# separate from the richer fixtures above (which drive the model-free
# bidirectional grader tests) because a single short paragraph is the only
# shape guaranteed to survive the real engine's chunk-merge + first-sentence
# summary behavior deterministically -- see the module docstring in
# tests/test_quality_gate.py for the full explanation.
INTEGRATION_FIXTURE = QualityGateFixture(
    fixture_id="qg_integration",
    doc_type="prose",
    source_text=(
        "The gotcontext Pro plan costs $49 per month and ships with the "
        "gc_9f3a2b1c reference API key example."
    ),
    answer_spans=["Pro plan costs $49 per month"],
    load_bearing_tokens=["$49", "gc_9f3a2b1c"],
    order_markers=[],
)


ALL_FIXTURES: List[QualityGateFixture] = [
    PROSE_FIXTURE,
    CODE_FIXTURE,
    MIXED_FIXTURE,
    JSON_FIXTURE,
    QA_FIXTURE,
]
