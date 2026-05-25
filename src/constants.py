"""
Configuration Constants for Semantic Modulator

This module centralizes all magic numbers and configuration defaults
used throughout the system. Each constant is documented with its purpose
and rationale.

Many constants support environment variable overrides for production
configuration without code changes.

Version: 0.7.0
"""

import os

# ============================================================================
# Embedding Models
# ============================================================================

DEFAULT_TEXT_MODEL = "all-MiniLM-L6-v2"
"""
Default embedding model for general text.
- Size: ~80MB download
- Dimensions: 384
- Speed: ~1000 sentences/sec on CPU
- Use case: Document compression, general semantic similarity
"""

DEFAULT_CODE_MODEL = "microsoft/codebert-base"
"""
Embedding model optimized for source code.
- Use case: Code compression, code similarity detection
- Fallback: Uses DEFAULT_TEXT_MODEL if codebert unavailable
"""

DEFAULT_IMAGE_MODEL = "clip-ViT-B-32"
"""
CLIP model for image-text embeddings.
- Use case: Multimodal compression (text + images)
- Architecture: Vision Transformer
"""

# ============================================================================
# Semantic Compression Parameters
# ============================================================================

SIMILARITY_THRESHOLD = 0.75
"""
Minimum cosine similarity for creating graph edges.
- Range: 0.0 to 1.0
- Higher values → Sparser graph → Faster PageRank, less compression
- Lower values → Denser graph → Slower PageRank, more compression
- Default 0.75 balances speed and quality
WHY: Below 0.75, too many spurious connections. Above 0.75, graph too sparse.
"""

SKELETON_RATIO = 0.2
"""
Percentage of nodes shown in skeleton view.
- Range: 0.0 to 1.0
- Higher values → Less compression → More context
- Lower values → More compression → Less context
- Default 0.2 = 80% token reduction target
WHY: Empirically found to preserve semantic structure while achieving high compression.
"""

DEFAULT_CHUNK_SIZE = 512
"""
Maximum tokens per semantic chunk.
- Smaller chunks → More granular retrieval, slower processing
- Larger chunks → Less granular retrieval, faster processing
- Default 512 balances granularity and performance
WHY: Matches typical paragraph size, fits well in embedding model context window.
"""

# ============================================================================
# AFM (Adaptive Focus Memory) Parameters
# ============================================================================

AFM_TAU_HIGH = 0.45
"""
Threshold for CRITICAL importance classification in AFM.
- Messages with importance ≥ tau_high are CRITICAL (always included)
- Example: Allergies, medical conditions, hard constraints
WHY: Empirically tuned to capture safety-critical information.
"""

AFM_TAU_MID = 0.25
"""
Threshold for RELEVANT vs TRIVIAL classification in AFM.
- Messages with importance ≥ tau_mid but < tau_high are RELEVANT
- Messages with importance < tau_mid are TRIVIAL (can be dropped)
WHY: Three-tier classification provides good granularity for dialogue compression.
"""

AFM_HALF_LIFE = 12
"""
Exponential decay half-life for temporal recency in AFM.
- Measured in turns (messages)
- After 12 turns, a message's recency score drops by 50%
- Older messages decay exponentially
WHY: Balances recent context with historical information preservation.
"""

# ============================================================================
# Resource Limits
# ============================================================================

MAX_DOCUMENT_SIZE_MB = float(os.getenv("MAX_DOCUMENT_SIZE_MB", "100.0"))
"""
Maximum size for a single document (MB).
- Prevents memory exhaustion from excessively large documents
- 100MB ≈ 25M tokens (word-based estimate)
- Environment variable: MAX_DOCUMENT_SIZE_MB
WHY: System tested up to 100MB documents without performance degradation.
"""

MAX_TEXT_LENGTH_BYTES = int(os.getenv("MAX_TEXT_LENGTH_BYTES", str(1_000_000)))
"""
Maximum text content length in bytes (default: 1MB).
- Prevents memory exhaustion from excessively large text inputs
- Applied during document ingestion
- Environment variable: MAX_TEXT_LENGTH_BYTES
WHY: 1MB is sufficient for most documents while preventing DoS attacks.
"""

# Validation for MAX_TEXT_LENGTH_BYTES
if MAX_TEXT_LENGTH_BYTES < 1000:
    raise ValueError("MAX_TEXT_LENGTH_BYTES must be >= 1000")

MAX_TOTAL_STORAGE_MB = 1024.0
"""
Maximum total storage across all documents (MB).
- Prevents unbounded disk usage
- 1GB default limit for cached documents and metadata
WHY: Reasonable limit for local development; configurable for production.
"""

MAX_DOCUMENTS = 1000
"""
Maximum number of documents that can be cached.
- Prevents excessive memory usage from metadata tracking
WHY: 1000 documents ×  100MB each = 100GB theoretical max (but total storage limit applies).
"""

MAX_MEMORY_MB = 2048.0
"""
Maximum memory usage for the server process (MB).
- Triggers warnings when exceeded
- 2GB default allows comfortable operation on typical developer machines
WHY: Most systems have 8GB+ RAM; 2GB leaves headroom for OS and other processes.
"""

# ============================================================================
# Context Window Management
# ============================================================================

MAX_CONTEXT_TOKENS = 100_000
"""
Default maximum context window size (tokens).
- Typical for Claude 3 Sonnet (200K actual, but use conservative estimate)
- Used for adaptive context window management
WHY: Conservative estimate ensures we don't exceed model limits.
"""

DEFAULT_ADAPT_MAX_TOKENS = 100_000
"""
Default max_tokens for adapt_to_context_window tool.
- Falls back to this if user doesn't specify max_tokens
WHY: Same as MAX_CONTEXT_TOKENS for consistency.
"""

# ============================================================================
# Validation Thresholds
# ============================================================================

MIN_TEXT_LENGTH = 1
"""
Minimum text length for document ingestion (characters).
- Empty or whitespace-only documents are rejected
- Prevents noise in the document store
WHY: Zero-length documents have no semantic content to compress.
"""

MIN_TOKEN_BUDGET = 1
"""
Minimum token budget for AFM context building.
- Zero or negative budgets are rejected
WHY: Cannot build context with zero tokens.
"""

RESOURCE_WARN_THRESHOLD = 0.8
"""
Warning threshold for resource usage (percentage).
- Warns when storage/memory usage exceeds 80% of limit
- Gives users time to clean up before hitting hard limit
WHY: Standard threshold used in ResourceManager.
"""

# ============================================================================
# Fidelity Level Token Budgets
# ============================================================================

FIDELITY_ABSTRACT_TOKENS = 10
"""
Token budget for ABSTRACT fidelity level.
- One-sentence summary only
- Minimal detail, maximum compression
"""

FIDELITY_OUTLINE_TOKENS = 30
"""
Token budget for OUTLINE fidelity level.
- Summary + section context
"""

FIDELITY_STRUCTURE_TOKENS = 50
"""
Token budget for STRUCTURE fidelity level.
- Summary + entities + metadata
"""

FIDELITY_DETAILED_TOKENS = 100
"""
Token budget for DETAILED fidelity level.
- Summary + entities + key excerpts
"""

# FIDELITY_RAW has no fixed budget (returns full original text)

# ============================================================================
# File Sync & Versioning
# ============================================================================

DEFAULT_DIFF_CONTEXT_LINES = 3
"""
Number of context lines in unified diffs.
- Standard git diff format uses 3 lines of context
WHY: Familiar to developers, provides sufficient context for understanding changes.
"""

DEFAULT_VERSION_RETENTION = 10
"""
Default number of versions to retain per document.
- Oldest versions auto-deleted when limit exceeded
- Set to 0 for unlimited retention (not recommended)
WHY: Balances version history utility with storage overhead.
"""

MAX_FILE_SYNC_ENTRIES = 1000
"""
Maximum number of file metadata entries to track in FileSyncManager (v0.4.2).
- Oldest entries auto-evicted (LRU) when limit exceeded
- Set to 0 for unlimited tracking (not recommended)
- Memory impact: ~170 bytes per file metadata entry
- Example: 1000 files = ~170KB total
WHY: Prevents unbounded memory growth in long-running servers while supporting
     typical project sizes (most projects have < 1000 tracked documents).
"""

MAX_ACE_CONTEXTS = int(os.getenv("MAX_ACE_CONTEXTS", "100"))
"""
Maximum number of ACE (Agentic Context Engineering) contexts to retain (v0.4.2).
- Oldest contexts auto-evicted (LRU) when limit exceeded
- Set to 0 for unlimited retention (not recommended)
- Memory impact: ~70KB per context (10 bullets with embeddings)
- Example: 100 contexts = ~7MB total
- Environment variable: MAX_ACE_CONTEXTS
WHY: Prevents unbounded memory growth in long-running servers while supporting
     extensive dialogue history. Profiling confirmed context deletion works correctly,
     this limit just prevents accumulation.
"""

# Validation for MAX_ACE_CONTEXTS
if MAX_ACE_CONTEXTS < 1:
    raise ValueError("MAX_ACE_CONTEXTS must be >= 1")

# ============================================================================
# Logging
# ============================================================================

LOG_LEVEL_DEFAULT = "INFO"
"""
Default logging level for the server.
- Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
WHY: INFO provides useful feedback without being too verbose.
"""

# ============================================================================
# Rate Limiting Configuration
# ============================================================================

RATE_LIMIT_INGEST = float(os.getenv("RATE_LIMIT_INGEST", "10.0"))
"""
Tokens per second for document ingestion.
- Environment variable: RATE_LIMIT_INGEST
WHY: 10 ingests/sec handles typical load; increase for high-throughput deployments.
"""

RATE_LIMIT_BATCH = float(os.getenv("RATE_LIMIT_BATCH", "2.0"))
"""
Tokens per second for batch operations.
- Environment variable: RATE_LIMIT_BATCH
WHY: 2 batches/sec prevents resource exhaustion from concurrent batch processing.
"""

RATE_LIMIT_COMPRESSION = float(os.getenv("RATE_LIMIT_COMPRESSION", "5.0"))
"""
Tokens per second for compression operations.
- Environment variable: RATE_LIMIT_COMPRESSION
WHY: 5 compressions/sec balances throughput with CPU utilization.
"""

# ============================================================================
# Health Check Configuration
# ============================================================================

HEALTH_CHECK_CACHE_SECONDS = int(os.getenv("HEALTH_CHECK_CACHE_SECONDS", "10"))
"""
Cache duration for health check results in seconds.
- Prevents expensive checks on every request
- Environment variable: HEALTH_CHECK_CACHE_SECONDS
WHY: 10 seconds balances responsiveness with performance overhead.
"""

# ============================================================================
# Version Information
# ============================================================================

VERSION = "0.11.0"
"""Current version of Token Saver 5000 (Single source of truth)"""

VERSION_STRING = f"Token Saver 5000 v{VERSION}"
"""Full version string for logging and display"""

# ============================================================================
# Tool Result Formatting (v0.11.0 - Token Optimization)
# ============================================================================

TOOL_RESULT_SOFT_LIMIT_CHARS = int(os.getenv("TOOL_RESULT_SOFT_LIMIT", "40000"))
"""
Soft limit for MCP tool result size in characters.
- When exceeded, metadata is stripped to reduce size
- Claude Code caps tool results at 50K chars; this keeps us safely under
- Environment variable: TOOL_RESULT_SOFT_LIMIT
WHY: 40K is 80% of Claude Code's 50K limit, leaving headroom for JSON wrapping.
"""

TOOL_RESULT_HARD_LIMIT_CHARS = int(os.getenv("TOOL_RESULT_HARD_LIMIT", "49000"))
"""
Hard limit for MCP tool result size in characters.
- When exceeded, response is paginated with a continuation token
- Environment variable: TOOL_RESULT_HARD_LIMIT
WHY: 49K stays under Claude Code's 50K per-tool cap with 1K safety margin.
"""

TOOL_RESULT_PREVIEW_CHARS = int(os.getenv("TOOL_RESULT_PREVIEW", "2000"))
"""
Preview size when response is paginated.
- First N chars shown with continuation instructions
- Matches Claude Code's preview size for persisted tool results
- Environment variable: TOOL_RESULT_PREVIEW
WHY: 2K preview matches Claude Code's disk-persisted preview convention.
"""

# ============================================================================
# Known Model Context Windows (v0.11.0 - Client Configuration)
# ============================================================================

KNOWN_MODEL_CONTEXT_WINDOWS = {
    # Anthropic Claude models (standard context)
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-1": 200_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    # Anthropic Claude models (1M context)
    "claude-opus-4-6[1m]": 1_000_000,
    "claude-sonnet-4-6[1m]": 1_000_000,
    "claude-opus-4-5[1m]": 1_000_000,
    "claude-sonnet-4-5[1m]": 1_000_000,
    # OpenAI models
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # Codex CLI models (v0.11.0 additions)
    "gpt-5.1-codex": 200_000,
    "codex-mini": 200_000,
    # Google Gemini models (legacy)
    "gemini-2.0-pro": 2_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    # Google Gemini models (v0.12.0 additions)
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-3.1-pro": 1_048_576,
    "gemini-3.1-flash": 1_048_576,
    "gemini-3.1-flash-lite": 1_048_576,
    # OpenCode additional models (cross-provider)
    "claude-4-opus": 200_000,
    "claude-4-sonnet": 200_000,
    "claude-4.5-sonnet": 200_000,
    "claude-3.7-sonnet": 200_000,
    "claude-3.5-sonnet": 200_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 200_000,
    "gpt-4.1-nano": 200_000,
    "o1-pro": 200_000,
    "gemini-2.0-flash-lite": 1_000_000,
    "groq-llama-4-scout": 512_000,
    "groq-llama-4-maverick": 512_000,
    "groq-deepseek-r1": 128_000,
    "groq-qwq": 128_000,
    "grok-3": 131_072,
    "grok-3-mini": 131_072,
    # Default fallback
    "default": 100_000,
}
"""
Known model context window sizes in tokens.
- Used by configure_for_client tool to auto-detect appropriate compression ratios
- Unknown models fall back to 'default' (100K tokens)
- Clients can override via explicit context_window_tokens parameter
WHY: Auto-tuning compression to available context improves token efficiency.
"""

KNOWN_MODEL_COMPRESSION_TRIGGERS = {
    # Gemini CLI compresses at 50% of context window
    "gemini-2.5-pro": 0.50,
    "gemini-2.5-flash": 0.50,
    "gemini-3.1-pro": 0.50,
    "gemini-3.1-flash": 0.50,
    "gemini-3.1-flash-lite": 0.50,
    # Claude Code compresses at ~93% of context window
    "claude-opus-4-6": 0.93,
    "claude-sonnet-4-6": 0.93,
    "claude-opus-4-6[1m]": 0.93,
    "claude-sonnet-4-6[1m]": 0.93,
    # GPT/other models don't auto-compress (ratio = 1.0)
    "gpt-4o": 1.0,
    # Codex CLI models compress at 80% (HISTORY_SOFT_CAP_RATIO = 0.8)
    "gpt-5.1-codex": 0.80,
    "codex-mini": 0.80,
    "o3": 0.80,
    "o4-mini": 0.80,
    # OpenCode trigger at 95% across all providers
    "claude-4-opus": 0.95,
    "claude-4-sonnet": 0.95,
    "claude-4.5-sonnet": 0.95,
    "claude-3.7-sonnet": 0.95,
    "claude-3.5-sonnet": 0.95,
    "gpt-4.1": 0.95,
    "gpt-4.1-mini": 0.95,
    "gpt-4.1-nano": 0.95,
    "o1-pro": 0.80,
    "groq-llama-4-scout": 0.95,
    "groq-llama-4-maverick": 0.95,
    "groq-deepseek-r1": 0.95,
    "groq-qwq": 0.95,
    "grok-3": 0.95,
    "grok-3-mini": 0.95,
    "default": 0.80,
}
"""
Compression trigger ratios by model.
Represents the % of context window at which the client typically starts compressing.
Lower values mean the client is more aggressive about compression.
Used to tune skeleton ratios: aggressive clients get more compressed output.
"""

DEFAULT_COMPRESSION_PROFILE = "balanced"
"""
Default compression profile name when no profile is explicitly set.
WHY: Balanced provides good compression without sacrificing too much detail.
"""

# ============================================================================
# Compression Profiles (v0.11.0 - Named Presets)
# ============================================================================

COMPRESSION_PROFILES = {
    "minimal": {
        "skeleton_ratio": 0.05,
        "fidelity": "ABSTRACT",
        "chunk_size": 256,
        "description": "Maximum compression, navigation only",
    },
    "summary": {
        "skeleton_ratio": 0.15,
        "fidelity": "OUTLINE",
        "chunk_size": 512,
        "description": "Quick overview, fits in compacted context",
    },
    "balanced": {
        "skeleton_ratio": 0.25,
        "fidelity": "STRUCTURE",
        "chunk_size": 512,
        "description": "Default, good for most tasks",
    },
    "detailed": {
        "skeleton_ratio": 0.50,
        "fidelity": "DETAILED",
        "chunk_size": 1024,
        "description": "Code review, deep analysis",
    },
    "full": {
        "skeleton_ratio": 0.80,
        "fidelity": "RAW",
        "chunk_size": 2048,
        "description": "Near-original, minimal compression",
    },
}
"""
Named compression presets bundling multiple parameters.
- Each profile defines skeleton_ratio, fidelity level, and chunk size
- Users set a profile via set_compression_profile tool
- Explicit parameters always override profile defaults
WHY: Simplifies UX by replacing 3+ parameter decisions with a single choice.
"""

# ============================================================================
# MIG Scoring Parameters (arXiv 2602.01719 — COMI)
# ============================================================================

MIG_DEFAULT_LAMBDA = float(os.getenv("MIG_DEFAULT_LAMBDA", "0.5"))
"""
Redundancy penalty weight for Marginal Information Gain scoring.
- Range: 0.0 (no redundancy penalty) to 1.0 (full penalty)
- Environment variable: MIG_DEFAULT_LAMBDA
WHY: 0.5 balances relevance maximisation and redundancy minimisation per COMI paper.
"""

MIG_MIN_CORPUS_TOKENS = int(os.getenv("MIG_MIN_CORPUS_TOKENS", "10"))
"""
Minimum token count before attempting TF-IDF vectorisation in MIG scorer.
- Below this threshold the scorer falls back to heuristic scoring.
- Environment variable: MIG_MIN_CORPUS_TOKENS
WHY: TF-IDF vocabulary is too sparse for very short inputs.
"""

# ============================================================================
# Quality Predictor Weights (PoC quality predictor)
# ============================================================================

QUALITY_ENTITY_WEIGHT = float(os.getenv("QUALITY_ENTITY_WEIGHT", "0.4"))
"""
Weight for entity-retention component in quality prediction.
- Environment variable: QUALITY_ENTITY_WEIGHT
WHY: Named entities are high-value semantic anchors; 0.4 gives them priority.
"""

QUALITY_COVERAGE_WEIGHT = float(os.getenv("QUALITY_COVERAGE_WEIGHT", "0.3"))
"""
Weight for sentence-coverage component in quality prediction.
- Environment variable: QUALITY_COVERAGE_WEIGHT
WHY: Coverage ensures broad topic representation; 0.3 is secondary to entities.
"""

QUALITY_RELEVANCE_WEIGHT = float(os.getenv("QUALITY_RELEVANCE_WEIGHT", "0.3"))
"""
Weight for query-relevance component in quality prediction.
- Environment variable: QUALITY_RELEVANCE_WEIGHT
WHY: Relevance is query-dependent and optional; 0.3 keeps it symmetrical with coverage.
"""

_RRF_K = int(os.getenv("_RRF_K", "60"))
"""
Reciprocal Rank Fusion k parameter (Cormack, Clarke, Buettcher — SIGIR 2009).
k=60 is the SOTA empirical default; higher k flattens the RRF score distribution.
- Environment variable: _RRF_K
WHY: k=60 was empirically shown to be robust across diverse retrieval tasks in the
     original paper. A smaller k (e.g. 10) gives more weight to top-ranked results;
     a larger k (e.g. 100) makes the fusion softer. 60 is the safe starting point.
"""

F11_RANKER_PATH = os.getenv("F11_RANKER_PATH", "a").lower().strip()
"""
F11 ranker path selector.
- "a" (default): dense cosine similarity only (backward-compatible Path A).
- "c": BM25+RRF hybrid retrieval (v1.34.35 Path C council patches).
- Environment variable: F11_RANKER_PATH
WHY: Path A is the existing default; Path C adds BM25 re-ranking via Reciprocal
     Rank Fusion without touching Path A's behavior when F11_RANKER_PATH != "c".
"""
