"""
Configuration Constants for Semantic Modulator

This module centralizes all magic numbers and configuration defaults
used throughout the system. Each constant is documented with its purpose
and rationale.

Version: 0.4.0
"""

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

MAX_DOCUMENT_SIZE_MB = 100.0
"""
Maximum size for a single document (MB).
- Prevents memory exhaustion from excessively large documents
- 100MB ≈ 25M tokens (word-based estimate)
WHY: System tested up to 100MB documents without performance degradation.
"""

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

MAX_ACE_CONTEXTS = 100
"""
Maximum number of ACE (Agentic Context Engineering) contexts to retain (v0.4.2).
- Oldest contexts auto-evicted (LRU) when limit exceeded
- Set to 0 for unlimited retention (not recommended)
- Memory impact: ~70KB per context (10 bullets with embeddings)
- Example: 100 contexts = ~7MB total
WHY: Prevents unbounded memory growth in long-running servers while supporting
     extensive dialogue history. Profiling confirmed context deletion works correctly,
     this limit just prevents accumulation.
"""

# ============================================================================
# Progress Indicators
# ============================================================================

PROGRESS_BAR_WIDTH = 40
"""
Character width for progress bars in CLI output.
- Fixed width for consistent terminal display
WHY: Fits comfortably in 80-column terminals.
"""

# ============================================================================
# Query Defaults
# ============================================================================

DEFAULT_QUERY_PRIORITY = 0.5
"""
Default query priority for adapt_to_context_window.
- Range: 0.0 (low) to 1.0 (high)
- 0.5 = medium priority (balanced approach)
WHY: Neutral default; user can override based on query importance.
"""

DEFAULT_SEARCH_TOP_K = 5
"""
Default number of results for semantic search.
- Returns top 5 most similar nodes by default
WHY: Small enough to be manageable, large enough to be useful.
"""

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
# Version Information
# ============================================================================

VERSION = "0.4.3"
"""Current version of Token Saver 5000"""

VERSION_STRING = f"Token Saver 5000 v{VERSION}"
"""Full version string for logging and display"""
