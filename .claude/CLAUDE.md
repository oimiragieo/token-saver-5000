# LLM Rules Production Pack

## Overview
- **Type**: Multi-platform agent configuration bundle
- **Stack**: Claude Code, Cursor 2.0, Factory Droid with shared rule base
- **Architecture**: Claude Projects as source of truth, mirrored role prompts for Cursor/Droid, technology-specific rule packs
- **Primary Model**: Claude Sonnet 4.6 for reasoning + vision, Claude Opus 4.6 for high-complexity tasks [1]

This CLAUDE.md is authoritative. Subdirectories extend these rules within the Claude Projects hierarchy.

## Codebase Audit Summary (2026-01-01)

### Audit Status: HEALTHY ✅
**No unused, orphaned, or legacy files detected. Codebase is clean and well-maintained.**

### Key Metrics (v0.10.0)
| Metric | Value | Status |
|--------|-------|--------|
| **Source Modules** | 71 Python files in src/ | ✅ All active (incl. semantic_modulator subpkg) |
| **Handler Modules** | 10 modules in src/handlers/ | ✅ All integrated |
| **MCP Tools** | 49 tools defined | ✅ All routed |
| **Test Files** | 78 test modules | ✅ 1,171+ tests |
| **Test Coverage** | 73%+ overall | ✅ Exceeds 70% threshold |
| **Documentation** | 49 markdown files | ⚠️ Minor updates needed |

### Version Alignment
- **pyproject.toml**: v0.10.0 ✅ (updated from 0.7.0)
- **README.md**: 44 MCP tools ✅ (updated from 35)
- **CHANGELOG.md**: Latest dated release v0.10.0 ✅ (added 2026-02-26)

### Architecture Summary
```
src/                              # 71 modules, ~25,000 lines
├── Core Compression (4 modules)  # semantic_compressor, code_compressor, adapter, multimodal
├── Embeddings (4 modules)        # 3-tier system: SBERT → ONNX → TF-IDF
├── Dialogue/Context (2 modules)  # AFM (dialogue), ACE (context)
├── File/Version (2 modules)      # file_sync_manager, version_manager
├── Persistence (1 module)        # ChromaDB + JSON fallback
├── Observability (4 modules)     # logging, metrics, tracing, health
├── Reliability (3 modules)       # timeout, circuit breaker, retry
├── Security (2 modules)          # path_validator, graceful_degradation
├── Experimental (3 modules)      # SCAR, TOON, training_utils
├── semantic_modulator/           # subpackage (additional modules)
└── handlers/ (10 modules)        # 49 MCP tool implementations
```

### Experimental Features Status
| Feature | Module | MCP Exposed | Production Ready |
|---------|--------|-------------|------------------|
| TOON Serialization | toon_serializer.py | ✅ v0.10.0 | ⚠️ API may change |
| SCAR Compression | scar_compressor.py | ✅ v0.10.0 | ❌ Untrained weights — results are meaningless |
| Multimodal | multimodal_compressor.py | ✅ v0.10.0 | ⚠️ Needs benchmarks |
| Training Utils | training_utils.py | ❌ Not exposed | ❌ Reference only |

### Claude Code Infrastructure (Implemented)
The following Claude Code integrations are active in this repository:
- `.claude/hooks/` - Lifecycle hooks (PreToolUse, PostToolUse, UserPromptSubmit)
- `.claude/commands/` - Custom slash commands (/review, /fix-issue)
- `.claude/subagents/` - 10 specialized agent definitions (analyst, pm, architect, developer, qa, ux-expert, product-owner, scrum-master, bmad-orchestrator, bmad-master)
- `.claude/agents/` - 9 alternate agent role definitions (legacy format, see subagents/ for authoritative routing)
- `.claude/workflows/` - 2 workflow definitions (greenfield-fullstack, brownfield-fullstack) — **declarative only**: Claude reads these as orchestration prompts; there is no automatic workflow runtime. The BMAD orchestrator subagent coordinates execution step-by-step.
- `.claude/config.yaml` - BMAD agent routing and model assignments
- `.claude/settings.json` - Tool permissions and hook configuration
- `.claude/.mcp.json` - 6 MCP server registrations (repo, artifacts, github, linear, slack, chrome-devtools)
- `.claude/schemas/` - 10 JSON validation schemas
- `.claude/templates/` - 9 reusable artifact templates
- `.claude/instructions/` - 12 guidance documents
- `.claude/context/` - Runtime artifacts, gate results, and session state
- `.claude/tools/gates/` - Quality gate validation script (gate.mjs)
- `skills/token-saver-context-compression/` - Python compression skill with 12 scripts

### Tech Debt Identified
1. **Documentation Tool Counts**: Some docs still reference old counts (HOW_IT_WORKS.md, API_REFERENCE.md)
2. **MCP_TOOLS_GUIDE.md**: 14 tools not fully documented in main guide
3. **GETTING_STARTED.md**: Contains 35 relative links that may not resolve correctly

### Zero Security Issues
- ✅ No suspicious network calls
- ✅ No credential leaks
- ✅ No path traversal vulnerabilities (CWE-22 hardened)
- ✅ All file I/O validated via PathValidator

---

## Critical Context: Claude Code Unique Capabilities

Claude Code has unique capabilities that set it apart from generic agent configurations:

1. **Strict Instruction Hierarchy**: CLAUDE.md content is treated as **immutable system rules** with strict priority over user prompts
2. **Hierarchical Memory System**: Reads CLAUDE.md files recursively UP from CWD to root, AND discovers them in subdirectories
3. **Hooks System**: Lifecycle hooks (PreToolUse, PostToolUse, UserPromptSubmit, Notification, Stop) for deterministic automation
4. **Model Context Protocol (MCP)**: Native integration with external tools, databases, and APIs
5. **Custom Slash Commands**: Repeatable workflows stored in `.claude/commands/`
6. **Subagents**: Specialized agents with isolated context windows and tool permissions
7. **Extended Thinking**: Can use long-form reasoning with extended context windows (1M+ tokens)

### Core Principles for CLAUDE.md

1. **CLAUDE.md is AUTHORITATIVE** - Treated as system rules, not suggestions
2. **Modular Sections** - Use clear markdown headers to prevent instruction bleeding
3. **Front-load Critical Context** - Large CLAUDE.md files provide better instruction adherence
4. **Hierarchical Strategy**: Root = universal rules; Subdirs = specific context
5. **Token Efficiency Through Structure** - Use sections to keep related instructions together
6. **Living Documentation** - Use `#` key during sessions to add memories organically

## Universal Development Rules

### Code Quality (MUST)
- **MUST** create a Plan Mode artifact before modifying more than one file; summarize dependencies and tests impacted.
- **MUST** generate or update automated tests covering critical paths before requesting merge.
- **MUST** keep security controls (authz, secrets, PII) unchanged unless explicitly tasked.
- **MUST** document decisions in Artifacts or repo ADRs when deviating from defaults.

### Collaboration (SHOULD)
- **SHOULD** use Claude Projects instructions for shared vocabulary, business context, and tone [2].
- **SHOULD** sync Cursor and Droid executions back into the Claude Project activity feed after major milestones.
- **SHOULD** promote Artifacts to versioned documents for UI/UX deliverables.
- **SHOULD** prefer Claude's built-in repo search and diff MCP skills over manual file browsing.

### Safeguards (MUST NOT)
- **MUST NOT** delete secrets, env files, or production infrastructure manifests.
- **MUST NOT** bypass lint/test hooks; rerun failed commands with context.
- **MUST NOT** push directly to protected branches; use reviewed pull requests.
- **MUST NOT** rely on hallucinated APIs—verify via docs or code search MCP.

## Platforms & Interfaces

### Claude Code (Primary)
- Launch tasks within the configured Claude Project; inherit root + path-specific CLAUDE.md instructions.
- Use Artifacts for live previews of code, docs, or UI prototypes and hand them off to collaborators [1].
- Apply Hooks under `.claude/hooks` to auto-run plan validation, linting, and artifact publication to Projects.

### Cursor IDE
- Composer provides low-latency multi-step coding; prefer Composer for iterative edits, and escalate to Claude for deep reasoning [3].
- Trigger Plan Mode (`Shift+Tab`) before large changes—Cursor researches the repo and stores a markdown plan under `.cursor/plans/` [4].
- Offload long-running refactors to Cloud Agents so work continues after the IDE closes [5].

### Factory Droid
- Invoke specialized Droids from CLI or IDE panes; layer repo context, design docs, and telemetry streams as contextual sources [6].
- Use checkpoint hooks (`hooks/`) to gate commits on test green, static analysis, and QA review.
- Share outputs back into Claude Projects to keep a single source of decision history.

## Hooks Policy
- **PreToolUse**: enforce plan creation, dependency diffing, and risk assessment before code generation.
- **UserPromptSubmit**: normalize prompts (role, tone, goal) and tag them for project analytics.
- **PostToolUse**: summarize changes, collect lint/test logs, publish Artifacts, and notify Factory Cloud agents when further work is required.

## Skills Inventory
- `.claude/skills/repo-rag.yaml`: registers repo/knowledge-base retrieval MCP endpoints via filesystem MCP server.
- `.claude/skills/artifact-publisher.yaml`: pushes generated artifacts to the Claude Project activity feed via filesystem MCP server.
- `.claude/skills/context-bridge.yaml`: syncs metadata across Claude, Cursor, and Droid sessions (requires GITHUB_TOKEN, LINEAR_API_KEY, SLACK_BOT_TOKEN).

## Rule Packs
- Framework-specific `.md` and `.yaml` files in `rules-library/` enforce language conventions, testing standards, and deployment steps.
- Rules share identifiers across platforms so Cursor `.cursorrules` and Droid guidelines match the Claude truth source.

## Common Bash Commands

### Development Workflow
- `pip install -r requirements.txt`: Install project dependencies
- `python scripts/quickstart.py`: One-command setup (recommended for first-time setup)
- `python scripts/check_setup.py`: Verify installation and model downloads
- `python -m src.server`: Start MCP server (stdio mode)
- `pytest tests/ -v`: Run all tests with verbose output
- `pytest tests/test_functional.py -v`: Run specific test file
- `pytest tests/ --cov=src --cov-report=term`: Run tests with coverage
- `black src/ tests/ examples/`: Format code (auto-fix)
- `ruff check src/ tests/ examples/`: Lint code

### Git Workflow
- `git status`: Check working directory status
- `git diff`: View unstaged changes
- `git log --oneline`: View commit history
- `gh issue view <number>`: View GitHub issue details
- `gh pr create`: Create pull request

### Code Quality
- Prefer running single test files for performance (`pytest tests/test_functional.py`)
- Always run linting before committing (`ruff check src/`)
- Use `black` for consistent code formatting

## Token Saver 5000 Project Specifics (v0.10.0 - Experimental Module Exposure - IN PROGRESS)

### Project Overview
**Token Saver 5000** is an MCP server implementing research-backed semantic compression for AI interactions. It achieves **85-90% token reduction (proven: 87.4%)** through graph-based semantic analysis.

> **Proven Performance:** 7.9× compression (485 → 61 tokens) on real quantum computing document. See `demo_proof.py`.

**Current Release (v0.10.0 - Experimental Module Exposure - IN PROGRESS):**
- ✅ **Experimental Handlers (5 new MCP tools):**
  - **New Handler File:** `src/handlers/experimental_handlers.py` (300+ lines)
  - **New Test File:** `tests/test_experimental_handlers.py` (30+ tests)
  - **TOON Tools (2):**
    * `toon_encode`: Encode data to TOON format (~40% smaller than JSON)
    * `toon_decode`: Decode TOON format back to structured data
    * Pure Python, always available, no external dependencies
  - **SCAR Tools (2):**
    * `scar_compress`: Compress embeddings using learnable compression
    * `scar_get_stats`: Get SCAR compressor statistics
    * Requires PyTorch, uses UNTRAINED random weights by default
  - **Multimodal Tool (1):**
    * `multimodal_ingest`: Ingest mixed content (text, code, images)
    * Requires Pillow for image support
    * Image paths validated via PathValidator (CWE-22 protection)
  - **All Experimental Responses Include:**
    * `"experimental": true` flag in every response
    * Graceful degradation when dependencies missing
    * Helpful error messages for missing PyTorch/Pillow
  - **Documentation:**
    * Feature Matrix added to MCP_TOOLS_GUIDE.md
    * SLA definitions: Production, Infrastructure, Experimental
    * Coverage and dependency notes for all modules
  - **Total MCP Tools:** 44 (was 39)

**Previous Release (v0.9.0 - Programmer UX Improvement Plan - COMPLETE):**
- ✅ **P0 CRITICAL: CodeSemanticCompressor Integration:**
  - **Issue:** Code files were using text chunking instead of AST-aware chunking
  - **Fix:** Created `CodeCompressionAdapter` (src/code_compression_adapter.py, 650+ lines)
  - **Features:**
    * Routes files to appropriate compressor based on file type
    * Lazy CodeBERT loading (~400MB saved on startup)
    * Environment variable opt-in: `PRELOAD_CODE_MODEL=true` for immediate loading
    * Unified API surface: `ingest_document()`, `generate_skeleton()`, `modulate_region()`, `search_semantic_with_scores()`
  - **Testing:** All 1063 tests passing with 72% coverage

- ✅ **P1 HIGH: Programmer UX Tools (3 new MCP tools):**
  - **`ingest_directory`** (src/handlers/compression_handlers.py):
    * Bulk ingest code files from directory using glob patterns
    * Default patterns: `*.py`, `*.js`, `*.ts`
    * Security: PathValidator prevents path traversal
    * Parallel processing via BatchCompressionManager (4× throughput)
    * Max 100 files per call
  - **`tool_help`** (src/handlers/help_handlers.py, NEW):
    * Detailed help, examples, and tips for all MCP tools
    * Structured JSON output with parameters, examples, tips, related tools
    * Tool discovery: call without tool_name to see all tools by category
  - **`check_environment`** (src/handlers/resource_handlers.py):
    * Comprehensive environment health check
    * Shows: models loaded, memory usage, cache hit ratio, stale documents, disk space
    * Returns recommendations for optimization

- ✅ **P1 HIGH: Search Semantic with Scores:**
  - **Method:** `search_semantic_with_scores()` in SemanticCompressor
  - **Returns:** `List[Tuple[node_id, similarity_score]]` instead of just node IDs
  - **Backward Compatible:** Original `search_semantic()` method preserved

- ✅ **P2 MEDIUM: Semantic Node IDs for Code Files:**
  - **Old pattern:** `{file_id}_func_{name}`, `{file_id}_class_{name}`
  - **New pattern:** `{file_id}::{name}` (e.g., `main.py::process_data`, `main.py::MyClass`)
  - **Benefits:** Cleaner, more semantic, easier to parse
  - **Server Support:** Updated `_extract_file_id_from_node()` to handle both patterns

- ✅ **Test Suite:** 1063 passed, 12 skipped, 72% coverage
- ✅ **Tool Count:** 38 MCP tools (was 35)

**v0.9.2 Hardening - Binary Detection & Security (COMPLETE):**
- ✅ **Binary File Detection Enhancement:**
  - `should_compress` tool now detects binary files (PDF, DOCX, images)
  - Returns `CONVERT_THEN_COMPRESS` recommendation with MarkItDown suggestion
  - Content sniffing via null byte ratio heuristic (>1% null bytes = binary)
  - Extension-based detection + content-based detection for unknown extensions
  - New response fields: `needs_conversion`, `is_text_readable`, `conversion_tool`

- ✅ **Security: Path Traversal Prevention in should_compress (CWE-22):**
  - **Issue:** `handle_should_compress` directly called `os.path.exists()` without PathValidator
  - **Attack Vector:** `../../etc/passwd` could bypass allowed directories
  - **Fix:** Added PathValidator call before any file I/O (lines 455-472)
  - **Graceful Degradation:** Logs warning if PathValidator unavailable in context

- ✅ **Error Handling: Binary Detection Read Errors:**
  - **Issue:** `is_binary_content()` returned `False` on read errors, silently treating unreadable files as text
  - **Fix:** Now returns `Tuple[bool, Optional[str]]` - (is_binary, error_message)
  - **Handler Updated:** Returns explicit error JSON when file can't be read

- ✅ **New Tests:** 3 path traversal security tests
  - `test_path_traversal_blocked` - Blocks `../../etc/passwd`
  - `test_path_traversal_with_absolute_path` - Blocks `/etc/passwd`
  - `test_valid_path_with_validator` - Validates allowed paths work
  - **Test Count:** 56 resource handler tests (was 53)

**Previous Release (v0.8.0 - Async Safety & Concurrency Audit - COMPLETE):**
- ✅ **Blocking Lock Audit Fix (Issue 3):**
  - **Issue:** ResourceManager and VersionManager used `threading.Lock` in handler-facing methods, blocking event loop under concurrent clients
  - **Solution:** ThreadPoolExecutor + `run_in_executor()` pattern for all handler-facing methods
  - **Changes:**
    * `version_manager.py`: Added `delete_versions_async()` wrapper (lines 557-568)
    * `version_manager.py`: Added lock protection to `delete_versions()` (line 541)
    * `compression_handlers.py`: Updated to use `add_version_async()` and `delete_versions_async()`
  - **Testing:** All 1063 tests passing with 75% coverage
  - **Pattern:** Sync method with lock → async wrapper using `run_in_executor` → handler awaits wrapper

**Previous Release (v0.7.0 - Enterprise Production Readiness - COMPLETE):**
- ✅ **Week 1-2 Complete:** Reliability Infrastructure (TimeoutManager, CircuitBreaker, RetryPolicy, RateLimiter, GracefulDegradation)
- ✅ **Week 3-4 Complete:** Comprehensive Testing Suite (100 new tests: integration, performance, chaos, E2E)
- ✅ **Week 5-6 Complete:** Observability & Monitoring (StructuredLogger, Prometheus metrics, OpenTelemetry tracing, HealthChecker)
- ✅ **Week 7-8 Complete:** DevOps & Operational Excellence
  - **HTTP Server for Production Monitoring** (src/http_server.py, 415 lines, 32 tests):
    * Optional aiohttp async server for Kubernetes health checks and metrics
    * 5 endpoints: /health/liveness, /health/readiness, /health/diagnostics, /metrics, / (root)
    * Environment-based configuration: HTTP_ENABLED (default: false), HTTP_HOST, HTTP_PORT
    * Zero-overhead when disabled (maintains stdio-only mode for local MCP use)
    * Integration with existing src/health.py and src/metrics.py modules
  - **Docker Multi-Stage Build** (Dockerfile, 134 lines):
    * Builder + runtime stages for <500MB target image (~450MB expected)
    * Security: Non-root user (uid 1000), read-only filesystem, dropped capabilities
    * Hybrid deployment: Supports both stdio mode (default) and HTTP mode (Kubernetes)
  - **Kubernetes Production Manifests** (deployment/kubernetes/, 14 files):
    * Core: Namespace, ConfigMap, Secret, Service, Deployment (2 replicas, 3-tier health probes)
    * Scaling: HPA (2-10 replicas, CPU 70%, memory 80%)
    * Monitoring: ServiceMonitor (Prometheus scraping), PrometheusRule (16 alerting rules)
    * High availability: Pod anti-affinity, zero-downtime rolling updates
  - **GitHub Actions CI/CD** (.github/workflows/, 10 files):
    * test.yml: Matrix testing (Python 3.10/3.11/3.12), pip caching, coverage enforcement
    * lint.yml: Black, Ruff, Bandit security scanning, complexity analysis
    * build.yml: Docker builds with layer caching, Trivy CVE scanning, SBOM generation
    * deploy.yml: Kubernetes deployment, staging auto-deploy, production approval-gated
  - **Test Coverage:** 32 new HTTP server tests (100% pass rate, 1,032 → 1,075 total tests)
  - **Test Suite Overhaul:** Fixed 87 test failures from API mismatches:
    * AFM handlers: Added async/await to all 35 tests (handlers are async def)
    * Chaos engineering: Fixed PersistenceManager, LRUEmbeddingCache, VersionManager API calls
    * Integration workflows: Removed unused imports and variables
  - **Dependencies:** Added aiohttp>=3.9.0 for HTTP server

**Previous Release (v0.6.1 - Security Hardening - COMPLETE):**
- 🔒 **Critical Security Fix: Path Traversal Prevention (CWE-22):**
  - **Issue:** File paths in `ingest_context()` were not validated, allowing arbitrary file read via `../../etc/passwd`
  - **Impact:** CVSS 7.5 HIGH - Arbitrary file read when combined with `refresh_document()` tool
  - **Attack Vector:** Local MCP server (stdio transport), requires file_path parameter
  - **Fix:** Multi-layer path validation with whitelist-based security model:
    * **Entry Point:** `handle_ingest()` validates all file paths via PathValidator
    * **Storage Layers:** Defense-in-depth checks in `file_sync_manager.py` and `version_manager.py`
    * **Allowed Directories:** Current working directory (`os.getcwd()`) and user home (`~`)
    * **Path Normalization:** Resolves symlinks, `..`, and redundant separators before validation
  - **Testing:** 31 security tests (29 passing, 2 skipped on Windows) with 96% coverage
  - **Verification:** All 735 tests passing with 72.69% overall coverage (up from 68%)

**Previous Release (v0.6.0-beta - COMPLETE):**
- ✅ **Track 1: Multi-Document Batch Processing (4× throughput improvement):**
  - BatchCompressionManager with concurrent document ingestion
  - Bounded parallelism with semaphore-based rate limiting (default: 4 concurrent)
  - Real-time progress tracking with callbacks
  - Error isolation (one failure doesn't block entire batch)
  - Automatic retry mechanism for transient failures
  - 18 comprehensive batch processing tests
  - Performance: 4 docs in 8.2s vs 32.8s sequential (4× improvement)
  - New MCP tool: batch_ingest_documents (handles 1-100 docs in single call)

- ✅ **Track 2: Interactive Graph Visualization (4 new MCP tools):**
  - GraphVisualizer with 4 export formats:
    * render_ascii(): Terminal-friendly text visualization with importance scores
    * export_json(): Structured JSON export for programmatic access
    * export_graphml(): GraphML format for Gephi, Cytoscape, NetworkX
    * visualize_html(): Interactive HTML with pyvis (draggable nodes, zoom, pan)
    * explain_compression_decision(): Detailed node analysis
  - 4 new MCP tools (35 total, was 31):
    * export_graph_json, visualize_graph_html, export_graph_graphml, explain_compression_decision
  - 16 visualization tests (15 passing, 1 skipped for optional pyvis dependency)
  - 68% coverage on GraphVisualizer

- ✅ **Track 3: Memory Optimization Engine (3-tier embedding system):**
  - **ONNX Tier** (60-70% memory reduction):
    * Quantized INT8 models (~150MB vs ~400MB)
    * 3-5× faster inference on CPU vs PyTorch
    * Automatic model download and caching
  - **TF-IDF Tier** (98% memory reduction):
    * Lightweight sklearn vectorization (~10MB)
    * 100-1000× faster than neural models
    * 70-80% quality correlation with SBERT
  - **LRU Embedding Cache** (60-80% hit rate):
    * Thread-safe OrderedDict-based cache with LRU eviction
    * Configurable capacity (default: 10k entries)
    * Optional disk persistence with msgpack serialization
    * TTL support for stale entry expiration
  - **Multi-Tier System:**
    * EmbeddingTier enum (STANDARD, ONNX, TFIDF)
    * Automatic fallback hierarchy: STANDARD → ONNX → TFIDF
    * Tier switching with set_tier() and get_tier()
    * Enhanced cache_stats(): tier info, LRU stats, memory breakdown
  - 24 memory optimization tests (all optional with graceful degradation)
  - Dependencies added: pyvis, onnxruntime, optimum, msgpack

- ✅ **1,171 comprehensive tests** (1,159 passed, 12 skipped, 73% coverage - v0.10.0 Audit)
  - Was 1,075 in v0.8.0 initial, 1,064 in v0.7.0 Week 7-8 initial, 1,032 in v0.7.0 Week 5-6, 864 in v0.7.0 Week 3-4, 764 in v0.7.0 Week 1-2, 735 in v0.6.1, 665 in Phase 1, 630 in post-ace, 591 in post-v0.6.0, 506 in v0.6.0, 446 in v0.5.0-beta, 427 in v0.4.3
  - 32 new tests added in v0.7.0 Week 7-8 DevOps infrastructure:
    * 32 HTTP server tests (endpoints, integration, configuration, lifecycle, edge cases - 100% pass rate)
  - 168 new tests added in v0.7.0 Week 5-6 observability infrastructure:
    * 43 structured logging tests (JSON/human formatters, async context propagation, OTEL integration - 91% coverage)
    * 29 Prometheus metrics tests (7 metrics, cardinality control, graceful degradation - 86% coverage)
    * 53 OpenTelemetry tracing tests (span creation, async propagation, exception handling - 85% coverage)
    * 43 health check tests (liveness/readiness/diagnostics, component health - 91% coverage)
  - 100 new tests added in v0.7.0 Week 3-4 comprehensive testing suite:
    * 50 integration workflow tests (complete workflows: ingest → compress → expand → refresh → versions)
    * 15 performance benchmark tests (throughput, latency, memory, cache, burst capacity)
    * 20 chaos engineering tests (disk failures, model crashes, network issues, data corruption)
    * 15 E2E scenario tests (research papers, codebase documentation, dialogue management)
  - 29 new tests added in v0.7.0 Week 1-2 reliability infrastructure:
    * 29 reliability tests (TimeoutManager, CircuitBreaker, RetryPolicy, RateLimiter, GracefulDegradation)
  - 70 tests added in v0.6.1 security hardening:
    * 31 path validation security tests (path_validator.py: 96% coverage)
    * 39 test fixes for path security compliance (absolute path requirements)
  - 159 tests added in Phase 1 production readiness:
    * 53 persistence comprehensive tests (persistence.py: 32% → 65% coverage)
    * 25 cache comprehensive tests (embedding_cache.py: 86% → 99% coverage)
    * 7 semantic fidelity benchmarks (semantic_ssim.py: 0% → 89%)
    * 39 ACE handlers comprehensive tests (ace_handlers.py: 37% → 100% coverage)
    * 35 AFM handlers comprehensive tests (afm_handlers.py: 40% → 100% coverage)
    * (v0.6.0 tests: 18 batch processing, 16 visualization, 24 memory optimization)
  - Zero tech debt introduced

- ✅ **Test Coverage Improvements (Phase 1 Handler Coverage - COMPLETE!):**
  - handlers/afm_handlers.py: 40% → **100%** ✅ (35 tests, 60pp improvement, exceeded 80% target by 20pp)
  - handlers/ace_handlers.py: 37% → **100%** ✅ (39 tests, 63pp improvement, exceeded 80% target by 20pp)
  - handlers/resource_handlers.py: 16% → **100%** ✅ (20 tests, full coverage achieved)
  - handlers/detection_handlers.py: 25% → **100%** ✅ (12 tests, full coverage achieved)
  - handlers/file_sync_handlers.py: 13% → **96%** ✅ (18 tests, 83pp improvement, exceeded 80% target by 16pp)
  - **Phase 1 Results:** 5/5 targeted handlers completed, 5/5 achieved ≥80% coverage!

- ✅ **Test Coverage Baseline (76.02% overall - EXCEEDED 70% production threshold!):**
  - **Excellent (90%+):** handlers/afm_handlers (100%), handlers/ace_handlers (100%), handlers/resource_handlers (100%), handlers/detection_handlers (100%), http_server (100%), mcp_core (100%), code_compressor (99%), semantic_compressor (99%), embedding_cache (99%), path_validator (96%), ace_framework (96%), reliability (96%), handlers/file_sync_handlers (96%), compression_advisor (91%), server (91%), health (91%), structured_logging (91%), rate_limiter (91%), version_manager (90%), fidelity_advisor (90%)
  - **Good (70-89%):** afm (87%), metrics (86%), file_sync_manager (86%), error_helpers (86%), observability (85%), embeddings_tfidf (84%), compression_handlers (84%), batch_manager (81%), scar_compressor (81%)
  - **Critical Gaps (<50%):**
    * embeddings_onnx.py (19%)
    * adaptive_rate_allocator.py (25%)
    * resource_manager.py (31%)
    * handlers/visualization_handlers.py (45%)
  - **Experimental (0% - not production critical):** multimodal_compressor, toon_serializer, training_utils

- ✅ **Code Quality:**
  - All code formatted with `black` (zero warnings)
  - All code linted with `ruff` (zero warnings)
  - Full backward compatibility maintained (all new features optional)

- ✅ **Previous Release Features (v0.5.0-beta and earlier):**
  - Async Support: Non-blocking embedding generation, prevents MCP timeouts
  - TypedDict handler hints for IDE autocomplete + type safety
  - File sync detection with MD5 checksums
  - Full version history with unified diffs
  - Memory Management: Version history pruning, file sync LRU eviction, ACE context LRU eviction
  - Real-time staleness detection for cached documents

### Key Test Files
- `tests/conftest.py` - Shared test fixtures and infrastructure (380 lines - v0.7.0 Week 3-4)
- `tests/test_functional.py` - Core compression features (19 tests)
- `tests/test_token_savings.py` - Compression benchmarks (21 tests)
- `tests/test_afm.py` - Dialogue memory (29 tests, allergy retention)
- `tests/test_file_sync.py` - File sync and version management (55 tests - v0.4.0)
- `tests/test_edge_cases.py` - Edge cases (50 tests - v0.4.0)
- `tests/test_ace_framework.py` - ACE Framework (34 tests - v0.4.0)
- `tests/test_mcp_routing.py` - MCP routing (24 tests, async - v0.5.0)
- `tests/test_compression_handlers.py` - Compression handlers (27 tests, async - v0.5.0)
- `tests/test_code_compressor.py` - AST-based code compression (47 tests - v0.4.3)
- `tests/test_server_unit.py` - Server initialization and validation (43 tests - v0.4.3)
- `tests/test_semantic_compressor_unit.py` - Semantic compressor unit tests (65 tests - v0.4.3)
- `tests/test_server_lifecycle.py` - Server lifespan management (6 tests - v0.4.4)
- `tests/test_async_operations.py` - Async operations validation (10 tests - v0.5.0)
- `tests/test_batch_processing.py` - Multi-document batch processing (18 tests - v0.6.0)
- `tests/test_visualization.py` - Graph visualization (16 tests - v0.6.0)
- `tests/test_memory_optimization.py` - Memory optimization tiers (24 tests - v0.6.0)
- `tests/test_persistence_comprehensive.py` - Persistence layer comprehensive (53 tests - Phase 1)
- `tests/test_cache_comprehensive.py` - LRU cache comprehensive (25 tests - Phase 1)
- `tests/test_semantic_fidelity.py` - Semantic SSIM fidelity benchmarks (7 tests - Phase 1)
- `tests/test_path_validator.py` - Path traversal security (31 tests - v0.6.1)
- `tests/test_http_server.py` - HTTP server endpoints and integration (32 tests - v0.7.0 Week 7-8)
- `tests/test_reliability.py` - Reliability infrastructure (29 tests - v0.7.0 Week 1-2)
- `tests/test_integration_workflows.py` - Integration workflows (50 tests - v0.7.0 Week 3-4)
- `tests/test_performance.py` - Performance benchmarks (15 tests - v0.7.0 Week 3-4)
- `tests/test_chaos_engineering.py` - Chaos engineering (20 tests - v0.7.0 Week 3-4)
- `tests/test_e2e_scenarios.py` - End-to-end scenarios (15 tests - v0.7.0 Week 3-4)
- `tests/test_structured_logging.py` - Structured logging (43 tests - v0.7.0 Week 5-6 - NEW!)
- `tests/test_metrics.py` - Prometheus metrics (29 tests - v0.7.0 Week 5-6 - NEW!)
- `tests/test_observability.py` - OpenTelemetry tracing (53 tests - v0.7.0 Week 5-6 - NEW!)
- `tests/test_health.py` - Health checks & diagnostics (43 tests - v0.7.0 Week 5-6 - NEW!)

### Important Implementation Details
**Path Validation Security (v0.6.1 - CWE-22 Prevention):**
- `src/path_validator.py` - Validates file paths against allowed directory whitelist (220 lines, 96% coverage)
- **Purpose:** Prevents path traversal attacks (CWE-22) via `../../etc/passwd` sequences
- **Architecture:** Multi-layer defense (entry point + storage layers)
- **Allowed Directories:** `os.getcwd()` + `os.path.expanduser("~")` (configurable whitelist)
- **Validation Steps:**
  1. Resolves symlinks and relative paths (`os.path.realpath()`)
  2. Normalizes redundant separators (`os.path.normpath()`)
  3. Validates against allowed directory whitelist
  4. Returns absolute path or raises ValueError
- **Integration:**
  * Entry point: `handle_ingest()` validates all file_path parameters
  * Defense-in-depth: `FileSyncManager.register_file()` and `VersionManager.add_version()` verify absolute paths
  * HandlerContext: PathValidator instance available to all handlers
- **Testing:** 31 comprehensive security tests covering exploit scenarios

**Observability & Monitoring Infrastructure (v0.7.0 Week 5-6 - <100ms Overhead):**
- **Structured Logging (`src/structured_logging.py` - 540 lines, 91% coverage):**
  - **Purpose:** Production-grade JSON and human-readable logging with async context propagation
  - **Key Features:**
    * Dual formatters: JSONFormatter (ISO 8601 timestamps) and HumanReadableFormatter (colored, aligned)
    * Async-aware context propagation via `contextvars.ContextVar`
    * Operation tracking context manager with automatic request ID generation
    * OpenTelemetry trace correlation (trace_id, span_id injected into logs)
    * Log sampling (1% DEBUG in production, configurable)
    * Singleton pattern for global logger access
  - **Performance:** <10ms overhead per log operation
  - **Usage:** `from src.structured_logging import get_logger; logger = get_logger(); logger.info("msg", key=value)`
  - **Testing:** 43 comprehensive tests covering formatters, async context, OTEL integration

- **Prometheus Metrics (`src/metrics.py` - 330 lines, 86% coverage):**
  - **Purpose:** Time-series metrics collection with cardinality control to prevent metric explosion
  - **Metrics Tracked (7 total):**
    * `compression_ratio` (Histogram): Compression effectiveness by fidelity level
    * `processing_latency_seconds` (Histogram): Operation latency by operation and fidelity
    * `documents_processed_total` (Counter): Processed documents by operation, fidelity, status
    * `cache_hit_ratio` (Gauge): Current cache hit rate (0.0-1.0)
    * `active_documents` (Gauge): Currently loaded documents
    * `errors_total` (Counter): Errors by error_type and operation
    * `batch_size` (Histogram): Batch processing sizes
  - **Cardinality Control:** Validated label values (fidelity: 5 values, operations: 5 values, status: 2 values)
  - **Graceful Degradation:** NoOp implementation when prometheus_client unavailable
  - **Export:** Prometheus text format via `generate_metrics_text()`
  - **Usage:** `from src.metrics import get_metrics; metrics = get_metrics(); metrics.record_compression_ratio(7.5, "BALANCED")`
  - **Testing:** 29 comprehensive tests covering all 7 metrics, cardinality control, graceful degradation

- **OpenTelemetry Tracing (`src/observability.py` - 717 lines, 85% coverage):**
  - **Purpose:** Distributed tracing with OTLP export for trace backends (Jaeger, Zipkin, etc.)
  - **Key Features:**
    * Async-safe context propagation via `contextvars.ContextVar`
    * Span creation context manager with attributes
    * Trace sampling (10% production, 100% development)
    * Integration with structured logging (trace correlation via trace_id/span_id)
    * Exception recording with span status tracking
    * OTLP/gRPC export (console fallback when OTLP unavailable)
    * Singleton pattern for global observability manager
  - **Performance:** <50ms overhead per traced operation
  - **Usage:** `from src.observability import get_observability; obs = get_observability(); with obs.trace("compress", doc_id="abc"): ...`
  - **Testing:** 53 comprehensive tests covering span creation, async propagation, exception handling, OTLP export

- **Health Checks & Diagnostics (`src/health.py` - 500 lines, 91% coverage):**
  - **Purpose:** Three-tier health checks (liveness, readiness, diagnostics) for production monitoring
  - **Health Tiers:**
    * `check_liveness()`: Simple alive/dead check (always returns healthy for running server)
    * `check_readiness()`: Component health check (embedding manager, persistence, cache, disk space)
    * `get_diagnostics()`: Detailed diagnostics (performance metrics, resource usage, cache stats)
  - **Component Health Monitoring:**
    * Embedding manager health (model loaded, inference working)
    * Persistence health (storage accessible, read/write working)
    * Cache health (LRU eviction working, hit rate tracking)
    * Disk space health (sufficient space for operations)
  - **Performance Metrics:** p50/p95/p99 latency percentiles (tracked via rolling window)
  - **Result Caching:** 10-second cache to avoid expensive checks
  - **Health Status:** healthy/degraded/unhealthy with detailed component messages
  - **Usage:** `from src.health import get_health; health = get_health(); readiness = health.check_readiness()`
  - **Testing:** 43 comprehensive tests covering liveness/readiness/diagnostics, component health, caching

**File Sync System:**
- `src/file_sync_manager.py` - Tracks file changes via mtime + MD5 with LRU eviction (v0.4.2)
- `src/version_manager.py` - Full version history with diffs and automatic pruning (v0.4.2)
- Storage: `.semantic_modulator_data/versions/{doc_id}.json`
- Coverage: 86% (file_sync), 90% (version_manager)

**Version History Automatic Pruning (v0.4.2 - Week 3 Day 13):**
- **Feature:** LRU-style automatic pruning to prevent unbounded memory growth
- **Configuration:** `DEFAULT_VERSION_RETENTION = 10` (keeps last 10 versions per document)
- **Memory Impact:** 880KB freed when pruning 50 → 10 versions (87% reduction, proven via profiling)
- **How it works:** Automatic pruning on every `add_version()` call when limit exceeded
- **Manual pruning:** `prune_old_versions(doc_id)` for cleanup of existing documents
- **Version tracking:** Uses separate counter (`_version_counters`) to preserve version_id sequence

**File Sync Metadata LRU Eviction (v0.4.2 - Week 3 Day 13):**
- **Feature:** Automatic LRU eviction to limit file metadata entries
- **Configuration:** `MAX_FILE_SYNC_ENTRIES = 1000` (default limit)
- **Memory Impact:** ~170 bytes per file metadata entry (1000 files = ~170KB total)
- **How it works:** Automatic eviction in `register_file()` when limit exceeded
- **Eviction strategy:** Oldest entries (by ingestion_time) evicted first
- **Statistics:** `get_stats()` provides tracking info and approaching_limit warning (>90%)

**ACE Context LRU Eviction (v0.4.2 - Week 3 Day 14):**
- **Feature:** Automatic LRU eviction to limit ACE (Agentic Context Engineering) contexts
- **Configuration:** `MAX_ACE_CONTEXTS = 100` (default limit)
- **Memory Impact:** ~70KB per context (10 bullets with embeddings), 100 contexts = ~7MB total
- **How it works:** `ACEContextManager` wraps OrderedDict for transparent LRU eviction
- **Access tracking:** Reading context via `__getitem__` marks it as recently used (move_to_end)
- **Update tracking:** Updating context via `__setitem__` moves it to end (most recently used)
- **Eviction strategy:** Oldest contexts (by creation/access time) evicted first
- **Statistics:** `get_stats()` provides tracking info and approaching_limit warning (>90%)
- **Note:** ACE deletion already worked correctly; this limit prevents accumulation over long server runs

**Phase 3 Refactoring (Completed):**
- ✅ Modular handler architecture with centralized routing
- ✅ `src/server.py` reduced from 1,911 → 299 lines (84% reduction)
- ✅ Created `src/handlers/` with 7 focused modules (v0.4.3 baseline; 10 modules as of v0.10.0):
  - `mcp_core.py` - Tool schemas and routing
  - `compression_handlers.py` - 9 document compression handlers
  - `afm_handlers.py` - 6 dialogue memory handlers
  - `ace_handlers.py` - 7 ACE Framework handlers (NEW in v0.4.0!)
  - `file_sync_handlers.py` - 4 file sync/version handlers
  - `detection_handlers.py` - 2 blind spot/hallucination handlers
  - `resource_handlers.py` - resource health + environment check handlers
  - (+ help_handlers.py, visualization_handlers.py, experimental_handlers.py added in v0.9.0–v0.10.0)
- ✅ `src/embeddings.py` - Singleton embedding manager (~70% memory reduction)
- ✅ `src/constants.py` - Centralized configuration with WHY documentation
- ✅ All 319 tests passing (v0.4.3 - Week 4 testing & refactoring)
- ✅ Code formatted (black) and linted (ruff) with zero issues

**Compression Reality (PROVEN via demo_proof.py):**
- **Real test:** 485-token quantum paper → 61 tokens (7.9×, 87.4% reduction) ✅ PROVEN
- Small documents (<100 tokens): 2-4×, may expand due to skeleton overhead
- Medium documents (500 tokens): 5-10× (proven range)
- Large documents (5000+ tokens): 15-20× (theoretical maximum)
- **Design intention:** System optimized for medium-to-large documents
- **Key insight:** Compression is size-dependent - larger docs compress better

### Critical Testing Notes
When updating tests or expectations:
1. Skeleton format has fixed overhead (headers, metadata, node IDs)
2. Small documents may show expansion, not compression (expected behavior)
3. Use realistic compression expectations, not theoretical maximums
4. AFM recency weight: Use `message.turn_index` not `manager.turn_counter` for "current" turn

### Experimental Features (Not in Core v0.4.3)
The following features are documented with working examples but not production-ready:

**Multimodal Compression (`src/multimodal_compressor.py`):**
- ⚠️  Status: 0% coverage, requires tests and integration
- Example: `examples/multimodal_example.py`
- Handles mixed image/code content compression
- Dependencies: Pillow, optional CLIP for image embeddings
- TODOs: Tests, MCP integration, quality benchmarks

**TOON Serialization (`src/toon_serializer.py`):**
- ⚠️  Status: 0% coverage, requires tests and benchmarking
- Example: `examples/toon_demo.py`
- Token-Oriented Object Notation for additional 40% token savings
- Pure Python (no external dependencies)
- TODOs: Tests, benchmarks vs JSON, MCP output option

**Training Utilities (`src/training_utils.py`):**
- ⚠️  Status: 0% coverage, requires tests and integration
- Reference: Mentioned in GETTING_STARTED.md
- SCAR training infrastructure (learnable compressor, alignment module)
- Dependencies: PyTorch, tqdm
- TODOs: Tests, MCP integration, training effectiveness benchmarks

All experimental features are KEPT (not deleted) to allow future production integration.

## Quick Find Commands (JIT Index)

### Code Navigation
```bash
# Find all MCP tool definitions
rg -n "Tool\(" src/handlers/mcp_core.py

# Find handler implementations
rg -n "async def handle_" src/handlers/

# Find test files
find tests -name "test_*.py" | head -20

# Find all Python modules in src/
find src -name "*.py" -not -path "*/handlers/*" | wc -l  # ~61 core modules

# Check handler module count
find src/handlers -name "*.py" | wc -l  # 11 handler modules
```

### Project Analysis
```bash
# Check version
grep "version" pyproject.toml

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term

# Check code quality
black --check src/ tests/
ruff check src/ tests/
```

### Dependency Analysis
```bash
# Check Python package dependencies
pip show <package-name>

# List all installed packages
pip list

# Check for security vulnerabilities
pip-audit
```

## Core Files and Utility Functions

### Configuration Files
- `.claude/CLAUDE.md`: This file (root instructions, 47KB)
- `pyproject.toml`: Python package metadata and dependencies (v0.10.0)
- `requirements.txt`: Runtime dependencies (39 packages)
- `config/claude_desktop_config.example.json`: Example MCP configuration for Claude Desktop

### MCP Configuration (User-Specific)
MCP server configuration is NOT stored in the repository. Users configure in their Claude Desktop settings:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- See `docs/guides/CLAUDE_CODE_SETUP.md` for integration instructions

### Actual Project Structure
```
.claude/                    # Claude Code configuration
├── CLAUDE.md               # Root instructions (this file)
├── config.yaml             # BMAD agent routing (10 agents)
├── settings.json           # Tool permissions and hooks
├── settings.local.json     # Local permission overrides
├── .mcp.json               # 6 MCP server registrations
├── agents/                 # 9 alternate role definitions (legacy)
├── subagents/              # 10 routable agent definitions (authoritative)
├── commands/               # 2 slash commands (/review, /fix-issue)
├── hooks/                  # 3 lifecycle hooks (pre/post tool use, prompt submit)
├── instructions/           # 12 guidance documents
├── schemas/                # 10 JSON validation schemas
├── skills/                 # 3 YAML skills + compression skill reference
├── templates/              # 9 artifact templates
├── workflows/              # 2 workflow definitions
├── context/                # Runtime artifacts, gates, session state
├── tools/gates/            # Quality gate validation (gate.mjs)
└── rules-library/          # 174 technology-specific rule sets

.github/workflows/          # CI/CD pipelines (test, lint, build, deploy)
config/                     # Example configurations
deployment/kubernetes/      # Production K8s manifests
docs/                       # 49 documentation files
examples/                   # 10 usage examples
scripts/                    # 8 utility scripts
skills/                     # Python compression skill (12 scripts)
src/                        # 71 source modules + handlers
tests/                      # 78 test files (1,171 tests)
```

## Developer Environment Setup

### Prerequisites
- **Python 3.10–3.13** (required; `<3.14` upper bound in pyproject.toml)
  - **Note:** ChromaDB is an optional extra (`pip install ".[chromadb]"`); base install uses JSON fallback storage
  - If ChromaDB is installed, Python 3.13+ may have numpy compatibility issues (numpy<2.0 required by chromadb)
- pip and virtualenv (for dependency management)
- GitHub CLI (`gh`) installed for GitHub integration (optional)
- Claude Code CLI installed and configured
- ~100MB disk space for embedding models (first-time setup)

### Initial Setup
1. **Quick Start (Recommended):**
   ```bash
   python scripts/quickstart.py
   ```
   This automates: dependency install, model download, tests, and demo.

2. **Manual Setup:**
   ```bash
   pip install -r requirements.txt
   python scripts/check_setup.py
   python -m src.server  # Start MCP server
   ```

3. **Claude Desktop Integration:**
   ```bash
   ./scripts/install_mcp.sh  # Auto-configure Claude Desktop
   ```
   Or manually edit `~/Library/Application Support/Claude/claude_desktop_config.json`

4. Enable hooks in Claude Code: `Preferences → Claude Code → Hooks`

### Environment Variables
- `GITHUB_TOKEN`: For GitHub MCP integration (optional)
- `LINEAR_API_KEY`: For Linear integration (optional)
- `SLACK_BOT_TOKEN`: For Slack integration (optional)

### Repository Etiquette
- **Branch naming**: `feature/<short-description>`, `fix/<issue-number>`
- **Commits**: Use conventional commits (feat:, fix:, docs:, etc.)
- **Pull requests**: Reference issues, include testing notes
- **Merges**: Prefer squash and merge for cleaner history

## Security & Secrets Management

### Secrets Management
- **NEVER** commit tokens, API keys, or credentials to version control
- Use `.env.local` for local secrets (already in .gitignore)
- Use environment variables for CI/CD secrets
- PII must be redacted in logs and artifacts
- **MUST NOT** edit `.env`, `.env.production`, or secret configuration files

### Protected Files
- **BLOCKED**: `.env*` files, `secrets/` directory, production configs
- **REQUIRE APPROVAL**: Database migrations, infrastructure changes, auth configs
- **REVIEW BEFORE**: Git force push, destructive database operations, production deploys

### Safe Operations
- Review generated bash commands before execution
- Confirm before: `git push --force`, `rm -rf`, database drops, `dd` commands
- Use staging environment for risky operations
- Validate file paths before batch operations

## Available Tools & Permissions

### Standard Tools Available
- **Read**: Any file in the repository
- **Write**: Code files, documentation, configuration (except protected files)
- **Bash**: Safe commands (build, test, lint, typecheck, git status/diff/log)
- **MCP Tools**: Repo search, artifact publishing, context bridging (see `.claude/.mcp.json`)

### Tool Permission Rules
- ✅ **Always Allowed**: Read, Search, Git status/diff/log
- ⚠️ **Require Confirmation**: Edit, Bash (git commit/push), MCP tool execution
- ❌ **Always Blocked**: 
  - Edit `.env*` files or secrets
  - Bash: `rm -rf`, `format *`, `dd *`, `mkfs *`, force push
  - Dangerous Git operations without approval

### MCP Server Access
Token Saver 5000 provides 49 MCP tools via stdio transport. Configure in your Claude Desktop settings:

**Tool Categories:**
- **Document Compression** (9 tools): ingest, read, search, modulate, batch operations
- **Dialogue Memory (AFM)** (6 tools): add_message, build_context, export/import
- **Context Engineering (ACE)** (7 tools): generate, reflect, curate, grow, refine
- **File Sync & Versioning** (4 tools): check_sync, refresh, diff, version_history
- **Visualization** (4 tools): export_json, visualize_html, export_graphml, explain
- **Detection** (2 tools): check_blind_spots, detect_hallucination
- **Health & Assessment** (3 tools): check_health, check_environment, should_compress
- **Experimental** (5 tools): TOON encode/decode, SCAR compress/stats, multimodal_ingest
- **Discovery** (3 tools): list_documents, delete_document, tool_help
- **Fidelity** (1 tool): recommend_fidelity

See `docs/guides/MCP_TOOLS_GUIDE.md` for complete tool documentation.

## Code Style Guidelines

### Python
- Follow PEP 8 style guide (enforced by `black` and `ruff`)
- Use type hints for function signatures: `def process(text: str) -> Dict[str, Any]:`
- Prefer explicit imports: `from src.semantic_compressor import SemanticCompressor`
- Use docstrings (Google style) for all public functions and classes
- Keep functions focused and under 50 lines when possible

### File Organization
- Keep related files together (e.g., `semantic_compressor.py` and `code_compressor.py` in `src/`)
- Use descriptive, explicit names: `blind_spot_detector.py` not `bsd.py`
- Follow module structure: `src/` (core), `tests/` (tests), `examples/` (demos), `scripts/` (utilities)

### Testing
- Write tests alongside implementation in `tests/` directory
- Use descriptive test names: `test_document_size_limit_exceeded` not `test_limits`
- Test both happy paths (test_functional.py) and edge cases (test_edge_cases.py - 50 comprehensive edge case tests)
- Maintain 75%+ coverage for core modules (`semantic_compressor.py`, `afm.py`, `server.py`)
- **Current Test Coverage (v0.6.0-beta):**
  - 506 comprehensive tests (492 passing, 14 skipped for optional dependencies)
  - 99% coverage: `code_compressor.py`, `semantic_compressor.py`
  - 96% coverage: `ace_framework.py`
  - 90% coverage: `server.py`, `version_manager.py`, `fidelity_advisor.py`
  - 86% coverage: `file_sync_manager.py`, `error_helpers.py`
  - 84% coverage: `tfidf_embeddings.py`
  - 83% coverage: `afm.py`
  - 81% coverage: `batch_manager.py`, `compression_handlers.py`, `scar_compressor.py`
  - 68% coverage: `graph_visualizer.py`
  - 60% coverage: Overall (includes optional features not installed in all environments)
- **Edge Case Coverage (v0.4.0):**
  - 50 comprehensive edge case tests across all core modules
  - Semantic Compressor: 4 advanced tests (large docs, circular refs, repetitive content)
  - AFM: 4 advanced tests (null handling, budget exhaustion, concurrent access)
  - File Sync: 7 tests (file deletion, permissions, symlinks, UNC paths, checksums)
  - Version Manager: 5 tests (corruption recovery, concurrent writes, binary diffs)

## Examples & Workflows

### Example: Running Agent Workflow
```
1. User: "I need a new full-stack web app for task management"
2. Orchestrator detects greenfield_fullstack workflow
3. Analyst creates project brief
4. PM creates PRD with user stories
5. UX Expert designs interface
6. Architect designs system architecture
7. QA creates test plan
8. Developer implements features
9. QA validates quality gates
```

### Example: Using Slash Commands
- `/review`: Comprehensive code review
- `/fix-issue <number>`: Automatically fix GitHub issue
- Custom commands available in `.claude/commands/`

### Example: Multi-File Changes
1. Generate Plan Mode artifact first
2. Review impacted files and dependencies
3. Implement changes systematically
4. Run tests to verify
5. Create artifact summary

## Claude Code Context Management

### Memory System Best Practices
- **Use `#` key during sessions** to add memories organically to CLAUDE.md
- Review and refactor CLAUDE.md monthly to remove stale instructions
- Keep sections modular to prevent instruction bleeding between contexts
- Use hierarchical CLAUDE.md files: root for universal rules, subdirectories for specific contexts

### Session Management
- **Use `/clear`** between unrelated tasks to reset context window
- **Use `/compact`** for long sessions to optimize token usage
- **Reference specific files** with `@filename` rather than reading entire directories
- **Use artifacts** for multi-file outputs instead of large inline responses

### Custom Commands Strategy
- Start with 3-5 most common workflows
- Use descriptive names (e.g., `/review`, `/fix-issue`, not `/r`, `/fi`)
- Include validation steps in commands
- Use `$ARGUMENTS` for parameterized commands
- See `.claude/commands/` for available slash commands

## Escalation Playbook
1. Flag blockers in the Claude Project feed; attach the current artifact or plan.
2. Page the appropriate subagent (Architect vs. QA vs. PM) via Claude subagent commands.
3. If automation fails, fall back to manual CLI with the same rules and document the resolution artifact.

## References

[1] Anthropic, "Claude 3.5 Sonnet" (Jun 2024).  
[2] Anthropic, "Projects" (Jun 2024).  
[3] Cursor, "Introducing Cursor 2.0 and Composer" (Oct 29, 2025).  
[4] Cursor, "Introducing Plan Mode" (Oct 7, 2025).  
[5] Cursor, "Cloud Agents" (Oct 30, 2025).  
[6] Sid Bharath, "Factory.ai: A Guide To Building A Software Development Droid Army" (Sep 30, 2025).
