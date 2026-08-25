# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Token Saver 5000 is an MCP server that performs semantic compression of text and code for AI context windows. It ingests documents, builds semantic graphs, ranks importance via PageRank, and outputs compressed "skeletons" that can be queried and expanded. Achieves 85-90% token reduction on medium-to-large documents.

**Package name:** `semantic-modulator` (v0.11.0)
**Python:** 3.10-3.13 (chromadb only works on 3.10-3.12)
**Transport:** stdio (default), optional HTTP for Kubernetes

## Monorepo context (gotcontext.ai)

When this engine is vendored inside the gotcontext.ai platform monorepo, the canonical **platform** documentation lives at the monorepo root `CLAUDE.md`. That file explains:

- How the FastAPI app mounts **MCP Streamable HTTP** at `https://api.gotcontext.ai/mcp` using this package’s `src.handlers` toolchain.
- How **plan gating** and **`GOTCONTEXT_SAAS_MODE`** strip dangerous tools in SaaS.
- How the **Docker image** sets `PYTHONPATH` to this folder instead of pip-installing the full torch/sentence-transformers stack.

Use **this** `CLAUDE.md` for day-to-day **library and MCP server** work, folder-guide indexes, and Python test commands. Use the **root** `CLAUDE.md` when changing deploy wiring, API gateway auth, or the Next.js dashboard.

**Hosted API embedding policy:** On gotcontext.ai, the API does not necessarily use the same default embedding tier as a local stdio server. Plan → tier mapping lives in **`api/app/services/plan_gating.py`** (`get_embedding_tier_for_plan`: e.g. `free` → TF-IDF, `pro` → ONNX, `enterprise` → standard/SBERT). The vendor layer respects **`EMBEDDING_TIER`** env and integration with `api/app/vendor/embeddings.py`.

## Codebase map (folder guides)

This file is the **master index** for humans and AI tools. Each first-class code directory also has a lowercase **`claude.md`** that lists **every file in that folder**, the **first line of the module docstring** (when present), and **top-level** `class` / `def` / `async def` names (nested helpers are omitted).

**How to use:** start here for intent and commands; open the folder `claude.md` when you need a file-by-file symbol index without loading whole modules.

**Regenerate** all folder guides after large refactors:

```bash
python scripts/generate_claude_folder_guides.py
```

**Verify** (same check as CI / pre-commit — compares only `**/claude.md` under `src/`, `tests/`, `scripts/`):

```bash
python scripts/check_claude_folder_guides_sync.py
```

**Determinism:** guides are UTF-8 with **LF-only** newlines (`pathlib.write_text(..., newline="\n")`), directory listings and `os.walk` branches are **sorted**, and `.gitattributes` sets `**/claude.md text eol=lf` so Git does not reintroduce CRLF on Windows.

| Directory | Folder guide | What lives here (summary) |
|-----------|----------------|---------------------------|
| `src/` | [src/claude.md](src/claude.md) | Main library: compression, embeddings, persistence, MCP-facing services, HTTP, CLIs, token economy, prompts/cache, multimodal, experiments—modules at repo “core” granularity. |
| `src/handlers/` | [src/handlers/claude.md](src/handlers/claude.md) | Async MCP **tool handlers** (`handle_*`), grouped by domain; delegates to `src/` services. |
| `src/semantic_modulator/` | [src/semantic_modulator/claude.md](src/semantic_modulator/claude.md) | Nested package root (`__init__` only). |
| `src/semantic_modulator/app/` | [src/semantic_modulator/app/claude.md](src/semantic_modulator/app/claude.md) | **Server wiring**: factory, router binding, lifecycle, tool profiles, ACE context manager, service adapter. |
| `src/semantic_modulator/api/` | [src/semantic_modulator/api/claude.md](src/semantic_modulator/api/claude.md) | API subpackage re-exports. |
| `src/semantic_modulator/api/mcp/` | [src/semantic_modulator/api/mcp/claude.md](src/semantic_modulator/api/mcp/claude.md) | MCP registry/router helpers for the `semantic_modulator.api` layer. |
| `src/cli_benchmark/` | [src/cli_benchmark/claude.md](src/cli_benchmark/claude.md) | Reusable **CLI benchmark** harness (corpus, pricing, providers, results). |
| `src/connectors/` | [src/connectors/claude.md](src/connectors/claude.md) | Optional **data connectors** (GitHub, S3, Slack export, web) built on `connectors/base.py`. |
| `src/proxy/` | [src/proxy/claude.md](src/proxy/claude.md) | **MCP proxy** server, upstream client, schema compression, response interception. |
| `tests/` | [tests/claude.md](tests/claude.md) | Pytest modules colocated at the test tree root (`test_*.py`, `conftest.py`). |
| `tests/fixtures/` | [tests/fixtures/claude.md](tests/fixtures/claude.md) | Shared static/json fixtures for integration-style tests. |
| `tests/fixtures/parity/` | [tests/fixtures/parity/claude.md](tests/fixtures/parity/claude.md) | Parity / golden-path fixture payloads. |
| `scripts/` | [scripts/claude.md](scripts/claude.md) | Maintainer **scripts**: setup check, benchmarks, proxy helper, quickstart, migration. |
| `scripts/benchmarks/` | [scripts/benchmarks/claude.md](scripts/benchmarks/claude.md) | Benchmark guard and benchmark runner entrypoints. |
| `scripts/skills/` | [scripts/skills/claude.md](scripts/skills/claude.md) | Skill-oriented CLIs (`compress_context`, workflow runner, evidence validation) also mirrored under `.claude/skills/`. |

**Related (not auto-generated):** `.claude/` holds Claude Code hooks, commands, and packaged skills; use that tree for editor integration, not runtime MCP packaging.

## Commands

```bash
# Install
pip install -r requirements.txt && pip install -e .

# Run MCP server
python -m src.server            # or: token-saver-mcp

# Tests
pytest tests/ -v                          # all tests (3400+)
pytest tests/ -q --no-cov --ignore=tests/test_performance.py --ignore=tests/test_mcp_routing.py  # fast suite
pytest tests/test_functional.py -v        # single file
pytest tests/ -k "test_name"              # single test by name
pytest tests/ --cov=src --cov-report=term # with coverage report

# Lint & format
black src/ tests/                # format (line-length=100)
ruff check src/ tests/           # lint
black --check src/ tests/        # format check only

# Setup verification
python scripts/check_setup.py    # full verification (downloads models)
token-saver-install-mcp --doctor --human  # quick status
```

## Architecture

### Request Flow

```
MCP Client → stdio → Server (src/server.py)
  → SemanticModulatorServer.__init__() builds all components via ServerFactoryService
  → bind_mcp_handlers() registers MCP protocol handlers
  → route_tool_call() in src/handlers/mcp_core/dispatch.py dispatches to handler modules
  → Handlers receive HandlerContext (TypedDict in src/types.py) with all server components
```

### Key Layers

**Server bootstrap** (`src/server.py` → `src/semantic_modulator/app/`):
- `ServerFactoryService.build_default()` wires all components (compressor, detectors, persistence, etc.)
- `router_binding.py` connects MCP protocol to the tool routing layer
- `server_service_adapter.py` bridges server methods to service layer
- Server uses async context manager (`__aenter__`/`__aexit__`) for lifespan management

**Tool routing** (`src/handlers/mcp_core/` — split from a single 3670-line `mcp_core.py` flat module, N2 slice 2, 2026-08-22; see `docs/design/2026-08-22-mcp-core-split.md`):
- `__init__.py` — re-exports the package's public surface (`SCOPE_PROPERTIES`, `SUPPORTED_TOOL_PROFILES`, `CORE_STABLE_TOOL_NAMES`, `setup_mcp_tools`, `route_tool_call`), including underscore-prefixed helpers nothing outside the file imports today (kept as insurance — one such helper turned out to be used by `registry.py`)
- `_constants.py` — module-level constants (`SCOPE_PROPERTIES`, `SUPPORTED_TOOL_PROFILES`, `CORE_STABLE_TOOL_NAMES`)
- `_profile.py` — tool-profile filtering helpers (`_normalize_tool_profile`, `_enabled_tool_names`, `_tools_for_profile`)
- `setup.py` — `setup_mcp_tools()`, returns all MCP tool schemas by concatenating the `schemas_*.py` modules below
- `dispatch.py` — `route_tool_call()`, dispatches by tool name to handler modules (the router dict, 1:1 with the schema list)
- `schemas_*.py` (8 files, one per tool-category group) — the actual `Tool(...)` schema literals, split out of the old file's single 3325-line list: `schemas_compression.py` (Document Compression), `schemas_afm_temporal.py` (AFM Dialogue + Temporal), `schemas_memory.py` (Memory handlers), `schemas_filesync_bundle.py` (File Sync + Handoff Bundles), `schemas_token_optimization.py` (Token Optimization), `schemas_multimodal_viz.py` (Multimodal + Visualization), `schemas_model_experiment.py` (Model + Experiment handlers), `schemas_experimental.py` (experimental / not production-ready), `schemas_prompts_ace.py` (Prompts + ACE), `schemas_misc.py` (Connector + Resource + Detection + Docs + Help)
- Tool profiles: `"full"` (all tools) or `"core_stable"` (7 essential tools), controlled by `MCP_TOOL_PROFILE` env var

**Handler modules** (`src/handlers/`):
- Each module handles a category: compression, AFM (dialogue memory), ACE (context engineering), file sync, visualization, detection, experimental, etc.
- All handlers are async, receive `HandlerContext` dict, return JSON-serializable dicts
- 20 handler files, ~127 MCP tools total
- **v1.8.0 addition:** `src/handlers/compress_manifest.py` — `handle_compress_manifest(params)` compresses an MCP `tools/list` response (shortens `description` fields, preserves `inputSchema` byte-for-byte). Consumed by gotcontext.ai's `gc_compress_manifest` platform tool in `api/app/mcp_gateway.py`. Plan-gated to Pro+ in `api/app/services/plan_gating.py::_PRO_TOOLS`.

**Core compression** (`src/semantic_compressor.py`, `src/code_compressor.py`):
- `SemanticCompressor`: text chunking → embedding → graph construction → PageRank → skeleton
- `CodeSemanticCompressor`: AST-aware code compression (preserves structure)
- `CodeCompressionAdapter`: routes files to text vs code compressor based on extension

**CLI Output Optimizer** (`src/cli_output_optimizer.py`):
- 11 command-specific filtering strategies (git_diff, test_output, lint_output, docker_output, etc.)
- Auto-detection of command type from output content
- User-defined `.gotcontext.toml` rules applied as post-filter stage via `FilterRuleEngine`
- Optional RTK binary fallback for additional coverage
- **Classification-gate under-compression fallback (#137c, 2026-07-21).** The classifier routes any `>10-line / <70%-unique` blob to the `log_output` strategy, whose `_log_dedup` collapses **only CONSECUTIVE-EXACT** duplicate lines — so a repetitive-but-not-adjacent-exact log (interleaved tracebacks, timestamp-cycling lines) fell through at ~0% savings. `log_output` now compares its `_log_dedup` output against `_generic_conservative` (frame-elision + timestamp/UUID-masked-run collapse + exact-dup + progress-drop + blank-collapse) and **keeps whichever is smaller when `_log_dedup` savings `< 5%`** (`_LOG_FALLBACK_MAX_SAVINGS_PCT=5.0`, module-level `_char_savings_pct`), gated on an explicit `fallback_strategy="generic"` opt-in (no silent wire-behavior move). **Load-bearing (Fable-audit correction):** `_generic_conservative` is NOT a strict superset of `_log_dedup` — it guards out short (`<8ch`) / oversized (`>2000ch`) / structured lines, so "fall back and it always wins" is WRONG; the compare-and-keep-smaller is what makes it safe. Envelope-preservation invariant kept (masked-run collapse keeps first + last line verbatim + a "N similar lines" count marker); reset-boundary detection deliberately NOT added (the fragile heuristic the #137-round-4 convergence guardrail forbids). Live-dogfooded: traceback **17.3%** (the identifier guard RE-INJECTS elided frames = a deliberate safety trade, so the endpoint regression threshold is `>12%` not `>20%`), timestamp-cycle **73.2%**. Tests: `tests/test_generic_conservative_strategy.py::TestLogOutputGenericFallback137c` (7) + endpoint `test_compress_tool_output.py::test_log_classified_traceback_blob_now_compresses`. See the `compression-quality-eval` skill.

**Token Economy** (`src/savings_tracker.py`, `src/savings_dashboard.py`, `src/budget_monitor.py`, `src/team_export.py`):
- `SavingsTracker`: per-session token/cost tracking with persistence
- `SavingsDashboard`: cross-session aggregation CLI (`token-saver-stats`)
- `TokenBudgetMonitor`: configurable per-session/daily/monthly budget limits
- `TeamExporter`: JSON/CSV/Prometheus team data export (user_id labels escaped)

**Filter Rules** (`src/filter_rules.py`):
- User-defined output filtering via `.gotcontext.toml` TOML DSL
- 8-stage pipeline with inline tests, safe regex compilation

**Tee/Recovery** (`src/tee_recovery.py`):
- Original content preservation before compression
- LRU-evicting store with 3 modes (failures/always/never)

**Missed Savings Discovery** (`src/savings_discover.py`):
- Directory scanning for compression opportunities
- Per-extension compression ratio estimates

**Embeddings** (`src/embeddings.py`, `src/embeddings_onnx.py`, `src/embeddings_tfidf.py`):
- 3-tier fallback: SBERT (best quality) → ONNX (60-70% less memory) → TF-IDF (98% less memory)
- Singleton `EmbeddingManager` with LRU cache (`src/embedding_cache.py`)

**Persistence** (`src/persistence.py`):
- ChromaDB (optional, `pip install ".[chromadb]"`) with JSON fallback
- Documents stored with embeddings for semantic search

### Multi-Tenant Scoping

All tools accept optional `workspace_id`, `user_id`, `agent_id`, `session_id` for tenant isolation. These are injected into tool schemas via `SCOPE_PROPERTIES` in `src/handlers/mcp_core/_constants.py`.

### Security

- `PathValidator` (`src/path_validator.py`) prevents CWE-22 path traversal. Whitelist: cwd + home dir.
- All file I/O in handlers goes through PathValidator before touching disk.

### Configuration

- `src/constants.py`: all tuning knobs with env var overrides and documented rationale
- `PRELOAD_CODE_MODEL=true`: preload CodeBERT (~400MB) instead of lazy-loading
- `HTTP_ENABLED=true`: start HTTP health/metrics endpoints (for Kubernetes)
- `MCP_TOOL_PROFILE=core_stable`: expose only 7 essential tools

## Testing Patterns

- `conftest.py` has shared fixtures; most tests mock components via `HandlerContext` dict
- Handler tests are async (`pytest-asyncio`)
- Coverage threshold enforced at 70% (`--cov-fail-under=70` in pyproject.toml)
- Optional dependency tests skip gracefully (ONNX, pyvis, chromadb)
- Markers: `@pytest.mark.slow`, `@pytest.mark.integration`

## Compression Behavior

Small documents (<100 tokens) may *expand* due to skeleton overhead. The system is optimized for medium-to-large documents (500+ tokens → 5-20x compression). This is by design.

## Skeleton output format — `Skeleton-Version: 2` (2026-07-02)

The skeleton render format is a **wire contract** consumed by the platform MCP (`gc_read_skeleton`), the dashboard, and any agent that reads a `[HIDDEN]` marker. As of the world-class compression sprint:

- The header carries a **`Skeleton-Version: 2`** marker + a single line `Hidden regions expand via modulate_region(node_id).` — the per-node `[HIDDEN] Detail hidden (use modulate_region to expand) - {summary}` boilerplate was **hoisted once to the header** to raise the compression ratio ceiling. Per-node hidden lines are now `[{node_id}] [HIDDEN] - {summary}` (or `[{node_id}] [HIDDEN]` when there is no summary). Locked by `tests/test_worldclass_batch1.py::test_hidden_boilerplate_hoisted_to_header_once` + the `[HIDDEN]` substring asserts in `test_functional.py` / `test_semantic_compressor_unit.py` / `test_read_skeleton_auto_fidelity.py`.
- **Perceived-quality fixes** (`_extract_key_entities` + `_generate_summary` in `src/semantic_compressor.py`): entity extraction now filters stopwords + strips trailing punctuation (killed the dogfood-visible `Key entities: This`), and summaries strip markdown noise (headings/inline-links/backticks) and pick the first substantive sentence. Locked model-free in `tests/test_worldclass_batch2.py`.
- **When you change the render format**, treat it as a contract change: `tg callers` the render path, blast-radius the platform consumers, and dogfood the LIVE MCP after the pin bump (an engine pin bump does NOT change the api version string, so the version is not a deploy signal — a live `gc_ingest`→`gc_read_skeleton` round-trip is).

## Model-free engine testing + HF-cache repair (2026-07-02)

- **Model-free testing:** pure text functions (`_extract_key_entities`, `_generate_summary`, entity/summary/render helpers) can be unit-tested WITHOUT a model load via `object.__new__(SemanticCompressor)` — they use only their arguments, no instance/model state. This is the pattern to reach for when the local HF cache is broken or on Python 3.14 where the event-loop teardown flakes real-model tests. See `tests/test_worldclass_batch2.py` + `tests/test_audit_compression_correctness.py::_make_compressor_with_chunks`.
- **HF-cache corruption repair:** if `SentenceTransformer(...)` / `SemanticCompressor().ingest_file(...)` fails at model load (`JSONDecodeError` on an empty `config.json`, or a downloader exit-5), the local `HF_HOME` cache JSON configs are likely 0-byte. This is NOT "offline" — the SentenceTransformer downloader chokes on the empty leftovers. Fix: `python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5')"` (public embedders need no token) repopulates cleanly; then the model loads. If a model still 404s, delete `HF_HOME/hub/models--<org>--<name>` first, then re-`snapshot_download`. CI is authoritative for real-ingest tests (clean cache + bge caching).
