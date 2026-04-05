# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Token Saver 5000 is an MCP server that performs semantic compression of text and code for AI context windows. It ingests documents, builds semantic graphs, ranks importance via PageRank, and outputs compressed "skeletons" that can be queried and expanded. Achieves 85-90% token reduction on medium-to-large documents.

**Package name:** `semantic-modulator` (v0.11.0)
**Python:** 3.10-3.13 (chromadb only works on 3.10-3.12)
**Transport:** stdio (default), optional HTTP for Kubernetes

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
  → route_tool_call() in mcp_core.py dispatches to handler modules
  → Handlers receive HandlerContext (TypedDict in src/types.py) with all server components
```

### Key Layers

**Server bootstrap** (`src/server.py` → `src/semantic_modulator/app/`):
- `ServerFactoryService.build_default()` wires all components (compressor, detectors, persistence, etc.)
- `router_binding.py` connects MCP protocol to the tool routing layer
- `server_service_adapter.py` bridges server methods to service layer
- Server uses async context manager (`__aenter__`/`__aexit__`) for lifespan management

**Tool routing** (`src/handlers/mcp_core.py`):
- `setup_mcp_tools()` returns all MCP tool schemas
- `route_tool_call()` dispatches by tool name to handler modules
- Tool profiles: `"full"` (all tools) or `"core_stable"` (7 essential tools), controlled by `MCP_TOOL_PROFILE` env var

**Handler modules** (`src/handlers/`):
- Each module handles a category: compression, AFM (dialogue memory), ACE (context engineering), file sync, visualization, detection, experimental, etc.
- All handlers are async, receive `HandlerContext` dict, return JSON-serializable dicts
- 19 handler files, ~121 MCP tools total

**Core compression** (`src/semantic_compressor.py`, `src/code_compressor.py`):
- `SemanticCompressor`: text chunking → embedding → graph construction → PageRank → skeleton
- `CodeSemanticCompressor`: AST-aware code compression (preserves structure)
- `CodeCompressionAdapter`: routes files to text vs code compressor based on extension

**CLI Output Optimizer** (`src/cli_output_optimizer.py`):
- 11 command-specific filtering strategies (git_diff, test_output, lint_output, docker_output, etc.)
- Auto-detection of command type from output content
- Optional RTK binary fallback for additional coverage

**Token Economy** (`src/savings_tracker.py`, `src/savings_dashboard.py`, `src/budget_monitor.py`, `src/team_export.py`):
- `SavingsTracker`: per-session token/cost tracking with persistence
- `SavingsDashboard`: cross-session aggregation CLI (`token-saver-stats`)
- `TokenBudgetMonitor`: configurable per-session/daily/monthly budget limits
- `TeamExporter`: JSON/CSV/Prometheus team data export

**Filter Rules** (`src/filter_rules.py`):
- User-defined output filtering via `.gotcontext.toml` TOML DSL
- 8-stage pipeline with inline tests

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

All tools accept optional `workspace_id`, `user_id`, `agent_id`, `session_id` for tenant isolation. These are injected into tool schemas via `SCOPE_PROPERTIES` in `mcp_core.py`.

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
