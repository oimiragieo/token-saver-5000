# Copilot Instructions for Token Saver 5000

## What This Is

Token Saver 5000 is an MCP server (`semantic-modulator` package) that performs semantic compression of text and code for AI context windows. It builds semantic graphs from documents, ranks importance via PageRank, and outputs compressed "skeletons" achieving 85-90% token reduction on medium-to-large documents.

- **Package name:** `semantic-modulator` (v0.11.0)
- **Python:** 3.10–3.13 (chromadb only works on 3.10–3.12)
- **Transport:** stdio (default), optional HTTP for Kubernetes
- **All processing is local** — no external API calls except one-time model download from Hugging Face

## Build, Test, and Lint

```bash
# Install
pip install -r requirements.txt && pip install -e .

# Run MCP server
python -m src.server            # or: token-saver-mcp

# Full test suite (skips performance + mcp routing)
python -m pytest tests/ -q --no-cov --ignore=tests/test_performance.py --ignore=tests/test_mcp_routing.py

# Single test file
pytest tests/test_functional.py -v

# Single test by name
pytest tests/ -k "test_name"

# With coverage (70% threshold enforced)
pytest tests/ --cov=src --cov-report=term

# Lint and format
black src/ tests/                # format (line-length=100)
black --check src/ tests/        # format check only
ruff check src/ tests/           # lint

# Setup verification
python scripts/check_setup.py            # full verification (downloads models)
token-saver-install-mcp --doctor --human # quick status check
```

## Architecture

### Compression Pipeline

Text chunking (~500 tokens/chunk) → `all-MiniLM-L6-v2` embeddings (384-dim) → NetworkX graph (edges where cosine similarity > 0.75) → PageRank importance scoring → skeleton generation → adaptive fidelity modulation (ABSTRACT / STRUCTURE / RAW).

Small documents (<100 tokens) may *expand* due to skeleton overhead. The system is optimized for 500+ token documents (5-20× compression).

### Request Flow

```
MCP Client → stdio → Server (src/server.py)
  → ServerFactoryService.build_default() wires all components
  → bind_mcp_handlers() registers MCP protocol handlers
  → route_tool_call() in src/handlers/mcp_core.py dispatches to handler modules
  → Handlers receive HandlerContext (TypedDict in src/types.py)
```

### Key Layers

- **Server bootstrap** (`src/server.py` → `src/semantic_modulator/app/`): `ServerFactoryService` wires compressor, detectors, persistence, etc. Server uses async context manager for lifespan.
- **Tool routing** (`src/handlers/mcp_core.py`): `setup_mcp_tools()` returns tool schemas; `route_tool_call()` dispatches by name to handler modules. Tool profiles: `"full"` (all ~121 tools) or `"core_stable"` (7 essential), via `MCP_TOOL_PROFILE` env var.
- **Handler modules** (`src/handlers/`): 19 files, each handling a category (compression, AFM, ACE, file sync, visualization, etc.). All async, receive `HandlerContext` dict, return JSON-serializable dicts.
- **Core compression** (`src/semantic_compressor.py`, `src/code_compressor.py`): `CodeCompressionAdapter` routes files to text vs code compressor by extension. Code compressor uses AST-aware chunking by functions/classes.
- **Embeddings** (`src/embeddings.py`, `src/embeddings_onnx.py`, `src/embeddings_tfidf.py`): 3-tier fallback: SBERT (best quality) → ONNX (60-70% less memory) → TF-IDF (98% less memory). Singleton `EmbeddingManager` with LRU cache.
- **Persistence** (`src/persistence.py`): ChromaDB (optional) with JSON fallback. Data stored in `.semantic_modulator_data/`.
- **Dialogue memory** (`src/afm.py`): Adaptive Focus Memory for multi-turn conversation compression (~66% token savings).

### Multi-Tenant Scoping

All tools accept optional `workspace_id`, `user_id`, `agent_id`, `session_id` for tenant isolation. These are injected via `SCOPE_PROPERTIES` in `mcp_core.py` and managed through helpers in `src/identity_scope.py`.

### Security

- `PathValidator` (`src/path_validator.py`) prevents CWE-22 path traversal. Whitelist: cwd + home dir. All file I/O in handlers must go through PathValidator before touching disk.
- No encryption at rest — rely on disk-level encryption if needed.
- No telemetry or external data transmission.

### Resource Limits

Configured in `src/resource_manager.py`: 100MB max per document, 1GB total storage, 1000 max documents. LRU eviction for file sync metadata (1000 entries), ACE contexts (100), and version history (10 per document). Monitor via `check_resource_health` tool.

### Configuration

`src/constants.py` centralizes all tuning knobs with env var overrides:
- `PRELOAD_CODE_MODEL=true` — preload CodeBERT (~400MB)
- `HTTP_ENABLED=true` — start HTTP health/metrics endpoints
- `MCP_TOOL_PROFILE=core_stable` — expose only 7 essential tools

## Key Conventions

### Handler Pattern

Every handler is an async function that receives a `HandlerContext` TypedDict and an `args` dict, and returns a JSON-serializable dict:

```python
async def handle_my_tool(context: HandlerContext, args: Dict[str, Any]) -> Dict[str, Any]:
    compressor = context["compressor"]
    # ... handler logic ...
    return {"status": "success", "result": ...}
```

Register new tools in `src/handlers/mcp_core.py` via `setup_mcp_tools()` and `route_tool_call()`.

### Error Handling

- `ReliabilityError` hierarchy in `src/error_types.py` (timeout, circuit breaker, retry exhaustion, rate limiting)
- `SmartError` in `src/error_helpers.py` for user-facing errors with suggestions
- Missing optional dependencies must return helpful JSON errors, not crash
- Rate limiters are shared globals (`src/rate_limiter.py`) — tests must reset them (see `conftest.py`)

### Testing Patterns

- `conftest.py` has shared fixtures and an autouse `_reset_shared_state` fixture that resets rate limiters and `EmbeddingManager` singleton between tests
- Handler tests mock components via `HandlerContext` dict using `unittest.mock`
- All handler tests are async (`pytest-asyncio`)
- Optional dependency tests (ONNX, pyvis, chromadb) skip gracefully
- Markers: `@pytest.mark.slow`, `@pytest.mark.integration`
- Coverage threshold: 70% (`--cov-fail-under=70`)
- Test naming: descriptive (`test_compression_ratio_exceeds_80_percent`, not `test_compression`)

### Experimental Modules

SCAR (`src/scar_compressor.py`), TOON (`src/toon_serializer.py`), and multimodal (`src/multimodal_compressor.py`) are research-only. SCAR uses untrained random weights. TOON is lossy. All experimental handler responses must include `"experimental": true`.

### Commit Messages

Use conventional commits: `feat(compressor): add PDF support`, `fix(afm): memory leak in session cleanup`. Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

### Logging

Use `structlog` via `src/structured_logging.py`:

```python
from src.structured_logging import get_logger
logger = get_logger("semantic-modulator")
logger.info("Document ingested", doc_id="abc123", token_count=500)
```

### Formatting

- Black with line-length=100, target Python 3.10
- Ruff for linting, same line-length
- Type hints required for all function signatures

## Gotchas

- **ChromaDB warnings are safe to ignore** — JSON fallback works with zero functionality loss.
- **Don't commit `.semantic_modulator_data/`** — it's ephemeral local state.
- **PageRank results are cached** by graph structure hash; subsequent lookups are O(1).
- **`should_compress` tool** recommends whether to compress based on token count: <100 skip, 100-500 direct read, ≥500 compress.
- **Embedding model download** is ~100MB one-time to `~/.cache/huggingface/`. If firewalled, this will fail silently and fall back to TF-IDF.
