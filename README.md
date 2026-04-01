# Token Saver 5000

Token Saver 5000 is a local semantic compression system for AI context.

In plain terms: it takes large text/code context, keeps the important parts, and gives you a smaller context that is cheaper to send to models.

## What This Project Actually Does

You give it a long document or codebase context.  
It builds a semantic graph, ranks importance, and outputs a compressed "skeleton" you can query and expand.

Core outcomes:

1. Lower token usage.
2. Faster context handling.
3. Better control over what information is kept vs omitted.

This is not tied to GitHub workflows. It works for any large context source (files, notes, transcripts, docs, code, or generated text).

## Who This Is For

Use this if you:

1. Work with large prompts/documents.
2. Need to cut token cost.
3. Want retrieval-oriented compression (not just naive summarization).

Common use cases:

1. RAG context compression before answer generation.
2. Long internal docs and wiki pages.
3. Customer support transcripts and call notes.
4. Legal/policy/contract text review prep.
5. Large code and architecture context for agents.
6. Multi-turn assistant memory compression.

Do not use this if you only have short prompts and token cost is irrelevant.

## Two Ways To Use It

There are two product surfaces in this repo:

1. MCP server (`src.server`) for Claude/Desktop and agent workflows.
2. Self-contained skill scripts (`skills/token-saver-context-compression`) that run locally without MCP.

## Multi-Tenant SaaS Deployment

Token Saver 5000 can also be used as a multi-tenant context service, not just a local MCP helper.

The core scope fields are:

1. `workspace_id`: isolates one customer or team workspace.
2. `user_id`: isolates a person within that workspace.
3. `agent_id`: isolates one automated agent or role.
4. `session_id`: isolates one short-lived interaction thread.

Use those fields consistently across memory, prompts, connector feeds, temporal exports, and handoff bundles when you expose the system behind a shared API gateway or multi-tenant worker.

If you are deploying for multiple customers, read `docs/deployment/SAAS_MULTI_TENANT.md`.

## Local vs Docker

You do not need Docker. Docker is optional.

Choose your runtime:

1. Local Python:
   - Best for development and quick usage.
   - Direct access to scripts and source.
   - Command: `token-saver-mcp`
2. Docker:
   - Best for reproducible deployment/team environments.
   - Avoids local dependency drift.
   - Command: `docker-compose up -d`

## First 10 Minutes (Recommended Path)

1. Get the code:

```bash
git clone https://github.com/oimiragieo/token-saver-5000.git
cd token-saver-5000
```

2. Install it like a tool:

**Option A: `uv` (recommended)**

```bash
uv tool install -e .
```

**Option B: `pipx`**

```bash
pipx install .
```

**Option C: developer/editable install**

```bash
pip install -r requirements.txt
pip install -e .
```

3. Run guided setup:

```bash
token-saver-setup --auto
```

That command picks the most likely target for your environment:

1. `desktop` for Claude Desktop-centric local use.
2. `portable-project` when you run it inside a repo/workspace that looks project-scoped.

If you want the low-level status report only:

```bash
token-saver-install-mcp --doctor --human
```

For a deeper, network-using verification pass that downloads the embedding model and runs a smoke test:

```bash
python scripts/check_setup.py
```

4. Run a local example:

```bash
python examples/example_usage.py
```

5. Try the self-contained skill scripts:

```bash
python skills/token-saver-context-compression/scripts/profile_tokens.py --file tests/fixtures/skill_context_sample.txt --output-format auto
python skills/token-saver-context-compression/scripts/compress_context.py --file tests/fixtures/skill_context_sample.txt --mode query_guided --query "what are the retry rules?" --output-format auto
python skills/token-saver-context-compression/scripts/validate_evidence.py --file tests/fixtures/skill_context_sample.txt --query "what are the retry rules?" --min-similarity 0.4
```

## How The Compression Flow Works

At a high level:

1. Ingest text (`ingest_context`).
2. Chunk + embed text.
3. Build semantic graph (nodes=chunks, edges=semantic similarity).
4. Rank nodes by importance.
5. Return compressed skeleton (`read_skeleton`).
6. Search/extract relevant regions (`search_semantic`, `modulate_region`).

If query-aware mode is used, scoring is biased toward the query.  
If evidence-aware mode is used, it checks whether selected context likely contains enough answer-supporting evidence.

`read_skeleton` now also returns a `pipeline` object so you can inspect which passes ran:

1. `baseline`
2. `query_guided`
3. `evidence_aware`

That makes it easier to debug why a document was compressed a certain way and to verify when evidence-aware retrieval expanded or changed the final anchor set.

## Core MCP Tools (The Ones Most Users Need)

If you are new, start with these 7:

1. `ingest_context`: add a document.
2. `read_skeleton`: view compressed structure.
3. `search_semantic`: find relevant nodes by query.
4. `modulate_region`: expand selected nodes at chosen fidelity.
5. `get_stats`: view compression stats.
6. `list_documents`: list ingested docs.
7. `delete_document`: remove a doc.

You can force this minimal surface with:

```bash
MCP_TOOL_PROFILE=core_stable python -m src.server
```

Or, after installing the tool:

```bash
MCP_TOOL_PROFILE=core_stable token-saver-mcp
```

## Client-Aware Token Optimization Tools

If you use Token Saver with a specific LLM client (Claude Code, Gemini CLI, etc.),
these tools auto-tune compression for your model's context window and behavior:

1. `configure_for_client`: set model ID or explicit context window size. Auto-tunes
   skeleton ratio based on window size and how aggressively the client compresses.
   Supports Claude, Gemini, GPT, and explicit overrides.
2. `estimate_tokens`: multi-method token estimation (tiktoken, fast len/4, Gemini-compatible,
   JSON-density, raw bytes). Use to budget context before ingestion.
3. `set_compression_profile`: named presets (minimal/summary/balanced/detailed/full) that
   bundle skeleton_ratio, fidelity, and chunk_size into one setting.
4. `get_compression_profile`: view the active profile and available profiles.

Example: configure for Gemini CLI (1M context, aggressive compression at 50%):

```bash
# Via MCP tool call
configure_for_client(model_id="gemini-2.5-pro")
# -> skeleton_ratio ~0.31 (vs ~0.50 for Claude with same window)
```

For details, see `docs/claude-code-token-optimization-enhancements.md`,
`docs/gemini-cli-token-optimization-enhancements.md`, and
`docs/codex-cli-token-optimization-enhancements.md`.

## Proven Benchmark Results

Real API measurements on a 2,206-line API reference document (16,461 tokens).
With token refinement enabled, Token Saver achieves **13x document compression** (up from 5.9x):

| Provider | Input Tokens (baseline) | Input Tokens (compressed) | Savings | Cost Savings |
|----------|------------------------|--------------------------|---------|-------------|
| Claude Code (Opus 4.6) | 61,399 | 45,206 | 26.4% | 13.5% |
| Codex (gpt-5.1-codex) | 37,504 | 23,179 | 38.2% | 36.3% |
| Gemini CLI (2.5 Flash) | 69,163 | 30,663 | **55.7%** | **54.4%** |

Token Saver compresses documents **13x** (semantic compression + token-level refinement).
Real-world API savings depend on system prompt overhead: smaller system prompts (Gemini,
Codex) see proportionally larger total savings.

Run benchmarks yourself:

```bash
# Dry run (no API calls, validates setup)
python scripts/benchmark_token_savings.py --dry-run --verbose

# Full benchmark across all providers
python scripts/benchmark_token_savings.py --mode skill --verbose --output results.json

# Single provider, single corpus size
python scripts/benchmark_token_savings.py --providers claude --sizes large --verbose
```

## Cache Optimization Features

Token Saver automatically optimizes for each provider's caching behavior:

1. **Cache-stable response ordering**: tool responses are key-ordered so stable metadata
   (status, file_id) sits at the prefix for Claude/Gemini cache hits, and is mirrored at
   the tail for Codex's middle-truncation pattern.
2. **Token-level refinement**: LLMLingua-inspired post-processing removes articles, fillers,
   and hedges from compressed skeletons (20-40% additional reduction). Preserves numbers,
   code identifiers, URLs, and sentence boundaries.
3. **TurboQuant-inspired embedding quantization**: 384-dim float32 embeddings compressed to
   96-dim int8 (13x memory reduction) using random orthogonal rotation + int8 quantization +
   1-bit residual error correction. >0.99 fidelity in the compressed subspace.

## Prompt Cache Observability Tools

If you are optimizing for prompt caching, the most relevant MCP tools are:

1. `audit_prompt_cacheability`: checks section ordering and volatility before provider calls.
2. `render_prompt_template`: produces a canonical cache-friendly prompt plus a `prompt_id`.
3. `assess_cache_compatibility`: checks whether Gemini CLI, Claude Code, Codex, or raw provider APIs expose enough cache telemetry to validate real reuse.
3. `capture_cache_telemetry`: normalizes provider cache-hit telemetry from Claude, OpenAI, and Gemini responses.
4. `diagnose_cache_miss`: explains likely causes of unexpected misses, partial reuse, section drift, and cache-creation churn.

The model-optimization layer now also exposes:

1. provider-specific cache threshold guidance via `optimize_for_model`
2. deterministic `prompt_cache_key` guidance for OpenAI/Codex-style routing stickiness
3. local extractive compression and history-compaction primitives for lower-latency context trimming
4. benchmark method comparisons between semantic and extractive baselines

For usage guidance, see `docs/guides/PROMPT_CACHING.md`.
For Gemini CLI, Claude, and Codex compatibility guidance, see `docs/guides/PROVIDER_CACHE_COMPATIBILITY.md`.

## Skill Scripts (No MCP Required)

Path: `skills/token-saver-context-compression/scripts/`

Main scripts:

1. `profile_tokens.py`: raw vs compressed token profile.
2. `compress_context.py`: baseline/query-guided/evidence-aware compression.
3. `validate_evidence.py`: checks if compressed output has enough evidence.
4. `run_skill_workflow.py`: profile + compress + evidence in one command.
5. `benchmark_toon_vs_json.py`: TOON/JSON token + quality guard checks.

All support local execution with no dependency on external MCP wrappers.

## Data Source Flexibility

You can feed Token Saver from any source as long as you provide text input:

1. Local files (`--file`).
2. Pasted text (`--text`).
3. Piped stdin from another command.
4. Upstream connectors that export text payloads.

The compressor itself is source-agnostic; GitHub is just one possible integration path, not a requirement.

## Output Formats (JSON vs TOON)

Skill scripts support:

1. `--output-format json`
2. `--output-format toon`
3. `--output-format auto`

`auto` behavior:

1. Select TOON only when data shape is TOON-friendly (uniform object arrays) and token-efficient.
2. Fall back to JSON otherwise.

## Repo Structure (Practical Map)

1. `src/` - core implementation.
2. `src/handlers/` - MCP tool handlers.
3. `src/semantic_modulator/` - app/api/service-layer architecture.
4. `skills/` - portable no-MCP skill package.
5. `scripts/` - benchmark/setup/dev scripts.
6. `tests/` - unit/integration/regression tests.
7. `docs/` - detailed guides and reference docs.

## Run The Server (MCP Mode)

```bash
token-saver-mcp
```

For web/API deployments, the repo also supports an HTTP server surface for health checks, metrics, and service-style runtime integrations. See `docs/deployment/DOCKER.md` and `docs/deployment/SAAS_MULTI_TENANT.md` for HTTP server, reverse proxy, and API gateway patterns.

Claude Desktop config example:

```json
{
  "mcpServers": {
    "token-saver": {
      "command": "token-saver-mcp",
      "args": [],
      "cwd": "/path/to/token-saver-5000"
    }
  }
}
```

The simplest setup path is:

```bash
token-saver-setup --auto
```

To install that entry automatically into Claude Desktop with the low-level installer:

```bash
token-saver-install-mcp
```

To generate a project-scoped `.claude\.mcp.json` for Claude Code or another MCP-aware workspace:

```bash
token-saver-install-mcp --project-config
```

To generate a **portable** project-scoped config for a shared repo using `${workspaceFolder}`:

```bash
token-saver-install-mcp --portable-project-config
```

If you want raw JSON instead of writing the project config file:

```bash
token-saver-install-mcp --print-config > .mcp.json
```

To inspect whether the command, Claude Desktop config, and project config are installed correctly:

```bash
token-saver-install-mcp --doctor --human
```

To uninstall cleanly:

```bash
token-saver-setup --uninstall-all
```

Or target just one surface:

```bash
token-saver-setup --uninstall --desktop
token-saver-setup --uninstall --portable-project
```

The MCP server now also exposes first-class prompts and resources:

1. Prompts for document compression, prompt-cache review, and MCP setup guidance.
2. Resources for tool catalogs, workflow instructions, install modes, and live install status.
3. A resource template at `token-saver://tool/{name}/help` for canonical per-tool help payloads.

## Test and Quality Commands

Run tests:

```bash
pytest tests/ -v
```

Run benchmark guard:

```bash
python scripts/benchmarks/run_benchmarks.py --compare baseline,query_guided,evidence_aware
python scripts/benchmarks/check_benchmark_guard.py --strict-case-set --summary-file artifacts/benchmarks/guard_summary.md
```

Lint/format:

```bash
python -m ruff check src tests scripts skills
python -m black src tests scripts skills
```

## Version and Requirements

1. Version: `0.10.0`
2. Python: `3.10-3.13`
3. Suggested RAM: `~4GB` for embedding workloads

Version source-of-truth:

1. `pyproject.toml`
2. `src/__init__.py`

## Documentation

Start here:

1. `docs/getting-started/GETTING_STARTED.md`
2. `docs/guides/HOW_IT_WORKS.md`
3. `docs/reference/ARCHITECTURE.md`
4. `docs/guides/MCP_TOOLS_GUIDE.md`
5. `docs/guides/WORKFLOW_ORCHESTRATION.md`
6. `docs/deployment/SAAS_MULTI_TENANT.md`
7. `CHANGELOG.md`

## License

MIT (`LICENSE`).
