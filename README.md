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

## Proven Results: 6 Real-World User Journeys

Every journey runs locally. No API keys needed. Verified with 40 tests.

```bash
python scripts/benchmark_cujs.py --verbose
```

| # | Journey | What Happens | Before | After | Savings |
|---|---------|-------------|--------|-------|---------|
| 1 | **Solo Dev: Codebase Compression** | Search 13-file project for "auth" → compress 6 matched files | 20,406 tokens | 767 tokens | **96.2%** |
| 2 | **Long Document Compression** | Compress 2,206-line API reference doc | 16,461 tokens | 1,269 tokens | **92.3%** |
| 3 | **CLI Output Filtering** | Filter git diff (320 lines) + pytest (86 lines) + npm install (28 lines) | 4,888 tokens | 240 tokens | **95.1%** |
| 4 | **Query-Focused Code Search** | Ask "how does caching work?" → find 3 relevant files → compress | 20,406 tokens | 439 tokens | **97.8%** |
| 5 | **Session Recovery** | Recover 7 events after conversation compaction | 26,600 tokens | 138 tokens | **99.5%** |
| 6 | **ROI Justification** | 10 compressions on Claude Opus → show savings report | $1.27 saved | 4.4x ROI | **$127/mo projected** |

**Aggregate: 179,161 input tokens → 8,473 output tokens (95.3% savings)**

### Journey Details

**CUJ 1: Solo Developer with a Codebase.**
You have 13 Python files. You ask about authentication. Token Saver searches for relevant
code (finds auth.py, middleware.py, tests), compresses only those files, skips everything
irrelevant. 96.2% savings vs reading every file.

**CUJ 2: Compressing Architecture Docs.**
You have a 2,206-line API reference. Token Saver builds a semantic graph, ranks importance
via PageRank, generates a 13x compressed skeleton preserving endpoints, parameters, and
error codes. 92.3% savings.

**CUJ 3: Cleaning Up CLI Noise.**
Your AI agent runs `git diff`, `pytest`, and `npm install`. Token Saver auto-detects each
command type and applies the right filter: stats extraction for git, failure focus for pytest,
summary for npm. 95.1% savings (434 lines → 22 lines).

**CUJ 4: "How Does Caching Work?"**
You ask a question about a codebase. Token Saver searches first (finds cache.py, config.py,
middleware.py), then compresses only those 3 files instead of all 13. 97.8% savings -- 75%
better than compressing everything.

**CUJ 5: Surviving Conversation Compaction.**
After a long Claude Code session, the conversation gets compacted and you lose context.
Token Saver's session journal recovers all your prior work (5 ingested files, model config,
compression profile, 26,600 tokens saved) in just 138 tokens.

**CUJ 6: Proving ROI to Your Manager.**
After 10 compressions on Claude Opus ($15/MTok), the savings tracker shows: $1.27 saved,
4.4x ROI vs the $29/mo Pro plan, breakeven at 228 operations. Projected $127/month savings.
The tool pays for itself on day 1.

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

**Document compression (locally measured, reproducible):**
Token Saver achieves **13x document compression** (16,461 tokens -> 1,269 tokens)
via semantic graph + PageRank + token-level refinement + lossless meta-tokens.

**API input token savings (real measurements, total content tokens):**

| Provider | Baseline | Compressed | Savings | Notes |
|----------|---------|------------|---------|-------|
| Codex (gpt-5.1-codex) | 37,514 | 23,189 | **~38%** | Most stable (consistent cache) |
| Gemini CLI (Flash) | 69,172 | 30,672 | **~56%** | Measured via prompt field (cache-independent) |
| Claude Code (Opus 4.6) | ~61,500 | ~45,300 | **~26%** | Approximate (cache state varies +/-3%) |

**Methodology:**
- **Validated across 3 independent runs** with zero variance (min = max = median)
- "Total content tokens" used (not billed tokens) to remove cache hit/miss variance
- "Doc Compression" = document-only reduction (92.3%) vs "Total API Savings" = including system prompt
- System prompt overhead varies by provider (~42K for Claude, ~16K for Codex, ~28K for Gemini)
- Cost savings NOT reported as headline (volatile due to cache pricing effects)
- Answer quality NOT formally evaluated (compression may affect response quality)

**All sizes (large corpus, 2206 lines):**

| Corpus | Claude | Codex | Gemini |
|--------|--------|-------|--------|
| small (156 lines) | ~2% | ~4% | ~8% |
| medium (479 lines) | ~9% | ~14% | ~26% |
| large (2,206 lines) | **~26%** | **~38%** | **~56%** |

Real-world savings depend on system prompt size: smaller system prompts (Codex, Gemini)
see proportionally larger savings from document compression.

Run benchmarks yourself:

```bash
# Dry run (no API calls, validates setup)
python scripts/benchmark_token_savings.py --dry-run --verbose

# Full benchmark across all providers
python scripts/benchmark_token_savings.py --mode skill --verbose --output results.json

# Single provider, single corpus size
python scripts/benchmark_token_savings.py --providers claude --sizes large --verbose
```

## Token Savings Tracker (NEW)

Every compression operation is tracked with exact dollar savings. See your ROI in real-time:

```
get_savings_report(session_id="my-session")
# -> {
#   "total_tokens_saved": 142,500,
#   "total_dollars_saved": 2.14,
#   "avg_compression_ratio": 13.0,
#   "monthly_projected_savings": 64.20,
#   "roi_vs_pro_plan": 2.2,
#   "breakeven_operations": 14,
#   "by_tool": {"ingest_context": {"operations": 8, "dollars_saved": 1.87}, ...}
# }
```

The tracker computes: tokens saved, dollar savings (model-aware pricing), compression ratios,
monthly projections, ROI vs the $29/mo Pro plan, and breakeven analysis. Persists to SQLite
so savings accumulate across sessions.

## CLI Output Optimizer (NEW -- RTK-Inspired)

Coding agent CLI output is the #1 token waster. Token Saver now auto-detects 10 command
types and applies optimal filtering:

| Command | Strategy | Typical Savings |
|---------|----------|----------------|
| `git diff` | Extract file list + stats summary | 90-99% |
| `pytest` / `jest` | Show only failures + summary | 94-99% |
| `npm install` / `pip install` | Keep summary, strip progress | 85-95% |
| Lint (ruff, eslint) | Group by rule, count occurrences | 80-90% |
| JSON output | Extract keys + types, first item | 80-95% |
| Logs | Deduplicate repeated lines | 70-85% |
| Any colored output | Strip ANSI escape codes | 10-30% |

Use directly via MCP tool:
```
filter_cli_output(text="<raw CLI output>")
```

Or automatically via the proxy (applies to all upstream tool responses):
```bash
token-saver-proxy npx some-server --provider anthropic
```

Falls back to [RTK](https://github.com/rtk-ai/rtk) when installed for maximum quality.

## MCP Proxy Mode (NEW)

Wrap ANY MCP server transparently -- compress tool responses automatically with zero code changes:

```bash
# Compress any MCP server's output
token-saver-proxy npx some-mcp-server --provider anthropic

# Enable schema compression (N tools -> 3 meta-tools, ~96% token reduction)
token-saver-proxy python -m my_server --schema-compression
```

In Claude Desktop config:
```json
{
  "mcpServers": {
    "my-server-compressed": {
      "command": "token-saver-proxy",
      "args": ["npx", "some-mcp-server", "--provider", "anthropic"]
    }
  }
}
```

The proxy applies TokenRefiner + MetaTokenCompressor to every tool response. Optional `--schema-compression` replaces all upstream tools with 3 meta-tools (search_tools, get_tool_schema, invoke_tool).

## Session Continuity (NEW)

Token Saver now survives conversation compaction. A SQLite-backed journal records all ingestions, configurations, and tool calls. After the CLI compacts your conversation:

```
# Call recover_session to get a compact summary of everything that happened
recover_session(session_id="my-session")
# -> {ingested_files: [...], client_config: {...}, active_profile: "balanced", total_tokens_saved: 14500}
```

## Cache Strategy Advisor (NEW)

Every LLM provider handles caching differently. The advisor tells you exactly what to do:

```
advise_cache_strategy(model_id="claude-4-sonnet")
# -> Anthropic: explicit cache, 90% discount, add ephemeral markers, 5min TTL

advise_cache_strategy(model_id="gpt-4.1")
# -> OpenAI: automatic cache, 50% discount, keep 1024+ token prefix stable

advise_cache_strategy(model_id="gemini-2.5-flash")
# -> Google: implicit cache, 90% discount, no client action needed

advise_cache_strategy(model_id="groq-llama-4-scout")
# -> Groq: no caching, focus on small prompts for fastest inference
```

Supports: Anthropic, OpenAI, Google Gemini, Groq, XAI (Grok), Azure, Bedrock, local/Ollama.

## Multi-Agent Setup (NEW)

Install Token Saver for any AI coding agent with a single command:

```bash
token-saver-setup --auto                    # Auto-detect your environment
token-saver-install-mcp --agent cursor      # Cursor
token-saver-install-mcp --agent windsurf    # Windsurf
token-saver-install-mcp --agent cline       # Cline / Roo Code
token-saver-install-mcp --agent codex       # OpenAI Codex CLI
token-saver-install-mcp --agent gemini      # Gemini CLI
token-saver-install-mcp --agent copilot     # VS Code Copilot
token-saver-install-mcp --doctor-all        # Check all agent configs
```

8 agents supported: Claude Desktop, Claude Code (project), Cursor, Windsurf, Cline, VS Code Copilot, Codex, and Gemini CLI.

## Savings Dashboard (NEW)

Track your token savings across sessions:

```bash
token-saver-stats                  # All-time summary
token-saver-stats --daily          # Day-by-day breakdown
token-saver-stats --weekly         # Weekly summary
token-saver-stats --by-tool        # Per-tool breakdown
token-saver-stats --cost           # Cost savings with model pricing
token-saver-stats --json           # Machine-readable output
token-saver-stats --csv            # Spreadsheet export
```

## ROI Calculator (NEW)

Calculate your return on investment via the `calculate_roi` MCP tool:

```
Input:  model=claude-opus-4-6, tokens_per_day=500000, team_size=10
Output:
  Without gotcontext: $1,650.00/mo
  With gotcontext:      $247.50/mo (85% savings)
  Pro plan cost:        $290.00/mo ($29/user × 10 users)
  Net savings:        $1,112.50/mo (5.7x ROI)
```

Supports 20+ models with real pricing data.

## Token Budget Monitoring (NEW)

Set per-session, daily, or monthly token budgets via `check_budget` MCP tool or environment variables:

```bash
TOKEN_BUDGET_SESSION=500000 TOKEN_BUDGET_DAILY=2000000 token-saver-mcp
```

Returns usage status, alert levels (ok/info/warning/critical), and projected usage.

## Team Dashboard Export (NEW)

Export aggregated team savings data via `export_team_data` MCP tool:

1. **JSON**: For API consumption and custom dashboards.
2. **CSV**: For spreadsheet analysis.
3. **Prometheus**: For Grafana/Datadog monitoring.

## Tee/Recovery System (NEW)

When compression drops information, the original is saved for recovery:

```bash
TEE_MODE=failures token-saver-mcp    # Tee on high compression (default)
TEE_MODE=always token-saver-mcp      # Tee everything
```

MCP tools: `get_original_output`, `list_tee_entries`, `tee_store_stats`.

## Custom Filter Rules (NEW)

Define project-specific output filtering rules in `.gotcontext.toml`:

```toml
[filters.my_build_output]
match_command = "my-build-tool"
strip_ansi = true
strip_lines_matching = ["^Progress:", "^\\s*$"]
keep_lines_matching = ["^ERROR", "^WARNING"]
head_lines = 50
tail_lines = 20
max_lines = 100
```

Supports 8-stage pipeline, inline tests, project + user-global precedence.

## Missed Savings Discovery (NEW)

The `discover_savings` MCP tool scans directories to find files that would benefit from compression:

```
discover_savings(directory="/path/to/project")
→ README.md: ~2,400 tokens → ~300 compressed (87% savings)
→ src/main.py: ~800 tokens → ~200 compressed (75% savings)
→ Total opportunity: ~12,000 tokens saveable
```

## Research-Backed Compression Techniques

Token Saver integrates techniques from recent AI research papers:

1. `compress_meta_tokens`: lossless LZ77-inspired compression (arXiv 2506.00307). Replaces
   repeated token subsequences with §N symbols + dictionary header. Fully reversible.
2. `recommend_compression`: quality-floor-based profile selection (arXiv 2603.19733). Specify
   minimum acceptable quality (e.g. 0.85) instead of manually choosing a compression profile.
   Auto-selects the most aggressive profile that meets your quality target.
3. **COMI MIG scoring** (arXiv 2602.01719): query-aware token refinement. When a query is
   provided, the token refiner uses Marginal Information Gain to keep relevant tokens and
   remove redundant ones, instead of simple filler-word removal.

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
