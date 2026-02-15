# Token Saver 5000

Semantic compression for AI context.  
This project helps you keep useful meaning while spending fewer tokens.

[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Tests](https://img.shields.io/badge/tests-1500%2B_passing-brightgreen)]()

## What This Is

Token Saver 5000 is a local-first system that:

1. Ingests big text/code context.
2. Builds a semantic graph of the content.
3. Produces a compact "skeleton" view.
4. Lets you pull back detail only where needed.

You can use it as:

1. An MCP server (`python -m src.server`) for Claude/Desktop workflows.
2. A self-contained skill package (`skills/token-saver-context-compression/`) with local scripts and no MCP requirement.

## How It Works (Plain English)

Think of it like turning a long book into:

1. A table of contents.
2. A map of important ideas.
3. A way to instantly open the exact page you need.

Flow:

1. `ingest_context`: split text into chunks and embed semantics.
2. Build a graph: chunks become nodes, semantic links become edges.
3. Rank importance (PageRank + relevance signals).
4. Return a compressed skeleton (high-value nodes only).
5. If you ask a specific question, query-guided/evidence-aware modes keep only likely answer-supporting spans.

Result: smaller prompt payloads with controlled quality risk.

## Quick Start

```bash
git clone https://github.com/oimiragieo/token-saver-5000.git
cd token-saver-5000
pip install -r requirements.txt
python scripts/check_setup.py
python examples/example_usage.py
```

## Run Modes

## 1) MCP Server Mode

```bash
python -m src.server
```

Claude Desktop config:

```json
{
  "mcpServers": {
    "token-saver": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/token-saver-5000"
    }
  }
}
```

### MCP Tool Profiles

Use `MCP_TOOL_PROFILE` to control exposed tools:

1. `full` (default): all tools.
2. `core_stable`: stable core tools only:
   - `ingest_context`
   - `read_skeleton`
   - `search_semantic`
   - `modulate_region`
   - `get_stats`
   - `list_documents`
   - `delete_document`

Example:

```bash
MCP_TOOL_PROFILE=core_stable python -m src.server
```

## 2) Self-Contained Skill Mode (No MCP Needed)

Portable skill folder:

- `skills/token-saver-context-compression/`

Core commands:

```bash
python skills/token-saver-context-compression/scripts/profile_tokens.py --file <path> --output-format auto
python skills/token-saver-context-compression/scripts/compress_context.py --file <path> --mode query_guided --query "<question>" --output-format auto
python skills/token-saver-context-compression/scripts/validate_evidence.py --file <path> --query "<question>" --min-similarity 0.4 --output-format json
python skills/token-saver-context-compression/scripts/run_skill_workflow.py --file <path> --mode evidence_aware --query "<question>" --output-format auto
python skills/token-saver-context-compression/scripts/benchmark_toon_vs_json.py
```

The skill scripts support:

1. `--output-format {json,toon,auto}`
2. Guarded TOON auto-selection (uniform tabular payloads only)
3. JSON fallback when TOON is not a win

## Project Layout

High-level map:

1. `src/`: core runtime and server implementation.
2. `src/handlers/`: MCP tool handlers.
3. `src/semantic_modulator/`: enterprise-style app/api layering and wiring.
4. `skills/`: portable skill package.
5. `scripts/`: utilities, benchmark runners, setup helpers.
6. `tests/`: unit/integration/regression suites.
7. `docs/`: guides, references, deployment, research notes.

## Architecture Snapshot

Core pipeline:

1. Chunk -> Embed -> Graph -> Rank -> Skeleton.
2. Query-guided and evidence-aware selection for tighter context targeting.
3. Tool routing via MCP registry/router + app-layer service adapters.
4. Persistence, sync, versioning, and resource controls as support services.

Design goal: keep a stable core path while allowing advanced/experimental capabilities.

## Testing and Quality

Run all tests:

```bash
pytest tests/ -v
```

Run benchmark guard workflow:

```bash
python scripts/benchmarks/run_benchmarks.py --compare baseline,query_guided,evidence_aware
python scripts/benchmarks/check_benchmark_guard.py --strict-case-set --summary-file artifacts/benchmarks/guard_summary.md
```

Formatting/lint:

```bash
python -m black src tests scripts skills
python -m ruff check src tests scripts skills
```

## Documentation Index

Getting started:

1. `docs/getting-started/GETTING_STARTED.md`
2. `docs/getting-started/CONTRIBUTING.md`

Guides:

1. `docs/guides/HOW_IT_WORKS.md`
2. `docs/guides/MCP_TOOLS_GUIDE.md`
3. `docs/guides/CLAUDE_CODE_SETUP.md`
4. `docs/guides/CLAUDE_SKILL_PACKAGING.md`
5. `docs/guides/TDD_MODERNIZATION_PLAN.md`
6. `docs/guides/TDD_EXECUTION_SPEC_2026-02-15.md`

Reference:

1. `docs/reference/ARCHITECTURE.md`
2. `docs/reference/API_REFERENCE.md`

Deployment:

1. `docs/deployment/DEPLOYMENT.md`
2. `docs/deployment/DOCKER.md`
3. `docs/deployment/SECURITY.md`

## Requirements

1. Python 3.10+
2. ~4GB RAM recommended (embedding model workloads)
3. Local disk for cache/persistence

## Version

Current version: `0.10.0`  
Source of truth: `pyproject.toml`, `src/__init__.py`

## License

MIT. See `LICENSE`.
