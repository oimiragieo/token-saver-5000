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

## Who This Is For

Use this if you:

1. Work with large prompts/documents.
2. Need to cut token cost.
3. Want retrieval-oriented compression (not just naive summarization).

Do not use this if you only have short prompts and token cost is irrelevant.

## Two Ways To Use It

There are two product surfaces in this repo:

1. MCP server (`src.server`) for Claude/Desktop and agent workflows.
2. Self-contained skill scripts (`skills/token-saver-context-compression`) that run locally without MCP.

## Local vs Docker

You do not need Docker. Docker is optional.

Choose your runtime:

1. Local Python:
   - Best for development and quick usage.
   - Direct access to scripts and source.
   - Command: `python -m src.server`
2. Docker:
   - Best for reproducible deployment/team environments.
   - Avoids local dependency drift.
   - Command: `docker-compose up -d`

## First 10 Minutes (Recommended Path)

1. Clone and install:

```bash
git clone https://github.com/oimiragieo/token-saver-5000.git
cd token-saver-5000
pip install -r requirements.txt
python scripts/check_setup.py
```

2. Run a local example:

```bash
python examples/example_usage.py
```

3. Try the self-contained skill scripts:

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

## Skill Scripts (No MCP Required)

Path: `skills/token-saver-context-compression/scripts/`

Main scripts:

1. `profile_tokens.py`: raw vs compressed token profile.
2. `compress_context.py`: baseline/query-guided/evidence-aware compression.
3. `validate_evidence.py`: checks if compressed output has enough evidence.
4. `run_skill_workflow.py`: profile + compress + evidence in one command.
5. `benchmark_toon_vs_json.py`: TOON/JSON token + quality guard checks.

All support local execution with no dependency on external MCP wrappers.

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
python -m src.server
```

Claude Desktop config example:

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
2. Python: `3.10+`
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
5. `CHANGELOG.md`

## License

MIT (`LICENSE`).
