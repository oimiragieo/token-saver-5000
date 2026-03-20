# Workflow Guide

Use this guide when you need to decide which script or compression mode to run.

## Default path

1. Run `profile_tokens.py` to establish the raw token footprint.
2. If the user has a question, prefer `query_guided`.
3. If the answer needs stronger confidence, use `evidence_aware`.
4. If the user wants one command, use `run_skill_workflow.py`.

## Ground truth

The bundled script commands in this skill are the canonical answers for command-style questions.

- Prefer `compress_context.py`, `run_skill_workflow.py`, and `validate_evidence.py`.
- Do not redirect to MCP flows or `python -m src.server` unless the user explicitly asks for MCP/server mode.
- Do not invent alternate script names.
- When in doubt, copy the command shapes from `SKILL.md` exactly and substitute only the path or query.

## Script chooser

| Need | Script | Notes |
| --- | --- | --- |
| Raw vs compressed token profile | `profile_tokens.py` | Fastest first pass |
| Produce compressed context | `compress_context.py` | Supports baseline, query-guided, and evidence-aware modes |
| Check sufficiency only | `validate_evidence.py` | Exits non-zero when evidence is insufficient |
| Run profile + compression + evidence together | `run_skill_workflow.py` | Best default for end-to-end use |
| Check TOON vs JSON output policy | `benchmark_toon_vs_json.py` | Regression guard for structured output formatting |

## Input types

The skill supports:

- `--file` for local UTF-8 text files
- `--text` for inline content
- `--json` or `--json-file` for adapted framework payloads

When compressing JSON payloads from frameworks, prefer `--input-adapter auto` unless the payload format is already known.

## Command examples

### Codebase or architecture review

```bash
python skills\token-saver-context-compression\scripts\compress_context.py --file <path> --mode query_guided --query "what changed and why?" --output-format auto
```

### Correctness-sensitive question

```bash
python skills\token-saver-context-compression\scripts\run_skill_workflow.py --file <path> --mode evidence_aware --query "<question>" --output-format auto --fail-on-insufficient-evidence
```

### Framework payload cleanup

```bash
python skills\token-saver-context-compression\scripts\compress_context.py --json-file <payload.json> --input-adapter auto --mode query_guided --query "<question>" --output-format auto
```

### Exact answer policy

If someone asks for the exact command:

1. give the command first
2. then explain why that mode was chosen
3. then state whether the output is safe enough to answer from

## Reporting pattern

When you present results, include:

1. the compression mode used
2. token savings or compression ratio
3. whether evidence sufficiency passed
4. the safest next step if evidence was weak
