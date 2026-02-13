---
name: token-saver-context
description: Compress large context before reasoning to reduce token usage while preserving answer-supporting evidence. Use for large files, long prompts, RAG context, and high-cost sessions.
argument-hint: [file-or-text-and-query]
---

# Token Saver Context Compression

Use this skill to reduce token usage without MCP by running local Python scripts bundled with the skill.
This package is self-contained and does not import project files outside this skill folder.

## When to use

- Context is large or expensive.
- You need query-targeted compression.
- You need evidence sufficiency checks before answering.

## Commands

Run from project root, or keep absolute paths as shown.

1. Profile token usage:

```bash
python "$CLAUDE_PROJECT_DIR/.claude/skills/token-saver-context-compression/scripts/profile_tokens.py" --file <path> --output-format auto
```

2. Compress context:

```bash
python "$CLAUDE_PROJECT_DIR/.claude/skills/token-saver-context-compression/scripts/compress_context.py" --file <path> --mode baseline --output-format auto
python "$CLAUDE_PROJECT_DIR/.claude/skills/token-saver-context-compression/scripts/compress_context.py" --file <path> --mode query_guided --query "<question>" --output-format auto
python "$CLAUDE_PROJECT_DIR/.claude/skills/token-saver-context-compression/scripts/compress_context.py" --file <path> --mode evidence_aware --query "<question>" --min-similarity 0.4 --output-format auto
```

3. Validate evidence:

```bash
python "$CLAUDE_PROJECT_DIR/.claude/skills/token-saver-context-compression/scripts/validate_evidence.py" --file <path> --query "<question>" --min-similarity 0.4 --output-format json
```

4. Run all steps in one command:

```bash
python "$CLAUDE_PROJECT_DIR/.claude/skills/token-saver-context-compression/scripts/run_skill_workflow.py" --file <path> --mode evidence_aware --query "<question>" --output-format auto --fail-on-insufficient-evidence
```

5. Run TOON-vs-JSON benchmark/guard checks:

```bash
python "$CLAUDE_PROJECT_DIR/.claude/skills/token-saver-context-compression/scripts/benchmark_toon_vs_json.py"
```

## Workflow policy

1. Profile first.
2. Prefer `query_guided` for QA tasks.
3. Use `evidence_aware` for correctness-sensitive tasks.
4. Prefer `--output-format auto` so TOON is chosen only when it is likely to reduce tokens.
5. If evidence is insufficient, reduce compression aggressiveness or broaden retrieval.

## Output contracts

- Scripts emit JSON to stdout.
- Scripts support `--output-format {json,toon,auto}`.
- In `auto`, TOON is selected only when a uniform object-array shape crosses a row threshold.
- Mixed or irregular structures automatically stay in JSON.
- `validate_evidence.py` exits `1` when insufficient evidence is detected.
- `run_skill_workflow.py` can fail on insufficient evidence when `--fail-on-insufficient-evidence` is set.
- `benchmark_toon_vs_json.py` exits non-zero if guard thresholds fail.

## Requirements

- Python 3.10+
- Optional: `tiktoken` for exact token counts (fallback counter works without it)
