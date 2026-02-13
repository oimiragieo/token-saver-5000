# Token Saver Context Compression Skill

## Purpose

Reduce token usage before heavy-context reasoning while preserving answer-supporting evidence.

## When to use

- Input context is large (>500 tokens) or likely to overflow budget.
- Task requires high factual grounding.
- You want query-targeted compression, not just global summaries.

## Workflow

1. Profile size:
   - `python scripts/skills/profile_tokens.py --file <path>`
2. Compress with selection mode:
   - Baseline: `python scripts/skills/compress_context.py --file <path> --mode baseline`
   - Query-guided: `python scripts/skills/compress_context.py --file <path> --mode query_guided --query "<question>"`
   - Evidence-aware: `python scripts/skills/compress_context.py --file <path> --mode evidence_aware --query "<question>" --min-similarity 0.4`
3. Validate evidence sufficiency when correctness risk is high:
   - `python scripts/skills/validate_evidence.py --file <path> --query "<question>" --min-similarity 0.4`
4. If evidence is insufficient, broaden retrieval or reduce compression aggressiveness.

## Outputs

- JSON payloads with compression metrics, skeleton text, and evidence diagnostics.
- `validate_evidence.py` exits non-zero when evidence is insufficient.

## Notes

- Default mode is conservative (`baseline`).
- Use `evidence_aware` for high-stakes responses.
- Keep `file_id` stable across iterative runs for traceability.
