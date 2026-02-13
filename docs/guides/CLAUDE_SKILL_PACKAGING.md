# Claude Skill Packaging

This project now includes a starter skill package for context compression:

- Skill file: `skills/token-saver-context-compression/SKILL.md`
- Scripts:
  - `scripts/skills/profile_tokens.py`
  - `scripts/skills/compress_context.py`
  - `scripts/skills/validate_evidence.py`

## Recommended Flow

1. Profile context size:
   - `python scripts/skills/profile_tokens.py --file path/to/context.txt`
2. Compress context:
   - `python scripts/skills/compress_context.py --file path/to/context.txt --mode evidence_aware --query "your question"`
3. Validate evidence sufficiency for high-stakes responses:
   - `python scripts/skills/validate_evidence.py --file path/to/context.txt --query "your question"`

## Integration Notes

- Use `query_guided` or `evidence_aware` mode when task intent is known.
- Keep baseline mode for generic summarization.
- In agent pipelines, treat non-zero exit from `validate_evidence.py` as a trigger to fetch more context or lower compression aggressiveness.
