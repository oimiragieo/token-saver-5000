# Token Saver 5000 - Claude Code Templates

Ready-to-use templates for integrating Token Saver 5000 with Claude Code.

## Quick Install

Copy all templates to your project:

```bash
# From your project root
cp -r /path/to/token-saver-5000/templates/claude-commands .claude/commands/
```

Or copy individual commands:

```bash
mkdir -p .claude/commands
cp /path/to/token-saver-5000/templates/claude-commands/compress.md .claude/commands/
```

## Available Slash Commands

| Command | Description |
|---------|-------------|
| `/compress` | Compress a document and show stats |
| `/search-docs` | Search compressed documents semantically |
| `/expand` | Expand a specific section to full detail |
| `/analyze` | Full document analysis workflow |
| `/remember` | Store critical info in dialogue memory |
| `/token-stats` | View comprehensive token savings stats |

## Usage Examples

```
/compress [paste your document here]

/search-docs authentication best practices

/expand paper_n3

/analyze [paste research paper]

/remember I have a severe peanut allergy

/token-stats
```

## CLAUDE.md Integration

Add this to your project's `CLAUDE.md` for automatic Token Saver usage:

```markdown
## Token Compression Rules
- Use `mcp__token-saver__ingest_context` for any document >500 tokens
- Use `mcp__token-saver__search_semantic` to find relevant sections
- Use `mcp__token-saver__check_blind_spots` after answering document questions
- Start with ABSTRACT fidelity, expand only as needed
- Target 80%+ token reduction on all large documents
```

## Full Documentation

See [CLAUDE_CODE_SETUP.md](../docs/guides/CLAUDE_CODE_SETUP.md) for complete setup instructions.
