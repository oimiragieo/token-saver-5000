# Hook Recipes: Automatic Knowledge Accumulation

This guide shows how to wire Claude Code lifecycle hooks to Token Saver's
knowledge management tools (`ingest_transcript`, `compile_knowledge`,
`lint_knowledge`) so that session insights accumulate automatically.

## Overview

Token Saver exposes three tiers of knowledge management:

| Tier | Tool | When to run |
|------|------|-------------|
| **Capture** | `ingest_transcript` | End of every session |
| **Compile** | `compile_knowledge` | Nightly or weekly |
| **Lint** | `lint_knowledge` | Before compile, or on-demand |

Hooks call these tools through the MCP server — no changes to the server
itself are needed.

## Recipe 1: Capture insights on session end

Create a Claude Code `Stop` hook that sends the session transcript to
`ingest_transcript` for extraction.

**.claude/hooks/stop-capture.sh**

```bash
#!/usr/bin/env bash
# Stop hook: extract insights from the ending session transcript.
# Requires the MCP server to be running (stdio transport).

TRANSCRIPT_FILE="${CLAUDE_SESSION_TRANSCRIPT:-/tmp/claude_session.txt}"

if [ ! -f "$TRANSCRIPT_FILE" ]; then
  exit 0
fi

# Call the MCP tool via the CLI wrapper
token-saver-mcp call ingest_transcript \
  --text "$(cat "$TRANSCRIPT_FILE")" \
  --mode all \
  --source "session-hook"
```

Register in `.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "command": "bash .claude/hooks/stop-capture.sh",
        "timeout": 10000
      }
    ]
  }
}
```

## Recipe 2: Log prompts to session journal

A `UserPromptSubmit` hook that logs each user prompt to the session
journal for later recovery.

**.claude/hooks/log-prompt.sh**

```bash
#!/usr/bin/env bash
# UserPromptSubmit hook: journal each prompt for session recovery.

SESSION_ID="${CLAUDE_SESSION_ID:-default}"

token-saver-mcp call recover_session \
  --session_id "$SESSION_ID"
```

## Recipe 3: Nightly compilation via cron

Set up a cron job (or Task Scheduler on Windows) to compile accumulated
memories into cross-linked markdown articles.

```bash
# crontab -e
# Run at 11 PM daily
0 23 * * * token-saver-mcp call compile_knowledge --write_files true
```

Or use Claude Code's built-in scheduling if available:

```json
{
  "hooks": {
    "Notification": [
      {
        "command": "bash -c 'token-saver-mcp call compile_knowledge --write_files true'",
        "schedule": "0 23 * * *"
      }
    ]
  }
}
```

## Recipe 4: Pre-compile lint gate

Run `lint_knowledge` before compilation to surface issues first.

**.claude/hooks/compile-with-lint.sh**

```bash
#!/usr/bin/env bash
# Lint then compile — stop on errors.

LINT_RESULT=$(token-saver-mcp call lint_knowledge --stale_days 30 2>&1)
ERRORS=$(echo "$LINT_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('errors',0))" 2>/dev/null)

if [ "$ERRORS" -gt 0 ]; then
  echo "Lint found $ERRORS error(s). Fix before compiling."
  echo "$LINT_RESULT"
  exit 1
fi

token-saver-mcp call compile_knowledge --write_files true
echo "Compilation complete."
```

## Recipe 5: Index-first Q&A for small knowledge bases

For projects with fewer than ~500 memories, index-first retrieval avoids
embedding overhead entirely:

```bash
# Returns the compiled index + matching articles
token-saver-mcp call search_memory_index --query "authentication flow"
```

This matches the claude-memory-compiler philosophy of "index beats RAG at
personal scale" while preserving Token Saver's embedding search as an
option for larger corpora.

## Architecture: two-stage pipeline

```
Session End                    Nightly
    |                             |
    v                             v
ingest_transcript          lint_knowledge
    |                             |
    v                             v
MemoryAPI.add_memory()     compile_knowledge
    |                             |
    v                             v
Flat memories              Markdown articles + index.md
(fast, <1s)                (slow, seconds)
```

The capture stage runs at session boundaries and is intentionally fast
and simple.  The compile stage runs on a schedule and performs heavier
processing (deduplication, cross-linking, file I/O).

## Relevant MCP tools

| Tool | Purpose |
|------|---------|
| `ingest_transcript` | Extract insights from transcript text |
| `compile_knowledge` | Group, deduplicate, and emit markdown articles |
| `get_knowledge_index` | Return compiled index without writing files |
| `lint_knowledge` | Quality checks (stale, duplicates, contradictions) |
| `search_memory_index` | Index-first retrieval for small knowledge bases |
| `add_memory` | Manual memory storage (existing) |
| `search_memory` | Embedding/token-overlap search (existing) |
