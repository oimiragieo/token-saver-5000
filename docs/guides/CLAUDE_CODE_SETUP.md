# Using Token Saver 5000 with Claude Code

This guide shows you how to set up Token Saver 5000 as an MCP server for Claude Code and configure it to work automatically.

## Table of Contents
1. [Quick Start](#quick-start-2-minutes)
2. [CLAUDE.md Configuration](#claudemd-configuration-copy-paste-ready)
3. [Slash Commands](#slash-commands)
4. [Hooks for Automation](#hooks-for-automation)
5. [Prompt Examples](#prompt-examples)
6. [Available Tools](#available-tools-121-total)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start (2 minutes)

### Step 1: Clone and Install

```bash
git clone https://github.com/oimiragieo/token-saver-5000.git
cd token-saver-5000
uv tool install -e .
```

Fallback options:

```bash
pipx install .
```

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 2: Configure Claude Code

Recommended:

```bash
token-saver-setup --portable-project
```

If you want the installer to choose automatically based on the current workspace:

```bash
token-saver-setup --auto
```

Advanced/manual fallback:

Create or edit `.mcp.json` in your project root:

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

**Important:** Replace `/path/to/token-saver-5000` with the actual absolute path.

Or generate the project-scoped `.claude/.mcp.json` automatically:

```bash
token-saver-install-mcp --project-config
```

For a team-shared config that uses `${workspaceFolder}` instead of a machine-specific absolute path:

```bash
token-saver-install-mcp --portable-project-config
```

To inspect the current Token Saver MCP setup state:

```bash
token-saver-install-mcp --doctor
token-saver-install-mcp --doctor --human
```

### Step 3: Restart Claude Code

```bash
claude
```

---

## CLAUDE.md Configuration (Copy-Paste Ready)

Add this to your project's `CLAUDE.md` or `~/.claude/CLAUDE.md` to teach Claude how to use Token Saver automatically:

```markdown
## Token Saver 5000 - Semantic Compression

### When to Use Token Saver
- **ALWAYS** use `mcp__token-saver__should_compress` FIRST when:
  - User mentions a file path (estimates tokens WITHOUT reading)
  - Before reading ANY file to check if compression is needed
  - Returns recommendation: SKIP, DIRECT_READ, COMPRESS, or CONVERT_THEN_COMPRESS
  - v0.9.2: Detects binary files (PDF, DOCX, images) that need conversion

- **ALWAYS** use `mcp__token-saver__ingest_context` when:
  - `should_compress` returns COMPRESS
  - User shares a document longer than 500 tokens
  - User asks to analyze, summarize, or work with a large file
  - User pastes code files, documentation, or research papers
  - Context window is getting full and you need to compress

- **ALWAYS** use `mcp__token-saver__search_semantic` when:
  - Looking for specific information in compressed documents
  - User asks questions about previously ingested content
  - Need to find relevant sections without expanding everything

- **ALWAYS** use `mcp__token-saver__check_blind_spots` when:
  - After answering questions about compressed documents
  - Before finalizing responses to ensure nothing was missed
  - User asks for comprehensive analysis

### Compression Workflow
0. **Pre-check**: Use `should_compress` with file_path to estimate tokens (doesn't read file!)
   - Check `needs_conversion` field - if true, use MarkItDown first
   - Check `recommendation` field for action: SKIP, DIRECT_READ, COMPRESS
1. **Convert (if binary)**: If `needs_conversion=true`, use MarkItDown or similar tool
2. **Ingest**: Use `ingest_context` with the document text and a unique file_id
3. **Review**: Use `read_skeleton` to see the compressed structure
4. **Search**: Use `search_semantic` to find relevant sections
5. **Expand**: Use `modulate_region` to get full detail when needed
6. **Verify**: Use `check_blind_spots` to catch missed context

### Handling Binary Files (v0.9.2)
When `should_compress` returns `needs_conversion=true`:
1. Use MarkItDown (or pandoc, pdftotext) to convert binary to text
2. Call `ingest_context` with the converted text
3. If conversion tool unavailable:
   - Try alternative: pandoc, docx2txt, pdftotext
   - Manual text extraction
   - Document file as "skipped - no converter available"

### Fidelity Levels (use progressively)
- `ABSTRACT`: 1-sentence summary (~10 tokens) - start here
- `STRUCTURE`: Headers + key points (~50 tokens) - for overview
- `BALANCED`: Important details (~150 tokens) - most common
- `DETAILED`: Full context (~300 tokens) - when needed
- `RAW`: Original text - only when absolutely necessary

### Dialogue Memory (AFM)
For multi-turn conversations with critical information:
- Use `afm_add_message` to store important user statements
- Mark allergies, preferences, constraints as role="user" content
- Use `afm_build_context` to retrieve relevant history
- Critical safety info is NEVER compressed

### Best Practices
- Prefer compressed skeletons over raw text (80-95% token savings)
- Use semantic search before expanding sections
- Check blind spots after complex queries
- Re-ingest files that have changed with `refresh_document`
```

### Minimal CLAUDE.md (Quick Version)

For a simpler setup, add this minimal section:

```markdown
## Token Compression Rules
- Use `mcp__token-saver__should_compress` BEFORE reading any file (checks size without reading!)
  - If `needs_conversion=true`: Use MarkItDown first (PDF, DOCX, images)
  - If `recommendation=COMPRESS`: Use ingest_context
  - If `recommendation=DIRECT_READ`: Read file directly
  - If `recommendation=SKIP`: File too small for compression
- Use `mcp__token-saver__ingest_context` for text files >500 tokens
- Use `mcp__token-saver__search_semantic` to find relevant sections
- Use `mcp__token-saver__check_blind_spots` after answering document questions
- Start with ABSTRACT fidelity, expand only as needed
- Target 80%+ token reduction on all large documents
```

---

## Slash Commands

Create these files in your project's `.claude/commands/` directory for quick access to Token Saver workflows.

### `/compress` - Compress a Document

Create `.claude/commands/compress.md`:

```markdown
Compress the following document for efficient token usage.

Steps:
1. Use `mcp__token-saver__ingest_context` with:
   - text: The document content below
   - file_id: Generate a short descriptive ID
2. Use `mcp__token-saver__read_skeleton` to show the compressed structure
3. Report compression stats (original tokens, compressed, ratio)

Document to compress:
$ARGUMENTS
```

### `/search-docs` - Semantic Search

Create `.claude/commands/search-docs.md`:

```markdown
Search compressed documents for relevant information.

Steps:
1. Use `mcp__token-saver__search_semantic` with the query below
2. Show top 5 results with relevance scores
3. Offer to expand any section with more detail

Query: $ARGUMENTS
```

### `/expand` - Expand Section

Create `.claude/commands/expand.md`:

```markdown
Expand a compressed section to full detail.

Steps:
1. Parse the node ID from the arguments
2. Use `mcp__token-saver__modulate_region` with:
   - node_ids: The specified node(s)
   - fidelity: "DETAILED" or "RAW" based on need
3. Display the expanded content

Section to expand: $ARGUMENTS
```

### `/analyze` - Full Document Analysis

Create `.claude/commands/analyze.md`:

```markdown
Perform comprehensive document analysis with token optimization.

Workflow:
1. **Ingest**: Compress the document with `ingest_context`
2. **Structure**: Show skeleton with `read_skeleton`
3. **Key Topics**: Identify main themes from compressed view
4. **Deep Dive**: Search for specific topics user might care about
5. **Blind Spots**: Run `check_blind_spots` to ensure completeness
6. **Summary**: Provide executive summary with token savings report

Document:
$ARGUMENTS
```

### `/remember` - Store Critical Information

Create `.claude/commands/remember.md`:

```markdown
Store critical information in dialogue memory for future reference.

Steps:
1. Use `mcp__token-saver__afm_add_message` with:
   - role: "user"
   - content: The information to remember
2. Confirm storage and explain retention priority

Information to remember: $ARGUMENTS
```

---

## Hooks for Automation

Create these hooks in `.claude/hooks/` to automate Token Saver usage.

### Auto-Compress Large Files

Create `.claude/hooks/auto-compress.yaml`:

```yaml
name: auto-compress-large-files
description: Automatically compress large files when read
trigger:
  event: PostToolUse
  tool: Read
conditions:
  - output_length > 2000
actions:
  - suggest: |
      This file is large ({{output_length}} chars).
      Consider compressing it with Token Saver:
      Use `mcp__token-saver__ingest_context` to reduce token usage by 80-95%.
```

### Blind Spot Check Reminder

Create `.claude/hooks/blind-spot-reminder.yaml`:

```yaml
name: blind-spot-reminder
description: Remind to check blind spots after document queries
trigger:
  event: PostToolUse
  tool: mcp__token-saver__search_semantic
actions:
  - remind: |
      Remember to use `mcp__token-saver__check_blind_spots`
      to verify no critical context was missed.
```

### Context Window Warning

Create `.claude/hooks/context-warning.yaml`:

```yaml
name: context-window-warning
description: Warn when context window is getting full
trigger:
  event: UserPromptSubmit
conditions:
  - context_usage > 0.7
actions:
  - suggest: |
      Context window is {{context_usage}}% full.
      Consider using Token Saver to compress documents:
      - `mcp__token-saver__ingest_context` for new documents
      - `mcp__token-saver__adapt_to_context_window` to fit budget
```

---

## Prompt Examples

### Example 1: Analyze a Research Paper

```
You: I have a 20-page research paper on transformer architectures.
     Can you analyze it and tell me the key innovations?

Claude: I'll compress this paper first to optimize our conversation.

[Uses mcp__token-saver__ingest_context]
Compressed: 8,500 tokens → 425 tokens (95% reduction)

[Uses mcp__token-saver__read_skeleton]
Key sections identified:
- [paper_n0] Abstract: Novel attention mechanism
- [paper_n3] Architecture: Multi-head sparse attention
- [paper_n7] Results: 15% improvement on benchmarks
- [paper_n12] Ablations: Component analysis

The key innovations are:
1. Sparse attention patterns reducing O(n²) to O(n log n)
2. Dynamic head allocation based on input complexity
3. Gradient checkpointing for memory efficiency

[Uses mcp__token-saver__check_blind_spots]
No critical blind spots detected.

Would you like me to expand any section for more detail?
```

### Example 2: Code Review with Compression

```
You: Review this 500-line Python file for issues.

Claude: I'll compress and analyze this code systematically.

[Uses mcp__token-saver__ingest_context with content_type="code"]
Compressed: 2,100 tokens → 210 tokens

[Uses mcp__token-saver__search_semantic query="error handling"]
Found 3 sections with potential issues:
- [code_n5] Missing exception handling in API calls
- [code_n12] Bare except clauses
- [code_n18] No timeout on network requests

[Uses mcp__token-saver__modulate_region for code_n5, fidelity="DETAILED"]
Here's the specific issue at line 45:
```python
response = requests.get(url)  # No error handling!
```

Recommendation: Add try/except with specific exceptions.

Want me to check for other issue categories (security, performance)?
```

### Example 3: Multi-Document Analysis

```
You: I have 5 design documents. Find inconsistencies between them.

Claude: I'll batch ingest all documents for cross-reference analysis.

[Uses mcp__token-saver__batch_ingest_documents]
Ingested 5 documents:
- design_v1: 1,200 → 120 tokens
- design_v2: 1,500 → 150 tokens
- api_spec: 2,000 → 200 tokens
- security_req: 800 → 80 tokens
- deployment: 1,100 → 110 tokens
Total savings: 6,600 → 660 tokens (90%)

[Uses mcp__token-saver__search_semantic query="authentication" across all docs]
Cross-referencing authentication approaches...

Inconsistency found:
- design_v1: "JWT with 24h expiry"
- security_req: "JWT with 1h expiry, refresh tokens"
- api_spec: "Session-based auth" (contradicts both!)

[Uses mcp__token-saver__check_blind_spots]
Additional concern: deployment.md doesn't mention auth at all.

Shall I expand the relevant sections for detailed comparison?
```

### Example 4: Dialogue Memory for Preferences

```
You: I'm vegetarian and allergic to nuts. Remember this for recipe suggestions.

Claude: I'll store this as critical information in dialogue memory.

[Uses mcp__token-saver__afm_add_message]
Stored with CRITICAL priority:
- Vegetarian diet preference
- Nut allergy (safety-critical)

This information will be retained at full fidelity across our entire
conversation and will never be compressed or forgotten.

Later...

You: What should I make for dinner?

Claude: [Uses mcp__token-saver__afm_build_context]
Retrieved your dietary requirements (vegetarian, nut-free).

Here are safe dinner suggestions:
1. Mushroom risotto (nut-free, vegetarian)
2. Spinach and cheese stuffed shells
3. Thai vegetable curry (no peanuts - using coconut)

All suggestions verified against your stored preferences.
```

---

## Available Tools (121 total)

### Document Compression (9 tools)
| Tool | Description |
|------|-------------|
| `ingest_context` | Compress and ingest a document |
| `read_skeleton` | View compressed document structure |
| `modulate_region` | Retrieve sections at variable fidelity |
| `search_semantic` | Find relevant sections by meaning |
| `check_blind_spots` | Detect missed critical context |
| `detect_hallucination` | Verify responses against source |
| `get_stats` | View compression statistics |
| `adapt_to_context_window` | Fit content within token budget |
| `multilevel_encode` | Generate multi-fidelity representations |

### Dialogue Memory - AFM (6 tools)
| Tool | Description |
|------|-------------|
| `afm_add_message` | Add message to dialogue history |
| `afm_build_context` | Build context with adaptive fidelity |
| `afm_get_stats` | View dialogue statistics |
| `afm_clear_history` | Clear dialogue history |
| `afm_export_history` | Export conversation state |
| `afm_import_history` | Import conversation state |

### Agentic Context Engineering - ACE (7 tools)
| Tool | Description |
|------|-------------|
| `ace_create_context` | Create new ACE context |
| `ace_add_bullet` | Add bullet to context |
| `ace_remove_bullet` | Remove bullet from context |
| `ace_get_context` | Retrieve context |
| `ace_reflect` | Trigger reflection cycle |
| `ace_curate` | Curate and deduplicate |
| `ace_generate_summary` | Generate context summary |

### File Sync & Versioning (4 tools)
| Tool | Description |
|------|-------------|
| `refresh_document` | Re-ingest modified file |
| `get_document_versions` | List version history |
| `get_version_diff` | View changes between versions |
| `check_file_staleness` | Check if file needs refresh |

### Batch Processing (1 tool)
| Tool | Description |
|------|-------------|
| `batch_ingest_documents` | Ingest 1-100 documents at once |

### Visualization (4 tools)
| Tool | Description |
|------|-------------|
| `export_graph_json` | Export graph as JSON |
| `visualize_graph_html` | Generate interactive HTML |
| `export_graph_graphml` | Export for Gephi/Cytoscape |
| `explain_compression_decision` | Explain node compression |

### Resource Management (5 tools)
| Tool | Description |
|------|-------------|
| `should_compress` | **Estimate tokens WITHOUT reading file** (call FIRST!). Returns: SKIP/DIRECT_READ/COMPRESS/CONVERT_THEN_COMPRESS. v0.9.2: Detects binary files needing conversion |
| `list_documents` | List all ingested documents |
| `delete_document` | Remove a document |
| `get_resource_usage` | View memory/storage usage |
| `get_health` | Check server health |

---

## Configuration Options

### Option A: Project-Level Configuration (Recommended)

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "token-saver": {
      "command": "token-saver-mcp",
      "args": [],
      "cwd": "/absolute/path/to/token-saver-5000",
      "env": {
        "PYTHONPATH": "/absolute/path/to/token-saver-5000"
      }
    }
  }
}
```

### Option B: User-Level Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "token-saver": {
      "command": "token-saver-mcp",
      "args": [],
      "cwd": "/absolute/path/to/token-saver-5000"
    }
  }
}
```

### Windows Configuration

```json
{
  "mcpServers": {
    "token-saver": {
      "command": "token-saver-mcp",
      "args": [],
      "cwd": "C:/dev/projects/token-saver-5000"
    }
  }
}
```

---

## Troubleshooting

### "MCP server not found"
1. Verify the path in `.mcp.json` is absolute
2. Check Python is in your PATH: `which python`
3. Restart Claude Code completely

### "Module not found: src.server"
1. Ensure you're using the full path to token-saver-5000
2. Add PYTHONPATH to the env config

### Server doesn't start
```bash
cd /path/to/token-saver-5000
token-saver-mcp
# Should hang (waiting for stdio) - this is correct!
```

### Check Claude Code MCP status
```bash
claude mcp list
```

---

## Quick Reference Card

| Action | Command/Tool |
|--------|--------------|
| **Check file size first** | `mcp__token-saver__should_compress` ⚡ |
| Compress document | `mcp__token-saver__ingest_context` |
| View structure | `mcp__token-saver__read_skeleton` |
| Search content | `mcp__token-saver__search_semantic` |
| Expand section | `mcp__token-saver__modulate_region` |
| Check coverage | `mcp__token-saver__check_blind_spots` |
| Store preference | `mcp__token-saver__afm_add_message` |
| Batch ingest | `mcp__token-saver__batch_ingest_documents` |

**Fidelity Levels:** ABSTRACT → STRUCTURE → BALANCED → DETAILED → RAW

**Expected Savings:** 80-95% on documents, 48-66% on dialogues

---

## Next Steps

- [MCP Tools Guide](MCP_TOOLS_GUIDE.md) - Detailed tool documentation
- [How It Works](HOW_IT_WORKS.md) - Technical explanation
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
