# Using Token Saver 5000 with Claude Code

This guide shows you how to set up Token Saver 5000 as an MCP server for Claude Code (the CLI tool).

## Quick Start (2 minutes)

### Step 1: Clone and Install

```bash
git clone https://github.com/oimiragieo/token-saver-5000.git
cd token-saver-5000
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python scripts/check_setup.py
```

You should see all checks passing.

### Step 3: Configure Claude Code

Create or edit `.mcp.json` in your project root:

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

**Important:** Replace `/path/to/token-saver-5000` with the actual absolute path.

### Step 4: Restart Claude Code

```bash
# Exit and restart Claude Code
claude
```

The Token Saver tools will now be available!

---

## Configuration Options

### Option A: Project-Level Configuration (Recommended)

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "token-saver": {
      "command": "python",
      "args": ["-m", "src.server"],
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
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/absolute/path/to/token-saver-5000"
    }
  }
}
```

### Windows Configuration

For Windows, use forward slashes or escaped backslashes:

```json
{
  "mcpServers": {
    "token-saver": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "C:/dev/projects/token-saver-5000"
    }
  }
}
```

---

## Available Tools (35 total)

Once configured, you'll have access to these MCP tools:

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

### Resource Management (4 tools)
| Tool | Description |
|------|-------------|
| `list_documents` | List all ingested documents |
| `delete_document` | Remove a document |
| `get_resource_usage` | View memory/storage usage |
| `get_health` | Check server health |

---

## Example Usage in Claude Code

### Compressing a Document

```
You: I have a long technical document. Can you compress it?

Claude: I'll use the ingest_context tool to compress your document.
[Uses ingest_context tool]

The document has been compressed:
- Original: 5,234 tokens
- Compressed: 523 tokens
- Compression ratio: 10x
- Token savings: 90%
```

### Searching Compressed Content

```
You: Find sections about "error handling" in the compressed document.

Claude: I'll search the compressed document semantically.
[Uses search_semantic tool]

Found 3 relevant sections:
1. [doc_n5] Error handling patterns (score: 0.89)
2. [doc_n12] Exception management (score: 0.76)
3. [doc_n8] Retry mechanisms (score: 0.71)
```

### Dialogue Memory

```
You: Remember that I have a severe peanut allergy.

Claude: I've noted your peanut allergy using AFM.
[Uses afm_add_message tool]

Your allergy information is stored as CRITICAL priority
and will be retained across all future messages.
```

---

## Troubleshooting

### "MCP server not found"

1. Verify the path in `.mcp.json` is absolute
2. Check Python is in your PATH: `which python`
3. Restart Claude Code completely

### "Module not found: src.server"

1. Ensure you're using the full path to token-saver-5000
2. Add PYTHONPATH to the env config:
   ```json
   "env": {
     "PYTHONPATH": "/path/to/token-saver-5000"
   }
   ```

### Server doesn't start

Test the server manually:
```bash
cd /path/to/token-saver-5000
python -m src.server
```

If it hangs (waiting for stdio input), the server is working correctly.

### Check Claude Code MCP status

```bash
claude mcp list
```

---

## Performance Tips

1. **Large documents**: The system works best on documents 500+ tokens
2. **Batch processing**: Use `batch_ingest_documents` for multiple files
3. **Memory**: First load takes ~5 seconds (model warm-up), subsequent calls are faster
4. **Fidelity levels**: Start with ABSTRACT, expand to RAW only when needed

---

## Comparison: Claude Code vs Claude Desktop

| Feature | Claude Code | Claude Desktop |
|---------|-------------|----------------|
| Config file | `.mcp.json` | `claude_desktop_config.json` |
| Config location | Project root or `~/.claude/` | `~/Library/Application Support/Claude/` (macOS) |
| Tools visible | `/mcp` command | Tools panel |
| Best for | Development, automation | Interactive use |

---

## Next Steps

- [MCP Tools Guide](MCP_TOOLS_GUIDE.md) - Detailed tool documentation
- [How It Works](HOW_IT_WORKS.md) - Technical explanation
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues

---

## Quick Reference

**Minimum config:**
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

**Verify installation:**
```bash
cd /path/to/token-saver-5000
python scripts/check_setup.py
```

**Test server:**
```bash
python -m src.server
# Should hang (waiting for stdio) - this is correct!
# Press Ctrl+C to exit
```
