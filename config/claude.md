# config/ Directory

## Overview
Configuration templates for integrating the Semantic Modulator MCP server with AI platforms like Claude Desktop.

## Files

### 1. **`claude_desktop_config.example.json`** (268 bytes)
**Purpose**: Example configuration for Claude Desktop MCP integration

**File Content**:
```json
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/token-saver-5000"
    }
  }
}
```

**Configuration Fields**:

- **`mcpServers`**: Object containing MCP server definitions
  - Key: Server name (can be any identifier)
  - Value: Server configuration object

- **`command`**: Executable to run
  - `"python"` or `"python3"` depending on system
  - Could also be `"uv"` if using uv package manager

- **`args`**: Command-line arguments
  - `["-m", "src.server"]`: Run src/server.py as module
  - Alternative: Full path to server.py

- **`cwd`**: Working directory
  - Must be absolute path to token-saver-5000 root
  - Replace `/path/to/token-saver-5000` with actual path

---

## Usage Instructions

### Step 1: Locate Claude Desktop Config

**macOS**:
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows**:
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux**:
```bash
~/.config/Claude/claude_desktop_config.json
```

### Step 2: Copy Example Config

```bash
cp config/claude_desktop_config.example.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

### Step 3: Update Paths

Edit the config file and replace:
```json
"cwd": "/path/to/token-saver-5000"
```

With your actual path, e.g.:
```json
"cwd": "/Users/username/projects/token-saver-5000"
```

### Step 4: Verify Python Command

Check which Python to use:
```bash
which python3
# or
which python
```

If using virtual environment:
```json
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "/Users/username/projects/token-saver-5000/.venv/bin/python",
      "args": ["-m", "src.server"],
      "cwd": "/Users/username/projects/token-saver-5000"
    }
  }
}
```

### Step 5: Restart Claude Desktop

Close and reopen Claude Desktop to load the new configuration.

### Step 6: Verify Connection

In Claude Desktop, check for MCP tools:
- `ingest_context`
- `read_skeleton`
- `modulate_region`
- `search_semantic`
- `analyze_blind_spots`
- etc.

---

## Alternative Configurations

### Using UV Package Manager
```json
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "uv",
      "args": ["run", "src/server.py"],
      "cwd": "/path/to/token-saver-5000"
    }
  }
}
```

### Direct Script Execution
```json
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "/usr/bin/python3",
      "args": ["/path/to/token-saver-5000/src/server.py"],
      "cwd": "/path/to/token-saver-5000"
    }
  }
}
```

### Multiple MCP Servers
```json
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "python3",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/token-saver-5000"
    },
    "other-mcp-server": {
      "command": "node",
      "args": ["server.js"],
      "cwd": "/path/to/other-server"
    }
  }
}
```

---

## Troubleshooting

### Issue: Claude Desktop doesn't show MCP tools
**Causes**:
1. Config file path incorrect
2. JSON syntax error
3. Python path incorrect
4. Dependencies not installed

**Debug Steps**:
```bash
# 1. Validate JSON syntax
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | python3 -m json.tool

# 2. Test server manually
cd /path/to/token-saver-5000
python3 -m src.server

# 3. Check dependencies
pip install -r requirements.txt
```

### Issue: Server starts but tools don't work
**Causes**:
1. Dependencies missing
2. Model download failed
3. Permissions issue

**Debug Steps**:
```bash
# Check dependencies
pip list | grep -E "mcp|sentence-transformers|networkx"

# Test compressor manually
python3 -c "from src.semantic_compressor import SemanticCompressor; c = SemanticCompressor()"
```

### Issue: "Module not found" error
**Fix**: Ensure `cwd` points to project root (where `src/` directory is)

### Issue: Permission denied
**Fix**: Make sure Python has execute permissions:
```bash
chmod +x /path/to/python3
```

---

## Environment Variables

Optional environment variables for configuration:

```bash
# Model cache directory (default: ~/.cache/huggingface/)
export TRANSFORMERS_CACHE=/path/to/cache

# MCP logging level
export MCP_LOG_LEVEL=INFO

# Embedding model override
export SEMANTIC_MODEL=all-MiniLM-L6-v2
```

Add to config:
```json
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "python3",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/token-saver-5000",
      "env": {
        "TRANSFORMERS_CACHE": "/custom/cache/path",
        "MCP_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

---

## Performance Tuning

### CPU-Only Mode (Faster startup)
```json
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "python3",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/token-saver-5000",
      "env": {
        "CUDA_VISIBLE_DEVICES": ""
      }
    }
  }
}
```

### Reduce Model Memory (Lower quality)
Edit `src/server.py`:
```python
self.compressor = SemanticCompressor(
    model_name="all-MiniLM-L12-v2",  # Smaller model
    ...
)
```

---

## Security Considerations

1. **File Permissions**: Config file contains paths
   ```bash
   chmod 600 ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. **No Credentials**: This config doesn't contain secrets (MCP uses stdio)

3. **Local Only**: Server runs locally, no network exposure

---

## Related Documentation

- Claude Desktop MCP Guide: https://docs.claude.com/
- MCP Protocol Spec: https://modelcontextprotocol.io/
- Server Implementation: `src/server.py`

---

## Future Configurations

**Planned**:
- Docker container config
- Systemd service config
- API server mode (HTTP endpoint)
- Configuration for other AI platforms
