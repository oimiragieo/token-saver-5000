# Deployment Guide - Token Saver 5000 MCP Server

Complete guide for deploying and configuring the Semantic Modulator MCP Server for production use.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [Verification](#verification)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)
- [Upgrading](#upgrading)

---

## Prerequisites

### System Requirements

- **Python**: 3.10, 3.11, 3.12, 3.13, or 3.14 (chromadb only works on 3.10-3.12)
- **Disk Space**: ~100MB for embedding models (first-time setup)
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Claude Desktop** or **Claude Code CLI**: Latest version

### Python Dependencies

The server automatically installs all required dependencies:

```bash
# Core dependencies
sentence-transformers>=2.2.0  # Semantic embeddings
networkx>=3.0                  # Graph operations
mcp>=1.0.0                     # Model Context Protocol

# Optional (auto-fallback if unavailable)
chromadb>=0.4.0               # Vector storage (uses JSON fallback)
tiktoken                      # Token counting (uses fallback)
```

**Note**: ChromaDB is optional. If unavailable, the server automatically uses JSON-based storage with no functionality loss.

---

## Quick Start

### One-Command Setup (Recommended)

The fastest way to get started:

```bash
# Clone the repository
git clone https://github.com/oimiragieo/token-saver-5000.git
cd token-saver-5000

# Run automated setup
python scripts/quickstart.py
```

This script:
1. ✅ Installs all dependencies
2. ✅ Downloads embedding models (~100MB)
3. ✅ Runs test suite (3,409+ tests)
4. ✅ Runs demo proof (7.9× compression verified)
5. ✅ Provides configuration instructions

**Estimated time**: 2-3 minutes (depending on internet speed for model download)

---

## Installation Methods

### Method 1: Tool-Style Local Installation (Recommended for Most Users)

```bash
# 1. Clone repository
git clone https://github.com/oimiragieo/token-saver-5000.git
cd token-saver-5000

# 2. Install the CLI/MCP tools
uv tool install -e .

# 3. Apply the recommended MCP config
token-saver-setup --auto

# 4. Optional deep verification
python scripts/check_setup.py
```

Fallback with `pipx`:

```bash
pipx install .
```

Fallback for development/editable installs:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Configuration

### Claude Desktop Configuration

Claude Desktop uses stdio transport for local MCP servers.

#### Configuration File Location

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

#### Setup Steps

1. **Open Configuration File**:
   - **Via Claude Desktop**: Settings → Developer → Edit Config
   - **Manually**: Navigate to the path above and open in text editor

2. **Add Server Configuration**:

```json
{
  "mcpServers": {
    "token-saver": {
      "command": "token-saver-mcp",
      "args": [],
      "cwd": "/absolute/path/to/token-saver-5000",
      "env": {}
    }
  }
}
```

**Simpler option**: run `token-saver-setup --desktop` or `token-saver-setup --auto` instead of editing this by hand.

**Example Paths**:
- macOS/Linux: `/Users/username/projects/token-saver-5000`
- Windows: `C:\\Users\\username\\projects\\token-saver-5000`

3. **Save and Restart**:
   - Save the configuration file
   - Completely quit and restart Claude Desktop

#### Multiple MCP Servers

If you already have other MCP servers configured:

```json
{
  "mcpServers": {
    "github": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"]
    },
    "token-saver": {
      "command": "token-saver-mcp",
      "args": [],
      "cwd": "/absolute/path/to/token-saver-5000"
    }
  }
}
```

### Claude Code CLI Configuration

Claude Code CLI offers more flexible configuration scoping.

#### Method 1: Using claude mcp add-json (Recommended)

```bash
# Navigate to your project directory
cd /path/to/your-project

# Add the server (project-scoped by default)
claude mcp add-json token-saver '{
  "command": "token-saver-mcp",
  "args": [],
  "cwd": "/absolute/path/to/token-saver-5000"
}'

# Verify
claude mcp list
```

#### Method 2: Manual .mcp.json Edit

Create or edit `.mcp.json` in your project directory:

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

Recommended guided setup:

```bash
token-saver-setup --portable-project
```

Or generate the project-scoped `.claude/.mcp.json` automatically:

```bash
token-saver-install-mcp --project-config
```

For a team-shared config that uses `${workspaceFolder}` instead of a machine-specific absolute path:

```bash
token-saver-install-mcp --portable-project-config
```

To inspect command availability plus desktop/project MCP wiring:

```bash
token-saver-install-mcp --doctor --human
```

#### Configuration Scopes

Claude Code supports three configuration scopes:

- **Local** (default): `.mcp.json` - Current project only, not committed to git
- **Project** (`-s project`): `.mcp.json` - Shared via git, team-wide
- **User** (`-s user`): Global config - Available across all projects

**Recommendation**: Use **local** scope for personal development, **project** scope for team projects.

```bash
# User-scoped (all projects)
claude mcp add-json -s user token-saver '{...}'

# Project-scoped (shared via git)
claude mcp add-json -s project token-saver '{...}'

# Add .mcp.json to .gitignore for local scope
echo ".mcp.json" >> .gitignore
```

---

## Verification

### 1. Check Server Status

**Claude Desktop**:
- Look for "Token Saver 5000" in the MCP servers section
- Check Claude Desktop logs: `~/Library/Logs/Claude/` (macOS) or `%APPDATA%\Claude\logs\` (Windows)

**Claude Code CLI**:
```bash
claude mcp list           # List all configured servers
claude mcp get token-saver       # Check specific server config
```

### 2. Test MCP Tools

Start a conversation with Claude and test the available tools:

```plaintext
You: List available MCP tools for token-saver-5000

Claude should show 30 tools including:
- ingest_context
- read_skeleton
- modulate_region
- search_semantic
- check_resource_health
...and 25 more
```

### 3. Run Integration Test

Test the core functionality:

```plaintext
You: Ingest this document into token-saver-5000:
[Paste a medium-length document, 500+ words]

Claude will:
1. Use ingest_context tool
2. Return compression stats
3. Show semantic graph structure

You: Now give me a skeleton summary

Claude will:
1. Use read_skeleton tool
2. Return compressed overview (~80-90% reduction)
```

### 4. Verify Health Status

```plaintext
You: Check the health of token-saver-5000

Claude will use check_resource_health tool and show:
- Resource usage (documents, memory, disk)
- System status (healthy/warning/critical)
- Performance metrics
```

---

## Performance Tuning

### Memory Management

The server includes automatic memory management:

**File Sync Metadata** (v0.4.2):
- LRU eviction: 1000 entries max (~170KB total)
- Automatic cleanup of oldest files

**ACE Contexts** (v0.4.2):
- LRU eviction: 100 contexts max (~7MB total)
- Automatic cleanup of least-recently-used contexts

**Version History** (v0.4.2):
- Automatic pruning: 10 versions per document
- 880KB saved per 50 → 10 version pruning

### Resource Limits

Default limits (configured in `src/resource_manager.py`):

```python
ResourceLimits(
    max_document_size_mb=100.0,    # Per-document limit
    max_total_storage_mb=1024.0,   # Total storage limit
    max_documents=1000,            # Document count limit
    max_memory_mb=2048.0,          # Memory usage limit
)
```

To monitor resource usage:
```plaintext
You: Check resource health for token-saver-5000
```

### Disk Space

**Storage Location**: `.semantic_modulator_data/` (in server directory)

```bash
# Check current disk usage
du -sh token-saver-5000/.semantic_modulator_data

# Clean old data (if needed)
rm -rf token-saver-5000/.semantic_modulator_data/versions
rm -rf token-saver-5000/.semantic_modulator_data/persistent_storage
```

**Typical Usage**:
- Empty: ~10KB (metadata only)
- 10 documents: ~50MB (with embeddings)
- 100 documents: ~500MB (with full version history)

### Embedding Model Cache

**Location**: `~/.cache/huggingface/` (automatic)

**Size**: ~100MB (one-time download for `all-MiniLM-L6-v2`)

```bash
# Check model cache size
du -sh ~/.cache/huggingface

# Clear cache (will re-download on next use)
rm -rf ~/.cache/huggingface
```

---

## Troubleshooting

### Server Not Starting

**Symptom**: Claude shows "Server failed to start" or no tools available

**Solutions**:

1. **Check Python Version**:
   ```bash
   python --version  # Should be 3.10-3.14
   ```

2. **Verify Installation**:
   ```bash
   cd /path/to/token-saver-5000
   python scripts/check_setup.py
   ```

3. **Test Server Manually**:
   ```bash
   token-saver-mcp
   # Should output: "🚀 Starting Semantic Modulator MCP Server"
   # Press Ctrl+C to stop
   ```

4. **Check Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Review Logs**:
   - **Claude Desktop**: `~/Library/Logs/Claude/mcp-server-token-saver-5000.log` (macOS)
   - **Windows**: `%APPDATA%\Claude\logs\mcp-server-token-saver-5000.log`

### Configuration Issues

**Symptom**: Server appears in list but tools don't work

**Solutions**:

1. **Validate JSON Syntax**:
   - Use a JSON validator: https://jsonlint.com/
   - Common error: Missing comma between server entries

2. **Check Absolute Path**:
   - Path must be absolute, not relative
   - Windows: Use double backslashes `C:\\Users\\...` or forward slashes `C:/Users/...`
   - macOS/Linux: Use full path `/Users/...` not `~/...`

3. **Verify Working Directory**:
   ```json
   "cwd": "/absolute/path/to/token-saver-5000"  # MUST be absolute
   ```

### Permission Errors

**Symptom**: "Access denied" or "Permission error" in logs

**Solutions**:

1. **Path Validation (v0.6.1 Security Feature)**:
   - File paths must be within allowed directories (CWD + user home)
   - Error message will show allowed directories
   - Use absolute paths, not `../` sequences

2. **File System Permissions**:
   ```bash
   # Check directory permissions
   ls -la /path/to/token-saver-5000

   # Fix if needed (macOS/Linux)
   chmod -R u+rw /path/to/token-saver-5000
   ```

### Model Download Issues

**Symptom**: "Failed to download model" or slow first startup

**Solutions**:

1. **Manual Download**:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

2. **Check Internet Connection**:
   - Model downloads from huggingface.co
   - ~100MB download size

3. **Firewall/Proxy**:
   - Ensure huggingface.co is accessible
   - Configure proxy if needed

### ChromaDB Issues (Optional Dependency)

**Symptom**: ChromaDB import warnings (non-critical)

**Background**: ChromaDB is optional. The server automatically uses JSON fallback.

**Solutions**:

1. **Ignore Warning** (Recommended):
   - JSON fallback works perfectly
   - No functionality loss

2. **Install ChromaDB** (Optional):
   ```bash
   pip install 'chromadb>=0.4.0'
   ```

   **Note**: Requires numpy<2.0, which conflicts with Python 3.13+

---

## Security Considerations

### Path Traversal Prevention (v0.6.1)

**Security Feature**: All file paths are validated to prevent path traversal attacks (CWE-22).

**Allowed Directories**:
- Current working directory (where server runs)
- User home directory

**Blocked Paths**:
- `../../etc/passwd` - Path traversal
- `/etc/passwd` - Outside allowed directories
- Symbolic links to restricted locations

**Error Messages**:
```
ValueError: Access denied: file_path must be within allowed directories.
Allowed directories: ['/path/to/server', '/home/user']
Attempted path: /etc/passwd
```

### Data Privacy

**Local-Only Processing**:
- ✅ All semantic processing is local (no external API calls)
- ✅ Uses `all-MiniLM-L6-v2` model (local embeddings)
- ✅ No network requests (except initial model download)
- ✅ No telemetry or usage tracking

**Data Storage**:
- Location: `.semantic_modulator_data/` (local directory)
- Format: JSON files + embeddings
- Encryption: Not encrypted at rest (use disk encryption if needed)

### Best Practices

1. **Run as Non-Root**: Never run as administrator/root
2. **Limit File Access**: Use principle of least privilege
3. **Review Documents**: Don't ingest sensitive documents without review
4. **Regular Updates**: Keep server updated for security fixes
5. **Monitor Logs**: Review logs for unexpected behavior

---

## Upgrading

### Upgrade Process

```bash
# 1. Backup data (optional)
cp -r .semantic_modulator_data .semantic_modulator_data.backup

# 2. Pull latest code
cd /path/to/token-saver-5000
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt --upgrade

# 4. Run tests
pytest tests/

# 5. Restart Claude Desktop/Claude Code
```

### Version Compatibility

The server maintains backward compatibility for:
- ✅ Persisted documents (`.semantic_modulator_data/persistent_storage/`)
- ✅ File sync metadata (`.semantic_modulator_data/file_sync/`)
- ✅ Version history (`.semantic_modulator_data/versions/`)

**Breaking Changes** (if any) are documented in:
- `CHANGELOG.md`
- `SECURITY.md` (for security-related changes)
- Release notes on GitHub

### Rollback

If issues occur after upgrade:

```bash
# 1. Checkout previous version
git checkout <previous-tag>  # e.g., v0.6.1

# 2. Reinstall dependencies
pip install -r requirements.txt

# 3. Restore backup (if needed)
rm -rf .semantic_modulator_data
mv .semantic_modulator_data.backup .semantic_modulator_data

# 4. Restart Claude Desktop/Claude Code
```

---

## Advanced Configuration

### Custom Allowed Directories (v0.6.1+)

To allow file access beyond CWD and home directory, modify `src/server.py`:

```python
# In SemanticModulatorServer.__init__() (line 172)
allowed_dirs = [
    os.getcwd(),
    os.path.expanduser("~"),
    "/custom/project/path",  # Add custom paths here
]
self.path_validator = PathValidator(allowed_base_dirs=allowed_dirs)
```

**Security Warning**: Only add trusted directories. Path traversal protection applies within these directories.

### Embedding Model Customization

To use a different Sentence-BERT model:

```python
# In src/server.py (line 128)
self.compressor = SemanticCompressor(
    model_name="all-mpnet-base-v2",  # Alternative: better quality, slower
    # model_name="all-MiniLM-L6-v2",  # Default: fast, good quality
    similarity_threshold=0.75,
    skeleton_ratio=0.2,
)
```

**Available Models**:
- `all-MiniLM-L6-v2` (default): Fast, 384-dim embeddings, ~100MB
- `all-mpnet-base-v2`: Better quality, 768-dim embeddings, ~420MB
- `paraphrase-multilingual-MiniLM-L12-v2`: Multilingual support, ~420MB

### Resource Limit Tuning

Modify limits in `src/server.py` (line 154):

```python
self.resource_manager = ResourceManager(
    ResourceLimits(
        max_document_size_mb=200.0,   # Increase for large documents
        max_total_storage_mb=2048.0,  # Increase for more documents
        max_documents=2000,           # Increase document limit
        max_memory_mb=4096.0,         # Increase memory limit
    )
)
```

---

## Support

### Getting Help

- **Documentation**: See `README.md`, `SECURITY.md`, `CLAUDE.md`
- **Issues**: https://github.com/oimiragieo/token-saver-5000/issues
- **Discussions**: https://github.com/oimiragieo/token-saver-5000/discussions

### Reporting Issues

When reporting issues, include:

1. **Environment**:
   - OS and version (macOS/Windows/Linux)
   - Python version (`python --version`)
   - Claude Desktop/Code CLI version

2. **Configuration**:
   ```bash
   # Sanitize paths/secrets before sharing
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

3. **Logs**:
   - Claude Desktop: `~/Library/Logs/Claude/mcp-server-token-saver-5000.log`
   - Server output: `token-saver-mcp` (capture output)

4. **Reproduction Steps**:
   - Minimal example to reproduce the issue
   - Expected vs actual behavior

---

## Version History

- **v0.6.1** (2025-01-26): Security hardening - Path traversal fix (CWE-22)
- **v0.4.3**: Testing improvements, 427 comprehensive tests
- **v0.4.2**: Memory management (LRU eviction, version pruning)
- **v0.4.0**: File sync and version management
- **v0.3.0**: ACE Framework, 30 MCP tools

See `CHANGELOG.md` for detailed release notes.

---

**Last Updated**: 2025-01-26 (v0.6.1)
**Status**: ✅ Production-Ready (Beta Quality)
