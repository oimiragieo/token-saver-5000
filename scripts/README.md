# Utility Scripts

This directory contains utility scripts for Token Saver 5000.

## Scripts

### `check_setup.py`

**Purpose:** Verify installation and check that all dependencies are correctly installed.

**Usage:**
```bash
python scripts/check_setup.py
```

**What it checks:**
- Python version (>= 3.10)
- Required packages (mcp, sentence-transformers, chromadb, etc.)
- Model availability (all-MiniLM-L6-v2)
- Storage directories (data/, config/)
- Import tests for all modules

**Expected output:**
```
🎉 All checks passed! Token Saver 5000 is ready to use.
```

**If checks fail:** See [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)

---

### `benchmark.py`

**Purpose:** Run performance benchmarks on document compression and AFM.

**Usage:**
```bash
python scripts/benchmark.py
```

**What it tests:**
- Compression ratios on various document sizes
- Processing time benchmarks
- Memory usage tracking
- AFM dialogue compression efficiency
- SSIM quality scores

**Output:**
```
=== Document Compression Benchmarks ===
Small (5K):    6.3× compression in 0.8s
Medium (20K):  16.7× compression in 1.9s
Large (45K):   19.5× compression in 3.2s

=== AFM Dialogue Benchmarks ===
Short (3 turns):  60% savings in 0.1s
Medium (9 turns): 67% savings in 0.3s
Long (20 turns):  67% savings in 0.5s
```

---

### `test_simulation.py`

**Purpose:** Run end-to-end simulation tests without requiring pytest.

**Usage:**
```bash
python scripts/test_simulation.py
```

**What it tests:**
- Full ingestion workflow
- Semantic search accuracy
- Fidelity modulation
- Blind spot detection
- AFM dialogue management
- Persistence (save/load)

**Use case:** Quick smoke test without full test suite.

---

### `install_mcp.sh`

**Purpose:** Automated MCP server installation for Claude Desktop.

**Usage:**
```bash
chmod +x scripts/install_mcp.sh
./scripts/install_mcp.sh
```

**What it does:**
1. Auto-detects your operating system (macOS, Windows, Linux)
2. Locates Claude Desktop config file
3. Backs up existing config
4. Merges MCP server configuration
5. Validates config syntax

**Supported platforms:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/claude/claude_desktop_config.json`

**Manual alternative:** See [GETTING_STARTED.md](../GETTING_STARTED.md#configure-claude-desktop)

---

## Development Scripts

These scripts are primarily for development and testing. For normal usage, see:

- [GETTING_STARTED.md](../GETTING_STARTED.md) - Installation guide
- [QUICKSTART.md](../QUICKSTART.md) - Quick reference
- [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) - Common issues

---

## Adding New Scripts

If you create new utility scripts, add them to this directory and document them here:

```markdown
### `your_script.py`

**Purpose:** Brief description

**Usage:**
```bash
python scripts/your_script.py [args]
```

**What it does:**
- Bullet points
```

---

**Last Updated:** 2024-11-22
