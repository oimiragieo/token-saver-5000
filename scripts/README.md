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

### `benchmarks/run_benchmarks.py`

**Purpose:** Run a fixed-corpus benchmark with golden thresholds to detect token-savings regressions.

**Usage:**
```bash
python scripts/benchmarks/run_benchmarks.py
python scripts/benchmarks/run_benchmarks.py --case medium_architecture
python scripts/benchmarks/run_benchmarks.py --output artifacts/benchmarks/baseline.json
python scripts/benchmarks/run_benchmarks.py --mode query_guided
python scripts/benchmarks/run_benchmarks.py --compare baseline,query_guided,evidence_aware
```

**What it tests:**
- Compression ratio per benchmark case
- Token savings percentage per benchmark case
- Pass/fail against minimum expected thresholds

**Output:**
- Console summary with pass/fail per case
- JSON report written to `artifacts/benchmarks/latest.json` by default

---

### `benchmarks/check_benchmark_guard.py`

**Purpose:** Validate generated benchmark reports against CI threshold config.

**Usage:**
```bash
python scripts/benchmarks/check_benchmark_guard.py
python scripts/benchmarks/check_benchmark_guard.py --summary-file artifacts/benchmarks/guard_summary.md
```

**Config:**
- Threshold file: `artifacts/benchmarks/golden_thresholds.json`
- Reports directory: `artifacts/benchmarks/`

---

### `skills/profile_tokens.py`

**Purpose:** Quick token profile for raw input vs compressed skeleton output.

**Usage:**
```bash
python scripts/skills/profile_tokens.py --file path/to/context.txt
python scripts/skills/profile_tokens.py --text "inline text"
```

---

### `skills/compress_context.py`

**Purpose:** Generate compressed context in baseline, query-guided, or evidence-aware mode.

**Usage:**
```bash
python scripts/skills/compress_context.py --file path/to/context.txt --mode baseline
python scripts/skills/compress_context.py --file path/to/context.txt --mode query_guided --query "auth flow"
python scripts/skills/compress_context.py --file path/to/context.txt --mode evidence_aware --query "auth flow" --min-similarity 0.4
```

---

### `skills/validate_evidence.py`

**Purpose:** Validate whether query evidence is sufficient before final answer generation.

**Usage:**
```bash
python scripts/skills/validate_evidence.py --file path/to/context.txt --query "auth flow"
```

**Exit codes:**
- `0`: evidence sufficient
- `1`: evidence insufficient

---

### `skills/run_skill_workflow.py`

**Purpose:** Run profile + compression + evidence validation in one workflow command.

**Usage:**
```bash
python scripts/skills/run_skill_workflow.py --file path/to/context.txt --query "auth flow" --mode evidence_aware
```

**Notes:**
- Emits one JSON payload containing `profile`, `compressed`, and `evidence_validation`.
- Use `--fail-on-insufficient-evidence` to return non-zero on insufficient evidence.

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
