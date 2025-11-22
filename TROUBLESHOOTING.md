# Troubleshooting Guide

**Common issues and solutions for Token Saver 5000**

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Model Download Issues](#model-download-issues)
- [MCP Server Issues](#mcp-server-issues)
- [Compression Issues](#compression-issues)
- [Performance Issues](#performance-issues)
- [Storage & Resource Issues](#storage--resource-issues)
- [AFM (Dialogue Memory) Issues](#afm-dialogue-memory-issues)
- [Testing Issues](#testing-issues)
- [Platform-Specific Issues](#platform-specific-issues)

---

## Installation Issues

### Issue: `pip install -r requirements.txt` fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0
```

**Solutions:**

**1. Check Python version:**
```bash
python --version  # Must be 3.10 or higher
```

If < 3.10, upgrade Python:
```bash
# macOS (Homebrew)
brew install python@3.11

# Linux (Ubuntu/Debian)
sudo apt update && sudo apt install python3.11

# Windows
# Download from python.org
```

**2. Use virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Install PyTorch separately (if GPU issues):**
```bash
# CPU-only (smaller, faster install)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install remaining dependencies
pip install -r requirements.txt
```

**4. Install dependencies one by one:**
```bash
# Identify which package is failing
pip install mcp>=0.9.0
pip install sentence-transformers>=2.2.0
pip install networkx>=3.0
# ... etc
```

---

### Issue: `ModuleNotFoundError: No module named 'mcp'`

**Symptoms:**
```
python -m src.server
ModuleNotFoundError: No module named 'mcp'
```

**Solutions:**

**1. Verify virtual environment is activated:**
```bash
which python  # Should show .venv/bin/python
pip list | grep mcp  # Should show mcp>=0.9.0
```

**2. Reinstall MCP:**
```bash
pip uninstall mcp
pip install mcp>=0.9.0
```

**3. Check Python path:**
```bash
python -c "import sys; print('\n'.join(sys.path))"
# Should include your project directory
```

**4. Set PYTHONPATH explicitly:**
```bash
export PYTHONPATH=/path/to/token-saver-5000:$PYTHONPATH
python -m src.server
```

---

### Issue: `check_setup.py` fails with missing dependencies

**Symptoms:**
```
python scripts/check_setup.py
❌ sentence-transformers not found
```

**Solution:**

Run the setup checker to see what's missing:
```bash
python scripts/check_setup.py
```

Install each missing dependency:
```bash
pip install sentence-transformers
pip install chromadb
pip install tiktoken
```

**If ChromaDB fails to install:**
```bash
# ChromaDB has C++ dependencies
# On Linux:
sudo apt-get install build-essential

# On macOS:
xcode-select --install

# Then retry
pip install chromadb
```

**Fallback:** The system will use JSON storage if ChromaDB is unavailable (slower but functional).

---

## Model Download Issues

### Issue: `sentence-transformers` model download hangs or fails

**Symptoms:**
```
Downloading all-MiniLM-L6-v2...
[Hangs indefinitely]
```

**Solutions:**

**1. Check internet connection:**
```bash
ping huggingface.co
```

**2. Download manually:**
```python
from sentence_transformers import SentenceTransformer

# This triggers the download
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Download complete!")
```

**3. Use offline mode (if model already downloaded):**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='~/.cache/torch')
```

**4. Use proxy if needed:**
```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
python scripts/check_setup.py
```

**5. Download to custom location:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    cache_folder='./models'  # Local directory
)
```

**Expected model size:** ~80MB (once downloaded, no further downloads needed)

---

### Issue: `tiktoken` encoding errors

**Symptoms:**
```
ValueError: Could not find encoding 'cl100k_base'
```

**Solutions:**

**1. Update tiktoken:**
```bash
pip install --upgrade tiktoken
```

**2. Manually download encoding:**
```python
import tiktoken
tiktoken.get_encoding("cl100k_base")  # Triggers download
```

**3. Use fallback word count:**
```python
# System automatically falls back to word count × 1.3 approximation
# if tiktoken unavailable (less accurate but functional)
```

---

## MCP Server Issues

### Issue: MCP server won't start

**Symptoms:**
```
python -m src.server
[No output, process hangs]
```

**Solutions:**

**1. Check for port conflicts:**
```bash
# MCP uses stdio, not TCP ports, but check anyway
lsof -i :8080  # Should be empty
```

**2. Run with verbose logging:**
```bash
python -m src.server 2>&1 | tee server.log
```

**3. Check for initialization errors:**
```python
# Add debug logging to src/server.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**4. Verify imports work:**
```bash
python -c "from src.server import SemanticModulatorServer; print('OK')"
```

**5. Check resource availability:**
```bash
# Ensure enough RAM (needs ~1-2GB)
free -h  # Linux
vm_stat  # macOS

# Ensure enough disk space (needs ~2GB)
df -h
```

---

### Issue: Claude Desktop doesn't see the MCP server

**Symptoms:**
- MCP server tools don't appear in Claude Desktop
- Claude Desktop shows "No MCP servers configured"

**Solutions:**

**1. Verify config file location:**

```bash
# macOS
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows
type %APPDATA%\Claude\claude_desktop_config.json

# Linux
cat ~/.config/claude/claude_desktop_config.json
```

**2. Check config syntax:**

```json
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/ABSOLUTE/PATH/to/token-saver-5000",
      "env": {}
    }
  }
}
```

**Common mistakes:**
- ❌ Relative path in `cwd` (must be absolute)
- ❌ Missing comma after `semantic-modulator` block
- ❌ Wrong Python command (`python3` vs `python`)
- ❌ Incorrect file path (spaces not escaped)

**3. Test Python command manually:**
```bash
cd /path/to/token-saver-5000
python -m src.server
# Should start without errors
```

**4. Use the automated installer:**
```bash
chmod +x scripts/install_mcp.sh
./install_mcp.sh
```

This script auto-detects your OS and config location.

**5. Restart Claude Desktop:**
- Quit Claude Desktop completely (not just close window)
- Relaunch
- Wait 10-15 seconds for MCP servers to initialize

**6. Check Claude Desktop logs:**

```bash
# macOS
tail -f ~/Library/Logs/Claude/mcp*.log

# Windows
type %LOCALAPPDATA%\Claude\logs\mcp*.log

# Linux
tail -f ~/.local/share/Claude/logs/mcp*.log
```

Look for errors like:
```
Failed to start MCP server: semantic-modulator
Error: python: command not found
```

---

### Issue: MCP tools timeout

**Symptoms:**
```
Error: Tool call timed out after 30 seconds
```

**Solutions:**

**1. Reduce document size:**
```python
# For large documents (>100K tokens), split into chunks
chunk_size = 50000  # tokens
chunks = split_document(large_doc, chunk_size)
for i, chunk in enumerate(chunks):
    await mcp_tools.ingest_context(
        text=chunk,
        file_id=f"large_doc_part_{i}"
    )
```

**2. Increase timeout (in Claude Desktop config):**
```json
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/token-saver-5000",
      "env": {},
      "timeout": 60000  // 60 seconds (default: 30000)
    }
  }
}
```

**3. Use async processing:**
```python
# Instead of waiting for ingestion to complete
await mcp_tools.ingest_context(...)  # May timeout

# Check status separately
stats = await mcp_tools.get_stats(file_id="my_doc")
```

---

## Compression Issues

### Issue: Poor compression ratio (<10× instead of 18-20×)

**Symptoms:**
```
Expected: 45,000 → 2,300 tokens (19.5× compression)
Actual: 45,000 → 6,500 tokens (6.9× compression)
```

**Causes & Solutions:**

**1. Document has low semantic redundancy:**

Documents with high diversity (like dictionaries, lists) compress less.

```python
# Check graph statistics
stats = await mcp_tools.get_stats(file_id="my_doc")
print(f"Graph density: {stats['graph_density']}")

# Low density (<0.03) = low redundancy = poor compression
```

**Solution:** This is expected. Not all documents compress equally.

**2. Similarity threshold too high:**

```python
# Default threshold: 0.75 (aggressive)
compressor = SemanticCompressor(similarity_threshold=0.75)

# Try lower threshold for denser graphs (more compression)
compressor = SemanticCompressor(similarity_threshold=0.65)
```

**3. Skeleton ratio too high:**

```python
# Default: 0.2 (20% anchor nodes)
compressor = SemanticCompressor(skeleton_ratio=0.2)

# Try lower ratio for more compression
compressor = SemanticCompressor(skeleton_ratio=0.10)  # 10% anchors
```

**4. Document structure not recognized:**

```python
# For code, use CodeSemanticCompressor
from src.code_compressor import CodeSemanticCompressor

code_comp = CodeSemanticCompressor()
code_comp.ingest_code_file(code, "my_code", language="python")
```

---

### Issue: Important information not in skeleton

**Symptoms:**
```
read_skeleton doesn't show critical section from original document
```

**Solutions:**

**1. Use semantic search to find it:**
```python
results = await mcp_tools.search_semantic(
    query="the missing concept",
    file_id="my_doc",
    top_k=10
)
```

**2. Check node importance:**
```python
# Low importance nodes are hidden
# But you can still retrieve them by ID
await mcp_tools.modulate_region(
    node_ids=["my_doc_n47"],  # Specific node
    fidelity_level="RAW"
)
```

**3. Lower skeleton ratio to show more nodes:**
```python
compressor = SemanticCompressor(skeleton_ratio=0.30)  # Show top 30%
```

**4. Use blind spot detection:**
```python
# System may auto-suggest missing nodes
result = await mcp_tools.check_blind_spots(
    ai_response=your_response,
    file_id="my_doc",
    retrieved_nodes=retrieved_so_far
)
```

---

### Issue: Semantic search returns irrelevant results

**Symptoms:**
```
Query: "error handling"
Results: Nodes about "user authentication" (unrelated)
```

**Solutions:**

**1. Use more specific queries:**
```python
# ❌ Too vague
search_semantic("handling")

# ✅ More specific
search_semantic("error handling and exception management in Python")
```

**2. Increase top_k to see more results:**
```python
# Sometimes relevant results are ranked lower
results = await mcp_tools.search_semantic(
    query="error handling",
    top_k=15  # Instead of default 5
)
```

**3. Check document actually contains the content:**
```python
# Verify with keyword search
import re
matches = re.findall(r'error', document_text, re.I)
print(f"Found {len(matches)} occurrences of 'error'")
```

**4. Re-ingest with better chunking:**
```python
# Use smaller chunks for better granularity
compressor = SemanticCompressor(chunk_size=300)  # Instead of 500
```

---

## Performance Issues

### Issue: Slow ingestion (>10 seconds for medium documents)

**Symptoms:**
```
Ingesting 20K token document...
[Takes 15+ seconds instead of ~2 seconds]
```

**Solutions:**

**1. Check CPU usage:**
```bash
top  # Should show Python process using ~100% CPU during ingestion
```

If CPU usage is low, something else is the bottleneck.

**2. Disable GPU if not needed:**
```python
import torch
torch.set_num_threads(4)  # Limit CPU threads

# Or disable CUDA entirely
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

**3. Use smaller embedding model:**
```python
# all-MiniLM-L6-v2 (default): 384-dim, ~80MB, fast
# If still too slow, try even smaller model:
compressor = SemanticCompressor(model_name='all-MiniLM-L12-v2')
```

**4. Reduce chunk overlap:**
```python
# Default: 50 tokens overlap
# Less overlap = fewer chunks = faster

# Edit src/semantic_compressor.py:
def _chunk_text(self, text, chunk_size=500, overlap=25):  # Reduced from 50
```

**5. Check disk I/O:**
```bash
# If using ChromaDB, ensure it's on fast SSD
iostat -x 1  # Monitor disk usage

# If slow, switch to JSON fallback
persistence = PersistenceManager(backend="json")
```

---

### Issue: High memory usage (>4GB RAM)

**Symptoms:**
```
System slows down, swap usage increases
Python process using 4-8GB RAM
```

**Solutions:**

**1. Check how many documents are loaded:**
```python
docs = await mcp_tools.list_documents()
print(f"Loaded: {len(docs['documents'])} documents")

# Each document uses ~2-5MB per 1000 nodes
```

**2. Delete unused documents:**
```python
# Free up memory
await mcp_tools.delete_document(
    file_id="old_doc",
    confirm=True
)
```

**3. Reduce max documents limit:**
```python
# Edit src/server.py ResourceManager settings:
ResourceManager(ResourceLimits(
    max_documents=100,  # Instead of 1000
    max_memory_mb=1024  # 1GB instead of 2GB
))
```

**4. Use float16 embeddings:**
```python
# Edit src/semantic_compressor.py:
embeddings = model.encode(texts, convert_to_numpy=True).astype('float16')
# 50% memory reduction, minimal accuracy loss
```

**5. Restart MCP server periodically:**
```bash
# Memory leaks may accumulate over long sessions
# Restart clears memory (documents auto-reload from persistence)
pkill -f "src.server"
python -m src.server
```

---

### Issue: ChromaDB performance degradation

**Symptoms:**
```
First document ingestion: 2 seconds
10th document ingestion: 8 seconds
```

**Solutions:**

**1. Check ChromaDB size:**
```bash
du -sh data/chromadb
# Should be <500MB for typical usage
```

**2. Compact ChromaDB:**
```python
# ChromaDB doesn't auto-compact deleted documents
# Manual vacuum:
import chromadb
client = chromadb.PersistentClient(path="./data/chromadb")
# Delete old collections
for collection in client.list_collections():
    if should_delete(collection):
        client.delete_collection(collection.name)
```

**3. Switch to JSON fallback:**
```python
# Faster for small deployments
persistence = PersistenceManager(backend="json")
```

**4. Increase ChromaDB batch size:**
```python
# Edit src/persistence.py:
collection.add(
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids,
    batch_size=1000  # Increase from default 100
)
```

---

## Storage & Resource Issues

### Issue: `ResourceLimitExceeded: Document exceeds 100MB limit`

**Solutions:**

**1. Split large document:**
```python
# For 250MB document
chunks = split_by_size(document, max_size_mb=90)
for i, chunk in enumerate(chunks):
    await mcp_tools.ingest_context(
        text=chunk,
        file_id=f"large_doc_part{i}"
    )
```

**2. Increase limit:**
```python
# Edit src/server.py:
ResourceManager(ResourceLimits(
    max_document_size_mb=250.0  # Increase from 100MB
))
```

**3. Compress document first:**
```bash
# Remove unnecessary whitespace, comments
python -c "import re; text = open('doc.txt').read(); \
  text = re.sub(r'\s+', ' ', text); \
  print(text)" > doc_compressed.txt
```

---

### Issue: `ResourceLimitExceeded: Total storage exceeds 1GB limit`

**Solutions:**

**1. List and delete old documents:**
```python
docs = await mcp_tools.list_documents()

# Sort by last accessed time
old_docs = sorted(docs['documents'], key=lambda d: d['last_accessed'])

# Delete oldest 25%
for doc in old_docs[:len(old_docs)//4]:
    await mcp_tools.delete_document(
        file_id=doc['file_id'],
        confirm=True
    )
```

**2. Increase storage limit:**
```python
# Edit src/server.py:
ResourceManager(ResourceLimits(
    max_total_storage_mb=5120.0  # 5GB instead of 1GB
))
```

**3. Clean ChromaDB manually:**
```bash
# Nuclear option: delete all persisted data
rm -rf data/chromadb/*
rm -rf data/json/*
# Then re-ingest only needed documents
```

---

### Issue: Persistent storage not working (documents lost on restart)

**Symptoms:**
```
Ingest document → Restart server → Document not found
```

**Solutions:**

**1. Check data directory exists:**
```bash
ls -la data/
# Should show chromadb/ or json/ subdirectories
```

**2. Verify write permissions:**
```bash
touch data/test.txt
# If error, fix permissions:
chmod -R u+w data/
```

**3. Check logs for persistence errors:**
```bash
python -m src.server 2>&1 | grep -i "persist"
```

**4. Force JSON fallback:**
```python
# Edit src/server.py:
persistence = PersistenceManager(backend="json")
```

**5. Manually trigger save:**
```python
# After ingestion
result = await mcp_tools.ingest_context(...)
# Persistence should auto-save, but verify:
docs = await mcp_tools.list_documents()
# Restart server
# Check again
docs = await mcp_tools.list_documents()
```

---

## AFM (Dialogue Memory) Issues

### Issue: AFM not preserving critical messages (e.g., allergies)

**Symptoms:**
```
User mentions peanut allergy at turn 1
At turn 20, allergy is dropped from context
```

**Solutions:**

**1. Verify importance classification:**
```python
# Add message
await mcp_tools.afm_add_message(
    role="user",
    content="I have a severe peanut allergy"
)

# Check stats
stats = await mcp_tools.afm_get_stats()
print(stats['importance_breakdown'])
# Should show CRITICAL: 1
```

**If not classified as CRITICAL:**

```python
# Check classification patterns in src/afm.py
# Add custom patterns if needed:

CRITICAL_PATTERNS = [
    r'\b(allerg|cannot|must not|never|severe|deadly)\b',
    r'\b(life-threatening|dangerous|toxic|harmful)\b',
    # Add your custom patterns here
]
```

**2. Check token budget:**
```python
# Budget too small may force dropping even CRITICAL messages
context = await mcp_tools.afm_build_context(
    current_query=query,
    budget_tokens=800  # Try increasing to 1500
)
```

**3. Verify AFM config:**
```python
# Edit src/server.py:
afm_config = AFMConfig(
    tau_high=0.45,  # Threshold for FULL fidelity
    tau_mid=0.25,   # Threshold for COMPRESSED fidelity
    half_life=12    # Recency decay rate
)
```

**Lower tau_high to preserve more messages at FULL fidelity:**
```python
afm_config = AFMConfig(tau_high=0.30)  # More messages at FULL
```

---

### Issue: AFM compression not saving enough tokens

**Symptoms:**
```
Expected: ~66% savings
Actual: ~30% savings
```

**Solutions:**

**1. Check message importance distribution:**
```python
stats = await mcp_tools.afm_get_stats()
print(stats['importance_breakdown'])

# If most messages are CRITICAL or RELEVANT:
# CRITICAL: 15, RELEVANT: 10, TRIVIAL: 2
# → Not much to compress!
```

**2. Adjust classification to be more aggressive:**
```python
# Edit src/afm.py classification patterns
# Make RELEVANT patterns more restrictive
# Make TRIVIAL patterns more inclusive
```

**3. Increase half-life decay:**
```python
# Edit src/server.py:
afm_config = AFMConfig(
    half_life=8  # Faster decay (default: 12)
)
# Older messages decay faster → more compression
```

**4. Reduce tau thresholds:**
```python
afm_config = AFMConfig(
    tau_high=0.40,  # Fewer FULL messages
    tau_mid=0.20    # More PLACEHOLDER messages
)
```

---

### Issue: AFM export/import not working

**Symptoms:**
```
await mcp_tools.afm_export_history(session_id="my_session")
await mcp_tools.afm_import_history(session_id="my_session")
Error: Session 'my_session' not found
```

**Solutions:**

**1. Check export directory:**
```bash
ls -la data/afm_sessions/
# Should contain my_session.json
```

**2. Verify export completed:**
```python
result = await mcp_tools.afm_export_history(session_id="my_session")
print(result)  # Should show success: true
```

**3. Check file permissions:**
```bash
ls -la data/afm_sessions/my_session.json
# Should be readable
chmod 644 data/afm_sessions/*.json
```

**4. Use default session:**
```python
# If session_id issues persist, use default
await mcp_tools.afm_export_history()  # Exports to "default"
await mcp_tools.afm_import_history()  # Imports from "default"
```

---

## Testing Issues

### Issue: Tests fail with `ModuleNotFoundError`

**Symptoms:**
```
pytest tests/
ImportError: No module named 'src'
```

**Solutions:**

**1. Install in editable mode:**
```bash
pip install -e .
```

**2. Set PYTHONPATH:**
```bash
export PYTHONPATH=/path/to/token-saver-5000:$PYTHONPATH
pytest tests/
```

**3. Run from project root:**
```bash
cd /path/to/token-saver-5000
pytest tests/  # Not from tests/ directory
```

---

### Issue: Test coverage below 70% target

**Symptoms:**
```
pytest tests/ --cov=src
Coverage: 62% (below 70% threshold)
```

**Solutions:**

**1. Check which modules have low coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View detailed coverage report
```

**2. Run specific test suites:**
```bash
# Test each module
pytest tests/test_semantic_compressor.py -v
pytest tests/test_afm.py -v
pytest tests/test_functional.py -v
```

**3. Temporarily lower threshold for development:**
```python
# Edit pyproject.toml:
[tool.pytest.ini_options]
addopts = [
    "--cov-fail-under=60",  # Lower from 70
]
```

---

## Platform-Specific Issues

### macOS: `xcrun: error: invalid active developer path`

**Symptoms:**
```
Error installing chromadb: xcrun: error
```

**Solution:**
```bash
xcode-select --install
# Then retry
pip install chromadb
```

---

### Windows: `error: Microsoft Visual C++ 14.0 is required`

**Symptoms:**
```
Building wheel for chromadb failed
error: Microsoft Visual C++ 14.0 is required
```

**Solution:**
```
1. Download "Build Tools for Visual Studio"
   https://visualstudio.microsoft.com/downloads/
2. Install "C++ build tools" workload
3. Restart terminal
4. pip install chromadb
```

---

### Linux: `libgomp.so.1: cannot open shared object file`

**Symptoms:**
```
ImportError: libgomp.so.1: cannot open shared object file
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libgomp1

# CentOS/RHEL
sudo yum install libgomp

# Arch
sudo pacman -S gcc-libs
```

---

## Getting More Help

If your issue isn't covered here:

1. **Check logs:**
   ```bash
   # MCP server logs
   python -m src.server 2>&1 | tee server.log

   # Claude Desktop logs (if using MCP)
   tail -f ~/Library/Logs/Claude/mcp*.log
   ```

2. **Run diagnostics:**
   ```bash
   python scripts/check_setup.py --verbose
   ```

3. **Create minimal reproduction:**
   ```python
   # Simplest code that triggers the issue
   from src.semantic_compressor import SemanticCompressor
   comp = SemanticCompressor()
   comp.ingest_file("Test text", "test")
   ```

4. **Report issue on GitHub:**
   - Include: Python version, OS, error message, reproduction steps
   - https://github.com/oimiragieo/token-saver-5000/issues

5. **Community discussions:**
   - https://github.com/oimiragieo/token-saver-5000/discussions

---

## Quick Diagnostic Commands

```bash
# Check installation
python scripts/check_setup.py

# Verify imports
python -c "from src.semantic_compressor import SemanticCompressor; print('OK')"

# Check dependencies
pip list | grep -E "mcp|sentence-transformers|chromadb|torch"

# Test MCP server
python -m src.server --help

# Check storage
du -sh data/

# Check resource usage
ps aux | grep python

# View logs
tail -f server.log
```

---

**Last Updated:** 2024-11-22
**For more help:** See [GETTING_STARTED.md](GETTING_STARTED.md) or [README.md](README.md)
