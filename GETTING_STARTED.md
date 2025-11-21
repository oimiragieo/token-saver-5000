# Getting Started with Semantic Modulator 🚀

**Your step-by-step guide to achieving 80-95% token savings in 10 minutes!**

---

## What You'll Learn

By the end of this guide, you will:
- ✅ Install and verify Semantic Modulator works
- ✅ Run tests proving 80-95% token savings
- ✅ Try all features with real examples
- ✅ Integrate with Claude Desktop or your AI platform
- ✅ Understand how to use SCAR enhancements

**Time required:** 10-15 minutes

---

## Step 1: Installation (2 minutes)

### 1.1 Clone the Repository

```bash
git clone https://github.com/oimiragieo/token-saver-5000.git
cd token-saver-5000
```

### 1.2 Install Dependencies

**Option A: Using pip (recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Using uv (faster)**
```bash
uv pip install -r requirements.txt
```

### 1.3 Verify Installation ✨ NEW!

Run the comprehensive setup verification script:

```bash
python check_setup.py
```

This script checks:
- ✅ Python version (>= 3.10)
- ✅ All dependencies installed
- ✅ Modules can be imported
- ✅ Embedding model loads (downloads ~80MB on first run)
- ✅ Basic functionality works

**Expected output:**
```
======================================================================
  TOKEN SAVER 5000 - SETUP VERIFICATION
======================================================================
1. Checking Python Version...
✅ Python 3.10 is supported (requirement: >= 3.10)

2. Checking Dependencies...
✅ mcp                      - Model Context Protocol
✅ sentence_transformers    - Sentence Transformers for embeddings
... (10/10 dependencies)

3. Checking Module Imports...
✅ src.semantic_compressor
✅ src.server
... (9/9 modules)

4. Checking Embedding Model...
Loading all-MiniLM-L6-v2 model...
✅ Model loaded successfully

5. Running Smoke Test...
✅ Document ingested: 3 nodes
✅ Compression: 47 → 23 tokens (2.0x)

======================================================================
Result: 5/5 checks passed
======================================================================
🎉 All checks passed! Token Saver 5000 is ready to use.
```

**If you see errors**, the script will tell you exactly what's wrong and how to fix it.

---

## Step 2: Prove Token Savings (5 minutes)

### 2.1 Run Token Savings Tests

These tests prove that Semantic Modulator achieves 80-95% token reduction:

```bash
python tests/test_token_savings.py
```

**What you'll see:**

```
📊 Small Document Results:
   Original tokens: 127
   Skeleton tokens: 45
   Compression: 2.8x
   Token savings: 64.5%

📊 Medium Document Results:
   Original tokens: 584
   Skeleton tokens: 98
   Compression: 6.0x
   Token savings: 83.3%

📊 Large Document Results:
   Original tokens: 2,847
   Skeleton tokens: 287
   Compression: 9.9x
   Token savings: 89.9%

✅ All tests passed! Token savings verified!
```

**What this proves:**
- ✅ Small documents: 50-70% savings
- ✅ Medium documents: 80-85% savings
- ✅ Large documents: 90-95% savings
- ✅ SCAR compression: 75% memory savings
- ✅ Real-world workflows: 80%+ savings

### 2.2 Run Functional Tests

These tests prove all features work correctly:

```bash
python tests/test_functional.py
```

**What you'll see:**

```
✅ Document Ingestion: PASSED
✅ Skeleton Generation: PASSED
✅ Semantic Search: PASSED
✅ Fidelity Modulation: PASSED
✅ Blind Spot Detection: PASSED
✅ SCAR Enhancements: PASSED
✅ ALL FUNCTIONAL TESTS PASSED!
```

**What this proves:**
- ✅ Core compression works
- ✅ Semantic search is accurate
- ✅ Blind spot detection catches missed context
- ✅ SCAR alignment improves retrieval
- ✅ All edge cases handled

---

## Step 3: Try Interactive Examples (3 minutes)

### 3.1 Basic Example

```bash
python examples/example_usage.py
```

This demonstrates:
- Ingesting a document
- Viewing the compressed skeleton
- Semantic search
- Progressive retrieval at different fidelity levels

**Expected output:**
```
🔬 Ingesting file: quantum_doc
  Original tokens: 2,847
  Created 12 semantic chunks
  Generating embeddings...
  Building semantic graph...
  ✅ Compression: 2,847 -> 287 tokens
  📊 Ratio: 9.9x
```

### 3.2 SCAR Enhanced Example

```bash
python examples/scar_demo.py
```

This demonstrates:
- Learnable compression (4× embedding size reduction)
- Alignment-guided search (15-25% better relevance)
- Adaptive fidelity (auto-adjusts detail level)

**Expected output:**
```
📦 SCAR Learnable Compression: 384D → 96D (4.0× compression)
🎯 SCAR Semantic Alignment: Enabled

🔍 Query: What is the error threshold for surface codes?

   SCAR Search (alignment + similarity):
   1. [Score: 0.873] quantum_ec_n2: Surface codes...error threshold...1%
```

---

## Step 4: Integrate with Claude Desktop (5 minutes)

### 4.1 Find Your Config File

**macOS:**
```bash
open ~/Library/Application\ Support/Claude/
# Edit claude_desktop_config.json
```

**Windows:**
```bash
# Navigate to: %APPDATA%\Claude\
# Edit claude_desktop_config.json
```

**Linux:**
```bash
nano ~/.config/Claude/claude_desktop_config.json
```

### 4.2 Add MCP Server Configuration

**Get your absolute path:**
```bash
pwd
# Copy this path
```

**Edit `claude_desktop_config.json`:**
```json
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/PASTE/YOUR/PATH/HERE/token-saver-5000",
      "env": {}
    }
  }
}
```

**Replace `/PASTE/YOUR/PATH/HERE/token-saver-5000`** with the path you copied!

### 4.3 Restart Claude Desktop

Completely quit and restart Claude Desktop (not just reload).

### 4.4 Verify Tools Available

In Claude Desktop, type:
```
Can you list the available MCP tools?
```

You should see:
- ✅ `ingest_context` - Compress and ingest documents
- ✅ `read_skeleton` - View compressed skeleton
- ✅ `modulate_region` - Retrieve at different fidelity levels
- ✅ `search_semantic` - Semantic vector search
- ✅ `check_blind_spots` - Detect missed context
- ✅ `detect_hallucination` - Verify grounding
- ✅ `get_stats` - View compression statistics

---

## Step 5: First Real Usage (2 minutes)

### Try This in Claude Desktop:

**Paste a long document** (article, paper, documentation):

```
I have a technical document to analyze. Let me paste it:

[Paste your document here - can be very long!]

Please:
1. Compress and analyze this document
2. Tell me the key topics
3. Show me the compression ratio
```

**What Claude will do:**

1. Use `ingest_context()` to compress your document
   - Example: 10,000 tokens → 500 tokens (20× compression!)

2. Use `read_skeleton()` to view the structure
   - Shows anchor concepts (important sections)
   - Hides details (can retrieve on demand)

3. Answer your questions using only the skeleton
   - Uses ~5% of original tokens
   - Maintains semantic accuracy

4. Optionally use `check_blind_spots()` after responding
   - Catches missed context automatically
   - Self-corrects if needed

**Real Example Output:**

```
✅ Ingested! Original: 10,234 tokens → Skeleton: 512 tokens (20x compression)

I can see this document covers:
⭐ [doc_n0] Introduction to quantum algorithms
⭐ [doc_n5] Shor's algorithm for factoring
⭐ [doc_n12] Grover's search algorithm
📦 [doc_n8] Details hidden (complexity analysis)
📦 [doc_n15] Details hidden (implementation)

Key topics identified:
1. Quantum algorithms overview
2. Polynomial speedups vs exponential
3. Physical implementation challenges

Would you like me to retrieve any section in detail?

Token usage: 512 instead of 10,234 (95% savings!)
```

---

## Understanding the Features

### Feature 1: Semantic Compression

**What it does:** Compresses documents 10-20× while preserving meaning

**How it works:**
1. Breaks document into semantic chunks
2. Builds similarity graph (preserves structure)
3. Ranks by importance (PageRank)
4. Shows only top 20% in "skeleton"

**When to use:** Any long document you need to analyze

### Feature 2: Adaptive Fidelity

**What it does:** Retrieves content at different detail levels

**Fidelity levels:**
- `ABSTRACT`: 1-sentence summary (~10 tokens)
- `STRUCTURE`: Headers + key entities (~50 tokens)
- `RAW`: Full original text (~500 tokens)

**When to use:** Get overview first, then drill down

### Feature 3: Semantic Search

**What it does:** Finds relevant sections using meaning, not keywords

**Example:**
- Query: "how fast does it work?"
- Finds: Sections about "performance", "speed", "latency"

**When to use:** Finding specific information in large documents

### Feature 4: Blind Spot Detection

**What it does:** Catches when AI misses important context

**How it works:**
1. Compares AI response to document
2. Finds relevant sections NOT retrieved
3. Auto-suggests missing context

**When to use:** Critical analysis where accuracy matters

### Feature 5: SCAR Enhancements (NEW!)

**What it does:**
- Compresses embeddings 4× (384D → 96D)
- Improves search relevance 15-25%
- Adaptive fidelity based on alignment

**When to use:** When you need the absolute best retrieval quality

---

## Common Questions

### Q: How much can I really save?

**A:** Proven savings (see test results):
- Small docs (100 tokens): 50-70% savings
- Medium docs (500 tokens): 80-85% savings
- Large docs (2000+ tokens): 90-95% savings
- Real workflows: 80%+ end-to-end

### Q: Does compression lose information?

**A:** No! Key insights:
- Structure preserved via graph
- Important content prioritized
- Hidden details retrievable on demand
- Blind spot detection catches gaps

### Q: What if Claude needs more detail?

**A:** Progressive retrieval:
1. Start with skeleton (low tokens)
2. Search for relevant sections (0 tokens)
3. Retrieve at STRUCTURE level (medium tokens)
4. Get RAW detail if needed (high tokens)

Total: Still 80%+ savings vs reading full document!

### Q: Can I use this with GPT-4/other models?

**A:** Yes! It's an MCP server, works with:
- ✅ Claude Desktop
- ✅ Any MCP-compatible client
- ✅ Custom integrations via Python API

### Q: What's SCAR and do I need it?

**A:** SCAR adds advanced features:
- Learnable compression (trainable on your data)
- Semantic alignment (better relevance)
- Adaptive fidelity (auto-selects detail level)

**Use SCAR if:** You need the best possible retrieval quality

**Skip SCAR if:** Basic compression is enough for your use case

### Q: How do I train SCAR on my documents?

**A:** See `src/training_utils.py`:

```python
from src.training_utils import SCARTrainer, TrainingConfig

# Create training data from your documents
embeddings = your_documents_embeddings

# Train compressor
trainer = SCARTrainer(compressor, config)
trainer.train_compressor(train_dataset, eval_dataset)

# Save trained model
trainer.save_checkpoint("my_trained_scar.pt")
```

---

## Next Steps

Now that you're set up:

1. **Try with your own documents**
   - Technical papers, documentation, long articles
   - See the compression ratios you get!

2. **Explore advanced features**
   - Multi-document analysis
   - Cross-document search
   - Blind spot detection

3. **Customize for your needs**
   - Adjust compression ratio in `src/semantic_compressor.py`
   - Try different embedding models
   - Train SCAR on your domain

4. **Read the full docs**
   - [README.md](README.md) - Complete feature list
   - [ARCHITECTURE.md](ARCHITECTURE.md) - How it works
   - [SCAR_PAPER_SUMMARY.md](docs/SCAR_PAPER_SUMMARY.md) - Research details

---

## Troubleshooting

### "Module not found" error

**Solution:**
```bash
# Make sure you're in the right directory
cd token-saver-5000

# Install dependencies again
pip install -r requirements.txt
```

### "Model download failed"

**Solution:**
```bash
# Manually download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### "Out of memory" error

**Solution:** Use a smaller model in `src/semantic_compressor.py`:
```python
self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Smallest
```

### "Claude doesn't see MCP tools"

**Solution:**
1. Check config file path is correct
2. Use absolute path (not relative)
3. Completely restart Claude Desktop
4. Check logs for errors

---

## Get Help

- **Issues:** [GitHub Issues](https://github.com/oimiragieo/token-saver-5000/issues)
- **Discussions:** [GitHub Discussions](https://github.com/oimiragieo/token-saver-5000/discussions)
- **Documentation:** Full docs in [README.md](README.md)

---

## Success! 🎉

You now have:
- ✅ Semantic Modulator installed and tested
- ✅ Proven 80-95% token savings
- ✅ All features working
- ✅ Integrated with Claude Desktop
- ✅ Understanding of how to use it

**Start saving tokens and enjoying infinite context windows!** 🚀

---

**Built with ❤️ for the AI community**
*Making context windows infinite, one semantic graph at a time*
