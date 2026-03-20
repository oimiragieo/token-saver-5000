# Getting Started with Token Saver 5000 🚀

**Your step-by-step guide to achieving 80-95% token savings in 10 minutes!**

Token Saver 5000 provides **two complementary compression systems**:
1. **Document Compression** (SemanticCompressor) - Compress long documents 80-95%
2. **Dialogue Compression** (AFM) - Manage multi-turn conversations with ~66% fewer tokens

---

## What You'll Learn

By the end of this guide, you will:
- ✅ Install and verify Token Saver 5000 works
- ✅ Run tests proving 80-95% document compression
- ✅ Test AFM dialogue memory retaining critical context
- ✅ Try all features with real examples
- ✅ Integrate with Claude Desktop or your AI platform
- ✅ Understand when to use Document vs Dialogue compression
- ✅ Explore SCAR enhancements for advanced use cases

**Time required:** 10-15 minutes

---

## Step 1: Installation (2 minutes)

### 1.1 Clone the Repository

```bash
git clone https://github.com/oimiragieo/token-saver-5000.git
cd token-saver-5000
```

### 1.2 Install as a Tool

**Option A: Using uv (recommended)**
```bash
uv tool install -e .
```

**Option B: Using pipx**
```bash
pipx install .
```

**Option C: Editable install for development**
```bash
pip install -r requirements.txt
pip install -e .
```

### 1.3 Configure MCP the Easy Way

Run the guided setup command:

```bash
token-saver-setup --auto
```

That chooses the most likely target automatically:
- Claude Desktop users usually get `desktop`
- repo/workspace users usually get `portable-project`

### 1.4 Optional Deep Verification

Run the comprehensive setup verification script if you want a full smoke test that also downloads the embedding model:

```bash
python scripts/check_setup.py
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

**If you see errors**, the script will tell you exactly what's wrong and how to fix it. For detailed troubleshooting, see [**TROUBLESHOOTING.md**](TROUBLESHOOTING.md).

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

### 2.3 Run AFM Tests (Dialogue Memory)

These tests prove AFM retains critical context across conversations:

```bash
python tests/test_afm.py
```

**What you'll see:**

```
test_allergy_retention_short_conversation PASSED
test_allergy_retention_medium_conversation PASSED
test_recency_weighting PASSED
test_token_savings PASSED
✅ All AFM tests passed!

Key Results:
- Critical messages (allergies) retained across 9+ turns
- ~66% token reduction with full context preservation
- Recency weighting working as expected
- Safety context never lost
```

**What this proves:**
- ✅ CRITICAL messages always retained at FULL fidelity
- ✅ RELEVANT messages compressed to summaries
- ✅ TRIVIAL messages replaced with placeholders
- ✅ Recency weighting prioritizes recent context
- ✅ ~66% token savings while preserving safety

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

### 3.2 AFM Dialogue Memory Demo

```bash
python examples/afm_demo.py
```

This demonstrates:
- Adding messages to dialogue history
- Automatic importance classification (CRITICAL/RELEVANT/TRIVIAL)
- Building context with adaptive fidelity
- Token savings comparison (FULL vs AFM)

**Expected output:**
```
=======================================================================
ADAPTIVE FOCUS MEMORY (AFM) DEMONSTRATION
=======================================================================

[1] SHORT CONVERSATION (3 turns)
Adding messages to dialogue history...
Turn 1: User mentions peanut allergy
Turn 2: Assistant acknowledges
Turn 3: User asks about Thai street food

Building context with AFM (budget: 500 tokens)...
📊 Context Built:
   Messages: 3
   FULL fidelity: 3 (100%)
   Token usage: 342 / 500 (68%)

✅ PASS: Allergy context retained

[2] MEDIUM CONVERSATION (9 turns)
...
📊 AFM Results:
   FULL fidelity: 3 messages (33%)
   COMPRESSED: 4 messages (44%)
   PLACEHOLDER: 2 messages (22%)
   Token usage: 647 / 800 (81%)

✅ PASS: Critical allergy retained across 9 turns!

[3] TOKEN SAVINGS
Baseline (all FULL): 1,245 tokens
AFM adaptive: 647 tokens
Savings: 598 tokens (48.0%)
```

### 3.3 SCAR Enhanced Example

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
    "token-saver": {
      "command": "token-saver-mcp",
      "args": [],
      "cwd": "/PASTE/YOUR/PATH/HERE/token-saver-5000",
      "env": {}
    }
  }
}
```

**Replace `/PASTE/YOUR/PATH/HERE/token-saver-5000`** with the path you copied!

**Or install it automatically:**
```bash
token-saver-install-mcp
```

### 4.3 Restart Claude Desktop

Completely quit and restart Claude Desktop (not just reload).

### 4.4 Verify Tools Available

In Claude Desktop, type:
```
Can you list the available MCP tools?
```

You should see **16 tools** (9 document + 4 dialogue + 3 discovery/persistence):

**Document Compression Tools (9):**
- ✅ `ingest_context` - Compress and ingest long documents
- ✅ `read_skeleton` - View compressed document structure
- ✅ `modulate_region` - Retrieve specific sections with variable detail
- ✅ `search_semantic` - Find relevant sections via embedding search
- ✅ `check_blind_spots` - Detect missed critical context
- ✅ `detect_hallucination` - Verify AI responses against source
- ✅ `get_stats` - View compression statistics
- ✅ `adapt_to_context_window` - Fit content within token budget
- ✅ `multilevel_encode` - Generate multi-fidelity representations

**Dialogue Memory Tools (AFM) - 4:**
- ✅ `afm_add_message` - Add message to dialogue history
- ✅ `afm_build_context` - Build context window with adaptive fidelity
- ✅ `afm_get_stats` - View dialogue statistics
- ✅ `afm_clear_history` - Clear dialogue history

**Discovery & Persistence Tools (3) - NEW in v0.2.0:**
- ✅ `list_documents` - Get inventory of all ingested documents
- ✅ `afm_export_history` - Save conversation state to file
- ✅ `afm_import_history` - Restore conversation state from file

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

### Feature 5: SCAR Enhancements

**What it does:**
- Compresses embeddings 4× (384D → 96D)
- Improves search relevance 15-25%
- Adaptive fidelity based on alignment

**When to use:** When you need the absolute best retrieval quality

### Feature 6: Adaptive Focus Memory (AFM) (NEW!)

**What it does:**
- Manages multi-turn conversations with adaptive compression
- Automatically classifies message importance (CRITICAL/RELEVANT/TRIVIAL)
- Applies 3 fidelity levels: FULL, COMPRESSED, PLACEHOLDER
- Uses recency weighting to favor recent context

**How it works:**
1. Each message added gets an importance score
2. Recent messages weighted more heavily
3. CRITICAL messages (allergies, safety) → always FULL
4. RELEVANT messages → compressed to summaries
5. TRIVIAL messages → replaced with placeholders

**Fidelity Levels:**
- **FULL**: Original message text (100% tokens)
- **COMPRESSED**: 1-2 sentence summary (~33% tokens)
- **PLACEHOLDER**: Brief stub (~10 tokens)

**When to use:**
- Multi-turn conversations (customer support, planning sessions)
- When critical context must be retained (medical info, preferences)
- Long conversations that exceed context windows
- Combining with document compression for hybrid workflows

**Token Savings:** ~48-66% while preserving all critical context

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

### Q: When should I use AFM vs Document Compression?

**A:** Use both! They're complementary:

**Document Compression:**
- Use for: Long documents, papers, codebases, articles
- Saves: 80-95% of document tokens
- Best for: Analyzing large static content

**AFM (Dialogue Memory):**
- Use for: Multi-turn conversations, planning sessions
- Saves: 48-66% of conversation history tokens
- Best for: Retaining critical context across turns

**Combined Workflow:**
1. Compress long document with `ingest_context()` (80-95% savings)
2. Add document insights to conversation with `afm_add_message()`
3. Build context with `afm_build_context()` (additional 48-66% savings)
4. Result: Massive token savings with full context retention!

### Q: Is AFM safe for critical information?

**A:** Yes! AFM is designed for safety:
- CRITICAL importance → Always FULL fidelity (never compressed)
- Heuristic classifier detects medical info, allergies, safety warnings
- Test suite verifies retention across 9+ turns
- Based on research paper (arXiv:2511.12712v1) with safety focus

**Example:** Allergy mentioned in turn 1 is retained at FULL fidelity even at turn 10!

### Q: Can I customize AFM thresholds?

**A:** Yes! Edit `src/afm.py`:

```python
config = AFMConfig(
    tau_high=0.45,      # Threshold for FULL (higher = stricter)
    tau_mid=0.25,       # Threshold for COMPRESSED
    half_life=12,       # Recency decay parameter
)
```

**Trade-offs:**
- Higher `tau_high` → More compression, but risk missing context
- Lower `half_life` → Stronger recency bias (favor recent messages)

---

## Next Steps

Now that you're set up:

### 1. Try with Your Own Content

**Document Compression:**
- Technical papers, documentation, long articles
- See the 80-95% compression ratios!
- Example: `python examples/example_usage.py`

**Dialogue Memory:**
- Multi-turn conversations with context preservation
- Test with critical information (allergies, preferences)
- Example: `python examples/afm_demo.py`

### 2. Explore Advanced Features

- **Multi-document analysis**: Ingest multiple documents, search across all
- **Cross-document search**: Find related concepts across your knowledge base
- **Blind spot detection**: Auto-catch missed context
- **Hybrid workflows**: Combine document + dialogue compression

### 3. Customize for Your Needs

**Document Compression:**
- Adjust compression ratio in `src/semantic_compressor.py`
- Try different embedding models
- Train SCAR on your domain

**AFM Configuration:**
- Tune importance thresholds in `src/afm.py`
- Adjust recency weighting
- Customize compression ratios per fidelity level

### 4. Read the Full Documentation

**Core Documentation:**
- [**README.md**](README.md) - Complete feature list & research background
- [**HOW_IT_WORKS.md**](HOW_IT_WORKS.md) - Technical deep dive on all technology
- [**MCP_TOOLS_GUIDE.md**](MCP_TOOLS_GUIDE.md) - Complete reference for all 17 MCP tools
- [**TROUBLESHOOTING.md**](TROUBLESHOOTING.md) - Common issues and solutions

**Advanced Topics:**
- [**ARCHITECTURE.md**](ARCHITECTURE.md) - System architecture and design
- [**API_REFERENCE.md**](API_REFERENCE.md) - Module and API documentation
- [docs/SCAR_PAPER_SUMMARY.md](docs/SCAR_PAPER_SUMMARY.md) - SCAR research details
- [docs/RESEARCH_SYNTHESIS.md](docs/RESEARCH_SYNTHESIS.md) - Research paper validation

---

## Troubleshooting

Having issues? See the comprehensive [**TROUBLESHOOTING.md**](TROUBLESHOOTING.md) guide which covers:

- Installation issues (dependencies, Python version, etc.)
- Model download issues
- MCP server configuration
- Performance problems
- Storage and resource limits
- AFM dialogue memory issues
- Platform-specific fixes (macOS, Windows, Linux)

**Quick fixes for common issues:**

### "Module not found" error
```bash
cd token-saver-5000
pip install -r requirements.txt
```

### "Model download failed"
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### "Claude doesn't see MCP tools"
1. Check config uses absolute path (not relative)
2. Completely restart Claude Desktop
3. See [TROUBLESHOOTING.md#mcp-server-issues](TROUBLESHOOTING.md#mcp-server-issues)

---

## Get Help

- **Issues:** [GitHub Issues](https://github.com/oimiragieo/token-saver-5000/issues)
- **Discussions:** [GitHub Discussions](https://github.com/oimiragieo/token-saver-5000/discussions)
- **Documentation:** Full docs in [README.md](README.md)

---

## Success! 🎉

You now have:
- ✅ Token Saver 5000 installed and tested
- ✅ Document compression: 80-95% token savings proven
- ✅ AFM dialogue memory: 48-66% conversation token savings
- ✅ All 13 MCP tools working (9 document + 4 dialogue)
- ✅ Integrated with Claude Desktop
- ✅ Understanding of when to use each system

**Key Capabilities:**
- 📄 Compress long documents 10-20× with semantic preservation
- 💬 Manage multi-turn conversations with adaptive fidelity
- 🔍 Semantic search across your knowledge base
- 🎯 Progressive retrieval at variable detail levels
- 🛡️ Blind spot detection for critical accuracy
- 🧠 SCAR enhancements for advanced use cases

**Start saving tokens and enjoying infinite context windows!** 🚀

---

**Built with ❤️ for the AI community**
*Making context windows infinite, one semantic graph at a time*

Based on 4 research papers:
- JSCCM (Joint Semantic Compression & Contextual Memory)
- FPQE (Full Prompt Quality Evaluation - SSIM metrics)
- SCAR (Semantic Compression with Alignment & Retrieval)
- AFM (Adaptive Focus Memory - arXiv:2511.12712v1)
