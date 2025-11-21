# 🚀 Quick Start Guide

Get up and running with Semantic Modulator in 5 minutes!

---

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager
- 4GB free RAM (for embedding model)
- Compatible with: Claude Desktop, OpenAI API, any MCP-compatible client

---

## Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/token-saver-5000.git
cd token-saver-5000

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using uv (recommended, faster)

```bash
# Clone the repository
git clone https://github.com/yourusername/token-saver-5000.git
cd token-saver-5000

# Install with uv
uv pip install -r requirements.txt
```

---

## Testing the Installation

### Run token savings tests (RECOMMENDED!)

These tests **prove** that Semantic Modulator achieves 80-95% token reduction:

```bash
python tests/test_token_savings.py
```

**Expected output:**
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

TOKEN SAVINGS VALIDATION REPORT
✅ All tests passed! Token savings verified:
   📈 Small documents (100 tokens):    50-70% savings
   📈 Medium documents (500 tokens):   80-85% savings
   📈 Large documents (2000+ tokens):  90-95% savings
```

### Run functional tests

These tests verify all features work correctly:

```bash
python tests/test_functional.py
```

**Expected output:**
```
✅ Document Ingestion: PASSED
✅ Skeleton Generation: PASSED
✅ Semantic Search: PASSED
✅ ALL FUNCTIONAL TESTS PASSED!
```

### Run example scripts

```bash
# Basic example
python examples/example_usage.py

# SCAR-enhanced retrieval
python examples/scar_demo.py
```

---

## Setting Up MCP Server

### For Claude Desktop

1. Find your config file location:
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

2. Edit the config file and add:

```json
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/absolute/path/to/token-saver-5000",
      "env": {}
    }
  }
}
```

3. Replace `/absolute/path/to/token-saver-5000` with your actual path

4. Restart Claude Desktop

5. Verify the tools are available by typing:
   ```
   Can you list the MCP tools available?
   ```

You should see:
- `ingest_context`
- `read_skeleton`
- `modulate_region`
- `search_semantic`
- `check_blind_spots`
- `detect_hallucination`
- `get_stats`

---

## First Usage with Claude

### Example 1: Analyze a Long Document

```
You: I have a 50-page technical document. Help me understand it efficiently.

[Paste your document content]

Claude: I'll use semantic compression to analyze this efficiently.

[Uses: ingest_context(text=your_content, file_id="doc1")]

✅ Compressed 45,000 tokens down to 2,300 tokens (19.5x)

[Uses: read_skeleton("doc1")]

Here's the structure:
⭐ [doc1_n0] ANCHOR: Introduction discusses X...
📦 [doc1_n1] Detail hidden - appears to cover Y
⭐ [doc1_n5] ANCHOR: Main findings show Z...

What would you like to explore in detail?

You: Tell me more about the main findings

[Claude uses: modulate_region(["doc1_n5"], "RAW")]

[After responding, Claude can use: check_blind_spots() to verify it didn't miss anything critical]
```

**Token usage:** ~3,000 tokens instead of 45,000 (93% savings)

---

### Example 2: Semantic Search

```
You: Find all sections discussing "error rates" in doc1

[Uses: search_semantic(query="error rates", file_id="doc1", top_k=5)]

Found 5 relevant sections:
1. [doc1_n12] - Error rate analysis methodology
2. [doc1_n18] - Experimental error measurements
3. [doc1_n25] - Comparison with theoretical bounds
...

Would you like me to retrieve any of these in full?
```

---

### Example 3: Blind Spot Detection

```
[After Claude generates a response]

You: Check if you missed any important context

[Uses: check_blind_spots(
  ai_response="my analysis...",
  file_id="doc1",
  retrieved_nodes=[...]
)]

🔍 BLIND SPOT ANALYSIS:
⚠️ CRITICAL: Found 1 highly relevant node that was NOT retrieved!
  • [doc1_n23] similarity=0.91: "Contradicts earlier claims about..."

🔧 AUTO-INJECTION SUGGESTED
Retrieving missed context...

[Claude retrieves the node and updates its analysis]
```

---

## Configuration Options

### Adjusting Compression Ratio

Edit `src/server.py`:

```python
self.compressor = SemanticCompressor(
    model_name="all-MiniLM-L6-v2",
    similarity_threshold=0.75,  # Higher = fewer edges, faster
    skeleton_ratio=0.2,  # 0.2 = show top 20% in skeleton
)
```

**Trade-offs:**
- Lower `skeleton_ratio` (e.g., 0.1) → Higher compression but more tool calls
- Higher `skeleton_ratio` (e.g., 0.3) → Lower compression but fewer tool calls
- Lower `similarity_threshold` (e.g., 0.6) → More connections, better structure preservation
- Higher `similarity_threshold` (e.g., 0.85) → Fewer connections, faster processing

---

### Using a Different Embedding Model

```python
# Options:
"all-MiniLM-L6-v2"           # Fast, 384 dim (default)
"all-mpnet-base-v2"          # Better quality, 768 dim
"all-distilroberta-v1"       # Balanced, 768 dim
"paraphrase-multilingual"    # Multilingual support
```

Trade-off: Larger models = better semantic understanding but slower processing

---

## Troubleshooting

### Issue: Model download fails

**Solution:** Manually download the model:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### Issue: Out of memory

**Solution:** Use a smaller embedding model or reduce batch size in `semantic_compressor.py`:
```python
embeddings = model.encode(raw_chunks, batch_size=16)  # Reduce from default 32
```

### Issue: MCP server not recognized in Claude

**Solution:**
1. Check absolute path in config is correct
2. Verify Python is in your PATH: `python --version`
3. Check logs: Look for MCP server errors in Claude Desktop logs
4. Restart Claude Desktop completely

### Issue: "Module not found" error

**Solution:**
```bash
# Ensure PYTHONPATH is set in config
"env": {
  "PYTHONPATH": "/absolute/path/to/token-saver-5000"
}
```

---

## Performance Tips

### 1. For very large documents (>100K tokens)

Increase chunk size to reduce node count:
```python
chunks = self._chunk_text(text, max_chunk_size=1024)  # Default is 512
```

### 2. For interactive usage

Use STRUCTURE fidelity first, then RAW only if needed:
```
modulate_region(["node_5"], "STRUCTURE")  # ~50 tokens
# If you need more detail:
modulate_region(["node_5"], "RAW")  # ~500 tokens
```

### 3. For batch processing

Ingest multiple documents, then search across all:
```python
ingest_context(doc1, "paper1")
ingest_context(doc2, "paper2")
ingest_context(doc3, "paper3")

# Search across all
search_semantic("quantum entanglement")  # Searches all files
```

---

## Using SCAR Enhancements (NEW!)

SCAR adds advanced features for even better performance:

### What SCAR Provides

1. **Learnable Compression**: 4× smaller embeddings (384D → 96D)
2. **Semantic Alignment**: 15-25% better retrieval relevance
3. **Adaptive Fidelity**: Auto-adjusts detail level based on query

### Try SCAR

```bash
python examples/scar_demo.py
```

This demonstrates:
- Embedding compression savings
- Alignment-guided search
- Adaptive fidelity modulation

### Training SCAR (Optional)

To train SCAR on your domain-specific documents:

```python
from src.training_utils import SCARTrainer, TrainingConfig

# See examples/scar_demo.py for full example
config = TrainingConfig(batch_size=32, num_epochs=10)
trainer = SCARTrainer(model, config)
trainer.train_compressor(train_dataset, eval_dataset)
trainer.save_checkpoint("my_scar_model.pt")
```

See [SCAR_PAPER_SUMMARY.md](docs/SCAR_PAPER_SUMMARY.md) for implementation details.

---

## Next Steps

1. ✅ **Run tests** to verify installation: `python tests/test_token_savings.py`
2. ✅ **Try examples** to see it in action: `python examples/scar_demo.py`
3. ✅ **Configure Claude Desktop** with the MCP server (see above)
4. ✅ **Read** [GETTING_STARTED.md](GETTING_STARTED.md) for complete walkthrough
5. 📖 **Explore** [README.md](README.md) for advanced features
6. 🧪 **Experiment** with blind spot detection and SCAR
7. 🚀 **Try** multi-document analysis

---

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/token-saver-5000/issues)
- **Documentation:** [README.md](README.md)
- **Examples:** [examples/](examples/)

---

**Happy compressing! 🧠🚀**
