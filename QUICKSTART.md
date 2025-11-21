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

### Run the example script

```bash
python examples/example_usage.py
```

This will:
1. Load the embedding model (~80MB download on first run)
2. Ingest a sample document
3. Demonstrate compression, search, and blind spot detection
4. Show ~90% token savings

**Expected output:**
```
Initializing Semantic Compressor...
Loading embedding model: all-MiniLM-L6-v2

Step 1: Ingesting Document with Semantic Compression
Original tokens: 2,847
Skeleton tokens: 287
Compression ratio: 9.9x
Token savings: 89.9%
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

## Next Steps

1. ✅ Run `examples/example_usage.py` to see it in action
2. ✅ Configure Claude Desktop with the MCP server
3. ✅ Try ingesting your own documents
4. 📖 Read the full [README.md](README.md) for advanced features
5. 🧪 Experiment with blind spot detection
6. 🚀 Explore multi-document analysis

---

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/token-saver-5000/issues)
- **Documentation:** [README.md](README.md)
- **Examples:** [examples/](examples/)

---

**Happy compressing! 🧠🚀**
