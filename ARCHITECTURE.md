# 🏗️ Architecture Documentation

## System Overview

Semantic Modulator implements a **three-layer architecture** for adaptive semantic compression:

```
┌─────────────────────────────────────────────────────────┐
│              Layer 3: MCP Interface                      │
│  (MCP Server exposing tools to AI agents)               │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │
┌─────────────────────────────────────────────────────────┐
│         Layer 2: Semantic Intelligence                   │
│  (Blind Spot Detection, Hallucination Detection)        │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │
┌─────────────────────────────────────────────────────────┐
│         Layer 1: Core Compression Engine                │
│  (Graph-based semantic compression with fidelity)       │
└─────────────────────────────────────────────────────────┘
```

---

## Layer 1: Core Compression Engine

**File:** `src/semantic_compressor.py`

### Components

#### 1.1 Text Chunker
```
Input: Raw text
Output: List of semantic chunks

Algorithm:
1. Split by paragraphs (\n\n)
2. Combine small chunks up to ~512 tokens
3. Split large chunks by sentences
4. Preserve semantic boundaries

Example:
"Long document..." → ["Chunk 1...", "Chunk 2...", ...]
```

#### 1.2 Embedding Generator
```
Input: List of text chunks
Output: Vector embeddings (384-dim)

Model: all-MiniLM-L6-v2 (local)
- Size: ~80MB
- Speed: ~1000 sentences/sec on CPU
- Quality: 0.68 Spearman correlation on STS

Example:
"Quantum computing..." → [0.23, -0.15, 0.87, ...]
```

#### 1.3 Graph Builder
```
Input: Chunks + Embeddings
Output: NetworkX graph with weighted edges

Algorithm:
1. Calculate pairwise cosine similarity
2. Create edge if similarity > threshold (0.75)
3. Weight edges by similarity

Purpose: Preserves global structure (FPQE principle)

Example Graph:
    [Node 0] ---- 0.87 ---- [Node 5]
       |                       |
      0.76                   0.82
       |                       |
    [Node 1] ---- 0.79 ---- [Node 10]
```

#### 1.4 Importance Calculator
```
Input: Graph
Output: Importance scores per node

Algorithm: PageRank (iterative)
- Nodes with many high-similarity connections = important
- Captures centrality in the semantic network

Example:
Node 0: importance = 0.052
Node 5: importance = 0.087 ← More central
Node 1: importance = 0.034
```

#### 1.5 Skeleton Generator
```
Input: Graph + Importance scores
Output: Compressed skeleton text

Algorithm:
1. Sort nodes by importance
2. Select top N% (default: 20%) as "anchors"
3. Show anchors with summaries + entities
4. Show others as references only

Example Output:
[node_0] ⭐ ANCHOR (importance: 0.087)
  Summary: Quantum computing requires error correction...
  Key entities: quantum, error, correction

[node_1] 📦 Detail hidden (use modulate_region)

Compression: 45,000 tokens → 2,300 tokens (19.5x)
```

#### 1.6 Adaptive Modulator
```
Input: Node IDs + Fidelity level
Output: Content at requested detail level

Fidelity Levels:
- ABSTRACT: 1-sentence summary (~10 tokens)
- STRUCTURE: Summary + entities + metadata (~50 tokens)
- RAW: Full original text (variable)

Purpose: Channel adaptation (JSCCM principle)
→ Only transmit the detail level actually needed
```

---

## Layer 2: Semantic Intelligence

**Files:**
- `src/blind_spot_detector.py`

### Components

#### 2.1 Blind Spot Detector

**Problem:** AI might generate response based on incomplete context

**Solution:** Post-hoc validation of what was retrieved vs. what's relevant

```
Algorithm:
1. Embed AI's generated response → vector
2. Compare to ALL nodes in document graph
3. Calculate: urgency = similarity × importance
4. Flag nodes with high urgency that weren't retrieved

Example:
AI says: "Gate fidelity is 99.7%"
Retrieved: [node_5]
Missed: [node_23] urgency=0.68 "Actually, fidelities appear
         higher than true due to cross-talk..."

Alert: ⚠️ CRITICAL blind spot detected!
```

**Urgency Scoring:**
```python
urgency = cosine_similarity(response, node) × node.importance

if urgency >= 0.6:  → CRITICAL (auto-inject)
elif urgency >= 0.4: → HIGH (suggest retrieval)
elif urgency >= 0.25: → MEDIUM (mention)
else: → LOW (ignore)
```

#### 2.2 Hallucination Detector

**Problem:** AI might fabricate content not in source

**Solution:** Check if response is grounded in any source node

```
Algorithm:
1. Embed AI response
2. Find max similarity to any document node
3. If max_similarity < threshold (0.3):
   → Likely hallucination

Example:
Response: "The document discusses Majorana fermions..."
Max similarity to any node: 0.12
Alert: 🚨 Response not grounded in source material!
```

---

## Layer 3: MCP Interface

**File:** `src/server.py`

### MCP Server Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          MCP Server (stdio transport)                        │
│                                                               │
│  Handlers:                                                    │
│  ┌───────────────────────────────────────────┐              │
│  │ @list_tools() → Return 16 tool schemas    │              │
│  │ @call_tool() → Execute tool logic         │              │
│  └───────────────────────────────────────────┘              │
│                                                               │
│  State Management:                                            │
│  • compressor: SemanticCompressor instance                   │
│  • blind_spot_detector: BlindSpotDetector                    │
│  • focus_manager: FocusManager (AFM)                         │
│  • persistence: PersistenceManager (NEW!)                    │
│  • resource_manager: ResourceManager (NEW!)                  │
│  • retrieval_history: Dict[file_id, node_ids]               │
│                                                               │
│  Features:                                                    │
│  • Auto-load documents on server start                       │
│  • Auto-save documents on ingest                             │
│  • Resource limit enforcement                                │
│  • AFM conversation export/import                            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Complete MCP Tool List (16 tools)

#### Document Compression Tools (9)
1. `ingest_context` - Ingest and compress documents
2. `read_skeleton` - View compressed skeleton
3. `modulate_region` - Retrieve at chosen fidelity
4. `search_semantic` - Vector similarity search
5. `check_blind_spots` - Detect missed context
6. `detect_hallucination` - Validate response grounding
7. `get_stats` - Document statistics
8. `adapt_to_context_window` - JSCCM adaptation
9. `multilevel_encode` - Multi-level encoding

#### Dialogue Memory Tools (4)
10. `afm_add_message` - Add dialogue turn
11. `afm_build_context` - Build optimized context
12. `afm_get_stats` - Dialogue statistics
13. `afm_clear_history` - Reset dialogue

#### Discovery & Persistence Tools (3)
14. `list_documents` - Document inventory
15. `afm_export_history` - Save conversation state
16. `afm_import_history` - Restore conversation state

### Tool Implementations

#### Tool: `ingest_context`
```
Flow:
1. Receive: text, file_id, metadata
2. Call: compressor.ingest_file()
3. Initialize: retrieval_history[file_id] = []
4. Return: Compression stats + skeleton preview
```

#### Tool: `read_skeleton`
```
Flow:
1. Receive: file_id
2. Call: compressor.read_skeleton()
3. Return: Formatted skeleton view
```

#### Tool: `modulate_region`
```
Flow:
1. Receive: node_ids[], fidelity_level
2. Track: Add to retrieval_history
3. Call: compressor.modulate_region()
4. Return: Content at requested fidelity
```

#### Tool: `check_blind_spots`
```
Flow:
1. Receive: ai_response, file_id, retrieved_nodes
2. Call: blind_spot_detector.analyze_response()
3. If critical spots found:
   → Auto-suggest modulate_region() calls
4. Return: Formatted report with recommendations
```

---

## Data Flow: Complete Example

### Scenario: AI analyzing a technical paper

```
┌─────────────────────────────────────────────────┐
│ 1. USER: Analyze this 100-page paper            │
└─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ 2. AI → ingest_context(text, "paper1")          │
└─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ 3. COMPRESSOR                                    │
│    • Chunk: 150 chunks                           │
│    • Embed: 150 × 384-dim vectors               │
│    • Graph: 150 nodes, 243 edges                │
│    • PageRank: Calculate importance             │
│    • Skeleton: Show top 30 nodes                │
│    • Result: 45,000 → 2,300 tokens (19.5x)     │
└─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ 4. AI → read_skeleton("paper1")                  │
│    Receives: Compressed structure                │
│    [paper1_n0] ⭐ ANCHOR: Introduction...        │
│    [paper1_n5] ⭐ ANCHOR: Main theorem...        │
│    [paper1_n10] 📦 Hidden...                     │
└─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ 5. AI → search_semantic("error correction")      │
│    Returns: [paper1_n5, paper1_n12, paper1_n23] │
└─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ 6. AI → modulate_region([paper1_n5], "RAW")     │
│    Receives: Full content of node 5             │
│    (Tracked in retrieval_history)               │
└─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ 7. AI generates response based on node 5        │
│    "The paper presents error correction..."     │
└─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ 8. AI → check_blind_spots(response, "paper1",   │
│                           ["paper1_n5"])         │
└─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ 9. BLIND SPOT DETECTOR                           │
│    • Embed response                              │
│    • Compare to all 150 nodes                    │
│    • Find: paper1_n23 has urgency=0.72          │
│      (high similarity but not retrieved)        │
│    • Alert: ⚠️ CRITICAL blind spot!              │
│    • Suggest: Retrieve paper1_n23               │
└─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ 10. AI → modulate_region([paper1_n23], "RAW")   │
│     Retrieves missed content                     │
│     Updates response with new context            │
└─────────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│ 11. Final result: Accurate analysis using        │
│     ~4,500 tokens instead of 45,000 tokens      │
│     (90% savings, 0% accuracy loss)             │
└─────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Chunking | O(n) | n = document length |
| Embedding | O(k) | k = number of chunks |
| Graph building | O(k²) | Pairwise similarity |
| PageRank | O(E × i) | E = edges, i = iterations (~20) |
| Skeleton generation | O(k log k) | Sorting by importance |
| Modulation | O(1) | Direct lookup |
| Blind spot detection | O(k) | Linear scan |

### Space Complexity

| Component | Space | Example (100-page doc) |
|-----------|-------|------------------------|
| Original text | O(n) | ~45,000 tokens |
| Chunks | O(k) | 150 chunks |
| Embeddings | O(k × d) | 150 × 384 floats = 230KB |
| Graph | O(k²) worst | ~22,500 edges max |
| Actual graph | O(E) | ~500 edges (sparse) |

### Benchmark Results

Measured on M1 MacBook Pro:

| Document Size | Chunks | Ingest Time | Skeleton Time | Search Time |
|---------------|--------|-------------|---------------|-------------|
| 5K tokens | 15 | 0.8s | 0.05s | 0.1s |
| 20K tokens | 60 | 2.1s | 0.15s | 0.3s |
| 50K tokens | 150 | 4.5s | 0.35s | 0.7s |
| 100K tokens | 300 | 9.2s | 0.80s | 1.5s |

**Note:** Ingest is one-time cost. Subsequent operations are fast.

---

## Scalability Considerations

### Current Limits (as of v0.2.0)
- **Persistent storage:** ChromaDB/JSON fallback for documents and AFM history
- **Resource limits:** 100MB per document, 1GB total, 1000 documents max
- **Single process:** No parallelization
- **Max document size:** 100MB (configurable)

### Optimization Strategies

#### For Large Documents (>100K tokens)
```python
# Option 1: Increase chunk size
compressor = SemanticCompressor(
    chunk_size=1024  # Instead of 512
)
# Result: Fewer nodes, less memory, faster graph

# Option 2: Reduce graph density
compressor = SemanticCompressor(
    similarity_threshold=0.85  # Instead of 0.75
)
# Result: Fewer edges, faster PageRank

# Option 3: Use approximate PageRank
# Fewer iterations for large graphs
```

#### For Multiple Documents
```python
# Option 1: Separate graphs per document
# Current approach: O(n) memory per doc

# Option 2: Shared embedding space
# Future: Cross-document semantic links
```

---

## Security & Privacy

### Local-First Design
- ✅ All compression happens locally
- ✅ No external API calls for embeddings
- ✅ No data leaves the system
- ✅ Model runs on CPU (no cloud GPU needed)

### Data Handling (v0.2.0)
- Documents: Persistent storage via ChromaDB or JSON fallback
- Auto-save: Documents automatically persisted on ingest
- Auto-load: Persisted documents restored on server start
- AFM History: Exportable/importable conversation state
- Location: `.semantic_modulator_data/` directory

---

## Extension Points

### 1. Custom Embedding Models
```python
class SemanticCompressor:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        # Swap in any sentence-transformer model
```

### 2. Custom Importance Metrics
```python
# Current: PageRank
importance = nx.pagerank(graph)

# Alternative: Betweenness centrality
importance = nx.betweenness_centrality(graph)

# Alternative: Custom scoring
importance = custom_scorer(graph, metadata)
```

### 3. Custom Fidelity Levels
```python
class FidelityLevel(Enum):
    ABSTRACT = 1
    STRUCTURE = 2
    RAW = 3
    DETAILED = 4  # Add custom level
```

### 4. Persistent Storage
```python
# Add ChromaDB backend
import chromadb

client = chromadb.Client()
collection = client.create_collection("semantic_chunks")

# Store embeddings persistently
collection.add(
    embeddings=embeddings,
    documents=chunks,
    ids=node_ids
)
```

---

## Future Architecture (Roadmap)

### ✅ Version 0.2.0 (COMPLETED)
```
┌─────────────────────────────────────┐
│       Persistent Storage Layer       │
│  • ChromaDB/JSON for document storage │
│  • Resource management & limits       │
│  • AFM export/import                 │
│  • Auto-load/save on server cycle    │
└─────────────────────────────────────┘
```

### Version 0.3.0 (In Progress)
```
┌─────────────────────────────────────┐
│    Cross-Document Intelligence       │
│  • Multi-doc graphs                  │
│  • Contradiction detection           │
│  • Citation tracking                 │
└─────────────────────────────────────┘
```

### Version 1.0.0
```
┌─────────────────────────────────────┐
│       Distributed Architecture       │
│  • Federated search across servers   │
│  • Streaming results (WebSocket)     │
│  • Collaborative knowledge graphs    │
└─────────────────────────────────────┘
```

---

## Debugging

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Visualize Graphs
```python
import networkx as nx
import matplotlib.pyplot as plt

graph = compressor.graphs["file_id"]
nx.draw(graph, with_labels=True)
plt.show()
```

### Inspect Node Details
```python
node = compressor.chunks["file_id_n5"]
print(f"Text: {node.text}")
print(f"Embedding: {node.embedding[:10]}...")  # First 10 dims
print(f"Importance: {node.importance}")
print(f"Metadata: {node.metadata}")
```

---

## References

### Research Papers
1. **Semantic Communication:** "Joint Semantic-Channel Coding and Modulation"
2. **Fidelity-Preserving Encoding:** "Structure-Preserving Quantization"

### Technologies
- **MCP:** Model Context Protocol (Anthropic)
- **Sentence Transformers:** SBERT for embeddings
- **NetworkX:** Graph analysis library
- **scikit-learn:** Cosine similarity calculations

---

## Contact

For architecture questions or contributions:
- **GitHub Issues:** Technical discussions
- **Pull Requests:** Code contributions welcome
- **Email:** architecture@example.com

---

**Last Updated:** 2025-11-22
**Version:** 0.2.0
**Status:** Production-Ready with Persistent Storage
