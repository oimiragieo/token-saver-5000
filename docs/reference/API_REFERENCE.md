# API Reference

**Complete Python API documentation for Token Saver 5000**

---

## Table of Contents

- [Core Modules](#core-modules)
- [SemanticCompressor](#semanticcompressor)
- [FocusManager (AFM)](#focusmanager-afm)
- [CodeSemanticCompressor](#codesemanticcompressor)
- [MultiModalSemanticCompressor](#multimodalsemanticcompressor)
- [BlindSpotDetector](#blindspotdetector)
- [MCP Server](#mcp-server)
- [Utility Modules](#utility-modules)

---

## Core Modules

### Module Structure

```
src/
├── semantic_compressor.py      # Main document compression
├── afm.py                      # Adaptive Focus Memory (dialogue)
├── code_compressor.py          # Code-specific compression
├── multimodal_compressor.py    # Text + Code + Images
├── blind_spot_detector.py      # Hallucination detection
├── scar_compressor.py          # Learnable compression
├── server.py                   # MCP server (44 tools)
├── persistence.py              # ChromaDB/JSON storage
├── resource_manager.py         # Resource limits
├── adaptive_rate_allocator.py  # Dynamic compression
├── semantic_ssim.py            # Quality metrics
└── training_utils.py           # Model training helpers
```

---

## SemanticCompressor

**File:** `src/semantic_compressor.py`

Main class for document compression with 80-95% token reduction.

### Class Definition

```python
class SemanticCompressor:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
        skeleton_ratio: float = 0.2
    )
```

**Parameters:**
- `model_name` (str): Sentence transformer model name. Default: `"all-MiniLM-L6-v2"`
- `similarity_threshold` (float): Cosine similarity threshold for graph edges (0-1). Default: `0.75`
- `skeleton_ratio` (float): Ratio of anchor nodes to show in skeleton (0-1). Default: `0.2` (20%)

**Attributes:**
- `model` (SentenceTransformer): Embedding model
- `chunks` (Dict[str, Chunk]): All ingested chunks (nodes)
- `graphs` (Dict[str, nx.DiGraph]): Semantic graphs by file_id
- `file_metadata` (Dict[str, dict]): Metadata by file_id

### Methods

#### `ingest_file()`

Ingest and compress a document into a semantic graph.

```python
def ingest_file(
    self,
    content: str,
    file_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> IngestionResult
```

**Parameters:**
- `content` (str): Raw document text
- `file_id` (str): Unique identifier (e.g., "paper_1")
- `metadata` (dict, optional): Custom metadata

**Returns:**
- `IngestionResult` with fields:
  - `file_id` (str)
  - `num_nodes` (int): Number of chunks created
  - `num_edges` (int): Number of semantic connections
  - `compression_ratio` (float): Original tokens / skeleton tokens
  - `token_savings_pct` (float): Percentage reduction
  - `original_tokens` (int)
  - `skeleton_tokens` (int)
  - `processing_time` (float): Seconds

**Example:**
```python
from src.semantic_compressor import SemanticCompressor

compressor = SemanticCompressor()
result = compressor.ingest_file(document_text, "research_paper")

print(f"Compression: {result.compression_ratio:.1f}×")
print(f"Token savings: {result.token_savings_pct:.1f}%")
```

---

#### `read_skeleton()`

Get compressed skeleton view (80-95% reduction).

```python
def read_skeleton(
    self,
    file_id: str,
    format: str = "text"
) -> str
```

**Parameters:**
- `file_id` (str): Document identifier
- `format` (str): Output format ("text", "json", "markdown"). Default: "text"

**Returns:**
- `str`: Formatted skeleton showing anchor nodes and hidden node count

**Example:**
```python
skeleton = compressor.read_skeleton("research_paper")
print(skeleton)
# Output:
# ⭐ research_paper_n0: Introduction to Quantum Computing
# ⭐ research_paper_n5: Error Correction Techniques
# ...
# 📦 72 hidden nodes available
```

---

#### `modulate_region()`

Retrieve specific nodes at chosen fidelity level.

```python
def modulate_region(
    self,
    node_ids: List[str],
    fidelity_level: FidelityLevel,
    file_id: Optional[str] = None
) -> str
```

**Parameters:**
- `node_ids` (List[str]): List of node IDs to retrieve
- `fidelity_level` (FidelityLevel): Detail level (ABSTRACT | OUTLINE | STRUCTURE | DETAILED | RAW)
- `file_id` (str, optional): Document ID (for validation)

**Returns:**
- `str`: Formatted content at specified fidelity

**FidelityLevel enum:**
```python
from src.semantic_compressor import FidelityLevel

FidelityLevel.ABSTRACT   # ~10 tokens/node
FidelityLevel.OUTLINE    # ~30 tokens/node
FidelityLevel.STRUCTURE  # ~50 tokens/node
FidelityLevel.DETAILED   # ~100 tokens/node
FidelityLevel.RAW        # Original content
```

**Example:**
```python
nodes = ["research_paper_n5", "research_paper_n12"]
content = compressor.modulate_region(
    node_ids=nodes,
    fidelity_level=FidelityLevel.STRUCTURE
)
```

---

#### `search_semantic()`

Semantic vector search across documents.

```python
def search_semantic(
    self,
    query: str,
    file_id: Optional[str] = None,
    top_k: int = 5
) -> List[SearchResult]
```

**Parameters:**
- `query` (str): Natural language search query
- `file_id` (str, optional): Limit to specific document (None = search all)
- `top_k` (int): Number of results. Default: 5

**Returns:**
- `List[SearchResult]` with fields:
  - `node_id` (str)
  - `similarity` (float): Cosine similarity score (0-1)
  - `preview` (str): First ~100 characters
  - `importance` (float): PageRank score
  - `tokens` (int)

**Example:**
```python
results = compressor.search_semantic(
    query="error correction techniques",
    file_id="research_paper",
    top_k=5
)

for result in results:
    print(f"{result.similarity:.2f} - {result.preview}")
```

---

#### `get_stats()`

Get document statistics.

```python
def get_stats(
    self,
    file_id: str
) -> Dict[str, Any]
```

**Returns:**
- Dictionary with:
  - `num_nodes` (int)
  - `num_anchors` (int)
  - `num_edges` (int)
  - `original_tokens` (int)
  - `skeleton_tokens` (int)
  - `compression_ratio` (float)
  - `graph_density` (float)
  - `ssim_score` (float): Structural similarity

---

## FocusManager (AFM)

**File:** `src/afm.py`

Adaptive Focus Memory for dialogue compression with ~66% token reduction.

### Class Definition

```python
class FocusManager:
    def __init__(
        self,
        config: AFMConfig
    )
```

**AFMConfig:**
```python
@dataclass
class AFMConfig:
    tau_high: float = 0.45        # Threshold for FULL fidelity
    tau_mid: float = 0.25         # Threshold for COMPRESSED fidelity
    half_life: int = 12           # Recency decay (turns until 50%)
    use_llm_importance: bool = False   # Use LLM for importance (expensive)
    use_llm_compression: bool = False  # Use LLM for compression (expensive)
```

**Attributes:**
- `messages` (List[Message]): All dialogue turns
- `turn_counter` (int): Current turn number
- `config` (AFMConfig): Configuration

### Methods

#### `add_message()`

Add message to dialogue history.

```python
def add_message(
    self,
    role: str,
    content: str,
    importance: Optional[str] = None
) -> Message
```

**Parameters:**
- `role` (str): "user", "assistant", or "system"
- `content` (str): Message text
- `importance` (str, optional): Override auto-classification ("CRITICAL", "RELEVANT", "TRIVIAL")

**Returns:**
- `Message` object with assigned importance

**Example:**
```python
from src.afm import FocusManager, AFMConfig

manager = FocusManager(AFMConfig())

manager.add_message("user", "I have a severe peanut allergy")
# Auto-classified as CRITICAL

manager.add_message("user", "What Thai food should I try?")
# Auto-classified as RELEVANT
```

---

#### `build_context()`

Build optimized context for current query.

```python
def build_context(
    self,
    current_query: str,
    budget_tokens: int,
    system_preamble: Optional[str] = None
) -> Tuple[str, ContextStats]
```

**Parameters:**
- `current_query` (str): User's current query
- `budget_tokens` (int): Maximum tokens allowed
- `system_preamble` (str, optional): System message to prepend

**Returns:**
- Tuple of:
  - `context` (str): Formatted context under budget
  - `stats` (ContextStats): Statistics about compression

**ContextStats:**
```python
@dataclass
class ContextStats:
    total_messages: int
    full_count: int           # Messages at FULL fidelity
    compressed_count: int     # Messages at COMPRESSED fidelity
    placeholder_count: int    # Messages at PLACEHOLDER fidelity
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float  # compressed / original
```

**Example:**
```python
context, stats = manager.build_context(
    current_query="What street food should I try?",
    budget_tokens=800
)

print(f"Token savings: {(1 - stats.compression_ratio) * 100:.0f}%")
print(f"Fidelity: {stats.full_count} FULL, {stats.compressed_count} COMPRESSED")
```

---

#### `export_history()` / `import_history()`

Save and restore dialogue state.

```python
def export_history(self, session_id: str = "default") -> dict:
    """Export conversation state to dictionary"""

def import_history(self, session_id: str = "default") -> None:
    """Restore conversation state from file"""
```

**Example:**
```python
# Save conversation
export_data = manager.export_history("my_session")
with open("session.json", "w") as f:
    json.dump(export_data, f)

# Restore later
manager.import_history("my_session")
```

---

## CodeSemanticCompressor

**File:** `src/code_compressor.py`

Language-aware code compression with AST parsing.

### Class Definition

```python
class CodeSemanticCompressor(SemanticCompressor):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        supported_languages: List[str] = ["python", "javascript", "typescript", "java", "cpp", "go"]
    )
```

### Methods

#### `ingest_code_file()`

Ingest code with language-specific parsing.

```python
def ingest_code_file(
    self,
    code: str,
    file_id: str,
    language: str = "python",
    filepath: Optional[str] = None
) -> IngestionResult
```

**Parameters:**
- `code` (str): Source code
- `file_id` (str): Identifier
- `language` (str): Programming language
- `filepath` (str, optional): Original file path

**Supported languages:**
- `"python"` - Uses `ast` module
- `"javascript"` / `"typescript"` - Uses tree-sitter
- `"java"`, `"cpp"`, `"go"` - Uses tree-sitter

**Example:**
```python
from src.code_compressor import CodeSemanticCompressor

code_comp = CodeSemanticCompressor()
result = code_comp.ingest_code_file(
    code=python_code,
    file_id="utils",
    language="python",
    filepath="src/utils.py"
)
```

---

#### `generate_code_skeleton()`

Generate code outline (signatures only).

```python
def generate_code_skeleton(
    self,
    file_id: str,
    include_docstrings: bool = True
) -> str
```

**Returns:**
- Code skeleton with:
  - All imports
  - Class definitions with method signatures
  - Function signatures
  - No implementations (replaced with `...`)

**Example:**
```python
skeleton = code_comp.generate_code_skeleton("utils")
# Returns:
# import numpy as np
#
# class DataProcessor:
#     def __init__(self, config: dict): ...
#     def process(self, data: List[float]) -> np.ndarray: ...
```

---

## MultiModalSemanticCompressor

**File:** `src/multimodal_compressor.py`

Cross-modal compression for text, code, and images.

### Class Definition

```python
class MultiModalSemanticCompressor(SemanticCompressor):
    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        clip_model: str = "openai/clip-vit-base-patch32"
    )
```

### Methods

#### `ingest_mixed_content()`

Ingest mixed-modality content.

```python
def ingest_mixed_content(
    self,
    content_items: List[Dict[str, Any]],
    doc_id: str
) -> IngestionResult
```

**Parameters:**
- `content_items` (List[dict]): List of content items with:
  - `type` (str): "text", "code", or "image"
  - `content` (str | bytes): Content data
  - `language` (str, optional): For code
  - `description` (str, optional): For images

**Example:**
```python
from src.multimodal_compressor import MultiModalSemanticCompressor

mm_comp = MultiModalSemanticCompressor()

content = [
    {'type': 'text', 'content': readme_text},
    {'type': 'code', 'content': python_code, 'language': 'python'},
    {'type': 'image', 'content': diagram_bytes, 'description': 'Architecture diagram'}
]

result = mm_comp.ingest_mixed_content(content, "my_project")
```

---

#### `search_cross_modal()`

Search across all modalities.

```python
def search_cross_modal(
    self,
    query: str,
    doc_id: str,
    top_k: int = 10
) -> List[SearchResult]
```

**Returns:**
- Mixed results from text, code, and images

---

## BlindSpotDetector

**File:** `src/blind_spot_detector.py`

Detects missing context in AI responses.

### Class Definition

```python
class BlindSpotDetector:
    def __init__(
        self,
        compressor: SemanticCompressor
    )
```

### Methods

#### `check_blind_spots()`

Detect if response mentions concepts not in retrieved context.

```python
def check_blind_spots(
    self,
    ai_response: str,
    retrieved_nodes: List[str],
    file_id: str
) -> BlindSpotResult
```

**Returns:**
- `BlindSpotResult` with:
  - `blind_spot_detected` (bool)
  - `confidence` (float)
  - `missing_concepts` (List[str])
  - `suggested_nodes` (List[SearchResult])

**Example:**
```python
from src.blind_spot_detector import BlindSpotDetector

detector = BlindSpotDetector(compressor)

result = detector.check_blind_spots(
    ai_response="Quantum error correction uses surface codes...",
    retrieved_nodes=["paper_n12", "paper_n13"],
    file_id="quantum_paper"
)

if result.blind_spot_detected:
    print(f"Missing concepts: {result.missing_concepts}")
    print(f"Suggested nodes: {[n.node_id for n in result.suggested_nodes]}")
```

---

## MCP Server

**File:** `src/server.py`

MCP server exposing 44 tools via stdio transport.

### Server Setup

```python
from src.server import SemanticModulatorServer

server = SemanticModulatorServer()
# Server auto-starts when run as module: python -m src.server
```

### Available Tools

See [**MCP_TOOLS_GUIDE.md**](MCP_TOOLS_GUIDE.md) for complete documentation of all 44 tools.

**Quick reference:**
- `ingest_context` - Ingest document
- `read_skeleton` - Get compressed view
- `modulate_region` - Retrieve at fidelity
- `search_semantic` - Semantic search
- `check_blind_spots` - Detect missing context
- `detect_hallucination` - Validate grounding
- `get_stats` - Document statistics
- `adapt_to_context_window` - Dynamic compression
- `multilevel_encode` - Progressive loading
- `afm_add_message` - Add dialogue turn
- `afm_build_context` - Build compressed context
- `afm_get_stats` - Dialogue statistics
- `afm_clear_history` - Reset dialogue
- `list_documents` - List all documents
- `afm_export_history` - Save conversation
- `afm_import_history` - Restore conversation
- `delete_document` - Delete document

---

## Utility Modules

### PersistenceManager

**File:** `src/persistence.py`

Handles document storage (ChromaDB or JSON fallback).

```python
from src.persistence import PersistenceManager

persistence = PersistenceManager(backend="chromadb")  # or "json"

# Auto-used by server for save/load
persistence.save_document(file_id, graph_data, embeddings)
data = persistence.load_document(file_id)
```

---

### ResourceManager

**File:** `src/resource_manager.py`

Enforces resource limits.

```python
from src.resource_manager import ResourceManager, ResourceLimits

limits = ResourceLimits(
    max_document_size_mb=100.0,
    max_total_storage_mb=1024.0,
    max_documents=1000,
    max_memory_mb=2048.0
)

manager = ResourceManager(limits)
manager.check_limits(content, file_id)  # Raises ResourceLimitExceeded if exceeded
```

---

### SemanticSSIM

**File:** `src/semantic_ssim.py`

Calculate structural similarity between graphs.

```python
from src.semantic_ssim import calculate_ssim

ssim_score = calculate_ssim(original_graph, compressed_graph)
# Returns float in [0, 1], target > 0.7
```

---

### TOONSerializer

**File:** `src/toon_serializer.py`

Serialize outputs to TOON format for additional ~40% token savings.

```python
from src.toon_serializer import TOONSerializer, OutputFormat

serializer = TOONSerializer()

# Serialize search results
toon_output = serializer.serialize_search_results(results)

# Or use format_response helper
from src.toon_serializer import format_response

output = format_response(
    data=results,
    format_type=OutputFormat.TOON  # or JSON, TEXT
)
```

---

## Type Definitions

### Common Types

```python
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Fidelity levels
class FidelityLevel(Enum):
    ABSTRACT = "ABSTRACT"
    OUTLINE = "OUTLINE"
    STRUCTURE = "STRUCTURE"
    DETAILED = "DETAILED"
    RAW = "RAW"

# Ingestion result
@dataclass
class IngestionResult:
    file_id: str
    num_nodes: int
    num_edges: int
    compression_ratio: float
    token_savings_pct: float
    original_tokens: int
    skeleton_tokens: int
    processing_time: float

# Search result
@dataclass
class SearchResult:
    node_id: str
    similarity: float
    preview: str
    importance: float
    tokens: int

# Chunk (node)
@dataclass
class Chunk:
    id: str
    text: str
    embedding: np.ndarray
    importance: float
    tokens: int
    metadata: Dict[str, Any]
```

---

## Example Workflows

### Basic Document Compression

```python
from src.semantic_compressor import SemanticCompressor, FidelityLevel

# Initialize
compressor = SemanticCompressor()

# Ingest
result = compressor.ingest_file(document_text, "my_doc")
print(f"Compressed: {result.compression_ratio:.1f}×")

# Read skeleton
skeleton = compressor.read_skeleton("my_doc")

# Search
results = compressor.search_semantic("error handling", "my_doc", top_k=3)

# Retrieve details
content = compressor.modulate_region(
    node_ids=[r.node_id for r in results],
    fidelity_level=FidelityLevel.STRUCTURE
)
```

### Dialogue Memory with Documents

```python
from src.semantic_compressor import SemanticCompressor
from src.afm import FocusManager, AFMConfig

# Setup
compressor = SemanticCompressor()
manager = FocusManager(AFMConfig())

# User uploads document
compressor.ingest_file(manual_text, "user_manual")

# Conversation
manager.add_message("user", "I uploaded our company manual")
manager.add_message("assistant", "Got it! What would you like to know?")
manager.add_message("user", "What's the vacation policy?")

# Build context with document augmentation
dialogue_context, stats = manager.build_context(
    current_query="What's the vacation policy?",
    budget_tokens=1000
)

# Search relevant manual sections
doc_results = compressor.search_semantic(
    query="vacation policy",
    file_id="user_manual",
    top_k=3
)

doc_context = compressor.modulate_region(
    node_ids=[r.node_id for r in doc_results],
    fidelity_level=FidelityLevel.STRUCTURE
)

# Combine both contexts
full_context = dialogue_context + "\n\n" + doc_context
```

---

## Performance Considerations

### Embedding Caching

Embeddings are computed once and cached:

```python
# First call: computes embeddings
result1 = compressor.search_semantic("query 1", "doc")  # ~100ms

# Subsequent calls: uses cached embeddings
result2 = compressor.search_semantic("query 2", "doc")  # ~10ms
```

### Memory Usage

Approximate memory per document:

```python
document_memory_mb = (
    num_nodes * embedding_dim * 4 bytes +  # Embeddings (float32)
    num_edges * 16 bytes +                  # Graph edges
    total_text_bytes                        # Original text
) / 1024 / 1024

# Example: 100 nodes, 500 edges, 100KB text
# = (100 * 384 * 4 + 500 * 16 + 100000) / 1024 / 1024
# ≈ 0.24 MB per document
```

### Speed Optimization

```python
# Batch ingestion
for doc in documents:
    # Uses same model instance (faster)
    compressor.ingest_file(doc.text, doc.id)

# Parallel search (for multiple queries)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(compressor.search_semantic, query, "doc")
        for query in queries
    ]
    results = [f.result() for f in futures]
```

---

## Error Handling

### Common Exceptions

```python
from src.semantic_compressor import (
    DocumentNotFoundError,
    NodeNotFoundError,
    ResourceLimitExceeded
)

try:
    skeleton = compressor.read_skeleton("unknown_doc")
except DocumentNotFoundError as e:
    print(f"Document not found: {e}")
    available = compressor.list_documents()

try:
    content = compressor.modulate_region(["invalid_node"], FidelityLevel.RAW)
except NodeNotFoundError as e:
    print(f"Node not found: {e}")

try:
    result = compressor.ingest_file(huge_document, "big_doc")
except ResourceLimitExceeded as e:
    print(f"Resource limit exceeded: {e}")
```

---

## For More Information

- [**HOW_IT_WORKS.md**](HOW_IT_WORKS.md) - Technical deep dive
- [**MCP_TOOLS_GUIDE.md**](MCP_TOOLS_GUIDE.md) - MCP tools reference
- [**ARCHITECTURE.md**](ARCHITECTURE.md) - System architecture
- [**Examples**](examples/) - Hands-on code examples

---

**Last Updated:** 2024-11-22
