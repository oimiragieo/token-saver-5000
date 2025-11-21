# src/ Directory

## Overview
Core source code implementing the Semantic Modulator MCP Server for adaptive semantic fidelity compression with 80-95% token reduction.

## Files

### Package Initialization
- **`__init__.py`** (216 bytes)
  - Package initialization file
  - Exports version: `0.1.0`
  - Minimal boilerplate for Python package structure

### Core Compression Modules

#### 1. **`semantic_compressor.py`** (18,537 bytes)
**Purpose**: Core semantic compression using embeddings and graph analysis

**Key Classes**:
- `FidelityLevel(Enum)`: 5 adaptive fidelity levels (ABSTRACT, OUTLINE, STRUCTURE, DETAILED, RAW)
  - ABSTRACT: ~10 tokens (1-sentence summary)
  - OUTLINE: ~30 tokens (summary + section markers)
  - STRUCTURE: ~50 tokens (headers + key entities)
  - DETAILED: ~100 tokens (summary + entities + excerpts)
  - RAW: Full original text (200-500 tokens)

- `SemanticNode(dataclass)`: Represents a semantic chunk in the graph
  - Fields: node_id, text, embedding, importance, metadata

- `SkeletonResponse(dataclass)`: Compressed skeleton view metadata
  - Tracks compression ratio, tokens saved, node mapping

- `SemanticCompressor`: Main compression engine
  - Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
  - Graph-based structure: NetworkX for semantic relationships
  - PageRank for importance scoring
  - Token counting: tiktoken (cl100k_base)

**Key Methods**:
- `ingest_file(text, file_id, metadata)`: Convert raw text → semantic graph
  - Intelligent chunking (preserves paragraphs, sentences)
  - Generates embeddings for each chunk
  - Builds similarity graph (edges where similarity > threshold)
  - Calculates PageRank importance scores
  - Returns compressed skeleton

- `read_skeleton(file_id)`: Get compressed skeleton view (~80-95% token reduction)
- `modulate_region(node_ids, fidelity_level)`: Adaptive content retrieval
- `search_semantic(query, file_id, top_k)`: Vector similarity search
- `_chunk_text(text, max_chunk_size)`: Semantic-aware text chunking
- `_extract_key_entities(text)`: Simple NER (capitalized words heuristic)
- `_generate_summary(text, max_length)`: Extractive summarization

**Inspired by**: JSCCM (rate adaptation) + FPQE (structure preservation)

---

#### 2. **`code_compressor.py`** (22,701 bytes)
**Purpose**: Specialized compression for source code using AST analysis

**Key Classes**:
- `CodeLanguage(Enum)`: Supported languages (Python, JS, TS, Java, C++, Go, Rust)
- `CodeChunk(dataclass)`: Semantic code unit
  - Fields: chunk_id, chunk_type (function/class/import/block), code, name, docstring, start_line, end_line, dependencies

- `CodeSemanticCompressor`: Code-specific compression
  - Embedding model: `microsoft/codebert-base` (fallback to all-MiniLM-L6-v2)
  - AST parsing for intelligent chunking
  - Dependency graph (imports, function calls)
  - Code-aware importance (public vs private, entry points)

**Key Methods**:
- `detect_language(filepath)`: Language detection from file extension
- `chunk_python_code(code, file_id)`: AST-based Python parsing
  - Extracts imports, functions, classes
  - Parses docstrings
  - Identifies dependencies (function calls)

- `chunk_javascript_code(code, file_id)`: Regex-based JS/TS parsing
  - Extracts imports/requires
  - Finds functions (including arrow functions)
  - Parses JSDoc comments

- `ingest_code_file(code, file_id, filepath, metadata)`: Process source code
- `generate_code_skeleton(file_id, show_top_n)`: Code skeleton with signatures
  - Shows import statements
  - Function signatures (not bodies)
  - Class definitions (not implementations)
  - Dependency indicators

- `search_code(query, file_id, top_k)`: Semantic code search
- `get_code_chunk(chunk_id, include_context)`: Retrieve full code

**Use Case**: Compress large codebases for AI context windows while preserving structure

---

#### 3. **`multimodal_compressor.py`** (18,259 bytes)
**Purpose**: Unified compression for text, code, AND images

**Key Classes**:
- `ModalityType(Enum)`: TEXT, CODE, IMAGE
- `MultiModalNode(dataclass)`: Can contain any modality
  - Content: string (text/code) or bytes (images)
  - Unified embedding space for cross-modal search

- `MultiModalCompressor`: Multi-modal semantic graph
  - Text encoder: all-MiniLM-L6-v2
  - Code encoder: CodeBERT (optional)
  - Image encoder: CLIP (clip-ViT-B-32) for vision-language alignment

**Key Methods**:
- `_encode_text(text)`: Text → embedding
- `_encode_code(code)`: Code → embedding
- `_encode_image(image_data)`: Image bytes → CLIP embedding

- `ingest_mixed_content(content_items, project_id)`: Process mixed content
  - content_items: List of {type, content, metadata}
  - Creates unified cross-modal semantic graph
  - Edges connect semantically similar items across modalities

- `search_cross_modal(query, query_type, filter_modality)`: Cross-modal search
  - Examples:
    - Text query → Find related images
    - Image query → Find related code
    - Code query → Find documentation

- `get_node_content(node_id)`: Retrieve with base64 for images
- `generate_multimodal_summary(project_id)`: Project overview

**Use Case**: Compress documentation with diagrams, code with screenshots, etc.

---

#### 4. **`scar_compressor.py`** (20,005 bytes)
**Purpose**: SCAR paper implementation - learnable semantic compression

**Inspired by**: "Semantic Context Matters: Improving Conditioning for Autoregressive Models" (arXiv:2511.14063v1)

**Key Classes**:
- `LearnableSemanticCompressor(nn.Module)`: Neural compression module
  - Architecture: Parallel downsampling branches (like SCAR's Pk from Eq. 3)
  - Input: 384D embeddings (sentence-transformers)
  - Output: 96D compressed embeddings (4× compression)
  - Training: Reconstruction loss (SCAR Eq. 4) - L2 preservation

**Network Architecture**:
```
Branch 1: Linear(384→256) → LayerNorm → GELU → Linear(256→96)
Branch 2: Linear(384→96)
Compressed = Branch1 + Branch2
Reconstruction: Linear(96→256) → LayerNorm → GELU → Linear(256→384)
```

- `SemanticAlignmentModule(nn.Module)`: Alignment guidance (SCAR Section 3.3)
  - Aligns retrieved nodes with query semantics
  - Loss: L2 alignment (SCAR Eq. 8): L_align = ||Hs - Pt||²
  - Improves retrieval quality via dense in-context guidance

- `SCAREnhancedCompressor`: Integration with SemanticCompressor
  - Learnable embedding compression (optional)
  - Semantic alignment for better search
  - Adaptive fidelity based on alignment scores

**Key Methods**:
- `compress_batch(embeddings)`: Compress embeddings 4×
- `search_with_alignment(query, file_id)`: Enhanced search using alignment
  - Combines cosine similarity + alignment score

- `adaptive_modulate(query, file_id, top_k)`: SCAR-style adaptive fidelity
  - High alignment → RAW (full detail)
  - Medium alignment → STRUCTURE
  - Low alignment → ABSTRACT

**Training**: Uses L2 reconstruction loss to preserve semantic information during compression

---

### Supporting Modules

#### 5. **`adaptive_rate_allocator.py`** (14,825 bytes)
**Purpose**: Dynamic skeleton ratio based on document complexity and context

**Inspired by**: JSCCM (Joint Semantic-Channel Coding) - arXiv:2511.15699v1

**Key Classes**:
- `AdaptiveRateAllocator(nn.Module)`: Learned rate allocation
  - Input features: [complexity, context_availability, query_priority]
  - Output: Optimal skeleton ratio (0.10-0.30)
  - Uses Gumbel-Softmax for differentiable rate selection

**Key Methods**:
- `calculate_complexity_score(graph)`: Measure document complexity
  - Graph density
  - Clustering coefficient
  - PageRank entropy

- `gumbel_softmax_rate_selection(logits)`: Differentiable discrete choice
  - Like JSCCM Eq. 17 for constellation points
  - Straight-through estimator for gradients

- `ContextWindowAdapter`: Adapt compression to "channel conditions"
  - Low context availability (like low SNR) → More compression
  - High context availability (like high SNR) → Less compression

- `MultiLevelSemanticEncoder`: Two-branch architecture (like JSCCM Fig. 3)
  - Main branch: Global structure (top 15%, always included)
  - Auxiliary branch: Local details (next 25%, if space allows)
  - Detail nodes: Only if plenty of space

**Analogy**: Context window = wireless channel, availability = SNR

---

#### 6. **`blind_spot_detector.py`** (11,610 bytes)
**Purpose**: Self-correcting context loop to prevent hallucination

**Concept**: "Holodeck Context" - detect when AI misses critical information

**Key Classes**:
- `BlindSpot(dataclass)`: Detected blind spot
  - Fields: node_id, similarity_to_response, was_retrieved, urgency, reason

- `BlindSpotReport(dataclass)`: Analysis report
  - total_blind_spots, critical_blind_spots, recommendations, auto_inject

- `BlindSpotDetector`: Fidelity preservation checker
  - Embeds AI's response
  - Compares to all document nodes
  - Finds high-similarity nodes that weren't retrieved
  - Ranks by urgency = similarity × importance

**Key Methods**:
- `analyze_response(ai_response, file_id, retrieved_node_ids)`: Detect blind spots
  - Returns list of missed critical content

- `_calculate_urgency(similarity, importance)`: Urgency levels
  - Critical: urgency ≥ 0.6
  - High: urgency ≥ 0.4
  - Medium: urgency ≥ 0.25
  - Low: urgency < 0.25

- `validate_response_fidelity(ai_response, file_id, retrieved_node_ids)`: Quick check
  - Returns (is_valid, correction_message)

- `HaloEffectDetector`: Hallucination detection
  - Detects claims with low similarity to ALL nodes
  - Identifies contradictions to high-importance content

**Use Case**: Post-response validation to ensure grounded answers

---

#### 7. **`semantic_ssim.py`** (13,676 bytes)
**Purpose**: Structural Similarity Index for semantic graphs

**Inspired by**: FPQE paper (arXiv:2511.15695v1) - "Structure preservation metrics (SSIM) better predict performance than MSE/PSNR"

**Key Class**:
- `SemanticSSIM`: Measure structure preservation quality
  - Based on visual SSIM formula: SSIM(x,y) = l^α · c^β · s^γ

**Components** (adapted from visual SSIM):
1. **Luminance (l)**: Average importance comparison
   - Visual: Brightness
   - Semantic: Information density
   - Formula: (2·μ_x·μ_y + c1) / (μ_x² + μ_y² + c1)

2. **Contrast (c)**: Variance in importance
   - Visual: Dynamic range
   - Semantic: Importance distribution
   - Formula: (2·σ_x·σ_y + c2) / (σ_x² + σ_y² + c2)

3. **Structure (s)**: Graph connectivity preservation
   - Edge preservation: Fraction of edges retained
   - Community preservation: Clustering coefficient similarity
   - Centrality preservation: PageRank correlation

**Key Methods**:
- `calculate_ssim(graph, original_nodes, compressed_nodes)`: Overall SSIM score
  - Returns: (ssim_score, components_dict)

- `calculate_embedding_ssim(original_embeddings, compressed_embeddings)`: Alternative without graph

- `interpret_ssim_score(ssim)`: Actionable guidance
  - SSIM > 0.9: Excellent preservation
  - SSIM 0.7-0.9: Good preservation
  - SSIM 0.5-0.7: Acceptable (some loss)
  - SSIM < 0.5: Poor (reduce compression)

**Use Case**: Validate that compression preserves semantic structure

---

#### 8. **`training_utils.py`** (17,597 bytes)
**Purpose**: Training utilities for learnable compression modules

**Key Functions**:
- Synthetic data generation for training SCAR compressor
- Loss computation for semantic preservation
- Training loops for LearnableSemanticCompressor
- Evaluation metrics for compression quality
- Model checkpointing and loading

**Use Case**: Train learnable components of SCAR compression

---

### MCP Server

#### 9. **`server.py`** (19,668 bytes)
**Purpose**: MCP (Model Context Protocol) server implementation

**Key Class**:
- `SemanticModulatorServer`: MCP server exposing compression tools

**MCP Tools Exposed**:
1. `ingest_context`: Ingest document → semantic graph
2. `read_skeleton`: Get compressed skeleton view
3. `modulate_region`: Retrieve specific sections at chosen fidelity
4. `search_semantic`: Semantic search across documents
5. `analyze_blind_spots`: Detect missed context in AI response
6. `adapt_to_context_window`: JSCCM-inspired context adaptation
7. `multilevel_encode`: Multi-branch encoding
8. `get_stats`: Document statistics

**Server Features**:
- stdio transport for MCP communication
- Async handlers for tool calls
- Context window monitoring
- Retrieval history tracking (for blind spot detection)
- Integration with all compression modules

**Usage**: Run as MCP server, integrate with Claude Desktop or other MCP clients

---

## Architecture Flow

```
1. INGEST:
   Text → SemanticCompressor.ingest_file()
   ↓
   Chunks → Embeddings → Graph → PageRank → Skeleton

2. RETRIEVE:
   Query → search_semantic() → Top-K nodes
   ↓
   Nodes → modulate_region(fidelity_level) → Content

3. VALIDATE:
   AI Response → BlindSpotDetector.analyze_response()
   ↓
   Detect missed critical content → Auto-inject if needed
```

## Dependencies

**Core**:
- `sentence-transformers`: Embeddings (all-MiniLM-L6-v2, CodeBERT, CLIP)
- `networkx`: Graph analysis and PageRank
- `torch`: Neural modules (SCAR)
- `scikit-learn`: Similarity metrics
- `tiktoken`: Token counting

**MCP**:
- `mcp`: Model Context Protocol SDK

**Optional**:
- `PIL`: Image processing for multimodal
- `chromadb`: Vector database (for persistence)

## Design Principles

1. **Local-first**: No external API calls, runs entirely local
2. **Fidelity preservation**: SSIM-based quality metrics
3. **Adaptive**: Adjusts to context availability (JSCCM-inspired)
4. **Research-backed**: Implements JSCCM, FPQE, SCAR papers
5. **Self-correcting**: Blind spot detection prevents hallucination
6. **Modular**: Each compressor can be used independently

## Performance Targets

- Token reduction: 80-95%
- SSIM score: > 0.7 (good preservation)
- Retrieval speed: < 100ms for typical queries
- Memory: < 500MB for embedding model
