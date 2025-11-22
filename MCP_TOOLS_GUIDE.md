# MCP Tools Guide: Complete Reference

**Comprehensive documentation for all 17 Token Saver 5000 MCP tools**

---

## Table of Contents

- [Overview](#overview)
- [Quick Reference](#quick-reference)
- [Document Compression Tools (9)](#document-compression-tools)
- [Dialogue Memory Tools (4)](#dialogue-memory-tools)
- [Discovery & Management Tools (4)](#discovery--management-tools)
- [Common Usage Patterns](#common-usage-patterns)
- [Error Handling](#error-handling)
- [Performance Tips](#performance-tips)

---

## Overview

Token Saver 5000 exposes **17 MCP tools** via the stdio transport protocol. These tools enable AI assistants to:

1. **Compress documents** with 80-95% token reduction
2. **Manage dialogue memory** with ~66% token savings and safety preservation
3. **Search semantically** across ingested content
4. **Detect blind spots** and hallucinations (self-correcting)
5. **Adapt compression dynamically** based on available context

All tools operate **locally** (no external API calls required) and use **persistent storage** (documents survive server restarts).

---

## Quick Reference

### Tool Categories

| Category | Tools | Primary Use Case |
|----------|-------|------------------|
| **Document Compression** | 9 tools | Process long documents (papers, manuals, code) |
| **Dialogue Memory (AFM)** | 4 tools | Manage multi-turn conversations efficiently |
| **Discovery & Management** | 4 tools | List, export, import, and delete documents |

### Typical Workflow

```
1. ingest_context          → Compress document into semantic graph
2. read_skeleton           → Get 80-95% compressed overview
3. search_semantic         → Find relevant sections
4. modulate_region         → Retrieve details at chosen fidelity
5. check_blind_spots       → Verify no critical context missed
```

---

## Document Compression Tools

### 1. `ingest_context`

**Purpose:** Ingest and compress a document into a semantic graph with 80-95% token reduction.

**Input Parameters:**

```json
{
  "text": "string (required)",          // Raw document text
  "file_id": "string (required)",       // Unique identifier (e.g., "paper_1")
  "metadata": {                         // Optional metadata
    "author": "string",
    "date": "string",
    "source": "string",
    "tags": ["string", ...]
  }
}
```

**Output:**

```json
{
  "success": true,
  "file_id": "paper_1",
  "original_tokens": 45000,
  "skeleton_tokens": 2300,
  "compression_ratio": 19.5,
  "token_savings_pct": 94.9,
  "num_nodes": 90,
  "num_anchors": 18,
  "processing_time_sec": 3.2
}
```

**Example Usage:**

```python
# Claude Desktop interaction
await mcp_tools.ingest_context(
    text=research_paper_text,
    file_id="quantum_paper_2024",
    metadata={
        "author": "Smith et al.",
        "date": "2024-11-15",
        "source": "arXiv:2411.12345",
        "tags": ["quantum computing", "error correction"]
    }
)
```

**Performance:**

- Small docs (5K tokens): ~0.8 seconds
- Medium docs (20K tokens): ~1.9 seconds
- Large docs (45K tokens): ~3.2 seconds
- Very large docs (100K tokens): ~6.8 seconds

**Resource Limits:**

- Max document size: 100MB (configurable)
- Max total documents: 1000 (configurable)
- Max total storage: 1GB (configurable)

**Tips:**

- Use descriptive `file_id` values (e.g., "manual_v2.0" not "doc1")
- Add metadata for better organization and searchability
- Documents are auto-saved and persist across server restarts

---

### 2. `read_skeleton`

**Purpose:** Get compressed skeleton view showing only high-importance "anchor" concepts.

**Input Parameters:**

```json
{
  "file_id": "string (required)"  // Document identifier
}
```

**Output:**

```
⭐ quantum_paper_2024_n0: Quantum Error Correction: A Comprehensive Survey
   [ABSTRACT] This paper reviews state-of-the-art error correction techniques...
   Importance: 0.95 | Tokens: 18

⭐ quantum_paper_2024_n5: Surface Codes and Logical Qubits
   [SECTION] Surface codes arrange physical qubits in 2D lattice...
   Importance: 0.87 | Tokens: 22

⭐ quantum_paper_2024_n12: Fault-Tolerant Quantum Gates
   [SECTION] Implementing universal gate set with error detection...
   Importance: 0.82 | Tokens: 20

... (15 more anchor nodes) ...

📦 72 hidden nodes available (use search_semantic + modulate_region to retrieve)

Total: 18 anchors shown, 72 hidden | Compression: 94.9% (45,000 → 2,300 tokens)
```

**Typical Compression Ratios:**

- Research papers: 18-20× (95% reduction)
- Technical documentation: 16-18× (94% reduction)
- Novels/narrative: 15-17× (93% reduction)
- Legal/contracts: 17-19× (94% reduction)

**When to Use:**

- **Always use FIRST** before requesting specific content
- Get high-level overview of document structure
- Identify important concepts and sections
- Decide what to retrieve at higher fidelity

**Tips:**

- Anchor nodes (⭐) are top 20% by PageRank importance
- Hidden nodes (📦) are accessible via `search_semantic` or by node ID
- Use this to "navigate" large documents efficiently

---

### 3. `modulate_region`

**Purpose:** Retrieve specific sections at a chosen fidelity level (adaptive semantic fidelity).

**Input Parameters:**

```json
{
  "node_ids": ["string", ...],  // List of node IDs (required)
  "fidelity_level": "string"    // ABSTRACT | OUTLINE | STRUCTURE | DETAILED | RAW
}
```

**Fidelity Levels:**

| Level | Tokens/Node | Content | Use Case |
|-------|-------------|---------|----------|
| **ABSTRACT** | ~10 | Single sentence summary | Quick overview, navigation |
| **OUTLINE** | ~30 | Summary + section context | Understanding structure |
| **STRUCTURE** | ~50 | Summary + entities + metadata | Detailed analysis |
| **DETAILED** | ~100 | Summary + entities + key excerpts | In-depth reading |
| **RAW** | Original | Complete original text | Exact quotes, verification |

**Output:**

```
=== Node: quantum_paper_2024_n12 ===
Fidelity: STRUCTURE (~50 tokens)

**Section:** Fault-Tolerant Quantum Gates
**Summary:** Implementing universal gate set using Clifford+T decomposition with magic state distillation for error detection and correction.

**Key Entities:**
- Clifford gates (stabilizer operations)
- T gate (π/8 rotation, non-Clifford)
- Magic state distillation
- Error detection circuits

**Related Sections:** n11 (Error Syndromes), n13 (Threshold Theorem)

**Importance Score:** 0.82 (anchor)
```

**Example Usage:**

```python
# After reading skeleton and identifying interesting nodes
await mcp_tools.modulate_region(
    node_ids=["quantum_paper_2024_n12", "quantum_paper_2024_n13"],
    fidelity_level="STRUCTURE"  # Start with STRUCTURE, upgrade to RAW if needed
)
```

**Strategy:**

```
1. read_skeleton               → Identify relevant nodes
2. modulate_region (STRUCTURE) → Get detailed summaries
3. modulate_region (RAW)       → Get full text if needed for quotes
```

**Tips:**

- Start with lower fidelity (STRUCTURE) to save tokens
- Upgrade to RAW only when you need exact quotes or full context
- ABSTRACT is useful for quickly scanning many nodes
- DETAILED provides good balance between tokens and detail

---

### 4. `search_semantic`

**Purpose:** Semantic vector search across ingested documents (finds conceptually similar content, not just keywords).

**Input Parameters:**

```json
{
  "query": "string (required)",       // Natural language search query
  "file_id": "string (optional)",     // Limit to specific document
  "top_k": 5                          // Number of results (default: 5)
}
```

**Output:**

```json
{
  "results": [
    {
      "node_id": "quantum_paper_2024_n47",
      "similarity": 0.89,
      "preview": "Error mitigation strategies include zero-noise extrapolation...",
      "importance": 0.74,
      "tokens": 485
    },
    {
      "node_id": "quantum_paper_2024_n23",
      "similarity": 0.85,
      "preview": "Noise models characterize decoherence channels in NISQ devices...",
      "importance": 0.68,
      "tokens": 512
    }
    // ... 3 more results
  ],
  "total_searched": 90,
  "query_time_ms": 45
}
```

**Example Queries:**

```python
# Conceptual search (not just keywords!)
await mcp_tools.search_semantic(
    query="error correction techniques",
    file_id="quantum_paper_2024",
    top_k=5
)
# Finds: "fault-tolerant gates", "noise mitigation", "syndrome extraction"
# Even if exact phrase "error correction" doesn't appear!

# Cross-document search
await mcp_tools.search_semantic(
    query="machine learning optimization",
    # No file_id → searches ALL ingested documents
    top_k=10
)
```

**How It Works:**

1. Query is encoded to 384-dim vector using `all-MiniLM-L6-v2`
2. Cosine similarity computed against all node embeddings
3. Results ranked by similarity score (0-1)
4. Top-k results returned with previews

**Performance:**

- Single document search: ~40-60ms
- Cross-document search (10 docs): ~200-400ms
- Scales linearly with number of nodes

**Tips:**

- Use natural language queries ("explain the concept of..." works great)
- Semantic search finds related concepts, not just keyword matches
- Combine with `modulate_region` to retrieve full content
- Higher `similarity` scores (>0.8) indicate very relevant matches

---

### 5. `check_blind_spots`

**Purpose:** Detect if your AI response missed critical context (self-correcting context loop).

**Input Parameters:**

```json
{
  "ai_response": "string (required)",    // Your generated response
  "file_id": "string (required)",        // Document being discussed
  "retrieved_nodes": ["string", ...]     // Node IDs you actually viewed
}
```

**Output:**

```json
{
  "blind_spot_detected": true,
  "confidence": 0.78,
  "missing_concepts": [
    "magic state distillation",
    "surface code boundaries",
    "logical qubit encoding"
  ],
  "suggested_nodes": [
    {
      "node_id": "quantum_paper_2024_n34",
      "relevance": 0.91,
      "preview": "Magic state distillation produces high-fidelity T states..."
    },
    {
      "node_id": "quantum_paper_2024_n56",
      "relevance": 0.87,
      "preview": "Surface code boundaries define logical qubit regions..."
    }
  ],
  "recommendation": "Auto-inject suggested nodes to improve response accuracy"
}
```

**How It Works:**

1. Encodes your AI response into embedding vector
2. Extracts key concepts from response text
3. Compares concepts to retrieved nodes
4. Identifies concepts mentioned in response but NOT in retrieved nodes
5. Searches full document for these missing concepts
6. Suggests nodes to retrieve to fill gaps

**Example Workflow:**

```python
# 1. Generate response
response = "Quantum error correction uses surface codes and magic states..."

# 2. Check for blind spots
result = await mcp_tools.check_blind_spots(
    ai_response=response,
    file_id="quantum_paper_2024",
    retrieved_nodes=["quantum_paper_2024_n12", "quantum_paper_2024_n13"]
)

# 3. If blind spot detected, retrieve suggested nodes
if result["blind_spot_detected"]:
    suggested_ids = [node["node_id"] for node in result["suggested_nodes"]]
    new_context = await mcp_tools.modulate_region(
        node_ids=suggested_ids,
        fidelity_level="STRUCTURE"
    )
    # 4. Regenerate response with additional context
```

**When to Use:**

- After generating responses to verify accuracy
- When response seems vague or uncertain
- For high-stakes queries requiring complete information
- To prevent hallucination by grounding in source material

**Tips:**

- Use BEFORE delivering final response to user
- Acts as a "safety net" for missing context
- Particularly valuable for complex technical documents
- Helps maintain response fidelity even with aggressive compression

---

### 6. `detect_hallucination`

**Purpose:** Check if a response is grounded in source material or potentially fabricated.

**Input Parameters:**

```json
{
  "ai_response": "string (required)",  // Response to validate
  "file_id": "string (required)"       // Source document
}
```

**Output:**

```json
{
  "hallucination_detected": false,
  "confidence": 0.92,
  "max_similarity": 0.89,
  "avg_similarity": 0.74,
  "grounding_score": 0.85,
  "most_similar_nodes": [
    {
      "node_id": "quantum_paper_2024_n12",
      "similarity": 0.89
    }
  ],
  "verdict": "GROUNDED - Response aligns well with source material"
}
```

**Interpretation:**

| Grounding Score | Verdict | Meaning |
|-----------------|---------|---------|
| > 0.7 | **GROUNDED** | Response closely matches source material |
| 0.5 - 0.7 | **UNCERTAIN** | Partial grounding, may have speculative elements |
| < 0.5 | **HALLUCINATION** | Likely fabricated or not from source |

**Example Usage:**

```python
response = """
Quantum error correction uses surface codes arranged in a
3D hypercubic lattice with magic state factories.
"""

result = await mcp_tools.detect_hallucination(
    ai_response=response,
    file_id="quantum_paper_2024"
)

if result["hallucination_detected"]:
    print(f"⚠️ Warning: Potential hallucination (score: {result['grounding_score']})")
    print(f"Recommendation: Review source material and regenerate")
```

**How It Works:**

1. Encodes response into embedding vector
2. Computes similarity to ALL nodes in document
3. If max similarity < threshold (0.5), flags as hallucination
4. Returns most similar nodes for verification

**Tips:**

- Use for high-stakes or technical responses
- Combine with `check_blind_spots` for comprehensive validation
- Low grounding scores suggest you should retrieve more context
- Helps maintain accuracy with heavy compression

---

### 7. `get_stats`

**Purpose:** Get statistics about ingested documents and compression efficiency.

**Input Parameters:**

```json
{
  "file_id": "string (optional)"  // Specific file, or omit for global stats
}
```

**Output (Single Document):**

```json
{
  "file_id": "quantum_paper_2024",
  "num_nodes": 90,
  "num_anchors": 18,
  "num_edges": 342,
  "original_tokens": 45000,
  "skeleton_tokens": 2300,
  "compression_ratio": 19.5,
  "token_savings_pct": 94.9,
  "avg_node_tokens": 500,
  "graph_density": 0.042,
  "ssim_score": 0.87,
  "metadata": {
    "author": "Smith et al.",
    "ingestion_time": "2024-11-22T10:30:15Z"
  }
}
```

**Output (Global Stats):**

```json
{
  "total_documents": 12,
  "total_nodes": 1247,
  "total_tokens": 543200,
  "total_compressed_tokens": 28400,
  "avg_compression_ratio": 19.1,
  "storage_used_mb": 87.3,
  "storage_limit_mb": 1024.0,
  "ram_used_mb": 234.1,
  "ram_limit_mb": 2048.0,
  "documents": ["paper_1", "paper_2", "manual_v2", ...]
}
```

**When to Use:**

- Monitor compression efficiency
- Check resource usage (storage, RAM)
- Verify document ingestion success
- Compare compression across documents

**Tips:**

- High `ssim_score` (>0.7) indicates good structural preservation
- `graph_density` shows how interconnected concepts are
- Use to identify which documents compress best

---

### 8. `adapt_to_context_window`

**Purpose:** Dynamically adjust compression based on available context window (JSCCM-inspired).

**Input Parameters:**

```json
{
  "file_id": "string (required)",
  "available_tokens": 2000,        // Currently available tokens
  "max_tokens": 100000,            // Max context window (default)
  "query_priority": 0.8            // Query importance 0-1 (default: 0.5)
}
```

**Output:**

```json
{
  "adaptive_skeleton": "...",
  "allocated_tokens": 1850,
  "skeleton_ratio": 0.15,
  "recommended_fidelity": "STRUCTURE",
  "snr_equivalent": 0.02,
  "compression_strategy": "HIGH_COMPRESSION"
}
```

**How It Works:**

Mimics JSCCM's channel adaptation:

```
Low available tokens (like low SNR)   → More compression (smaller skeleton)
High available tokens (like high SNR) → Less compression (larger skeleton)
```

**Example:**

```python
# Context window almost full (only 2K tokens left)
result = await mcp_tools.adapt_to_context_window(
    file_id="large_manual",
    available_tokens=2000,
    query_priority=0.8  # Important query
)
# Returns: Highly compressed skeleton (~15% of nodes)

# Context window mostly empty (80K tokens available)
result = await mcp_tools.adapt_to_context_window(
    file_id="large_manual",
    available_tokens=80000,
    query_priority=0.5  # Normal query
)
# Returns: Less compressed skeleton (~40% of nodes)
```

**Tips:**

- Use when context window is constrained
- Higher `query_priority` → allocates more tokens
- Automatically selects optimal skeleton ratio
- Inspired by wireless communication's SNR adaptation

---

### 9. `multilevel_encode`

**Purpose:** Generate skeleton with 3 priority levels for progressive loading (JSCCM-inspired).

**Input Parameters:**

```json
{
  "file_id": "string (required)",
  "available_tokens": 5000
}
```

**Output:**

```json
{
  "main_branch": {
    "nodes": ["n0", "n5", "n12", ...],
    "tokens": 1200,
    "coverage": "Top 15% (critical concepts)"
  },
  "auxiliary_branch": {
    "nodes": ["n3", "n8", "n15", ...],
    "tokens": 2500,
    "coverage": "Next 25% (important details)"
  },
  "detail_branch": {
    "nodes": ["n23", "n34", ...],
    "tokens": 1300,
    "coverage": "Remaining 60% (supplementary)"
  },
  "loading_strategy": "Include main (always), auxiliary (if >1500 tokens left), detail (if >4000 tokens left)"
}
```

**Progressive Loading Strategy:**

```python
# Start with main branch (always fits)
context = multilevel_result["main_branch"]

# Add auxiliary if space allows
if available_tokens > 1500:
    context += multilevel_result["auxiliary_branch"]

# Add detail if plenty of space
if available_tokens > 4000:
    context += multilevel_result["detail_branch"]
```

**When to Use:**

- Building context incrementally
- Uncertain how much space is available
- Want layered compression (core → important → details)
- Implementing progressive disclosure UI

**Tips:**

- Inspired by JSCCM's parallel encoder architecture
- Main branch contains absolute essentials
- Auxiliary adds important supporting context
- Detail provides comprehensive coverage

---

## Dialogue Memory Tools

### 10. `afm_add_message`

**Purpose:** Add message to dialogue history with automatic importance classification.

**Input Parameters:**

```json
{
  "role": "user | assistant | system",
  "content": "string (required)"
}
```

**Output:**

```json
{
  "message_id": "msg_12",
  "turn_number": 12,
  "importance": "CRITICAL",
  "estimated_tokens": 32,
  "concepts_extracted": ["peanut allergy", "severe", "life-threatening"]
}
```

**Importance Classification:**

| Level | Score | Examples | Retention |
|-------|-------|----------|-----------|
| **CRITICAL** | 1.0 | Allergies, safety constraints, requirements | Always preserved |
| **RELEVANT** | 0.6 | Main topics, preferences, questions | Preserved if recent or query-relevant |
| **TRIVIAL** | 0.2 | Acknowledgments, "ok", "thanks" | First to compress/drop |

**Example Usage:**

```python
# User discloses allergy
await mcp_tools.afm_add_message(
    role="user",
    content="I have a severe peanut allergy - even trace amounts can cause anaphylaxis"
)
# → Classified as CRITICAL (score = 1.0)

# User asks question
await mcp_tools.afm_add_message(
    role="user",
    content="What's the best Thai street food to try?"
)
# → Classified as RELEVANT (score = 0.6)

# Assistant acknowledges
await mcp_tools.afm_add_message(
    role="assistant",
    content="Got it, I'll keep that in mind."
)
# → Classified as TRIVIAL (score = 0.2)
```

**Automatic Classification:**

AFM uses pattern matching to classify importance:

```python
CRITICAL patterns:
- Allergies: "allerg", "severe", "deadly", "life-threatening", "cannot"
- Constraints: "must not", "never", "required", "mandatory"
- Safety: "danger", "toxic", "harmful"

RELEVANT patterns:
- Preferences: "prefer", "like", "want", "interested"
- Questions: "how", "what", "why", "when", "where"
- Main topics: "plan to", "need to", "should"

TRIVIAL patterns:
- Acknowledgments: "ok", "thanks", "got it", "I see"
- Confirmations: "yes", "no", "sure", "alright"
```

**Tips:**

- Messages auto-embed with `all-MiniLM-L6-v2`
- Importance can be manually overridden if needed
- CRITICAL messages never decay with recency
- Build history sequentially (turn order matters)

---

### 11. `afm_build_context`

**Purpose:** Build optimized context for current query with ~66% token reduction and safety preservation.

**Input Parameters:**

```json
{
  "current_query": "string (required)",
  "budget_tokens": 800,
  "system_preamble": "string (optional)"
}
```

**Output:**

```
=== CONTEXT (Budget: 800 tokens, Used: 754 tokens, Savings: 67%) ===

[Turn 1 - FULL]
User: I have a severe peanut allergy - even trace amounts can cause anaphylaxis.
(Score: 1.00, Relevance: 0.89, Importance: CRITICAL)

[Turn 2 - COMPRESSED]
User: Planning Thailand trip
(Score: 0.45, Relevance: 0.72, Importance: RELEVANT)

[Turn 3-8 - PLACEHOLDER]
[Turns 3-8: Hotel and flight discussion]

[Turn 9 - FULL]
User: What's the best street food vendor near Chatuchak Market?
(Score: 0.82, Relevance: 0.94, Importance: RELEVANT)

[Turn 10 - COMPRESSED]
Assistant: Chatuchak has great food options
(Score: 0.38, Relevance: 0.88, Importance: RELEVANT)

[CURRENT QUERY]
User: What Thai street food should I try?

=== STATISTICS ===
Original turns: 10
Messages included: 10 (3 FULL, 2 COMPRESSED, 5 PLACEHOLDER)
Original tokens: 2,280
Compressed tokens: 754
Compression: 67% savings
Allergy preserved: ✅ Yes (CRITICAL importance → FULL fidelity)
```

**Fidelity Assignment:**

```
Score >= 0.45 (tau_high)  → FULL (verbatim)
Score >= 0.25 (tau_mid)   → COMPRESSED (50% compression)
Score < 0.25              → PLACEHOLDER (single line stub)
```

**Score Calculation:**

```python
score = base_importance × recency_weight × query_relevance_boost

# Example: Allergy message (turn 1, current turn 11)
base_importance = 1.0  # CRITICAL
recency_weight = 0.5^((11-1)/12) = 0.5  # But ignored for CRITICAL!
query_relevance = cosine_sim("allergy", "Thai food") = 0.75
boost = 1 + 0.75 = 1.75

score = 1.0 × 1.0 × 1.75 = 1.75  # Capped at 1.0
→ FULL fidelity (always preserved)
```

**Performance:**

```
| Scenario | Original Tokens | AFM Tokens | Savings | Time |
|----------|-----------------|------------|---------|------|
| 3 turns  | 245             | 98         | 60%     | 0.1s |
| 9 turns  | 1,847           | 612        | 67%     | 0.3s |
| 20 turns | 4,120           | 1,350      | 67%     | 0.5s |
```

**Tips:**

- CRITICAL messages always preserved (allergies, constraints)
- Messages packed chronologically (preserves conversation flow)
- Query-relevant messages get score boost
- If message doesn't fit at FULL, tries COMPRESSED automatically
- Achieves ~66% savings while maintaining safety

---

### 12. `afm_get_stats`

**Purpose:** Get dialogue statistics and importance breakdown.

**Input Parameters:** None

**Output:**

```json
{
  "total_messages": 20,
  "current_turn": 20,
  "importance_breakdown": {
    "CRITICAL": 2,
    "RELEVANT": 13,
    "TRIVIAL": 5
  },
  "total_tokens": 4120,
  "avg_tokens_per_message": 206,
  "oldest_message": "Turn 1: I have a severe peanut allergy...",
  "newest_message": "Turn 20: What Thai street food should I try?"
}
```

**When to Use:**

- Monitor dialogue state
- Check if history is getting too long
- Verify importance distribution
- Debug AFM behavior

---

### 13. `afm_clear_history`

**Purpose:** Clear dialogue history and reset turn counter.

**Input Parameters:** None

**Output:**

```json
{
  "success": true,
  "messages_cleared": 20,
  "message": "Dialogue history cleared. Ready for new conversation."
}
```

**When to Use:**

- Starting a new conversation
- Topic has completely changed
- Testing AFM with fresh slate
- User explicitly requests to "forget" previous conversation

**Tips:**

- Irreversible (export history first if you want to preserve it)
- Resets turn counter to 0
- Clears all message embeddings and metadata

---

## Discovery & Management Tools

### 14. `list_documents`

**Purpose:** Get inventory of all ingested documents with metadata.

**Input Parameters:** None

**Output:**

```json
{
  "total_documents": 12,
  "documents": [
    {
      "file_id": "quantum_paper_2024",
      "num_nodes": 90,
      "original_tokens": 45000,
      "skeleton_tokens": 2300,
      "compression_ratio": 19.5,
      "metadata": {
        "author": "Smith et al.",
        "date": "2024-11-15",
        "source": "arXiv:2411.12345",
        "tags": ["quantum computing", "error correction"]
      },
      "ingestion_time": "2024-11-22T10:30:15Z",
      "last_accessed": "2024-11-22T14:22:33Z"
    },
    // ... 11 more documents
  ],
  "storage_used_mb": 87.3,
  "storage_limit_mb": 1024.0
}
```

**When to Use:**

- Discover what documents are available
- Check compression statistics across documents
- Find documents by metadata
- Monitor storage usage

**Tips:**

- Documents persist across server restarts (auto-loaded)
- Use descriptive `file_id` values for easy discovery
- Add tags in metadata for better organization

---

### 15. `afm_export_history`

**Purpose:** Export dialogue history to JSON for later resumption.

**Input Parameters:**

```json
{
  "session_id": "default"  // Optional session identifier
}
```

**Output:**

```json
{
  "success": true,
  "session_id": "default",
  "messages_exported": 20,
  "export_data": {
    "version": "1.0",
    "session_id": "default",
    "turn_counter": 20,
    "messages": [
      {
        "turn": 1,
        "role": "user",
        "content": "I have a severe peanut allergy...",
        "importance": "CRITICAL",
        "embedding": [0.23, -0.45, ...]
      }
      // ... 19 more messages
    ],
    "exported_at": "2024-11-22T14:30:00Z"
  }
}
```

**Example Usage:**

```python
# Export current conversation
export_result = await mcp_tools.afm_export_history(
    session_id="thailand_trip_planning"
)

# Save to file (user or assistant does this)
with open("session_thailand_trip.json", "w") as f:
    json.dump(export_result["export_data"], f)
```

**Tips:**

- Export before clearing history if you want to preserve
- Can have multiple named sessions
- Exports include embeddings (ready for import)
- Use for conversation checkpoints

---

### 16. `afm_import_history`

**Purpose:** Import previously exported dialogue history.

**Input Parameters:**

```json
{
  "session_id": "default"  // Session to load
}
```

**Output:**

```json
{
  "success": true,
  "session_id": "default",
  "messages_imported": 20,
  "turn_counter_restored": 20,
  "message": "Dialogue history restored from session 'default'"
}
```

**Example Usage:**

```python
# Resume previous conversation
result = await mcp_tools.afm_import_history(
    session_id="thailand_trip_planning"
)

# Continue conversation from where you left off
await mcp_tools.afm_add_message(
    role="user",
    content="I've decided to visit Bangkok first"
)
```

**Tips:**

- Replaces current dialogue history (export first if needed)
- Restores turn counter (preserves recency calculations)
- Embeddings are restored (no re-encoding needed)
- Perfect for resuming long conversations

---

### 17. `delete_document`

**Purpose:** Permanently delete an ingested document from memory and storage.

**Input Parameters:**

```json
{
  "file_id": "string (required)",
  "confirm": true  // Must be true (safety check)
}
```

**Output:**

```json
{
  "success": true,
  "file_id": "old_document",
  "nodes_deleted": 90,
  "storage_freed_mb": 12.3,
  "message": "Document 'old_document' permanently deleted"
}
```

**Example Usage:**

```python
# Delete outdated document
result = await mcp_tools.delete_document(
    file_id="outdated_manual_v1",
    confirm=True  # Must explicitly confirm
)
```

**When to Use:**

- Remove outdated documents (old versions)
- Free up storage space
- Manage resource limits
- Clean up test documents

**⚠️ Warning:**

- **Irreversible operation** - cannot be undone
- Deletes from both memory and persistent storage
- Must set `confirm=True` to proceed (safety check)
- Document cannot be recovered after deletion

**Tips:**

- Use `list_documents` first to verify which documents exist
- Check `storage_used_mb` to see if deletion is needed
- Consider exporting important documents before deletion

---

## Common Usage Patterns

### Pattern 1: Basic Document Analysis

```python
# 1. Ingest document
await mcp_tools.ingest_context(
    text=document_text,
    file_id="my_doc"
)

# 2. Get overview
skeleton = await mcp_tools.read_skeleton(file_id="my_doc")

# 3. Search for specific topic
results = await mcp_tools.search_semantic(
    query="error handling",
    file_id="my_doc",
    top_k=3
)

# 4. Retrieve details
await mcp_tools.modulate_region(
    node_ids=[r["node_id"] for r in results],
    fidelity_level="STRUCTURE"
)
```

### Pattern 2: Self-Correcting Response Generation

```python
# 1. Generate initial response
response = generate_response(query, context)

# 2. Check for blind spots
blind_spot_result = await mcp_tools.check_blind_spots(
    ai_response=response,
    file_id="my_doc",
    retrieved_nodes=retrieved_node_ids
)

# 3. If blind spot detected, retrieve missing context
if blind_spot_result["blind_spot_detected"]:
    missing_nodes = [n["node_id"] for n in blind_spot_result["suggested_nodes"]]
    additional_context = await mcp_tools.modulate_region(
        node_ids=missing_nodes,
        fidelity_level="STRUCTURE"
    )
    # 4. Regenerate with complete context
    response = generate_response(query, context + additional_context)

# 5. Validate grounding
hallucination_check = await mcp_tools.detect_hallucination(
    ai_response=response,
    file_id="my_doc"
)
```

### Pattern 3: Dialogue with Document Context

```python
# Conversation with document augmentation
while True:
    user_input = get_user_input()

    # Add to dialogue history
    await mcp_tools.afm_add_message(role="user", content=user_input)

    # Build optimized dialogue context
    dialogue_context = await mcp_tools.afm_build_context(
        current_query=user_input,
        budget_tokens=2000
    )

    # Search relevant document sections
    doc_results = await mcp_tools.search_semantic(
        query=user_input,
        file_id="my_doc",
        top_k=3
    )

    # Retrieve document context
    doc_context = await mcp_tools.modulate_region(
        node_ids=[r["node_id"] for r in doc_results],
        fidelity_level="STRUCTURE"
    )

    # Generate response with both dialogue and document context
    response = generate_response(
        query=user_input,
        dialogue_context=dialogue_context,
        document_context=doc_context
    )

    # Add response to dialogue history
    await mcp_tools.afm_add_message(role="assistant", content=response)
```

### Pattern 4: Adaptive Compression Based on Context Window

```python
# Monitor context window usage
context_window_max = 100000
context_used = count_tokens(current_context)
available = context_window_max - context_used

# Adapt compression dynamically
if available < 5000:
    # Low space: high compression
    skeleton = await mcp_tools.adapt_to_context_window(
        file_id="large_doc",
        available_tokens=available,
        query_priority=0.8
    )
elif available > 20000:
    # Plenty of space: less compression
    skeleton = await mcp_tools.multilevel_encode(
        file_id="large_doc",
        available_tokens=available
    )
    # Include all three branches
else:
    # Medium space: standard skeleton
    skeleton = await mcp_tools.read_skeleton(file_id="large_doc")
```

---

## Error Handling

### Common Errors

**1. Document Not Found**

```json
{
  "error": "Document 'paper_99' not found",
  "available_documents": ["paper_1", "paper_2", "manual_v2"],
  "suggestion": "Use ingest_context() to add documents first or list_documents() to see available documents"
}
```

**2. Resource Limit Exceeded**

```json
{
  "error": "Document exceeds 100MB size limit",
  "document_size_mb": 156.3,
  "limit_mb": 100.0,
  "suggestion": "Split document into smaller parts or increase limit in ResourceManager"
}
```

**3. Invalid Node ID**

```json
{
  "error": "Node 'invalid_n99' not found in document 'paper_1'",
  "valid_range": "paper_1_n0 to paper_1_n89",
  "suggestion": "Use read_skeleton() or search_semantic() to get valid node IDs"
}
```

**4. Storage Limit Reached**

```json
{
  "error": "Total storage limit reached (1024MB)",
  "current_usage_mb": 1015.2,
  "suggestion": "Use delete_document() to remove old documents or increase storage limit"
}
```

### Error Handling Best Practices

```python
try:
    result = await mcp_tools.ingest_context(text=doc, file_id="my_doc")
except Exception as e:
    if "not found" in str(e):
        # Document doesn't exist - handle accordingly
        available = await mcp_tools.list_documents()
    elif "limit" in str(e):
        # Resource limit hit - clean up or increase limit
        stats = await mcp_tools.get_stats()
    else:
        # Other error - log and retry
        logger.error(f"Unexpected error: {e}")
```

---

## Performance Tips

### 1. Start with Low Fidelity

```python
# ❌ Don't start with RAW fidelity
expensive = await mcp_tools.modulate_region(nodes, "RAW")  # Uses lots of tokens

# ✅ Start with STRUCTURE, upgrade to RAW only if needed
cheap = await mcp_tools.modulate_region(nodes, "STRUCTURE")  # ~50 tokens/node
# Later, if you need exact quotes:
full = await mcp_tools.modulate_region([specific_node], "RAW")
```

### 2. Use Semantic Search Before Modulation

```python
# ❌ Don't retrieve all nodes
all_nodes = range(90)  # Retrieves everything!
content = await mcp_tools.modulate_region(all_nodes, "DETAILED")

# ✅ Search first, retrieve only relevant nodes
results = await mcp_tools.search_semantic(query="error handling", top_k=5)
relevant_nodes = [r["node_id"] for r in results]
content = await mcp_tools.modulate_region(relevant_nodes, "STRUCTURE")
```

### 3. Leverage Multi-Level Encoding

```python
# ✅ Progressive loading based on available space
result = await mcp_tools.multilevel_encode(
    file_id="large_doc",
    available_tokens=5000
)

# Use only main branch if space is tight
if available_tokens < 2000:
    use_context(result["main_branch"])
# Add auxiliary if more space
elif available_tokens < 4000:
    use_context(result["main_branch"] + result["auxiliary_branch"])
# Include everything if plenty of space
else:
    use_context(result["main_branch"] + result["auxiliary_branch"] + result["detail_branch"])
```

### 4. Batch Operations When Possible

```python
# ❌ Multiple individual searches
for query in queries:
    await mcp_tools.search_semantic(query=query, file_id="doc")

# ✅ Combine queries for better performance
combined_query = " ".join(queries)
results = await mcp_tools.search_semantic(query=combined_query, top_k=10)
```

### 5. Use AFM for Long Conversations

```python
# ❌ Send entire conversation history (grows indefinitely)
context = entire_conversation_history  # 10,000+ tokens after 50 turns

# ✅ Use AFM to compress intelligently
dialogue_context = await mcp_tools.afm_build_context(
    current_query=query,
    budget_tokens=800  # Fixed size, ~66% savings
)
# Only 800 tokens, with CRITICAL info preserved!
```

### 6. Monitor Resource Usage

```python
# Regularly check stats
stats = await mcp_tools.get_stats()

if stats["storage_used_mb"] > 900:  # Approaching 1GB limit
    # Delete old documents
    docs = await mcp_tools.list_documents()
    oldest_docs = sorted(docs, key=lambda d: d["last_accessed"])[:5]
    for doc in oldest_docs:
        await mcp_tools.delete_document(file_id=doc["file_id"], confirm=True)
```

---

## Conclusion

The 17 MCP tools provide comprehensive capabilities for:

- **Document compression** (80-95% reduction)
- **Dialogue memory** (~66% reduction with safety preservation)
- **Semantic search** (concept-based, not keyword-based)
- **Self-correction** (blind spot and hallucination detection)
- **Adaptive compression** (JSCCM-inspired dynamic allocation)
- **Resource management** (persistent storage, limits, cleanup)

**Key Principles:**

1. **Always start with `read_skeleton`** - get the overview first
2. **Use semantic search** - find what matters, not just keywords
3. **Modulate fidelity progressively** - start low, upgrade as needed
4. **Validate responses** - use blind spot and hallucination detection
5. **Leverage AFM for conversations** - preserve safety, reduce tokens
6. **Monitor resources** - check stats, delete old documents

**For more information:**

- [**HOW_IT_WORKS.md**](HOW_IT_WORKS.md) - Technical deep dive
- [**README.md**](README.md) - Project overview
- [**ARCHITECTURE.md**](ARCHITECTURE.md) - System architecture
- [**Examples**](examples/) - Hands-on demonstrations

---

**Built with the Model Context Protocol (MCP) • Powered by semantic communication theory** 🚀
