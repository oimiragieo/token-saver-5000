# Token-Saver-5000: Complete End-to-End Flow Analysis

**Version**: 0.7.0+ Analysis Document

## EXECUTIVE SUMMARY

AI clients using token-saver-5000 experience a complete pipeline:
1. **Ingestion** (handle_ingest): Text → Chunks → Embeddings → Semantic Graph → PageRank Scoring → Skeleton (80-95% compression)
2. **Skeleton Reading** (handle_read_skeleton): Query-aware node selection, 3 modes (baseline/query-guided/evidence-aware)
3. **Search** (handle_search_semantic): Cosine similarity + importance ranking with evidence-aware expansion
4. **Fidelity Modulation** (modulate_region): 5 levels ABSTRACT→OUTLINE→STRUCTURE→DETAILED→RAW

**Phase 5 Integration**: 6 modules (AccessTracker, CompressionReplayLog, keyword_anchoring, query_adaptive, semantic_chunking, intra_doc_dedup) are IMPLEMENTED but NOT WIRED into main handlers.

---

## QUICK REFERENCE: COMPLETE DATA FLOWS

### INGESTION FLOW (handle_ingest)

**Input**: text, file_id, file_path, metadata
**Output**: SkeletonResponse + JSON with compression_ratio, token_savings

**Steps**:
1. Rate limit check + text validation (CWE-22 path traversal check)
2. Resource pre-check (memory limits)
3. Compression estimate (CompressionAdvisor)
4. ingest_file_async() →
   - _chunk_text(): Paragraph→Sentence→512-token chunks
   - _encode_async(): ThreadPoolExecutor (4 workers) → 384-dim embeddings
   - Semantic graph: cosine_similarity > 0.75 threshold creates edges
   - PageRank: Importance scoring (cached, O(1) on hit)
   - _generate_skeleton(): Select top N% nodes, build skeleton text
5. Register with resource manager, persistence, file sync, version history
6. Return JSON with token_savings = total_tokens - skeleton_tokens

**Data Structure** (SkeletonResponse):
- file_id, total_nodes, total_tokens, skeleton_tokens, compression_ratio
- skeleton_text (readable format), node_map (node_id → description)

---

### SKELETON READING FLOW (handle_read_skeleton)

**Input**: file_id, selection_mode (baseline/query_guided/evidence_aware), optional query, top_k, min_similarity
**Output**: SkeletonResponse JSON

**Selection Modes**:
- **baseline** (PageRank only): Sort by importance, keep top N%
- **query_guided** (hybrid): 35% importance + 65% relevance, MMR redundancy penalty
- **evidence_aware** (with diagnostics): EvidenceResult with sufficiency detection + expanded search on insufficiency

**Returns**:
- skeleton_text: High-importance [ANCHOR] nodes shown with summaries
- Hidden nodes: [HIDDEN] with guidance to modulate_region()
- evidence: (optional) {sufficient, best_score, used_expanded_search, message}
- staleness_warning: (optional) File has changed on disk

---

### SEARCH FLOW (handle_search_semantic)

**Input**: query, optional file_id, top_k, evidence_aware flag
**Output**: Array of {node_id, similarity, importance, summary, tokens}

**Algorithm**:
1. Embed query (384-dim)
2. For each candidate node: cosine_similarity(query_embedding, node.embedding)
3. Sort by similarity (highest first)
4. If evidence_aware: Run retrieve_evidence() for sufficiency detection
   - If best_score < min_similarity: expand search (top_k *= 2) and retry
5. Return results with both similarity AND importance scores

**Returns**: List of nodes ranked by relevance with diagnostics

---

### TOKEN COMPRESSION MATH (Does It Actually Work?)

**Real Example**:
`
Input: 4,500 tokens
Chunks: 25 semantic nodes
Skeleton ratio: 0.2 (keep 5 nodes, hide 20)

Skeleton output:
- 5 [ANCHOR] nodes with summaries: ~450 tokens
- 20 [HIDDEN] references: ~100 tokens total
= 550 skeleton_tokens

Compression: 4,500 / 550 = 8.2x
Savings: 3,950 tokens (87.8%)

Cost impact (GPT-4): .135 → .0165 per request
`

**SkeletonResponse contains**:
- total_tokens: 4,500 (original)
- skeleton_tokens: 550 (compressed)
- compression_ratio: 8.2 (multiplier)
- skeleton_text: Actual readable skeleton
- node_map: Navigation for AI to find specific content

When modulating region to RAW:
- 1 node full (~180 tokens) + skeleton (450) = 630 total
- Still 86% savings vs. loading full document

---

## PHASE 5 FEATURE INTEGRATION STATUS

### Module Inventory

| Module | Location | Implemented | Wired to Handlers |
|--------|----------|-------------|------------------|
| AccessTracker | context_decay.py:14-56 | ✅ Yes | ❌ No |
| CompressionReplayLog | compression_replay.py:12-49 | ✅ Yes | ❌ No |
| keyword_anchoring | keyword_anchoring.py:11-53 | ✅ Yes | ❌ No |
| query_adaptive | query_adaptive.py:14-63 | ✅ Yes | ❌ No |
| semantic_chunking | semantic_chunking.py:14-80 | ✅ Yes | ❌ No |
| intra_doc_dedup | intra_doc_dedup.py:14-80 | ✅ Yes | ❌ No |

### What's NOT Happening

**In handle_ingest() [Line 346 calls ingest_file_async]**:
- ❌ No AccessTracker.record_access(file_id)
- ❌ No CompressionReplayLog.record(doc_id, input_tokens, output_tokens, ratio, fidelity_score)
- ❌ _chunk_text() does NOT use semantic_chunking.detect_semantic_boundaries()
- ❌ _select_skeleton_nodes() does NOT use keyword_anchoring.apply_keyword_anchoring()
- ❌ Skeleton generation does NOT use query_adaptive.compute_section_ratios()
- ❌ No intra_doc_dedup.find_intra_duplicates() to collapse repetitive content

**In handle_read_skeleton() [Line 515 calls _generate_skeleton]**:
- ❌ Does NOT return fidelity_score in response
- ❌ Does NOT use keyword_anchoring

**Standalone tools created for Phase 5 but not used in main flow**:
- ✅ handle_prune_by_relevance (Line 1703)
- ✅ handle_multi_level_skeleton (Line 1704)
- ✅ handle_evict_stale (Line 1705)
- ✅ handle_advise_context (Line 1706)
- ✅ handle_get_compression_insights (Line 1707)
- ✅ handle_generate_rewrite_prompt (Line 1708)

These are separate tools, not integrated into ingestion/skeleton flow.

---

## MCP TOOL DISCOVERY

### How AI Gets Tool List

1. **Server Init** (src/server.py:51-74):
   - ServerFactoryService builds components
   - Tool profile resolved from MCP_TOOL_PROFILE env var (default: "full")

2. **Tool Definition** (src/handlers/mcp_core.py:72-1592):
   - setup_mcp_tools(profile) returns List[Tool]
   - 48 total tools defined with schemas
   - Filtered by profile: "full" or "core_stable"

3. **Profile Filtering**:
   - core_stable: 7 tools only (ingest_context, read_skeleton, search_semantic, modulate_region, get_stats, list_documents, delete_document)
   - full: All 48 tools

4. **Tool Dispatch** (src/handlers/mcp_core.py:1595-1747):
   `python
   router = {
     "ingest_context": ch.handle_ingest,
     "read_skeleton": ch.handle_read_skeleton,
     "search_semantic": ch.handle_search_semantic,
     "modulate_region": ch.handle_modulate_region,
     # ... 44 more tools
   }
   `
   - AI calls tool → route_tool_call() looks up handler → handler executes → JSON returned

### All 48 Available Tools

**Document Compression (9)**: ingest_context, read_skeleton, modulate_region, search_semantic, get_stats, list_documents, delete_document, adapt_to_context_window, multilevel_encode

**Batch/Directory (2)**: batch_ingest_documents, ingest_directory

**Fidelity (1)**: recommend_fidelity

**Detection (2)**: check_blind_spots, detect_hallucination

**AFM Dialogue (6)**: afm_add_message, afm_build_context, afm_get_stats, afm_clear_history, afm_export_history, afm_import_history

**File Sync (4)**: check_file_sync, diff_cached_file, refresh_document, get_version_history

**Resources (3)**: check_resource_health, check_environment, should_compress

**Help (1)**: tool_help

**Visualization (4)**: export_graph_json, visualize_graph_html, export_graph_graphml, explain_compression_decision

**ACE Framework (7)**: ace_generate, ace_reflect, ace_curate, ace_grow_context, ace_refine_context, ace_get_playbook, ace_execute_cycle

**Utilities (6)**: diff_reingest, find_duplicates, get_compression_presets, check_context_budget, prune_by_relevance, get_compression_insights

**Phase 5 (6)**: prune_by_relevance, get_multi_level_skeleton, evict_stale, advise_context, get_compression_insights, generate_rewrite_prompt

**Experimental (5)** (not for production): toon_encode, toon_decode, scar_compress, scar_get_stats, multimodal_ingest

**ASG-SI (4)**: verify_compression, calculate_reward, get_evidence_stats, generate_synthetic_tests

---

## KEY FILES & LINE NUMBERS

| What | File | Lines |
|------|------|-------|
| handle_ingest | compression_handlers.py | 262-461 |
| handle_read_skeleton | compression_handlers.py | 464-563 |
| handle_search_semantic | compression_handlers.py | 639-711 |
| ingest_file_async | semantic_compressor.py | 399-406 |
| _ingest_file_impl | semantic_compressor.py | 442-566 |
| _generate_skeleton | semantic_compressor.py | 657-751 |
| _chunk_text | semantic_compressor.py | 295-341 |
| search_semantic_with_scores | semantic_compressor.py | 941-974 |
| retrieve_evidence | semantic_compressor.py | 763-815 |
| SkeletonResponse | semantic_compressor.py | 62-71 |
| setup_mcp_tools | mcp_core.py | 72-1592 |
| route_tool_call | mcp_core.py | 1595-1747 |
| AccessTracker | context_decay.py | 14-56 |
| CompressionReplayLog | compression_replay.py | 12-49 |

---

## INTEGRATION CHECKLIST FOR PHASE 5

**To wire Phase 5 features, implement**:

1. **AccessTracker in handle_ingest()**:
   `python
   # Add to SemanticCompressor.__init__():
   self._access_tracker = AccessTracker()
   
   # In handle_ingest() after skeleton:
   compressor._access_tracker.record_access(file_id)
   `

2. **CompressionReplayLog in handle_ingest()**:
   `python
   # Add to SemanticCompressor.__init__():
   self._compression_replay = CompressionReplayLog()
   
   # After skeleton generation:
   replay_log.record(
     doc_id=file_id,
     content_type=infer_type(metadata),
     input_tokens=skeleton.total_tokens,
     output_tokens=skeleton.skeleton_tokens,
     ratio=skeleton.compression_ratio,
     fidelity_score=compute_fidelity(text, skeleton_text)
   )
   `

3. **keyword_anchoring in read_skeleton()**:
   - Add parameter: anchored_keywords: List[str]
   - Pass to _select_skeleton_nodes()
   - Use apply_keyword_anchoring() if provided

4. **semantic_chunking in _chunk_text()**:
   - Replace fixed-size fallback with semantic boundaries
   - Use detect_semantic_boundaries() for embeddings-based chunks

5. **query_adaptive in _generate_skeleton()**:
   - Compute per-node compression ratios
   - Add to SkeletonResponse
   - Use in skeleton_text generation

6. **Fidelity scoring**:
   - Compute via semantic similarity or ROUGE
   - Return in SkeletonResponse
   - Record in CompressionReplayLog

---

## CONCLUSION

**Working Well**:
- Ingestion achieves 80-95% compression reliably
- Skeleton reading offers 3 selection modes with evidence diagnostics
- Search returns both similarity and importance scores
- MCP discovery exposes 48 tools to AI clients
- SkeletonResponse accurately reports compression stats

**Missing Integration**:
- Phase 5 modules implemented but not wired to main handlers
- Compression events not logged (AccessTracker/CompressionReplayLog)
- Advanced node selection features unused
- Semantic chunking not replacing fixed-size approach
- No intra-document deduplication

**Effort to Complete**: 5-10 small code changes across SemanticCompressor and handlers to fully integrate Phase 5 features.