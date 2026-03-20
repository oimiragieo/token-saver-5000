# IMPLEMENTATION QUICK-REFERENCE: File-to-Gap Matrix

This matrix maps each file/module to which gaps it affects, for quick navigation.

## Core Files Affected by ALL Gaps

| File | Current Purpose | Gaps Affected | Changes Required |
|------|-----------------|----------------|------------------|
| src/types.py | Handler context TypedDict | 1,2,3,4,5,6,7,8,9 | Add ~10 new fields (workspace_id, memory_service, prompt_registry, etc.) |
| src/handlers/mcp_core.py | Tool registration + routing | 1,2,3,4,5,6,7,8,9 | Register ~30 new tools + route to new handler modules |
| src/server.py | MCP server initialization | 1,2,3,4,5,6,7,8,9 | Instantiate all new services in _build_context() |
| src/persistence.py | Storage layer | 1,2,3,4,5,6,7,8 | Tenant-scope storage paths + save/load all new models |

## Gap-Specific Files

### Gap 1: Multi-Tenant Isolation
**Key Files to Modify:**
- src/persistence.py (lines 48-150): Add workspace_id/user_id/agent_id path segments
- src/types.py (lines 34-130): Add tenant fields to HandlerContext
- src/handlers/mcp_core.py (~line 1300): Extract tenant from request metadata
- src/semantic_modulator/app/server_service_adapter.py: Thread tenant context

**New Files:**
- src/tenancy.py (NEW) - TenantContext + TenantValidator

**Integration:**
- Storage path changes immediately affect all handlers
- Validation middleware prevents cross-tenant data access

---

### Gap 2: Explicit Memory APIs
**Key Files to Modify:**
- src/afm.py (lines 200-300): Add add_memory, search_memory, list_memories, delete_memory, summarize_user_memory
- src/memory_hooks.py (lines 14-68): Upgrade MemoryHookManager to full MemoryStore with search

**New Files:**
- src/memory_api.py (NEW) - MemoryAPIService + MemoryEntry
- src/handlers/memory_handlers.py (NEW) - Handler implementations

**Integration:**
- Use EmbeddingManager for memory search embeddings
- Store in ChromaDB via persistence.py
- Add 5 new MCP tools to handlers/mcp_core.py:setup_mcp_tools()

---

### Gap 3: Prompt Management
**Key Files to Modify:**
- src/compression_presets.py (lines 12-80): Add version, created_at, deployment_status, variant_id

**New Files:**
- src/prompt_registry.py (NEW) - PromptRegistry + CompressionTemplate + PromptVariant + DeploymentLabel
- src/handlers/prompt_handlers.py (NEW) - Handler implementations

**Integration:**
- Store/load templates from ChromaDB
- Select active template by deployment label
- Add 6 new MCP tools

---

### Gap 4: Experiment Tracking
**Key Files to Modify:**
- src/benchmark_harness.py (lines 35-76): Add experiment_id, run_timestamp, variant_id, metadata, delta_vs_baseline
- src/compression_verifier.py: Integrate with experiment tracking

**New Files:**
- src/experiment_tracker.py (NEW) - ExperimentTracker + ExperimentRun + analytics
- src/handlers/experiment_handlers.py (NEW) - Handler implementations
- OPTIONAL: src/handlers/experiment_dashboard.py (HTTP endpoints for web UI)

**Integration:**
- Track all benchmark runs with experiment metadata
- Provide comparison + trend analysis
- Add 7 new MCP tools

---

### Gap 5: Connector Layer (HIGHEST EFFORT)
**Key Files to Modify:**
- src/file_sync_manager.py (lines 41-130): Refactor to abstract connector pattern

**New Files:**
- src/connectors/__init__.py (NEW) - DataConnector ABC + ConnectorItem
- src/connectors/local_file_connector.py (NEW) - LocalFileConnector
- src/connectors/s3_connector.py (NEW) - S3Connector
- src/connectors/github_connector.py (NEW) - GitHubConnector
- src/connectors/slack_connector.py (NEW) - SlackConnector
- src/connectors/notion_connector.py (NEW) - NotionConnector
- src/connectors/gmail_connector.py (NEW) - GmailConnector
- src/connector_registry.py (NEW) - ConnectorRegistry + credential mgmt
- src/handlers/connector_handlers.py (NEW) - Handler implementations

**Integration:**
- Abstract connector pattern allows pluggable implementations
- Credential management in persistence layer (encrypted)
- Watch/polling mechanisms for change detection
- Add 5 new MCP tools

---

### Gap 6: Temporal Context & Fact Invalidation
**Key Files to Modify:**
- src/semantic_compressor.py (~lines 150-200): Add created_at, modified_at, valid_until, lifecycle_stage to nodes
- src/context_decay.py (lines 59-80): Add fact validity check in compute_decay_score
- src/afm.py: Use fact freshness in importance scoring

**New Files:**
- src/temporal_graph.py (NEW) - TemporalFactGraph + TemporalFact + Thread
- src/handlers/temporal_handlers.py (NEW) - Handler implementations

**Integration:**
- Facts have explicit lifecycle: active → deprecated → archived
- Decay scoring incorporates validity (0.0 if expired)
- Thread APIs for managing facts over time
- Add 7 new MCP tools

---

### Gap 7: Production Multimodal Ingestion
**Key Files to Modify:**
- src/multimodal_compressor.py (lines 1-100): Remove [EXPERIMENTAL] label, add error handling, PDF/OCR support

**New Files:**
- src/multimodal_ingestion.py (NEW) - MultiModalIngestionPipeline + DocumentFormat
- src/handlers/multimodal_handlers.py (NEW) - Handler implementations

**Integration:**
- Auto-detect format from file extension + MIME type
- Extract text from PDFs (PyPDF2/pdfplumber)
- OCR for scanned images (pytesseract)
- Create unified semantic graph from all modalities
- Add 4 new MCP tools

**Dependencies to Add:**
- pdfplumber or PyPDF2
- pytesseract (+ tesseract binary)

---

### Gap 8: Structured Handoff Bundles
**Key Files to Modify:**
- src/toon_serializer.py: Extend TOON encoding for bundle metadata
- src/evidence_bundle.py (lines 59+): Add purpose, recipients, expiration, is_reusable

**New Files:**
- src/structured_bundles.py (NEW) - HandoffBundle + BundleStore
- src/handlers/bundle_handlers.py (NEW) - Handler implementations

**Integration:**
- Bundle includes: skeleton + evidence trail + multimodal metadata
- Integrity via HMAC signature
- Optional encryption for sensitive bundles
- Reuse tracking (usage_count)
- Add 6 new MCP tools

---

### Gap 9: Model-Aware Optimization
**Key Files to Modify:**
- src/metrics.py (lines 80-91): Expand MODEL_PRICING dict with more models + separate input/output costs
- src/token_threshold.py: Make token limits dynamic per model
- src/fidelity_advisor.py: Consider model in fidelity recommendations

**New Files:**
- src/model_optimizer.py (NEW) - ModelOptimizer + ModelProfile
- src/handlers/model_handlers.py (NEW) - Handler implementations

**Integration:**
- ModelProfile tracks: cost per token, context window, special token rules, capabilities
- Estimate cost for any (model, input_tokens, output_tokens)
- Recommend best model given cost/fidelity constraints
- Optimize skeleton for model's strengths
- Add 6 new MCP tools

---

## Handler Registration Checklist

All new MCP tools must be registered in **handlers/mcp_core.py:setup_mcp_tools()** (lines 94-1200):

```python
def setup_mcp_tools(profile: str = "full") -> List[Tool]:
    all_tools = [
        # Existing tools (9+)
        
        # NEW: Gap 2 - Memory APIs (5 tools)
        Tool(name="memory_add", ...),
        Tool(name="memory_search", ...),
        Tool(name="memory_list", ...),
        Tool(name="memory_delete", ...),
        Tool(name="memory_summarize", ...),
        
        # NEW: Gap 3 - Prompt Management (6 tools)
        Tool(name="prompt_create_template", ...),
        Tool(name="prompt_list_templates", ...),
        Tool(name="prompt_get_template", ...),
        Tool(name="prompt_update_template", ...),
        Tool(name="prompt_create_variant", ...),
        Tool(name="prompt_set_deployment", ...),
        
        # NEW: Gap 4 - Experiment Tracking (7 tools)
        # ... etc
    ]
```

And corresponding routes in **handlers/mcp_core.py:route_tool_call()** (~line 1300+):

```python
async def route_tool_call(tool_name, args, context):
    # Existing routing
    if tool_name == "ingest_context":
        return await ch.handle_ingest(...)
    
    # NEW: Gap 2
    elif tool_name == "memory_add":
        return await memory_handlers.handle_add_memory(args, context)
    elif tool_name == "memory_search":
        return await memory_handlers.handle_search_memory(args, context)
    # ... etc for all new tools
```

---

## Development Order Recommendation

**Phase 1 (Weeks 1-4): Foundational**
1. Gap 1 - Tenancy (mandatory for SaaS)
2. Gap 2 - Memory APIs (builds on tenancy)

**Phase 2 (Weeks 5-8): Product Differentiation**
3. Gap 3 - Prompts
4. Gap 4 - Experiments

**Phase 3 (Weeks 9-16): Market Expansion**
5. Gap 5 - Connectors (high effort, high impact)

**Phase 4 (Weeks 17+): Nice-to-Haves**
6. Gap 6 - Temporal
7. Gap 7 - Multimodal (already partial)
8. Gap 8 - Bundles
9. Gap 9 - Model Optimization

---

## Testing Requirements by Gap

| Gap | Unit Tests | Integration Tests | Security Tests | Performance Tests |
|-----|-----------|------------------|----------------|------------------|
| 1 | tenancy.py | context isolation | cross-tenant access prevention | N/A |
| 2 | memory_api.py | memory + persistence | memory boundary checks | search latency |
| 3 | prompt_registry.py | template versioning | N/A | variant selection |
| 4 | experiment_tracker.py | run lifecycle | N/A | trend computation |
| 5 | connectors/*.py | connector integrations | OAuth/credential mgmt | batch ingestion |
| 6 | temporal_graph.py | fact lifecycle | fact visibility rules | timeline queries |
| 7 | multimodal_ingestion.py | format detection | PDF parsing robustness | OCR latency |
| 8 | structured_bundles.py | bundle creation | signature/encryption | bundle size limits |
| 9 | model_optimizer.py | model profiles | N/A | cost calculation |

