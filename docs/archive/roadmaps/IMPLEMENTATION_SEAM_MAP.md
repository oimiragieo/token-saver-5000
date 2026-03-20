# Implementation Seam Map: Competitor Parity Features
## Token Saver 5000 - Feature Gap Analysis & Codebase Integration

Generated: 2025-03-16

This document maps 9 major feature gaps to specific files, classes, functions in the codebase.

Each gap includes:
1. Current State (what exists)
2. Gap Analysis (what competitors have)
3. Target Files (exact files/classes/line ranges)
4. New Modules Needed (new files to create)
5. Integration Points (how pieces wire)
6. Complexity (Small/Medium/Large/XL)

---

## Gap 1: Multi-Tenant Memory Isolation (CRITICAL)

TARGET FILES:
- src/persistence.py (PersistenceManager, lines 48-150): Add workspace_id/user_id/agent_id to storage paths
- src/types.py (HandlerContext, lines 34-130): Add tenant fields (workspace_id, user_id, agent_id)
- src/handlers/mcp_core.py (route_tool_call, line ~1300+): Extract tenant from MCP request
- src/semantic_modulator/app/server_service_adapter.py (build_context): Thread tenant through context

NEW MODULE: src/tenancy.py
- TenantContext: Immutable tenant ID bundle
- TenantValidator: Isolation enforcement + cross-tenant prevention

CURRENT STORAGE: .semantic_modulator_data/{doc_id}
NEW STORAGE: .semantic_modulator_data/{workspace_id}/{user_id}/{agent_id}/{doc_id}

COMPLEXITY: XL (affects every handler, persistence, context mgmt)

---

## Gap 2: Explicit Memory APIs (CRITICAL)

TARGET FILES:
- src/afm.py (FocusManager, lines 200-300): Add methods:
  - add_memory(category, content, metadata)
  - search_memory(query, top_k=5)
  - list_memories(category=None)
  - delete_memory(memory_id)
  - summarize_user_memory(workspace_id, user_id)
- src/memory_hooks.py (MemoryHookManager, lines 14-68): Upgrade to full MemoryStore with:
  - Search via cosine similarity
  - Metadata filtering
  - Persistence hooks

NEW MODULES:
- src/memory_api.py (MemoryAPIService + MemoryEntry dataclass)
- src/handlers/memory_handlers.py (handle_add_memory, handle_search_memory, etc.)

MCP TOOLS TO ADD: 5 tools
- memory/add
- memory/search  
- memory/list
- memory/delete
- memory/summarize

COMPLEXITY: Large (new service + handlers + embedding integration)

---

## Gap 3: Prompt Management (CRITICAL)

TARGET FILES:
- src/compression_presets.py (lines 12-80): Upgrade CompressionPreset with:
  - version: int (incremental)
  - created_at: datetime
  - deployment_status: str ("draft", "staging", "production")
  - variant_id: Optional[str] (A/B support)
  - tags: List[str]

NEW MODULES:
- src/prompt_registry.py (PromptRegistry + CompressionTemplate + PromptVariant + DeploymentLabel)
- src/handlers/prompt_handlers.py (handle_create_prompt_template, handle_list_prompt_templates, etc.)

MCP TOOLS TO ADD: 6 tools
- prompt/create-template
- prompt/list-templates
- prompt/get-template
- prompt/update-template
- prompt/create-variant
- prompt/set-deployment

PERSISTENCE: ChromaDB collections for templates + variants

COMPLEXITY: Large (versioning, deployment tracking, A/B split logic)

---

## Gap 4: Experiment Tracking (CRITICAL)

TARGET FILES:
- src/benchmark_harness.py (BenchmarkResult, lines 35-76): Add fields:
  - experiment_id: str
  - run_timestamp: datetime
  - variant_id: str
  - metadata: Dict[str, Any]
  - delta_vs_baseline: float
- src/compression_verifier.py: Integrate with experiment tracking

NEW MODULES:
- src/experiment_tracker.py (ExperimentTracker + ExperimentRun + analytics)
  - Methods: create_experiment, start_run, complete_run, compare_runs, get_trend
  - Properties: pass_rate, avg_compression_ratio
- src/handlers/experiment_handlers.py (handle_create_experiment, handle_complete_run, etc.)

MCP TOOLS TO ADD: 7 tools
- experiment/create
- experiment/start-run
- experiment/complete-run
- experiment/get-run
- experiment/list-runs
- experiment/compare-runs
- experiment/get-trend

OPTIONAL: Web dashboard (src/handlers/experiment_dashboard.py for HTTP endpoints)

COMPLEXITY: Large (tracking system, analytics, optional UI)

---

## Gap 5: Connector Layer (CRITICAL)

TARGET FILES:
- src/file_sync_manager.py (lines 41-130): Refactor to use abstract connector pattern
  - Create abstract DataConnector base class
  - Move local file logic to LocalFileConnector
  - Keep FileSyncManager as coordinator

NEW MODULES:
- src/connectors/__init__.py (DataConnector ABC + ConnectorItem dataclass)
- src/connectors/local_file_connector.py (LocalFileConnector implementation)
- src/connectors/s3_connector.py (S3Connector with boto3)
- src/connectors/github_connector.py (GitHubConnector with PyGithub)
- src/connectors/slack_connector.py (SlackConnector with slack-sdk)
- src/connectors/notion_connector.py (NotionConnector with notion-client)
- src/connectors/gmail_connector.py (GmailConnector with google-auth)
- src/connector_registry.py (ConnectorRegistry + credential mgmt + watch/polling)
- src/handlers/connector_handlers.py (handle_authenticate_connector, handle_list_connector_items, etc.)

MCP TOOLS TO ADD: 5 tools
- connector/list-available
- connector/authenticate
- connector/list-items
- connector/ingest-item
- connector/watch-path

HIGH-VALUE CONNECTORS (priority order):
1. S3 (cloud storage)
2. GitHub (code repos)
3. Slack (exports)
4. Notion (wikis)
5. Gmail (emails)

COMPLEXITY: XL (6+ implementations, OAuth/credential mgmt, polling/webhooks, security)

---

## Gap 6: Temporal Context & Fact Invalidation (MEDIUM)

TARGET FILES:
- src/semantic_compressor.py (Node creation, ~lines 150-200): Add temporal metadata:
  - created_at: datetime
  - modified_at: datetime
  - valid_until: Optional[datetime]
  - lifecycle_stage: str ("active", "deprecated", "archived")
- src/context_decay.py (compute_decay_score, lines 59-80): Add fact validity check
  - Return 0.0 if expired, else decay based on age
- src/afm.py: Use fact freshness in importance scoring

NEW MODULES:
- src/temporal_graph.py (TemporalFactGraph + TemporalFact + Thread)
  - Methods: add_fact, invalidate_fact, supersede_fact, search_timeline, get_fact_timeline
  - Features: freshness_score, is_valid checks
- src/handlers/temporal_handlers.py (handle_create_thread, handle_add_temporal_fact, etc.)

MCP TOOLS TO ADD: 7 tools
- thread/create
- thread/add-fact
- thread/invalidate-fact
- thread/supersede-fact
- thread/get
- thread/search-timeline
- thread/get-fact-timeline

COMPLEXITY: Large (temporal fact model, invalidation logic, integration with decay scoring)

---

## Gap 7: Production Multimodal Ingestion (MEDIUM)

TARGET FILES:
- src/multimodal_compressor.py (lines 1-100): Production hardening:
  - Add error handling for image decode failures
  - Support PDF extraction (PyPDF2/pdfplumber)
  - Support OCR (pytesseract)
  - Add batch processing
  - Add progress tracking
  - Move from EXPERIMENTAL to stable

NEW MODULES:
- src/multimodal_ingestion.py (MultiModalIngestionPipeline + DocumentFormat)
  - Methods: ingest_file, ingest_url, batch_ingest
  - Support: TEXT, MARKDOWN, CODE, PDF, IMAGE, IMAGE_SCANNED, VIDEO (stub), AUDIO (stub)
- src/handlers/multimodal_handlers.py (handle_ingest_multimodal_file, handle_search_cross_modal, etc.)

MCP TOOLS TO ADD: 4 tools
- multimodal/ingest-file
- multimodal/ingest-url
- multimodal/get-document-modalities
- multimodal/search-cross-modal

DEPENDENCIES: PyPDF2 or pdfplumber, pytesseract (OCR)

COMPLEXITY: Large (PDF/OCR processing, error handling, testing)

---

## Gap 8: Structured Handoff Bundles (MEDIUM)

TARGET FILES:
- src/toon_serializer.py: Extend TOON for bundle encoding (add bundle_id, metadata)
- src/evidence_bundle.py (EvidenceBundle, lines 59+): Add fields:
  - purpose: str ("agent_handoff", "team_sharing", "audit_trail")
  - recipients: List[str]
  - expiration: Optional[datetime]
  - is_reusable: bool

NEW MODULES:
- src/structured_bundles.py (HandoffBundle + BundleStore)
  - Fields: bundle_id, purpose, skeleton, evidence, recipients, signature, encryption_key
  - Methods: get_toon_encoding, to_json, from_json
- src/handlers/bundle_handlers.py (handle_create_handoff_bundle, handle_share_bundle, etc.)

MCP TOOLS TO ADD: 6 tools
- bundle/create-handoff
- bundle/get
- bundle/list
- bundle/share
- bundle/import
- bundle/export-toon

FEATURES: Integrity via HMAC signature, optional encryption for sensitive content, reuse tracking

COMPLEXITY: Large (bundle creation, sharing, encryption, TOON integration)

---

## Gap 9: Model-Aware Optimization (LOW)

TARGET FILES:
- src/metrics.py (MODEL_PRICING, lines 80-91): Extend dict with:
  - More model variants (claude-3, gpt-4o, gemini-2, etc.)
  - Input vs output token pricing (separate costs)
  - Context window sizes
  - Max batch/concurrent limits
- src/token_threshold.py: Make thresholds dynamic per model
- src/fidelity_advisor.py: Consider model when recommending fidelity

NEW MODULES:
- src/model_optimizer.py (ModelOptimizer + ModelProfile)
  - Methods: get_model_profile, optimize_skeleton, estimate_cost, select_best_model, get_cost_breakdown
  - MODELS dict: gpt-4o, claude-opus-4.6, gemini-2.0, etc.
- src/handlers/model_handlers.py (handle_get_model_profile, handle_estimate_model_cost, etc.)

MCP TOOLS TO ADD: 6 tools
- model/get-profile
- model/list-supported
- model/estimate-cost
- model/select-best
- model/get-cost-breakdown
- model/optimize-for-model

COMPLEXITY: Medium (model database maintenance, cost calculation, optimization rules)

---

## Critical Integration Points

### 1. HandlerContext Extension (types.py)
Add new service references:
  - workspace_id, user_id, agent_id (Gap 1)
  - memory_service (Gap 2)
  - prompt_registry (Gap 3)
  - experiment_tracker (Gap 4)
  - connector_registry (Gap 5)
  - temporal_graph (Gap 6)
  - multimodal_pipeline (Gap 7)
  - bundle_store (Gap 8)
  - model_optimizer (Gap 9)

### 2. MCP Tool Registration (handlers/mcp_core.py:setup_mcp_tools)
Register ~30 new tools in Tool() objects (lines 94-1200)

### 3. Handler Routing (handlers/mcp_core.py:route_tool_call)
Dispatch to new handler modules:
  - if tool_name in memory_handlers.TOOLS: ...
  - elif tool_name in prompt_handlers.TOOLS: ...
  - etc.

### 4. Server Context Building (server.py:_build_context)
Instantiate all new services before building context dict

### 5. Persistence Extensibility (persistence.py)
All new services use PersistenceManager for storage:
  - ChromaDB collections for each service
  - Tenant-scoped storage paths (Gap 1)

---

## Implementation Roadmap (Priority Order)

TIER 1 (BLOCKING FOR SAAS):
1. Gap 1 - Tenancy (XL) - Required for multi-user
2. Gap 2 - Memory APIs (Large) - Foundation for stateful features
3. Gap 3 - Prompts (Large) - Product differentiation
4. Gap 4 - Experiments (Large) - Close Langfuse gap

TIER 2 (MARKET EXPANSION):
5. Gap 5 - Connectors (XL) - GTM requirement, Graphlit gap

TIER 3 (NICE-TO-HAVES):
6. Gap 6 - Temporal (Large) - Close Zep gap
7. Gap 7 - Multimodal (Large) - Production-grade ingestion
8. Gap 8 - Bundles (Large) - Close Thread Transfer gap
9. Gap 9 - Model Opt (Medium) - ROI/cost claims

ESTIMATED EFFORT: 3-4 person-months for all 9 gaps with full testing

---

## Summary Table

| Gap | Priority | Complexity | Est. LOC | New Modules |
|-----|----------|-----------|---------|------------|
| 1. Tenancy | CRITICAL | XL | 800-1200 | tenancy.py |
| 2. Memory APIs | CRITICAL | Large | 600-900 | memory_api.py, memory_handlers.py |
| 3. Prompts | CRITICAL | Large | 700-1000 | prompt_registry.py, prompt_handlers.py |
| 4. Experiments | CRITICAL | Large | 800-1100 | experiment_tracker.py, experiment_handlers.py |
| 5. Connectors | CRITICAL | XL | 2000-3000 | connectors/*, connector_registry.py, connector_handlers.py |
| 6. Temporal | MEDIUM | Large | 700-1000 | temporal_graph.py, temporal_handlers.py |
| 7. Multimodal | MEDIUM | Large | 800-1200 | multimodal_ingestion.py, multimodal_handlers.py |
| 8. Bundles | MEDIUM | Large | 600-900 | structured_bundles.py, bundle_handlers.py |
| 9. Model Opt | LOW | Medium | 500-800 | model_optimizer.py, model_handlers.py |

