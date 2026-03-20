# IMPLEMENTATION SEAM MAP - COMPLETE INDEX

This folder contains historical roadmap artifacts that were moved out of the repository root to keep the top level focused on product entrypoints.

## Overview

This is a detailed actionable implementation roadmap for closing 9 feature gaps identified in the competitor audit (`competitor-audit/feature-matrix.txt`). The maps specific files, classes, functions, and new modules needed to achieve market parity with competitors.

**Generated**: 2025-03-16  
**Codebase Version**: v0.10.0  
**Total Documentation**: ~40 KB, 3 documents  

---

## The Three Documents

### 1. IMPLEMENTATION_SEAM_MAP.md (11.6 KB, 270 lines)
**Main reference document** - Comprehensive mapping of all 9 gaps

**Contains:**
- Executive summary of all gaps
- Current state analysis for each gap
- Target files with exact line numbers
- New modules to create
- Integration points and wiring diagrams
- Complexity ratings (Small/Medium/Large/XL)
- Summary table with effort estimates
- Testing strategy
- Deployment considerations

**Best for**: Getting complete picture of what needs building

---

### 2. QUICK_REFERENCE_FILE_MATRIX.md (9.4 KB, 204 lines)
**Developer quick-reference** - File-to-gap mapping for navigation

**Contains:**
- Core files affected by ALL gaps (types.py, handlers/mcp_core.py, server.py, persistence.py)
- Gap-specific file modifications with line ranges
- New files to create
- MCP tool registration checklist
- Development order recommendation
- Testing requirements matrix
- Handler registration code template

**Best for**: Quick lookup while implementing; "which file do I modify for Gap 3?"

---

### 3. CODE_EXAMPLES.md (18.9 KB, 615 lines)
**Actionable code templates** - Copy-paste-able code signatures

**Contains:**
- Complete class/method signatures for all 9 gaps
- Example implementations for Gap 1 (Tenancy)
- Example implementations for Gap 2 (Memory APIs)
- Example implementations for Gap 3 (Prompts)
- Example implementations for Gap 4 (Experiments)
- Example implementations for Gap 5 (Connectors - abstract + S3 example)
- Partial examples for remaining gaps

**Best for**: Starting implementation; understand exact method signatures and patterns

---

## The 9 Gaps At A Glance

| # | Gap | Priority | Complexity | LOC | Files Modified | New Modules | MCP Tools |
|---|-----|----------|-----------|-----|-----------------|------------|-----------|
| 1 | Multi-Tenant Isolation | 🔴 CRITICAL | XL | 800-1200 | persistence.py, types.py, handlers/mcp_core.py | tenancy.py | - |
| 2 | Explicit Memory APIs | 🔴 CRITICAL | Large | 600-900 | afm.py, memory_hooks.py | memory_api.py, memory_handlers.py | 5 |
| 3 | Prompt Management | 🔴 CRITICAL | Large | 700-1000 | compression_presets.py | prompt_registry.py, prompt_handlers.py | 6 |
| 4 | Experiment Tracking | 🔴 CRITICAL | Large | 800-1100 | benchmark_harness.py | experiment_tracker.py, experiment_handlers.py | 7 |
| 5 | Connector Layer | 🔴 CRITICAL | XL | 2000-3000 | file_sync_manager.py | connectors/*, connector_registry.py, connector_handlers.py | 5 |
| 6 | Temporal Context | 🟡 MEDIUM | Large | 700-1000 | semantic_compressor.py, context_decay.py | temporal_graph.py, temporal_handlers.py | 7 |
| 7 | Multimodal Ingestion | 🟡 MEDIUM | Large | 800-1200 | multimodal_compressor.py | multimodal_ingestion.py, multimodal_handlers.py | 4 |
| 8 | Structured Bundles | 🟡 MEDIUM | Large | 600-900 | toon_serializer.py, evidence_bundle.py | structured_bundles.py, bundle_handlers.py | 6 |
| 9 | Model-Aware Optimization | 🟢 LOW | Medium | 500-800 | metrics.py, token_threshold.py | model_optimizer.py, model_handlers.py | 6 |

**Total**: ~8,800-12,200 lines of new code + modifications | ~41 new MCP tools | ~12 new modules

---

## How To Use These Documents

### Scenario 1: "I'm implementing Gap 3 (Prompt Management)"

1. **Start**: Open IMPLEMENTATION_SEAM_MAP.md → Jump to "Gap 3: Prompt Management"
   - Understand what exists vs. what's missing
   - See which files to modify (compression_presets.py)

2. **Get file locations**: Open QUICK_REFERENCE_FILE_MATRIX.md → "Gap 3: Prompt Management"
   - Lines to modify: compression_presets.py (lines 12-80)
   - New files: prompt_registry.py, prompt_handlers.py
   - Integration: Store/load via persistence.py

3. **Start coding**: Open CODE_EXAMPLES.md → "Gap 3: Prompt Management"
   - Copy CompressionTemplate dataclass signature
   - Copy PromptRegistry class skeleton
   - Adapt to your needs

4. **Register MCP tools**: QUICK_REFERENCE_FILE_MATRIX.md → "Handler Registration Checklist"
   - Add 6 new Tool() objects to handlers/mcp_core.py:setup_mcp_tools()
   - Add corresponding routes in route_tool_call()

---

### Scenario 2: "Where do I start? What's the order?"

1. Open QUICK_REFERENCE_FILE_MATRIX.md → "Development Order Recommendation"
   - Phase 1 (Weeks 1-4): Tenancy + Memory APIs
   - Phase 2 (Weeks 5-8): Prompts + Experiments
   - Phase 3 (Weeks 9-16): Connectors
   - Phase 4 (Weeks 17+): Temporal, Multimodal, Bundles, Model Opt

2. Open IMPLEMENTATION_SEAM_MAP.md → "Critical Integration Points"
   - All gaps converge on: types.py, handlers/mcp_core.py, server.py, persistence.py
   - Implement in order to minimize rework

---

### Scenario 3: "I need to understand how all the pieces fit together"

1. Open IMPLEMENTATION_SEAM_MAP.md → "Critical Integration Points" section
   - Shows how all services wire through HandlerContext
   - Shows MCP tool registration flow
   - Shows how new services integrate with persistence layer

2. Open CODE_EXAMPLES.md → Look at Gap 1 implementation
   - TenantContext shows how to structure scoping
   - TenantValidator shows isolation pattern
   - Extended HandlerContext shows how new services integrate

---

### Scenario 4: "I need to implement a specific method/class"

1. Open CODE_EXAMPLES.md
   - Find the gap you're working on
   - Copy the method/class signature
   - Adapt for your implementation
   - Use as reference for expected behavior

---

## Critical Files (Appear in All Gaps)

These files MUST be modified to integrate all new features:

1. **src/types.py** (HandlerContext TypedDict)
   - Add fields for all 9 gaps' services
   - ~10 new ReadOnly fields
   - Lines affected: 34-130

2. **src/handlers/mcp_core.py** (Tool registration + routing)
   - Register ~30 new tools in setup_mcp_tools()
   - Add routes for all new handlers in route_tool_call()
   - Lines affected: 94-1200 (tools), ~1300+ (routing)

3. **src/server.py** (Server initialization)
   - Instantiate all new services in _build_context()
   - Thread tenant context if implementing Gap 1
   - Lines affected: 98-130

4. **src/persistence.py** (Storage layer)
   - Add save/load methods for all new models
   - Tenant-scope storage paths if implementing Gap 1
   - All gaps depend on this for durability

---

## Integration Checklist

When implementing each gap:

- [ ] Create new modules in src/
- [ ] Implement handler module in src/handlers/
- [ ] Add methods to persistence.py for storage
- [ ] Add service reference to types.py:HandlerContext
- [ ] Register MCP tools in handlers/mcp_core.py:setup_mcp_tools()
- [ ] Add routes in handlers/mcp_core.py:route_tool_call()
- [ ] Instantiate service in server.py:_build_context()
- [ ] Write unit tests in tests/
- [ ] Write integration tests with context + persistence
- [ ] Document in README.md or tool_help handler

---

## Estimated Timeline

| Phase | Gaps | Duration | Team Size |
|-------|------|----------|-----------|
| 1 | Tenancy + Memory APIs | 4 weeks | 2 engineers |
| 2 | Prompts + Experiments | 4 weeks | 2 engineers |
| 3 | Connectors | 8 weeks | 2 engineers (high complexity) |
| 4 | Temporal + Multimodal + Bundles + Model Opt | 8 weeks | 2 engineers (parallel) |
| **Total** | All 9 gaps | **3-4 months** | **2-3 engineers** |

---

## Key Implementation Insights

### 1. Tenancy First (Gap 1)
- **Why**: All other gaps need user/workspace isolation
- **What**: Add workspace_id/user_id/agent_id scoping to storage + handlers
- **Impact**: Foundation for multi-user SaaS model
- **Effort**: High but essential

### 2. Memory & Prompts (Gaps 2-3)
- **Why**: Enable stateful, customizable operations
- **What**: Add CRUD APIs for memories + prompt templates
- **Impact**: Product differentiation vs. Langfuse
- **Effort**: Medium

### 3. Connectors (Gap 5)
- **Why**: Highest market value per effort
- **What**: Abstract connector pattern + 5-6 implementations
- **Impact**: Close Graphlit gap, enable GTM
- **Effort**: Highest, but highest ROI

### 4. Temporal & Bundles (Gaps 6, 8)
- **Why**: Competitive features from Zep + Thread Transfer
- **What**: Fact lifecycle + reusable handoff bundles
- **Impact**: Niche but differentiating
- **Effort**: Medium

### 5. Multimodal & Model Opt (Gaps 7, 9)
- **Why**: Nice-to-haves, lower market priority
- **What**: Production-grade multimodal + model-aware cost optimization
- **Impact**: Medium
- **Effort**: Medium

---

## Testing Strategy

Each gap needs 3 types of tests:

1. **Unit tests** (src feature in isolation)
   - Test new services independently
   - Mock dependencies (persistence, embeddings, etc.)
   - Target: 80%+ coverage

2. **Integration tests** (with HandlerContext + persistence)
   - Test handlers with real HandlerContext
   - Use in-memory or test database
   - Test isolation (Gap 1), security (Gaps 1, 5)

3. **E2E tests** (full MCP tool call)
   - Ingest document → compress → search → export
   - Test full flow for new gaps

Example test locations:
```
tests/
  test_tenancy.py
  test_memory_api.py
  test_prompt_registry.py
  test_experiment_tracker.py
  test_connectors.py
  test_temporal_graph.py
  test_multimodal_ingestion.py
  test_structured_bundles.py
  test_model_optimizer.py
```

---

## Backwards Compatibility Notes

- All new gaps are **additive** (no breaking changes)
- Existing compression_presets.py hardcoded presets remain as fallback
- Existing handlers continue to work without tenant scoping
- New handlers/tools can be gated with feature flags if needed

---

## Next Steps

1. **Read** IMPLEMENTATION_SEAM_MAP.md for complete picture
2. **Reference** QUICK_REFERENCE_FILE_MATRIX.md while coding
3. **Copy** CODE_EXAMPLES.md code signatures as starting templates
4. **Follow** the "Development Order Recommendation" to minimize rework
5. **Test** thoroughly with unit + integration + E2E tests
6. **Document** each feature in tool_help handler + README

---

## Questions?

- "Where do I modify for Gap X?" → QUICK_REFERENCE_FILE_MATRIX.md
- "What does service Y look like?" → CODE_EXAMPLES.md
- "How does everything integrate?" → IMPLEMENTATION_SEAM_MAP.md:Critical Integration Points
- "What's the full picture?" → IMPLEMENTATION_SEAM_MAP.md:Executive Summary

