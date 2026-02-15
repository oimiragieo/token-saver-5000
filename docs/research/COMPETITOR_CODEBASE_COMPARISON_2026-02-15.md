# Competitor Codebase Comparison (2026-02-15)

This report compares Token Saver 5000 against four external codebases cloned into:

- `artifacts/competitor-analysis/LLMLingua`
- `artifacts/competitor-analysis/Selective_Context`
- `artifacts/competitor-analysis/langchain`
- `artifacts/competitor-analysis/Contextual-Compression`

## Repos Reviewed

1. LLMLingua
   - Repo: https://github.com/microsoft/LLMLingua
   - Signals: `README.md`, `llmlingua/prompt_compressor.py`
2. Selective Context
   - Repo: https://github.com/liyucheng09/Selective_Context
   - Signals: `readme.md`, `selective_context.py`, `context_manager.py`
3. LangChain (contextual compression components)
   - Repo: https://github.com/langchain-ai/langchain
   - Signals: `langchain_classic/retrievers/contextual_compression.py`, document compressor modules
4. Contextual-Compression (community RAG experiments)
   - Repo: https://github.com/SrGrace/Contextual-Compression
   - Signals: `README.md`, notebooks, `main.py`

## High-Level Findings

1. Token Saver 5000 already has a stronger integrated product surface:
   - MCP server, app-layer service architecture, portable skill package, large test suite.
2. LLMLingua is stronger in model-backed prompt compression strategy breadth.
3. LangChain is stronger in composable compressor pipelines and ecosystem compatibility.
4. Selective Context contributes a simple but useful self-information scoring framing.
5. Community contextual-compression repo is useful mainly for evaluation style ideas.

## Side-by-Side Feature Matrix

Legend:

- `Strong`: mature/explicit in code
- `Partial`: exists but limited
- `Missing`: no clear support

| Capability | Token Saver 5000 | LLMLingua | Selective Context | LangChain CC | Contextual-Compression |
|---|---|---|---|---|---|
| Graph-based semantic compression | Strong | Missing | Missing | Missing | Partial |
| Query-guided compression | Strong | Strong | Partial | Strong | Strong |
| Evidence sufficiency loop | Strong | Partial | Missing | Partial | Partial |
| MCP tool server surface | Strong | Missing | Missing | Missing | Missing |
| Portable no-MCP skill package | Strong | Partial (library) | Partial (library) | Missing | Missing |
| Structured-tag prompt controls | Partial | Strong | Missing | Missing | Missing |
| Composable compression pipeline | Partial | Partial | Missing | Strong | Partial |
| Benchmark guard automation | Strong | Partial | Partial | Partial | Partial |
| Test coverage depth | Strong | Partial | Missing | Strong | Missing |

## Concrete Improvements We Can Borrow

## From LLMLingua

Observed in:

- `artifacts/competitor-analysis/LLMLingua/README.md`
- `artifacts/competitor-analysis/LLMLingua/llmlingua/prompt_compressor.py`

Ideas:

1. Structured segment directives (per-segment compression policy).
2. More explicit target-budget controls and preservation controls.
3. Clearer multi-strategy compression entry points (context-level + token-level).

## From Selective Context

Observed in:

- `artifacts/competitor-analysis/Selective_Context/readme.md`
- `artifacts/competitor-analysis/Selective_Context/selective_context.py`

Ideas:

1. Transparent self-information ranking explanation per lexical unit.
2. Simple deterministic "reduce ratio" semantics for user-facing UX.
3. Language-specific tokenization hooks that are easy to reason about.

## From LangChain Contextual Compression

Observed in:

- `artifacts/competitor-analysis/langchain/libs/langchain/langchain_classic/retrievers/contextual_compression.py`
- `artifacts/competitor-analysis/langchain/libs/langchain/langchain_classic/retrievers/document_compressors/base.py`
- `artifacts/competitor-analysis/langchain/libs/langchain/langchain_classic/retrievers/document_compressors/chain_extract.py`
- `artifacts/competitor-analysis/langchain/libs/langchain/langchain_classic/retrievers/document_compressors/embeddings_filter.py`

Ideas:

1. First-class compressor pipelines (chain multiple filters/extractors/transforms).
2. Uniform sync/async interfaces for compressor implementations.
3. Plug-and-play compatibility adapters for external retriever stacks.

## From Contextual-Compression Repo

Observed in:

- `artifacts/competitor-analysis/Contextual-Compression/README.md`

Ideas:

1. Explicit retrieval metric reporting (precision/recall/F1) in comparisons.
2. Side-by-side compressor strategy bake-offs in one run.

## Priority Gap List (ROI Order)

1. Add first-class composable compression pipeline in Token Saver app layer.
2. Add structured segment controls for skill and server compression paths.
3. Expand benchmark harness to include retrieval precision/recall/F1, not only token savings.
4. Publish per-step explainability payloads (why each segment was kept/dropped).
5. Add external stack adapters (LangChain/LlamaIndex-friendly wrappers).

## Current Competitive Position

Token Saver 5000 is currently strongest on:

1. Integrated operational surface (server + skill + tests + guards).
2. Contract-hardening and app-layer architecture discipline.
3. Evidence-aware retrieval-aware compression productization.

To outperform on the full stack, we need to close gaps in:

1. Strategy composability.
2. Explainable scoring outputs.
3. Broader benchmark comparability metrics.
