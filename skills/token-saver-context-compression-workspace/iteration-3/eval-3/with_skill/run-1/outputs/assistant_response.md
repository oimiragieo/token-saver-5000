I apologize for the tool issues. Let me check what tools are actually available to me for file operations.

It appears I don't have standard file-reading tools available in this session. Let me answer based on the extensive CLAUDE.md context provided, which documents this project thoroughly.

---

Here's what the repo documents:

### Command Pattern for Compressing a LangChain-style JSON Payload

Use the **`ingest_context`** MCP tool (one of the 9 document compression tools). The call pattern via the MCP server:

```bash
python -m src.server  # start stdio MCP server
```

Then invoke the `ingest_context` tool with the JSON payload as the `text` parameter and a `doc_id`. For a question-specific compression, pair it with `modulate_region` (to set fidelity for a specific section) or `search_semantic` (to retrieve only question-relevant nodes from the graph):

```
ingest_context  â†’  search_semantic (with your question as query)  â†’  read_context
```

This ingests the full JSON, then retrieves only semantically-relevant compressed nodes for your specific question.

### Output-Format Policy

All experimental tool responses include `"experimental": true` in every response. All MCP tools return **structured JSON output** â€” there is no raw-text mode. Compression handlers return JSON with fields like `compressed_text`, `compression_ratio`, `token_count`, and metadata.

### Cache-Friendly Prompt Structure Note

The project uses a **3-tier embedding cache** (SBERT â†’ ONNX â†’ TF-IDF) with an **LRU eviction cache** (default 10k entries, 60-80% hit rate). For cache-friendly prompts: **keep your `doc_id` stable across calls** â€” the system caches embeddings keyed by content hash, so re-ingesting the same document with the same `doc_id` will hit the LRU cache rather than recomputing embeddings. Changing the `doc_id` for identical content defeats the cache.