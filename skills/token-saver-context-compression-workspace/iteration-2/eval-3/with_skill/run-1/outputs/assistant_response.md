The LSP server isn't cooperating. Let me try reading files via the Hugging Face doc fetch for local context â€” that won't work either. Let me try the available tools differently.

Based on the CLAUDE.md documentation already loaded in context, here's what I can provide:

---

### Command Pattern for LangChain-style JSON Compression

Use the **`ingest_context`** MCP tool (via `python -m src.server` stdio transport) with a query-specific call:

```
Tool: ingest_context
{
  "content": "<your LangChain JSON payload as string>",
  "doc_id": "langchain-payload",
  "query": "your specific question here"
}
```

Then retrieve the compressed version with:

```
Tool: read_context
{
  "doc_id": "langchain-payload",
  "fidelity": "BALANCED"
}
```

The `query` parameter in `ingest_context` activates **query-adaptive compression** (`src/query_adaptive.py`), which allocates higher fidelity to nodes semantically relevant to your question and aggressively prunes irrelevant ones â€” ideal for large LangChain chain/agent JSON where only a subset matters.

### Output-Format Policy

All MCP tool responses return **structured JSON** with these guaranteed fields:
- `"content"` â€” the compressed/processed result
- `"metadata"` â€” compression stats (ratio, token counts, fidelity level)
- Experimental tools additionally include `"experimental": true`

Fidelity levels (`MINIMAL`, `LOW`, `BALANCED`, `HIGH`, `LOSSLESS`) control the compressionâ€“detail tradeoff in every response.

### Cache-Friendly Prompt-Structure Note

**Place static/stable content (system prompts, tool definitions, schema definitions) at the front of your prompt, and variable/query-specific content at the end.** Provider prompt caches (Anthropic, OpenAI) use prefix-matching â€” content that shares a common prefix across calls gets cache hits. Reordering a LangChain payload so the chain definition is first and the user question is last maximizes cache reuse across multiple queries against the same chain. (Documented in `docs/guides/PROMPT_CACHING.md`.)