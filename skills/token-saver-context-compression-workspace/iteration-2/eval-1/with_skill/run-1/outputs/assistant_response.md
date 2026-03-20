I don't have file read/write/bash tools available in this session, only LSP and Hugging Face tools. Based on the CLAUDE.md context provided, here's what I can tell you:

---

**Exact command** (query-targeted mode via MCP tool):

```bash
python -m src.server
```

Then call the MCP tool `ingest_context` with the file, followed by `read_context` with `query="what are the retry rules?"`. The query-adaptive mode is handled by `src/query_adaptive.py` which automatically adjusts compression ratios based on the query.

Alternatively, if you want to use the skill scripts directly:

```bash
python skills/token-saver-context-compression/scripts/compress.py \
  --file tests/fixtures/skill_context_sample.txt \
  --query "what are the retry rules?"
```

However, I **cannot verify the exact script name or arguments** without reading the files â€” I don't have file read tools available in this session. The fixture file `tests/fixtures/skill_context_sample.txt` also needs to exist (it's not listed in git status as tracked or untracked).

**Safety assessment:** Compressed output from semantic compression should be treated as **not safe to answer from directly** without caveats. The compression is lossy (87.4% token reduction means ~87% of content is discarded). Query-adaptive mode improves relevance to the specific question, but:
- Nodes below the similarity threshold are pruned entirely
- Semantic relationships may be lost
- The "retry rules" answer could be incomplete or missing edge cases

You should treat compressed results as a **summary/pointer** â€” useful for identifying which sections are relevant, but verify against the original document before giving a definitive answer.