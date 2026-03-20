# Eval 3 Response

Use this command pattern for LangChain-style payloads:

`python skills\token-saver-context-compression\scripts\compress_context.py --json-file <payload.json> --input-adapter auto --mode query_guided --query "retry policy" --output-format auto`

This run resolved the adapter to `langchain_json` for 2 documents. For cache-friendly prompting, keep the compressed context before dynamic metadata and the latest user question.

Output-format policy: prefer `auto` by default. It keeps JSON for irregular structures and only switches to TOON when the shape is uniform enough to save tokens.