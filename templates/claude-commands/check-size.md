Check if a file needs compression WITHOUT reading its content.

This tool estimates token count from file size, saving tokens by not
reading large files just to decide if they should be compressed.

Steps:
1. Use `mcp__token-saver__should_compress` with the file path
2. Report the estimated token count and recommendation
3. Based on result:
   - NO_COMPRESS: File is small, read directly
   - RECOMMEND_COMPRESS: Consider using ingest_context
   - STRONGLY_RECOMMEND: Should definitely compress first
   - MUST_COMPRESS: File is too large, must compress

Example output:
```
File: /path/to/document.md
Size: 45,000 bytes
Estimated tokens: ~11,800
Recommendation: MUST_COMPRESS
Potential savings: ~10,000 tokens with compression

Suggested action: Use /compress with this file content.
```

File to check: $ARGUMENTS
