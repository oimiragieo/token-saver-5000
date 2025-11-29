Compress the following document for efficient token usage.

Steps:
1. Use `mcp__token-saver__ingest_context` with:
   - text: The document content below
   - file_id: Generate a short descriptive ID based on the content
2. Use `mcp__token-saver__read_skeleton` to show the compressed structure
3. Report compression stats:
   - Original tokens
   - Compressed tokens
   - Compression ratio
   - Token savings percentage

After compression, offer to:
- Search for specific topics with `/search-docs`
- Expand any section with `/expand`
- Analyze the full document with `/analyze`

Document to compress:
$ARGUMENTS
