Expand a compressed section to show full detail.

Steps:
1. Parse the node ID from the arguments (format: docname_nX)
2. Use `mcp__token-saver__modulate_region` with:
   - node_ids: The specified node(s)
   - fidelity: Start with "DETAILED", use "RAW" if more detail needed
3. Display the expanded content with clear formatting
4. After showing content, use `mcp__token-saver__check_blind_spots` to verify completeness

If no node ID provided, list available documents with `mcp__token-saver__list_documents`.

Section to expand: $ARGUMENTS
