Show comprehensive token savings statistics across all compressed documents.

Steps:
1. Use `mcp__token-saver__list_documents` to get all documents
2. Use `mcp__token-saver__get_stats` for detailed compression metrics
3. Use `mcp__token-saver__get_resource_usage` for memory stats
4. Use `mcp__token-saver__afm_get_stats` for dialogue memory stats

Report Format:

## Document Compression Stats
| Document | Original | Compressed | Ratio | Savings |
|----------|----------|------------|-------|---------|
| (per doc)|          |            |       |         |
| **Total**|          |            |       |         |

## Dialogue Memory Stats
- Messages stored: X
- Critical (full fidelity): X
- Compressed: X
- Token usage: X

## Resource Usage
- Memory: X MB / 1 GB limit
- Documents: X / 1000 limit
- Cache hit rate: X%

## Recommendations
- Suggest pruning stale documents if approaching limits
- Suggest compression opportunities if large docs uncompressed
