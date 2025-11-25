# File Sync & Version Management - Quick Reference

## 🚨 The Problem

**Without file sync:**
```
1. Ingest README.md → Cache compressed version
2. [Edit README.md in VS Code]
3. Ask Claude about README → Gets OLD version ❌
```

**With file sync:**
```
1. Ingest README.md → Cache + track metadata
2. [Edit README.md in VS Code]
3. Ask Claude about README → MCP warns: "⚠️ File changed 10min ago"
4. Claude refreshes cache → Gets CURRENT version ✅
```

---

## ⚡ Quick Commands

### **Check if cached file is up-to-date**
```python
# MCP Tool
check_file_sync(doc_id="readme")

# Output:
# ✅ readme is in sync
# OR
# ⚠️ readme is OUT OF SYNC - file changed 2 hours ago
```

### **See what changed**
```python
# MCP Tool
diff_cached_file(doc_id="readme", context_lines=3)

# Output:
# --- readme (cached v1)
# +++ README.md (current on disk)
# @@ -10,7 +10,9 @@
#  ## Features
#  - Semantic compression
# -- TOON integration
# +- TOON integration (NEW!)
# +- File sync detection
# +- Version history
```

### **Update cache from file**
```python
# MCP Tool
refresh_document(doc_id="readme")

# Output:
# ✅ Refreshed readme from /path/to/README.md
# New stats: 10,000 tokens → 500 tokens (20x compression)
# Version history: 3 versions
```

---

## 🎯 Real-World Scenarios

### **Scenario 1: You Edit Files Manually**

```
You: "Ingest my project documentation"
→ MCP caches all docs

[You spend 2 hours editing docs in VS Code]

You: "Claude, what's in the architecture doc?"

Claude → read_skeleton("architecture")
→ MCP: "⚠️ File changed 37 minutes ago"

Claude: "The cache is stale. Let me refresh it first."
→ Uses refresh_document("architecture")
→ Answers with current content ✅
```

---

### **Scenario 2: Claude Edits Files**

```
You: "Claude, refactor main.py to use async/await"

Claude → Uses Edit tool to modify main.py on disk

You: "Show me what you changed"

Claude → diff_cached_file("main_py")
→ Shows unified diff:
   - def process(data):
   + async def process(data):
       ...
```

---

### **Scenario 3: Rollback After Bad Edit**

```
Claude: [Edits config.py, breaks the app]

You: "Undo that change!"

Claude → version_manager.get_version("config", version_id=1)
→ Retrieves original content
→ Writes back to disk
→ Refreshes cache

✅ Rolled back to version 1
```

---

## 📊 How It Works

```
┌────────────────────────────────────┐
│  Your File: README.md (on disk)    │
└──────────────┬─────────────────────┘
               │
               │ Modified in VS Code
               ▼
┌────────────────────────────────────┐
│  FileSyncManager                   │
│  - Tracks mtime, checksum          │
│  - Detects changes                 │
│  - Warns when stale                │
└──────────────┬─────────────────────┘
               │
               │ On ingestion/refresh
               ▼
┌────────────────────────────────────┐
│  VersionManager                    │
│  - Stores full content             │
│  - Maintains version history       │
│  - Enables diff & rollback         │
│                                     │
│  Version 1: Original (2 days ago)  │
│  Version 2: Updated (1 day ago)    │
│  Version 3: Latest (now)           │
└──────────────┬─────────────────────┘
               │
               │ Used by
               ▼
┌────────────────────────────────────┐
│  SemanticCompressor (MCP)          │
│  - read_skeleton() → checks sync   │
│  - modulate_region() → checks sync │
│  - refresh_document() → updates    │
└────────────────────────────────────┘
```

---

## 🛠️ New MCP Tools (3 added)

### **Total: 17 → 20 tools**

**Tool 18: `check_file_sync`**
- Purpose: Check if cached version matches file on disk
- Input: `doc_id`
- Output: Sync status (in_sync: bool, reason: str)

**Tool 19: `diff_cached_file`**
- Purpose: Show differences between cache and current file
- Input: `doc_id`, `context_lines` (optional)
- Output: Unified diff

**Tool 20: `refresh_document`**
- Purpose: Re-ingest file from disk, update cache
- Input: `doc_id`
- Output: New compression stats, version number

---

## 💾 Storage Impact

### **File Metadata (FileSyncManager)**

```json
{
  "readme": {
    "file_path": "/path/to/README.md",
    "mtime": 1705881600.0,
    "checksum": "a1b2c3d4...",
    "ingestion_time": 1705878000.0,
    "size_bytes": 10240
  }
}
```

**Storage:** ~200 bytes per document

---

### **Version History (VersionManager)**

```json
{
  "version_id": 3,
  "doc_id": "readme",
  "content": "Full document content...",
  "timestamp": 1705881600.0,
  "checksum": "a1b2c3d4...",
  "file_path": "/path/to/README.md",
  "compression_stats": {
    "original_tokens": 10000,
    "compressed_tokens": 500,
    "compression_ratio": 20.0
  }
}
```

**Storage:** ~100KB per version (typical doc)

**Default Limit:** 10 versions per document (configurable)

**Total:** ~1MB per document with full history

---

## ⚙️ Configuration

Add to `.semantic_modulator_config.json`:

```json
{
  "version_management": {
    "enabled": true,
    "max_versions_per_doc": 10,
    "max_version_storage_mb": 500,
    "auto_cleanup_days": 30
  },
  "file_sync": {
    "enabled": true,
    "auto_refresh_on_read": false,
    "warn_on_stale": true,
    "check_interval_seconds": 300
  }
}
```

---

## 🔔 Automatic Warnings

When enabled, MCP automatically warns on stale cache:

```python
# Reading skeleton of stale document
read_skeleton("config")

# Output:
⚠️  WARNING: Cache may be stale!

File content changed on disk
Cached: 2 hours ago
Current: Modified 10 minutes ago

💡 Use refresh_document('config') to update
💡 Use diff_cached_file('config') to see changes

Proceeding with cached version...
---
[skeleton content follows]
```

---

## 📈 Performance

**Staleness check:**
- File stat: ~0.1ms
- Checksum (if needed): ~1ms per MB
- **Total:** <2ms for typical file

**Diff generation:**
- Read file: ~5ms per MB
- Compute diff: ~10ms per MB
- **Total:** ~15ms for 1MB file

**Refresh document:**
- Same as initial ingestion
- Compression: ~100ms per MB
- Version storage: ~10ms
- **Total:** ~110ms for 1MB file

---

## ✅ Best Practices

1. **Ingest with `file_path`** - Always provide file path when possible
   ```python
   ingest_context(
       content=readme_text,
       doc_id="readme",
       file_path="/path/to/README.md"  # Include this!
   )
   ```

2. **Check sync before long operations** - Avoid wasted work on stale data
   ```python
   # Good: Check first
   check_file_sync("config")
   if stale:
       refresh_document("config")
   # Now proceed with analysis
   ```

3. **Refresh after edits** - If Claude edits a file, refresh cache
   ```python
   # After editing main.py
   refresh_document("main_py")
   ```

4. **Use diffs for review** - Before accepting changes
   ```python
   # Before committing
   diff_cached_file("important_config")
   # Review changes
   ```

---

## 🐛 Troubleshooting

**"Document not registered"**
→ File wasn't ingested with `file_path` parameter
→ Re-ingest with file path

**"Cannot generate diff"**
→ Version history not enabled
→ Check storage directory permissions

**"Refresh failed: file not found"**
→ Source file was moved/deleted
→ Update file_path or re-ingest from new location

---

## 🚀 Future Enhancements

**Phase 2:**
- Auto-refresh on read (configurable)
- Watch mode (monitor files for changes)
- Git integration (track with commits)

**Phase 3:**
- Compressed version storage (store diffs, not full content)
- Semantic diffing (show changes at concept level)
- Conflict resolution (merge changes)

---

## 📚 Full Documentation

- **Implementation Guide:** `FILE_SYNC_IMPLEMENTATION.md`
- **Module Reference:** `src/file_sync_manager.py`, `src/version_manager.py`
- **Test Suite:** `tests/test_file_sync.py`

---

## 💡 Summary

**Before:**
```
Ingest file → Cache forever → Hope it doesn't change ❌
```

**After:**
```
Ingest file → Track metadata → Detect changes → Warn/refresh → Always current ✅
```

**Impact:**
- ✅ No more stale cache surprises
- ✅ See exactly what changed
- ✅ Rollback bad edits
- ✅ Audit trail of all versions
- ✅ Safe file editing workflows

**Cost:**
- +200 bytes per document (metadata)
- +~1MB per document (version history)
- +<2ms per read (staleness check)

**Trade-off:** Worth it for production use!
