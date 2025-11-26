# Security Audit Report - Token Saver 5000 MCP Server

**Version:** 0.6.0
**Date:** 2025-01-26
**Auditor:** Claude Code
**Status:** IN PROGRESS

## Executive Summary

This document provides a comprehensive security audit of the Token Saver 5000 MCP server implementation, focusing on input validation, error handling, secrets management, and general security best practices for local MCP servers.

### 🚨 CRITICAL FINDINGS

**1. Path Traversal Vulnerability (CWE-22) - SEVERITY: HIGH**
- **Status:** IDENTIFIED - NOT YET FIXED
- **CVSS Score:** 7.5 (HIGH)
- **Impact:** Arbitrary file read with user privileges
- **Attack Vector:** MCP tool `ingest_context(file_path="../../etc/passwd")` → `refresh_document()` reads arbitrary files
- **Affected Files:**
  - `src/handlers/compression_handlers.py` (entry point)
  - `src/file_sync_manager.py` (no path validation)
  - `src/version_manager.py` (no path validation)
  - `src/handlers/file_sync_handlers.py` (exploit triggers)
- **Mitigation:** Implement PathValidator with allowed directory whitelist (detailed fix provided)
- **Timeline:** Fix required IMMEDIATELY before any public deployment

### Audit Summary

| Category | Status | Findings |
|----------|--------|----------|
| Input Validation | 🚨 CRITICAL | Path traversal vulnerability found |
| Error Handling | 🔍 In Progress | Partial review completed |
| Secrets Management | 🔍 Pending | Not yet reviewed |
| Logging Security | 🔍 Pending | Not yet reviewed |
| Resource Limits | ✅ VERIFIED | LRU limits in place (v0.4.2) |
| File System Security | 🚨 CRITICAL | Path traversal vulnerability found |

**Overall Security Posture:** Currently NOT SUITABLE for production deployment due to critical path traversal vulnerability. All other security aspects appear reasonable pending full audit.

## Threat Model

**Deployment Context:** Local MCP Server (stdio transport)
- Runs with user privileges on local machine
- Accessed via Claude Desktop through stdio IPC
- Has file system access within user's permissions
- No network exposure (stdio-only)

**Primary Security Risks:**
1. **Local Server Compromise** ⚠️ HIGH RISK
   - Arbitrary code execution with user privileges
   - Data exfiltration via file system access
   - Path traversal attacks in file operations
   - Malicious input causing crashes or resource exhaustion

2. **Confused Deputy Problem** ✅ NOT APPLICABLE
   - No OAuth proxying in current implementation

3. **Session Hijacking** ✅ NOT APPLICABLE
   - Stdio transport, no session management

4. **Token Passthrough** ✅ NOT APPLICABLE
   - Local server, no token forwarding

**Attack Surface:**
- 30 MCP tool handlers accepting various input types
- File system operations (4 file sync handlers)
- In-memory data structures (potential DoS)
- Error messages (potential information disclosure)
- Logging (potential secrets leakage)

## MCP Tools Inventory (30 Tools)

### Compression Handlers (9 tools)
1. `compress_document` - args: `file_id` (str), `text` (str), `compression_level` (str)
2. `decompress_document` - args: `file_id` (str)
3. `get_compression_stats` - args: `file_id` (str)
4. `list_compressed_documents` - args: none
5. `delete_compressed_document` - args: `file_id` (str)
6. `modulate_region` - args: `file_id` (str), `node_ids` (List[str])
7. `compress_code` - args: `code` (str), `language` (str)
8. `compress_batch` - args: `documents` (List[Dict])
9. `check_freshness` - args: `file_id` (str)

### AFM Handlers (6 tools)
10. `afm_add_message` - args: `role` (str), `content` (str)
11. `afm_build_context` - args: `current_query` (str), `budget_tokens` (int), `system_preamble` (str, optional)
12. `afm_get_stats` - args: none
13. `afm_clear_history` - args: none
14. `afm_export_history` - args: none
15. `afm_import_history` - args: `history_data` (dict)

### ACE Handlers (7 tools)
16. `ace_create_context` - args: `context_id` (str), `scenario_type` (str)
17. `ace_add_bullet` - args: `context_id` (str), `text` (str), `category` (str)
18. `ace_retrieve_context` - args: `context_id` (str), `query` (str), `top_k` (int)
19. `ace_list_contexts` - args: none
20. `ace_delete_context` - args: `context_id` (str)
21. `ace_export_context` - args: `context_id` (str)
22. `ace_import_context` - args: `context_id` (str), `context_data` (dict)

### File Sync Handlers (4 tools) ⚠️ CRITICAL
23. `sync_file` - args: `doc_id` (str), `file_path` (str)
24. `get_file_sync_status` - args: `doc_id` (str)
25. `list_synced_files` - args: none
26. `get_version_history` - args: `doc_id` (str)

### Detection Handlers (2 tools)
27. `check_blind_spots` - args: `ai_response` (str), `file_id` (str), `retrieved_nodes` (List[str])
28. `detect_hallucination` - args: `ai_response` (str), `file_id` (str)

### Resource Handlers (1 tool)
29. `check_resource_health` - args: none

### Visualization Handlers (1 tool)
30. `visualize_graph` - args: `file_id` (str), `output_format` (str)

## Security Audit Findings

### 1. Input Validation

#### 1.1 File Sync Handlers (CRITICAL - Path Traversal Risk)

**Status:** 🚨 **CRITICAL VULNERABILITY FOUND**

**Vulnerability Summary:**
- **CVE Classification:** CWE-22 (Path Traversal)
- **Severity:** HIGH (CVSS 7.5)
- **Attack Vector:** Local (requires MCP tool access)
- **Impact:** Arbitrary file read with user privileges

**Root Cause Analysis:**

1. **Entry Point:** `src/handlers/compression_handlers.py:146`
   ```python
   file_path = args.get("file_path")  # NO VALIDATION
   context["sync_manager"].register_file(file_id, file_path, text)  # Line 211
   context["version_manager"].add_version(..., file_path=file_path, ...)  # Line 212
   ```

2. **Vulnerable Storage:** `src/file_sync_manager.py:73-131`
   ```python
   def register_file(self, doc_id: str, file_path: Optional[str], content: str):
       # Line 97-98: NO PATH VALIDATION
       if file_path and os.path.exists(file_path):
           stat = os.stat(file_path)
   ```

3. **Exploit Trigger:** `src/handlers/file_sync_handlers.py:175`
   ```python
   # handle_refresh_document() - ARBITRARY FILE READ
   with open(metadata.file_path, "r", encoding="utf-8") as f:
       content = f.read()
   ```

4. **Second Exploit:** `src/version_manager.py:288`
   ```python
   # diff_with_current_file() - ARBITRARY FILE READ
   with open(cached.file_path, "r", encoding="utf-8") as f:
       current_content = f.read()
   ```

**Proof of Concept:**
```python
# Step 1: Ingest with malicious path
ingest_context(
    text="dummy content",
    file_id="exploit",
    file_path="../../etc/passwd"  # Path traversal
)

# Step 2: Trigger arbitrary file read
refresh_document(file_id="exploit")
# Result: Reads /etc/passwd and returns content

# Alternative exploit via diff:
diff_cached_file(file_id="exploit")
# Result: Shows diff including /etc/passwd content
```

**Affected Files:**
- ✅ `src/handlers/compression_handlers.py` - Entry point (line 146, 211-212)
- ✅ `src/file_sync_manager.py` - No validation (line 73-131)
- ✅ `src/version_manager.py` - No validation (line 96-104)
- ✅ `src/handlers/file_sync_handlers.py` - Exploit triggers (line 175, 288)

**Audit Notes:**
- [x] ❌ `handle_ingest()` - NO path validation before register_file()
- [x] ❌ `FileSyncManager.register_file()` - NO path validation
- [x] ❌ `VersionManager.add_version()` - NO path validation
- [x] ❌ `handle_refresh_document()` - Opens arbitrary file (line 175)
- [x] ❌ `VersionManager.diff_with_current_file()` - Opens arbitrary file (line 288)

#### 1.2 String Input Validation

**Status:** 🔍 PENDING REVIEW

**Handlers with String Inputs:**
- `compress_document` - `file_id`, `text`, `compression_level`
- `decompress_document` - `file_id`
- `get_compression_stats` - `file_id`
- `delete_compressed_document` - `file_id`
- `modulate_region` - `file_id`
- `compress_code` - `code`, `language`
- `afm_add_message` - `role`, `content`
- `afm_build_context` - `current_query`, `system_preamble`
- `ace_create_context` - `context_id`, `scenario_type`
- `ace_add_bullet` - `context_id`, `text`, `category`
- `ace_retrieve_context` - `context_id`, `query`
- `ace_delete_context` - `context_id`
- `ace_export_context` - `context_id`
- `ace_import_context` - `context_id`
- `check_blind_spots` - `ai_response`, `file_id`
- `detect_hallucination` - `ai_response`, `file_id`
- `visualize_graph` - `file_id`, `output_format`

**Security Requirements:**
- ✅ String length limits to prevent DoS
- ✅ Validate enum values (e.g., `compression_level`, `scenario_type`, `category`, `output_format`)
- ✅ Sanitize special characters if used in file operations
- ✅ No SQL injection risk (using in-memory/file storage, not SQL)

**Audit Notes:**
- [ ] Review string length limits across all handlers
- [ ] Review enum validation for all categorical inputs
- [ ] Review file_id sanitization (used in file operations)

#### 1.3 Numeric Input Validation

**Status:** 🔍 PENDING REVIEW

**Handlers with Numeric Inputs:**
- `afm_build_context` - `budget_tokens` (int)
- `ace_retrieve_context` - `top_k` (int)

**Security Requirements:**
- ✅ Positive integer validation
- ✅ Upper bounds to prevent resource exhaustion
- ✅ Type checking (reject floats, strings)

**Audit Notes:**
- [ ] Review `budget_tokens` validation (already has ValueError for <= 0)
- [ ] Review `top_k` validation for upper bounds

#### 1.4 Complex Input Validation

**Status:** 🔍 PENDING REVIEW

**Handlers with Complex Inputs:**
- `modulate_region` - `node_ids` (List[str])
- `compress_batch` - `documents` (List[Dict])
- `afm_import_history` - `history_data` (dict)
- `ace_import_context` - `context_data` (dict)
- `check_blind_spots` - `retrieved_nodes` (List[str])

**Security Requirements:**
- ✅ List length limits to prevent DoS
- ✅ Nested structure validation
- ✅ Type checking for all fields
- ✅ Reject malformed data gracefully

**Audit Notes:**
- [ ] Review list length limits (node_ids, documents, retrieved_nodes)
- [ ] Review `afm_import_history` validation (12 validation paths tested)
- [ ] Review `ace_import_context` validation
- [ ] Review `compress_batch` validation

### 2. Error Handling

**Status:** 🔍 PENDING REVIEW

**Security Requirements:**
- ✅ Error messages must not leak sensitive data (file paths, internal structure)
- ✅ Stack traces must not be exposed to client
- ✅ Consistent error format across all handlers
- ✅ Graceful degradation on non-critical errors

**Audit Notes:**
- [ ] Review error messages in all 30 handlers
- [ ] Review exception handling patterns
- [ ] Check for information disclosure in error responses

### 3. Secrets Management

**Status:** 🔍 PENDING REVIEW

**Security Requirements:**
- ✅ No hardcoded secrets in code
- ✅ Environment variables for sensitive config
- ✅ Secrets not logged or exposed in error messages
- ✅ .env files in .gitignore

**Audit Notes:**
- [ ] Review all environment variable usage
- [ ] Review .gitignore for secrets patterns
- [ ] Scan codebase for hardcoded credentials

### 4. Logging Security

**Status:** 🔍 PENDING REVIEW

**Security Requirements:**
- ✅ No secrets logged (API keys, tokens, passwords)
- ✅ No PII logged (unless necessary and documented)
- ✅ Log sanitization for user inputs
- ✅ Structured logging with appropriate levels

**Audit Notes:**
- [ ] Review all logger.debug/info/warning/error calls
- [ ] Check for accidental credential logging
- [ ] Review user input logging patterns

### 5. Resource Limits

**Status:** 🔍 PENDING REVIEW

**Security Requirements:**
- ✅ Document size limits enforced (prevent DoS)
- ✅ Memory limits enforced (LRU eviction in place)
- ✅ Rate limiting (not applicable for local stdio server)
- ✅ Batch operation limits

**Audit Notes:**
- [x] File sync metadata LRU: 1000 entry limit ✅ VERIFIED (v0.4.2)
- [x] ACE context LRU: 100 context limit ✅ VERIFIED (v0.4.2)
- [x] Version history pruning: 10 versions per document ✅ VERIFIED (v0.4.2)
- [ ] Review document size limits in compression handlers
- [ ] Review batch operation limits in compress_batch
- [ ] Review list length limits in all handlers

### 6. File System Security

**Status:** 🔍 PENDING REVIEW

**Security Requirements:**
- ✅ Path traversal prevention (critical)
- ✅ Symbolic link handling
- ✅ Permission error handling
- ✅ Safe file deletion
- ✅ Atomic file operations where possible

**Audit Notes:**
- [ ] Review file_sync_manager.py for path validation
- [ ] Review version_manager.py for safe file operations
- [ ] Review persistence.py for safe storage operations

## Recommendations

### 🚨 CRITICAL PRIORITY (IMMEDIATE ACTION REQUIRED)

#### 1. Fix Path Traversal Vulnerability (CWE-22)

**Affected Functions:**
- `src/handlers/compression_handlers.py:handle_ingest()` (line 146, 211-212)
- `src/file_sync_manager.py:FileSyncManager.register_file()` (line 73-131)
- `src/version_manager.py:VersionManager.add_version()` (line 96-104)

**Required Fix - Option 1 (Recommended): Add Path Validation Helper**

Create `src/path_validator.py`:
```python
"""Path validation for file sync operations"""
import os
from pathlib import Path
from typing import Optional

class PathValidator:
    """Validates file paths to prevent path traversal attacks"""

    def __init__(self, allowed_base_dirs: Optional[list[str]] = None):
        """
        Args:
            allowed_base_dirs: List of allowed base directories (None = current working directory only)
        """
        if allowed_base_dirs is None:
            # Default: only allow files in current working directory and subdirectories
            self.allowed_base_dirs = [os.getcwd()]
        else:
            # Resolve all base dirs to absolute paths
            self.allowed_base_dirs = [os.path.abspath(d) for d in allowed_base_dirs]

    def validate(self, file_path: str) -> str:
        """
        Validate file path and return absolute path if safe.

        Args:
            file_path: Path to validate

        Returns:
            Absolute path if valid

        Raises:
            ValueError: If path is invalid or outside allowed directories
        """
        if not file_path:
            raise ValueError("file_path cannot be empty")

        # Resolve to absolute path (resolves .. and symlinks)
        try:
            abs_path = os.path.abspath(os.path.realpath(file_path))
        except Exception as e:
            raise ValueError(f"Invalid file path: {e}")

        # Check if path is within allowed directories
        is_allowed = any(
            abs_path.startswith(base_dir)
            for base_dir in self.allowed_base_dirs
        )

        if not is_allowed:
            raise ValueError(
                f"Access denied: file_path must be within allowed directories\n"
                f"   Allowed: {', '.join(self.allowed_base_dirs)}\n"
                f"   Attempted: {abs_path}\n"
                f"💡 Tip: Use relative paths or paths within the current directory"
            )

        return abs_path
```

**Apply Fix in compression_handlers.py:**
```python
# At top of file
from ..path_validator import PathValidator

# In handle_ingest() after line 146:
file_path = args.get("file_path")

# Add validation:
if file_path:
    path_validator = context.get("path_validator")
    if not path_validator:
        raise RuntimeError("Path validator not configured - file sync disabled for security")
    try:
        file_path = path_validator.validate(file_path)  # Returns absolute path
    except ValueError as e:
        raise ValueError(f"Invalid file_path: {e}")
```

**Apply Fix in file_sync_manager.py:**
```python
# Update register_file() line 73-131:
def register_file(self, doc_id: str, file_path: Optional[str], content: str,
                  validate_path: bool = True) -> FileMetadata:
    """
    Register a file when it's ingested.

    Args:
        doc_id: Document ID
        file_path: Path to source file (must be already validated as absolute path)
        content: Content that was ingested
        validate_path: Whether to verify path is absolute (security check)
    """
    # Security check: Ensure file_path is absolute (validation should happen at entry point)
    if file_path and validate_path:
        if not os.path.isabs(file_path):
            raise ValueError(
                f"file_path must be absolute (got: {file_path})\n"
                "This indicates a security misconfiguration - contact administrator"
            )
```

**Apply Fix in version_manager.py:**
```python
# Update add_version() line 96-104:
def add_version(self, doc_id: str, content: str, checksum: str,
                file_path: Optional[str] = None,
                validate_path: bool = True,
                metadata: Optional[Dict] = None,
                compression_stats: Optional[Dict] = None) -> DocumentVersion:
    """
    Add a new version of a document.

    Args:
        ...
        file_path: Source file path (must be already validated as absolute path)
        validate_path: Whether to verify path is absolute (security check)
        ...
    """
    # Security check: Ensure file_path is absolute
    if file_path and validate_path:
        if not os.path.isabs(file_path):
            raise ValueError(
                f"file_path must be absolute (got: {file_path})\n"
                "This indicates a security misconfiguration - contact administrator"
            )
```

**Initialize PathValidator in server.py:**
```python
# In serve() function, initialize path_validator
from .path_validator import PathValidator

# Allow current directory and user's home directory
allowed_dirs = [
    os.getcwd(),
    os.path.expanduser("~")  # User's home directory
]
path_validator = PathValidator(allowed_base_dirs=allowed_dirs)

# Add to handler context
handler_context["path_validator"] = path_validator
```

**Testing Requirements:**
1. Test path traversal rejection: `../../etc/passwd` → ValueError
2. Test absolute path outside allowed dirs: `/etc/passwd` → ValueError
3. Test symlink traversal: `ln -s /etc/passwd safe.txt; ingest safe.txt` → Should resolve and reject
4. Test allowed paths: `./docs/test.txt` → Should work
5. Test relative paths: `docs/test.txt` → Should resolve to absolute and validate

### HIGH PRIORITY

2. **Add Input Length Limits** across all handlers
   - Define MAX_STRING_LENGTH constant (e.g., 1MB)
   - Define MAX_LIST_LENGTH constant (e.g., 1000 items)
   - Validate before processing

3. **Sanitize Error Messages**
   - Review all ValueError/Exception messages
   - Remove file paths, internal structure details
   - Use generic error messages for client-facing errors

### MEDIUM PRIORITY
4. **Implement Structured Logging**
   - Use structured logging with context
   - Add log levels appropriately
   - Sanitize user inputs in logs

5. **Add Health Check Endpoint**
   - Expose resource health via MCP tool
   - Monitor for anomalies
   - Alert on resource exhaustion

### LOW PRIORITY
6. **Document Security Posture**
   - Add SECURITY.md to repository
   - Document threat model
   - Provide security guidelines for users

## Audit Progress

- [x] 1.1 File Sync Handlers Path Traversal Review - **CRITICAL VULNERABILITY FOUND**
- [ ] 1.2 String Input Validation Review
- [ ] 1.3 Numeric Input Validation Review
- [ ] 1.4 Complex Input Validation Review
- [ ] 2. Error Handling Review
- [ ] 3. Secrets Management Review
- [ ] 4. Logging Security Review
- [x] 5. Resource Limits Review - **VERIFIED: LRU limits in place (v0.4.2)**
- [ ] 6. File System Security Review

## Implementation Tasks

### IMMEDIATE (Critical Security Fix)

1. **Create PathValidator Module** `src/path_validator.py`
   - [ ] Implement PathValidator class with validate() method
   - [ ] Add unit tests for path validation (8 test cases)
   - [ ] Test path traversal rejection (`../../etc/passwd`)
   - [ ] Test absolute path outside allowed dirs (`/etc/passwd`)
   - [ ] Test symlink resolution and validation
   - [ ] Test relative path resolution

2. **Apply Path Validation to Entry Points**
   - [ ] Update `src/handlers/compression_handlers.py:handle_ingest()`
   - [ ] Add path_validator to HandlerContext TypedDict
   - [ ] Add path validation before register_file() call
   - [ ] Add path validation before add_version() call

3. **Add Security Checks to Storage Layers**
   - [ ] Update `src/file_sync_manager.py:register_file()` with path validation
   - [ ] Update `src/version_manager.py:add_version()` with path validation
   - [ ] Ensure both verify paths are absolute (defense in depth)

4. **Initialize PathValidator in Server**
   - [ ] Update `src/server.py:serve()` to create PathValidator
   - [ ] Configure allowed base directories (cwd + home dir)
   - [ ] Add path_validator to handler_context

5. **Testing and Verification**
   - [ ] Create `tests/test_path_validator.py` (8+ tests)
   - [ ] Create `tests/test_security_path_traversal.py` (exploit tests)
   - [ ] Verify existing tests still pass
   - [ ] Document security fix in CHANGELOG.md

### HIGH PRIORITY (Post-Critical Fix)

6. **Input Length Limits**
   - [ ] Add MAX_STRING_LENGTH to constants.py
   - [ ] Add MAX_LIST_LENGTH to constants.py
   - [ ] Validate string lengths in all handlers
   - [ ] Validate list lengths in all handlers

7. **Error Message Sanitization**
   - [ ] Review all ValueError messages for info disclosure
   - [ ] Remove file paths from error messages (or sanitize)
   - [ ] Add generic error messages for security-sensitive errors

8. **Logging Security Audit**
   - [ ] Scan all logger.debug/info/warning/error calls
   - [ ] Remove any accidental credential logging
   - [ ] Sanitize user inputs in logs

## Next Steps

1. ✅ **COMPLETED:** Identified critical path traversal vulnerability
2. **IN PROGRESS:** Create implementation plan for path validation
3. **NEXT:** Implement PathValidator module with comprehensive tests
4. **THEN:** Apply path validation to all entry points
5. **FINALLY:** Re-run full test suite and update security documentation

---

**Document Version:** 1.0
**Last Updated:** 2025-01-26
