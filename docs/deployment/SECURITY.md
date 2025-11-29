# Security Policy

## Supported Versions

| Version | Supported          | Security Status |
| ------- | ------------------ | --------------- |
| 0.6.1+  | ✅ Yes             | Path traversal vulnerability fixed |
| 0.6.0   | :x: No             | Critical path traversal (CWE-22) |
| < 0.6   | :x: Unsupported    | Not recommended |

## Security Status

**Current Status:** ✅ **v0.6.1 SECURITY FIX RELEASED**

Critical path traversal vulnerability (CWE-22) has been **FIXED** in version 0.6.1.

**Affected Versions:** All versions up to and including 0.6.0

**Fix Version:** v0.6.1 (released 2025-01-26)

## Known Vulnerabilities

### 1. Path Traversal (CWE-22) - ✅ FIXED in v0.6.1

**Severity:** HIGH (CVSS 7.5)

**Status:** ✅ **FIXED** in v0.6.1 (2025-01-26)

**Description:** The `ingest_context` MCP tool in v0.6.0 and earlier accepted a `file_path` parameter without proper validation. An attacker with MCP tool access could provide path traversal sequences (e.g., `../../etc/passwd`) to read arbitrary files accessible to the user running the MCP server.

**Affected Components (v0.6.0 and earlier):**
- `src/handlers/compression_handlers.py` - Entry point (now patched)
- `src/file_sync_manager.py` - No path validation (now patched)
- `src/version_manager.py` - No path validation (now patched)
- `src/handlers/file_sync_handlers.py` - Exploit triggers (now patched)

**Proof of Concept (v0.6.0 - NO LONGER WORKS in v0.6.1+):**
```python
# Step 1: Ingest with malicious path
ingest_context(
    text="dummy",
    file_id="exploit",
    file_path="../../etc/passwd"  # BLOCKED in v0.6.1+
)

# Step 2: Trigger file read
refresh_document(file_id="exploit")
# v0.6.0: Returns contents of /etc/passwd
# v0.6.1: ValueError - Access denied
```

**Fix Implementation (v0.6.1):**
- ✅ **PathValidator module** (`src/path_validator.py`) - 220 lines, 96% test coverage
- ✅ **Entry point validation** - `handle_ingest()` validates all file paths
- ✅ **Defense-in-depth** - Storage layers verify absolute paths
- ✅ **Comprehensive testing** - 31 security tests covering exploit scenarios
- ✅ **Whitelist-based security** - File access restricted to allowed directories

**Timeline:**
- Identified: 2025-01-26
- Fix Released: v0.6.1 (2025-01-26)
- Verification: 735 tests passing, 72.69% coverage

## Threat Model

### Deployment Context

Token Saver 5000 is a **local MCP server** that runs on the user's machine and communicates via stdio transport with Claude Desktop. It has:

- **Privileges:** Runs with user's file system permissions
- **Access:** Can read/write files accessible to the user
- **Network:** No network exposure (stdio-only communication)
- **Attack Surface:** 30 MCP tool handlers accepting various inputs

### Primary Threats

1. **Local Server Compromise** ✅ MITIGATED (v0.6.1)
   - ~~Arbitrary file read (path traversal vulnerability)~~ **FIXED**
   - Potential arbitrary code execution through malicious inputs ⚠️ ONGOING REVIEW
   - Data exfiltration via file system access ✅ RESTRICTED to allowed directories

2. **Resource Exhaustion** ✅ MITIGATED (v0.4.2)
   - LRU eviction in place for file sync metadata (1000 entry limit)
   - LRU eviction for ACE contexts (100 context limit)
   - Automatic version pruning (10 versions per document)

3. **Information Disclosure** ⚠️ MEDIUM RISK
   - Full file paths displayed in error messages (non-critical for local stdio server)
   - Potential sensitive data leakage in logs

4. **Input Validation** ✅ IMPROVED (v0.6.1)
   - ✅ Path validation with whitelist-based security
   - String length limits not consistently enforced ⚠️ LOW PRIORITY
   - Complex input structures need validation ⚠️ ONGOING
   - Enum values partially validated ✅ ACCEPTABLE for stdio deployment

## Security Best Practices for Users

### Current Recommendations (v0.6.1+)

**DO:**
- ✅ Run the MCP server in a restricted user account (defense-in-depth)
- ✅ Monitor file system access logs
- ✅ Keep the software updated (subscribe to security advisories)
- ✅ Review MCP tool calls in Claude Desktop before approval
- ✅ **v0.6.1+:** File sync is now SAFE to use with path validation
- ✅ **v0.6.1+:** Allowed directories: current working directory + user home directory

**DO NOT:**
- ❌ Use v0.6.0 or earlier versions (path traversal vulnerability)
- ❌ Disable path validation or modify allowed directories without review
- ❌ Run as administrator/root (principle of least privilege)
- ⚠️ Deploy to production environments (beta software, not yet production-ready)

### Security Best Practices

**Path Validation (v0.6.1+):**
- File access restricted to allowed base directories (CWD + user home)
- All paths validated at entry point and storage layers
- Path traversal sequences (../, symlinks) blocked
- Use absolute paths for clarity and security

**Data Protection:**
- Regular backups of `.semantic_modulator_data/` directory
- Review version history for sensitive data exposure
- Monitor resource usage and disk space

## Reporting a Vulnerability

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via:
- Email: [security contact to be added]
- GitHub Security Advisories: [Create a private security advisory](https://github.com/oimiragieo/token-saver-5000/security/advisories/new)

### What to Include

1. Description of the vulnerability
2. Steps to reproduce
3. Proof of concept (if available)
4. Potential impact assessment
5. Suggested fix (if available)

### Response Timeline

- **Initial Response:** Within 48 hours
- **Severity Assessment:** Within 5 business days
- **Fix Timeline:**
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next minor release

## Security Audit History

| Date | Auditor | Findings | Status |
|------|---------|----------|--------|
| 2025-01-26 | Claude Code | Path traversal (CWE-22, CVSS 7.5) | ✅ Fixed in v0.6.1 |

## Changelog

### Security Fixes

#### v0.6.1 (2025-01-26) - SECURITY RELEASE
- ✅ **CRITICAL FIX:** Path traversal vulnerability (CWE-22, CVSS 7.5 HIGH)
- ✅ Added PathValidator with allowed directory whitelist (`src/path_validator.py`)
- ✅ Entry point validation in `handle_ingest()`
- ✅ Defense-in-depth checks in storage layers
- ✅ Comprehensive security testing (31 tests, 96% coverage for PathValidator)
- ✅ **Verification:** All 735 tests passing, 72.69% overall coverage

#### v0.4.2 (2025-01-XX)
- ✅ Added LRU eviction for file sync metadata (prevents DoS)
- ✅ Added LRU eviction for ACE contexts (prevents memory exhaustion)
- ✅ Added automatic version pruning (prevents unbounded growth)

## Contact

For security concerns, please contact:
- Repository: https://github.com/oimiragieo/token-saver-5000
- Security Issues: Use GitHub Security Advisories

---

**Last Updated:** 2025-01-26 (v0.6.1 Security Release)
**Security Status:** ✅ Critical Path Traversal Vulnerability Fixed - Beta Quality
