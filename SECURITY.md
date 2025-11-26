# Security Policy

## Supported Versions

| Version | Supported          | Security Status |
| ------- | ------------------ | --------------- |
| 0.6.x   | :warning: Development | Critical vulnerability identified |
| < 0.6   | :x: Unsupported    | Not recommended |

## Security Status

**Current Status:** ⚠️ **NOT SUITABLE FOR PRODUCTION**

A critical path traversal vulnerability (CWE-22) has been identified in version 0.6.0 and earlier. This vulnerability allows arbitrary file read with user privileges through the file sync functionality.

**Affected Versions:** All versions up to and including 0.6.0

**Fix Status:** In development (tracked in [docs/SECURITY_AUDIT.md](docs/SECURITY_AUDIT.md))

## Known Vulnerabilities

### 1. Path Traversal (CWE-22) - CRITICAL

**Severity:** HIGH (CVSS 7.5)

**Description:** The `ingest_context` MCP tool accepts a `file_path` parameter without proper validation. An attacker with MCP tool access can provide path traversal sequences (e.g., `../../etc/passwd`) to read arbitrary files accessible to the user running the MCP server.

**Affected Components:**
- `src/handlers/compression_handlers.py` - Entry point
- `src/file_sync_manager.py` - No path validation
- `src/version_manager.py` - No path validation
- `src/handlers/file_sync_handlers.py` - Exploit triggers

**Proof of Concept:**
```python
# Step 1: Ingest with malicious path
ingest_context(
    text="dummy",
    file_id="exploit",
    file_path="../../etc/passwd"
)

# Step 2: Trigger file read
refresh_document(file_id="exploit")
# Result: Returns contents of /etc/passwd
```

**Mitigation:**
- **Immediate:** Do NOT use file sync features (`file_path` parameter) in untrusted environments
- **Long-term:** Path validation implementation in progress (see SECURITY_AUDIT.md)

**Timeline:**
- Identified: 2025-01-26
- Fix In Progress: Yes
- Expected Fix: v0.6.1

## Threat Model

### Deployment Context

Token Saver 5000 is a **local MCP server** that runs on the user's machine and communicates via stdio transport with Claude Desktop. It has:

- **Privileges:** Runs with user's file system permissions
- **Access:** Can read/write files accessible to the user
- **Network:** No network exposure (stdio-only communication)
- **Attack Surface:** 30 MCP tool handlers accepting various inputs

### Primary Threats

1. **Local Server Compromise** ⚠️ CURRENT RISK
   - Arbitrary file read (path traversal vulnerability)
   - Potential arbitrary code execution through malicious inputs
   - Data exfiltration via file system access

2. **Resource Exhaustion** ✅ MITIGATED
   - LRU eviction in place for file sync metadata (1000 entry limit)
   - LRU eviction for ACE contexts (100 context limit)
   - Automatic version pruning (10 versions per document)

3. **Information Disclosure** ⚠️ MEDIUM RISK
   - Full file paths displayed in error messages
   - Potential sensitive data leakage in logs

4. **Input Validation Gaps** ⚠️ UNDER REVIEW
   - String length limits not consistently enforced
   - Complex input structures need validation
   - Enum values partially validated

## Security Best Practices for Users

### Current Recommendations

**DO:**
- ✅ Run the MCP server in a restricted user account
- ✅ Monitor file system access logs
- ✅ Keep the software updated
- ✅ Review MCP tool calls in Claude Desktop before approval

**DO NOT:**
- ❌ Use file sync features (`file_path` parameter) until v0.6.1+
- ❌ Run in environments with sensitive files accessible
- ❌ Grant file system access to untrusted directories
- ❌ Deploy to production environments (not suitable yet)

### Future Recommendations (Post-v0.6.1)

Once path traversal is fixed:
- Define allowed base directories for file sync
- Use relative paths within project directories only
- Enable logging to monitor file access patterns
- Regular security audits of ingested documents

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
| 2025-01-26 | Claude Code | Path traversal (CWE-22) | Fix in progress |

## Changelog

### Security Fixes

#### Planned for v0.6.1
- [ ] Fix path traversal vulnerability in file sync
- [ ] Add PathValidator with allowed directory whitelist
- [ ] Sanitize file paths in error messages
- [ ] Add comprehensive path validation tests

#### v0.4.2 (2025-01-XX)
- ✅ Added LRU eviction for file sync metadata (prevents DoS)
- ✅ Added LRU eviction for ACE contexts (prevents memory exhaustion)
- ✅ Added automatic version pruning (prevents unbounded growth)

## Contact

For security concerns, please contact:
- Repository: https://github.com/oimiragieo/token-saver-5000
- Security Issues: Use GitHub Security Advisories

---

**Last Updated:** 2025-01-26
**Security Status:** Under Active Development - Not Production Ready
