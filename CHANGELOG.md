# Changelog

All notable changes to Token Saver 5000.

## [0.4.3] - 2025-11-25

### Added
- TypedDict handler hints for IDE autocomplete and type safety
- Comprehensive test coverage improvements (427 tests, 59% overall coverage)
- Code compressor unit tests (47 tests, 99% coverage)
- Server unit tests (43 tests, 88% coverage)
- Semantic compressor unit tests (65 tests, 99% coverage)

### Changed
- Refactored AFM._pack_messages() into focused helper functions
- Standardized all handler signatures to use HandlerContext
- Updated sentence-transformers to >=3.1.0 (CVE fixes)
- Version synchronized across all files to 0.4.3

### Fixed
- Critical handler signature bug in compression_handlers.py
- Security vulnerabilities CVE-2024-11392/11393/11394
- Documentation metric inaccuracies

## [0.4.0] - 2025-11-24

### Added
- ACE Framework (Agentic Context Engineering) - 32% quality boost with 4× compression
- 7 new MCP tools for ACE operations (generate, reflect, curate)
- File sync with staleness detection (mtime + MD5 checksums)
- Full version history with diffs
- LRU eviction for version history, file sync metadata, ACE contexts
- Singleton embedding manager (~70% memory reduction)

### Changed
- Handler architecture refactoring (server.py: 1,911 → 299 lines, 84% reduction)
- Modular handler system with centralized routing
- Total MCP tools: 16 → 30

## [0.2.0] - 2025-11-22

### Added
- Persistent storage (ChromaDB + JSON fallback)
- Resource management (100MB/doc, 1GB total, 1000 docs max)
- AFM export/import for conversation state
- Automated MCP installation script

### Changed
- Total MCP tools: 13 → 16

## [0.1.0] - Initial Release

### Features
- Semantic compression (80-95% token reduction)
- 5 adaptive fidelity levels
- Graph-based structure preservation with PageRank
- Code compression with AST parsing
- Multi-modal support (text, code, images)
- Blind spot detection
- 13 MCP tools

### Research Foundations
- JSCCM (arXiv:2511.15699v1)
- FPQE (arXiv:2511.15695v1)
- SCAR (arXiv:2511.14063v1)
- AFM (arXiv:2511.12712v1)
- ACE (arXiv:2510.04618v1)
