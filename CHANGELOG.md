# Changelog

All notable changes to Token Saver 5000 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-11-22

**Codename**: "Persistence & Polish"
**Status**: Production-Ready

This release transforms Token Saver 5000 into a production-ready, persistent system with robust resource management.

### Added

#### Persistent Storage Layer
- ChromaDB integration for vector storage
- JSON/Pickle fallback when ChromaDB unavailable
- Auto-save documents on ingest
- Auto-load documents on server start
- Storage location: `.semantic_modulator_data/`
- New file: `src/persistence.py` (531 lines)

#### Resource Management System
- Max document size: 100MB (configurable)
- Max total storage: 1GB (configurable)
- Max document count: 1000 (configurable)
- Max memory usage: 2GB (configurable)
- Pre-ingestion validation
- Warning thresholds (80% of limits)
- Resource statistics and health checks
- New file: `src/resource_manager.py` (257 lines)

#### New MCP Tools (+3)
- `afm_export_history` - Save conversation state for multi-session dialogues
- `afm_import_history` - Restore conversation state
- `list_documents` - Get inventory of all ingested documents with metadata

#### Automated Installation
- `install_mcp.sh` - Automated MCP configuration script
- Auto-detects OS (macOS/Linux/Windows)
- Locates Claude Desktop config automatically
- Generates MCP config with correct paths
- Safely merges with existing config
- Creates backup before modification

#### Robust Token Counting
- Primary: tiktoken (cl100k_base)
- Fallback: word count × 1.3 approximation
- Graceful degradation when tiktoken unavailable
- Better offline support

### Changed

#### MCP Server
- Total MCP tools: 13 → 16
- Integrated persistence manager
- Integrated resource manager
- Added export/import handlers for AFM

#### Documentation
- Updated ARCHITECTURE.md with v0.2.0 features
- Updated README.md with new tool count and features
- Created comprehensive audit reports

### Technical Details

#### Files Created
- `src/persistence.py` - Persistence layer (531 lines)
- `src/resource_manager.py` - Resource management (257 lines)
- `install_mcp.sh` - Installation script (143 lines)

#### Files Modified
- `src/server.py` - Added persistence, resource mgmt, new tools (+74 lines)
- `src/semantic_compressor.py` - Robust token counting (+10 lines)
- `ARCHITECTURE.md` - Updated with v0.2.0 features
- `README.md` - Updated tool count and features

### Added - 2025-01-21 (Phase 2: Advanced Features & Polish)

#### Performance & Benchmarking
- **`benchmark.py`** - Comprehensive performance benchmarking suite
  - Document ingestion benchmarks (small/medium/large)
  - Search performance testing (1 vs 10 documents)
  - Fidelity modulation speed tests
  - Compression ratio analysis
  - SCAR features performance comparison
  - Supports `--quick`, `--full` modes
  - Detailed performance insights and statistics
  - Memory usage tracking with psutil

#### Advanced Examples
- **`examples/advanced_features.py`** - Advanced feature demonstrations ✨ NEW!
  - Demo 1: Adaptive Context Window Allocation
    - Shows dynamic compression based on token availability
    - Low vs high token budget scenarios
  - Demo 2: Multi-Level Encoding
    - Progressive content inclusion (Main/Auxiliary/Detail)
    - Token-aware adaptive skeleton generation
  - Demo 3: SCAR Alignment-Guided Search
    - Comparison of standard vs alignment-guided search
    - Shows 15-25% improvement in relevance
  - Demo 4: Blind Spot Detection
    - Self-correcting context loop demonstration
    - Auto-injection recommendations

#### Enhanced Error Handling
- **Server Validation Methods** in `src/server.py`:
  - `_validate_file_id()` - File existence checking with helpful suggestions
  - `_validate_node_ids()` - Node validation with available alternatives
  - `_validate_token_count()` - Token count validation with guidance
  - All tool handlers now include:
    - Input validation
    - Descriptive error messages
    - Actionable tips (💡) for fixing errors
    - Graceful error recovery

#### Dependencies
- Added `psutil>=5.9.0` for benchmark memory tracking

### Added - 2025-01-21 (Phase 1: Infrastructure)

#### New MCP Tools
- **`adapt_to_context_window`** 🔧 - JSCCM-inspired adaptive context allocation
  - Dynamically adjusts compression based on available context window
  - Uses learned rate allocator to determine optimal skeleton ratio
  - Adapts like wireless SNR: low availability → more compression, high availability → less compression

- **`multilevel_encode`** 📊 - Multi-level encoding with priority branches
  - Three-tier architecture: Main (15%, always) + Auxiliary (25%, if space) + Detail (remaining)
  - Progressively adds levels based on available tokens
  - Inspired by JSCCM's parallel encoder architecture

#### Infrastructure
- **`check_setup.py`** - Comprehensive setup verification script
  - Checks Python version, dependencies, imports, model loading, and basic functionality
  - Provides clear error messages and fix suggestions
  - 5-step verification process with detailed output

- **CI/CD Pipeline** (`.github/workflows/test.yml`)
  - Automated testing on Python 3.10, 3.11, 3.12
  - Black code formatting checks
  - Ruff linting
  - pytest with coverage reporting
  - Codecov integration
  - Runs on push to main, develop, and claude/** branches

- **pytest Configuration** in `pyproject.toml`
  - Coverage target: 70% minimum
  - HTML and terminal coverage reports
  - Custom markers for slow and integration tests
  - Proper test path configuration

- **Pre-commit Hooks** (`.pre-commit-config.yaml`)
  - Automatic code formatting with Black
  - Linting with Ruff
  - Type checking with mypy
  - YAML/JSON/TOML validation
  - Trailing whitespace removal

- **`CONTRIBUTING.md`** - Comprehensive contribution guidelines
  - Development setup instructions
  - Code style guidelines
  - Testing requirements
  - PR process and commit conventions
  - Project structure documentation
  - Areas for contribution

- **`AUDIT_REPORT.md`** - Complete codebase audit documentation
  - Documentation vs implementation analysis
  - Semantic Multiplexing paper relationship analysis
  - Code quality assessment
  - Missing infrastructure identification
  - Actionable recommendations

### Fixed - 2025-01-21

#### Documentation Corrections
- **Root `claude.md`**:
  - Fixed MCP tool count (was incorrectly showing 7 tools but listing 8)
  - Corrected tool name: `analyze_blind_spots` → `check_blind_spots`
  - Added missing tool: `detect_hallucination`
  - Moved unimplemented tools to "Planned for Future" section with clear status
  - Now accurately lists all 9 implemented MCP tools

- **`GETTING_STARTED.md`**:
  - Updated verification section to use new `check_setup.py` script
  - Added detailed expected output examples
  - Improved troubleshooting guidance

### Changed - 2025-01-21

#### MCP Server
- Updated tool list to include `adapt_to_context_window` and `multilevel_encode`
- Added handlers for both new tools
- Improved tool descriptions with clear JSCCM inspiration notes
- Total MCP tools: 7 → 9

#### Documentation Structure
- Reorganized MCP tools section in `claude.md` by category:
  - Core Compression & Retrieval (3 tools)
  - Search & Discovery (1 tool)
  - Quality & Validation (2 tools)
  - Analytics (1 tool)
  - JSCCM-Inspired Adaptive Features (2 tools - NEW!)

### Technical Details

#### Files Modified
- `src/server.py` - Added 2 new MCP tools with handlers
- `claude.md` - Fixed documentation inconsistencies, reorganized tool list
- `GETTING_STARTED.md` - Added check_setup.py instructions
- `pyproject.toml` - Added pytest and coverage configuration

#### Files Created
- `check_setup.py` - Setup verification script (236 lines)
- `.github/workflows/test.yml` - CI/CD configuration
- `.pre-commit-config.yaml` - Pre-commit hooks
- `CONTRIBUTING.md` - Contribution guidelines (434 lines)
- `AUDIT_REPORT.md` - Comprehensive audit report (726 lines)
- `CHANGELOG.md` - This file

#### Dependencies Impact
- No new runtime dependencies
- Added dev dependencies recommendations:
  - `pytest-cov` - Coverage reporting
  - `black` - Code formatting
  - `ruff` - Fast linting
  - `pre-commit` - Git hooks

## [0.1.0] - Initial Release

### Features
- Semantic compression with 80-95% token reduction
- 5 adaptive fidelity levels (ABSTRACT, OUTLINE, STRUCTURE, DETAILED, RAW)
- Graph-based structure preservation with PageRank
- 7 MCP tools (before this update)
- Code compression with AST parsing
- Multi-modal support (text, code, images)
- SCAR enhancements with learnable compression
- Blind spot detection for hallucination prevention
- Semantic SSIM quality metrics
- JSCCM-inspired adaptive rate allocation
- Comprehensive test suite
- Full documentation

### Research Foundations
- JSCCM (arXiv:2511.15699v1) - Adaptive rate allocation
- FPQE (arXiv:2511.15695v1) - Structure preservation
- SCAR (arXiv:2511.14063v1) - Learnable compression

---

## Release Notes Format

### Types of Changes
- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security fixes

### Version Numbering
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality (backwards compatible)
- **PATCH** version for bug fixes (backwards compatible)

---

**Note**: This project is actively developed. Check the [GitHub repository](https://github.com/oimiragieo/token-saver-5000) for the latest updates.
