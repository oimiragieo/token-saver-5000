# Contributing to Token Saver 5000

Thank you for your interest in contributing to Token Saver 5000! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Documentation](#documentation)

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- pip or uv for package management

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/token-saver-5000.git
   cd token-saver-5000
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/oimiragieo/token-saver-5000.git
   ```

## Development Setup

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov pytest-asyncio black ruff pre-commit
```

### 3. Verify Setup

```bash
python scripts/check_setup.py
```

This script checks:
- Python version
- All dependencies installed
- Module imports working
- Embedding model loads correctly
- Basic functionality works

Expected output: `🎉 All checks passed!`

## Code Style

### Python Code Formatting

We use **Black** for code formatting with a line length of 100 characters.

```bash
# Format all code
black src/ tests/ examples/

# Check formatting without modifying
black --check src/ tests/ examples/
```

### Linting

We use **Ruff** for fast Python linting.

```bash
# Run linter
ruff check src/ tests/ examples/

# Fix auto-fixable issues
ruff check --fix src/ tests/ examples/
```

### Code Style Guidelines

1. **Line Length**: Maximum 100 characters
2. **Type Hints**: Use type hints for function signatures
3. **Docstrings**: All public functions/classes should have docstrings
4. **Naming Conventions**:
   - Functions/variables: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_SNAKE_CASE`
   - Private methods: `_leading_underscore`

### Example Code

```python
from typing import List, Optional

def process_document(text: str, file_id: str, metadata: Optional[dict] = None) -> dict:
    """
    Process a document into a semantic graph.

    Args:
        text: Raw document text
        file_id: Unique identifier for the document
        metadata: Optional metadata dictionary

    Returns:
        Dictionary containing processing results

    Example:
        >>> result = process_document("Sample text", "doc1")
        >>> print(result["compression_ratio"])
        15.2
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_functional.py -v

# Run specific test class
pytest tests/test_functional.py::TestBasicFunctionality -v

# Run tests matching a pattern
pytest -k "test_compression" -v
```

### Writing Tests

1. Place tests in the `tests/` directory
2. Test files should match `test_*.py`
3. Test functions should start with `test_`
4. Use descriptive test names:
   - ✅ `test_compression_ratio_exceeds_80_percent`
   - ❌ `test_compression`

### Test Structure

```python
import pytest
from src.semantic_compressor import SemanticCompressor

class TestDocumentIngestion:
    """Tests for document ingestion functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.compressor = SemanticCompressor()

    def test_basic_ingestion(self):
        """Test that documents can be ingested successfully"""
        text = "Sample document text"
        result = self.compressor.ingest_file(text, "test_doc")

        assert result.total_nodes > 0
        assert result.compression_ratio > 1.0

    def test_compression_ratio(self):
        """Test that compression ratio meets targets"""
        text = "A" * 1000  # 1000 character document
        result = self.compressor.ingest_file(text, "test_doc")

        assert result.compression_ratio >= 5.0  # At least 5x compression
```

### Coverage Requirements

- Target: **70% minimum coverage** (enforced in CI/CD)
- New code should maintain or increase coverage
- Focus on testing core functionality
- Edge cases should have dedicated tests

## Submitting Changes

### Branch Naming Convention

- Feature: `feature/descriptive-name`
- Bug fix: `fix/issue-description`
- Documentation: `docs/what-changed`
- Refactor: `refactor/component-name`

### Commit Message Format

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(compressor): add support for PDF documents

Implement PDF text extraction and integration with semantic compressor.
Adds new dependency on pypdf2.

Closes #42
```

```
fix(server): correct MCP tool name mismatch

Changed 'analyze_blind_spots' to 'check_blind_spots' to match implementation.
Updated documentation accordingly.
```

### Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make Your Changes**
   - Write code
   - Add/update tests
   - Update documentation

3. **Verify Everything Works**
   ```bash
   # Run tests
   pytest tests/ -v

   # Check formatting
   black --check src/ tests/ examples/

   # Run linter
   ruff check src/ tests/ examples/

   # Verify setup
   python scripts/check_setup.py
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin feature/my-new-feature
   ```

6. **Create Pull Request**
   - Go to GitHub and create a PR from your fork
   - Fill out the PR template
   - Link any related issues

### PR Requirements

- ✅ All tests pass
- ✅ Code is formatted with Black
- ✅ No linting errors
- ✅ Coverage maintained or increased
- ✅ Documentation updated
- ✅ CHANGELOG.md updated (if applicable)

## Documentation

### Code Documentation

- **Docstrings**: All public functions and classes
- **Inline comments**: For complex logic
- **Type hints**: All function signatures

### Project Documentation

When making changes, update relevant documentation:

- **README.md**: User-facing changes, new features
- **claude.md**: Codebase structure changes
- **ARCHITECTURE.md**: Architectural decisions
- **docs/**: Research paper implementations
- **examples/**: New feature demonstrations

### Documentation Style

- Use clear, concise language
- Include code examples where helpful
- Update table of contents when adding sections
- Keep line length reasonable (prefer 80-100 chars)

## Project Structure

```
token-saver-5000/
├── src/                    # Core source code
│   ├── semantic_compressor.py      # Base semantic compression
│   ├── code_compressor.py          # Code-specific compression
│   ├── multimodal_compressor.py    # Multi-modal support
│   ├── scar_compressor.py          # SCAR enhancements
│   ├── adaptive_rate_allocator.py  # Adaptive rate allocation
│   ├── blind_spot_detector.py      # Hallucination detection
│   ├── semantic_ssim.py            # Quality metrics
│   ├── training_utils.py           # Training utilities
│   └── server.py                   # MCP server
├── tests/                  # Test suite
│   ├── test_functional.py          # Feature tests
│   └── test_token_savings.py       # Benchmark tests
├── examples/               # Usage examples
│   ├── example_usage.py            # Basic usage
│   ├── scar_demo.py                # SCAR features
│   ├── code_compression_example.py # Code compression
│   └── multimodal_example.py       # Multi-modal demo
├── docs/                   # Research documentation
├── config/                 # Configuration templates
└── .github/workflows/      # CI/CD configuration
```

## Areas for Contribution

We welcome contributions in these areas:

### 🔥 High Priority

1. **Better Chunking Strategies**
   - Improve semantic boundary detection
   - Language-specific chunking
   - Adaptive chunk sizing

2. **Alternative Embedding Models**
   - Support for domain-specific models
   - Multilingual embeddings
   - Fine-tuned models for specific use cases

3. **Enhanced Entity Extraction**
   - Better NER integration
   - Relationship extraction
   - Knowledge graph construction

### 🌟 Medium Priority

4. **Persistent Storage**
   - ChromaDB integration
   - FAISS vector index
   - Incremental updates

5. **Cross-Document Analysis**
   - Multi-document reasoning
   - Citation tracking
   - Contradiction detection

6. **Web UI**
   - Visualization of semantic graphs
   - Interactive fidelity control
   - Real-time compression statistics

### 💡 Nice to Have

7. **Performance Optimization**
   - Batch processing
   - GPU acceleration
   - Caching strategies

8. **Additional Modalities**
   - Audio transcription
   - Video frame analysis
   - LaTeX/equations

9. **Integration Examples**
   - Jupyter notebook extension
   - VS Code extension
   - API server mode

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/oimiragieo/token-saver-5000/issues)
- **Discussions**: [GitHub Discussions](https://github.com/oimiragieo/token-saver-5000/discussions)
- **Documentation**: See README.md and docs/

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Token Saver 5000! 🚀
