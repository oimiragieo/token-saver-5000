# tests/ Directory

## Overview
Comprehensive test suite validating token reduction (80-95% target), functional correctness, blind spot detection, multi-modal support, and SCAR implementation.

## Files

### 1. **`__init__.py`** (388 bytes)
**Purpose**: Test suite documentation and setup instructions

Contains:
- Package marker for pytest discovery
- Instructions for running tests
- Test suite overview

---

### 2. **`test_functional.py`** (18,436 bytes)
**Purpose**: Functional tests for all core features

**Test Coverage**:

#### TestBasicFunctionality
- `test_document_ingestion`: Document → semantic graph conversion
  - Validates SkeletonResponse fields
  - Checks compression ratio > 1.0
  - Verifies node creation

- `test_skeleton_generation`: Compressed skeleton view
  - Checks for ANCHOR markers
  - Validates skeleton structure
  - Measures character reduction

- `test_semantic_graph_creation`: NetworkX graph construction
  - Validates nodes and edges
  - Checks PageRank scores

- `test_importance_scores`: PageRank calculation
  - Ensures importance values in [0, 1]
  - Validates distribution across nodes

#### TestFidelityModulation
- `test_abstract_fidelity`: ABSTRACT level (~10 tokens/node)
- `test_outline_fidelity`: OUTLINE level (~30 tokens/node)
- `test_structure_fidelity`: STRUCTURE level (~50 tokens/node)
- `test_detailed_fidelity`: DETAILED level (~100 tokens/node)
- `test_raw_fidelity`: RAW level (full content)
- `test_progressive_token_increase`: Validates token scaling across levels

#### TestSemanticSearch
- `test_semantic_search_basic`: Vector similarity search
- `test_semantic_search_relevance`: Top results are relevant
- `test_semantic_search_empty_query`: Edge case handling

#### TestBlindSpotDetection
- `test_blind_spot_detector_initialization`: Setup validation
- `test_detect_critical_blind_spots`: Finds missed critical content
- `test_no_blind_spots_when_complete`: Validates complete retrieval
- `test_blind_spot_urgency_calculation`: Urgency = similarity × importance
- `test_auto_injection_recommendations`: Auto-inject suggestions

#### TestSCAREnhancements
- `test_scar_initialization`: SCAR modules load correctly
- `test_learnable_compression`: 4× embedding compression
- `test_alignment_guided_search`: Semantic alignment improves search
- `test_adaptive_modulation`: Fidelity adapts to alignment score

#### TestCodeCompression
- `test_python_code_ingestion`: AST parsing
- `test_code_skeleton_generation`: Signature extraction
- `test_code_semantic_search`: Code-specific search

#### TestMultimodalCompression
- `test_multimodal_ingestion`: Text + code + images
- `test_cross_modal_search`: Cross-modality retrieval
- `test_multimodal_summary`: Project overview

**Sample Document**:
```python
SAMPLE_DOCUMENT = """
Quantum Error Correction Overview

Quantum computers require error correction to be practical. Surface codes are
a leading approach, offering approximately 1% error thresholds.

Error Detection Mechanisms

Stabilizer measurements detect errors without collapsing quantum states.
Syndrome extraction identifies error locations through repeated measurements.

Implementation Challenges

The main challenges include high qubit overhead and fast syndrome extraction.
Decoherence times limit how quickly measurements must occur.

Recent Progress

Lattice surgery techniques enable logical gate operations.
Code concatenation methods reduce the required physical qubit count.
"""
```

**Run Command**:
```bash
pytest tests/test_functional.py -v
```

---

### 3. **`test_token_savings.py`** (22,692 bytes)
**Purpose**: Benchmark tests proving 80-95% token reduction

**Test Coverage**:

#### TestTokenReductionTargets
- `test_small_document_compression`: Small doc (< 100 tokens)
  - Target: > 60% reduction

- `test_medium_document_compression`: Medium doc (100-500 tokens)
  - Target: > 70% reduction

- `test_large_document_compression`: Large doc (> 500 tokens)
  - Target: > 80% reduction

- `test_compression_consistency`: Multiple runs produce consistent results

#### TestFidelityTokenBudgets
- `test_abstract_token_budget`: ABSTRACT ~10 tokens/node
- `test_outline_token_budget`: OUTLINE ~30 tokens/node
- `test_structure_token_budget`: STRUCTURE ~50 tokens/node
- `test_detailed_token_budget`: DETAILED ~100 tokens/node
- `test_raw_maintains_fidelity`: RAW preserves original content

#### TestCompressionQuality
- `test_semantic_ssim_scores`: SSIM > 0.7 (good preservation)
- `test_important_content_preserved`: High-importance nodes in skeleton
- `test_low_importance_hidden`: Low-importance nodes compressed
- `test_entity_extraction`: Key entities extracted correctly

#### TestSCARCompressionBenchmarks
- `test_scar_embedding_compression`: 4× compression (384D → 96D)
- `test_scar_reconstruction_quality`: Low reconstruction loss
- `test_scar_alignment_improvement`: Alignment improves retrieval

#### TestRealWorldScenarios
- `test_research_paper_compression`: Academic paper (5000+ tokens)
  - Target: > 85% reduction
  - Validates structure preservation

- `test_code_documentation_compression`: Code + docs
  - Target: > 80% reduction
  - Validates function signature extraction

- `test_multimodal_project_compression`: Text + code + images
  - Target: > 75% reduction (images count as heavy)
  - Validates cross-modal connections

#### TestCompressionRatioReporting
- `test_compression_ratio_calculation`: Accurate ratio reporting
- `test_token_savings_percentage`: % saved matches ratio
- `test_compression_stats_accuracy`: Stats match actual tokens

**Sample Documents**:

1. **SMALL_DOCUMENT** (~50 tokens): Quantum computing basics
2. **MEDIUM_DOCUMENT** (~200 tokens): Quantum error correction intro
3. **LARGE_DOCUMENT** (~800 tokens): Comprehensive QEC review

**Run Command**:
```bash
pytest tests/test_token_savings.py -v
python tests/test_token_savings.py  # Detailed benchmarks
```

**Expected Output**:
```
SMALL_DOCUMENT:
  Original: 48 tokens → Skeleton: 15 tokens
  Reduction: 68.8% ✅

MEDIUM_DOCUMENT:
  Original: 183 tokens → Skeleton: 42 tokens
  Reduction: 77.0% ✅

LARGE_DOCUMENT:
  Original: 827 tokens → Skeleton: 89 tokens
  Reduction: 89.2% ✅

Overall: 80-95% token reduction achieved ✅
```

---

## Running All Tests

### Quick Test
```bash
pytest tests/ -v
```

### With Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Specific Test Class
```bash
pytest tests/test_functional.py::TestBlindSpotDetection -v
```

### Verbose Benchmark
```bash
python tests/test_token_savings.py
```

---

## Test Dependencies

**Required**:
- `pytest`: Test framework
- `pytest-asyncio`: Async test support
- `numpy`: Numerical assertions

**Installed via**:
```bash
pip install pytest pytest-asyncio pytest-cov
```

---

## Test Success Criteria

### Functional Tests
✅ All features work without errors
✅ Edge cases handled gracefully
✅ Blind spot detection catches critical misses
✅ SCAR enhancements improve quality

### Token Savings Tests
✅ Small docs: > 60% reduction
✅ Medium docs: > 70% reduction
✅ Large docs: > 80% reduction
✅ Overall target: 80-95% reduction
✅ SSIM scores: > 0.7 (good quality)

---

## Test Data Philosophy

- **Quantum computing domain**: Chosen for rich technical content with structure
- **Varying document sizes**: Tests scalability
- **Real-world scenarios**: Research papers, code + docs, multimodal projects
- **Ground truth validation**: Known entities, importance ranking

---

## Continuous Integration

Tests should be run:
1. Before committing changes
2. In CI/CD pipeline
3. Before releases
4. After dependency updates

Target: **100% test pass rate** in main branch
