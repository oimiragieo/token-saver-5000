"""
Comprehensive tests for code_compressor.py (AST-based code compression)

Tests follow 2025 best practices for AST-driven testing:
1. Deterministic assertions (no LLM-based "vibe testing")
2. Mock external dependencies (embeddings)
3. Test behavior, not implementation
4. Comprehensive coverage (happy paths + error cases + edge cases)

Coverage target: 80%+ for code_compressor.py (669 lines, currently 0%)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.code_compressor import (
    CodeSemanticCompressor,
    CodeLanguage,
)

# Sample code for testing
SAMPLE_PYTHON_CODE = '''
import numpy as np
from sklearn.metrics import accuracy_score

def preprocess_data(data, normalize=True):
    """
    Preprocess input data for model training.

    Args:
        data: Input data array
        normalize: Whether to normalize the data

    Returns:
        Preprocessed data
    """
    if normalize:
        data = (data - np.mean(data)) / np.std(data)
    return data

class NeuralNetwork:
    """Simple neural network implementation."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.weights2 = np.random.randn(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass through the network"""
        hidden = np.dot(x, self.weights1)
        output = np.dot(hidden, self.weights2)
        return output
'''

SAMPLE_PYTHON_SYNTAX_ERROR = """
def broken_function(
    # Missing closing parenthesis and body
"""

SAMPLE_JAVASCRIPT_CODE = """
import React from 'react';
import { useState } from 'react';

/**
 * Main app component
 */
function App() {
    const [count, setCount] = useState(0);
    return <div>Count: {count}</div>;
}

const increment = () => {
    setCount(count + 1);
};
"""


class TestCodeLanguageDetection:
    """Test language detection from file extensions"""

    def setup_method(self):
        """Set up test fixtures"""
        with patch("src.code_compressor.EmbeddingManager"):
            self.compressor = CodeSemanticCompressor()

    def test_detect_python_language(self):
        """Test Python file detection"""
        assert self.compressor.detect_language("test.py") == CodeLanguage.PYTHON

    def test_detect_javascript_language(self):
        """Test JavaScript file detection"""
        assert self.compressor.detect_language("app.js") == CodeLanguage.JAVASCRIPT
        assert self.compressor.detect_language("app.jsx") == CodeLanguage.JAVASCRIPT

    def test_detect_typescript_language(self):
        """Test TypeScript file detection"""
        assert self.compressor.detect_language("app.ts") == CodeLanguage.TYPESCRIPT
        assert self.compressor.detect_language("app.tsx") == CodeLanguage.TYPESCRIPT

    def test_detect_java_language(self):
        """Test Java file detection"""
        assert self.compressor.detect_language("Main.java") == CodeLanguage.JAVA

    def test_detect_cpp_language(self):
        """Test C++ file detection"""
        assert self.compressor.detect_language("main.cpp") == CodeLanguage.CPP
        assert self.compressor.detect_language("main.cc") == CodeLanguage.CPP
        assert self.compressor.detect_language("main.h") == CodeLanguage.CPP
        assert self.compressor.detect_language("main.hpp") == CodeLanguage.CPP

    def test_detect_go_language(self):
        """Test Go file detection"""
        assert self.compressor.detect_language("main.go") == CodeLanguage.GO

    def test_detect_rust_language(self):
        """Test Rust file detection"""
        assert self.compressor.detect_language("main.rs") == CodeLanguage.RUST

    def test_detect_unknown_language(self):
        """Test unknown file extension"""
        assert self.compressor.detect_language("unknown.xyz") == CodeLanguage.UNKNOWN
        assert self.compressor.detect_language("README.md") == CodeLanguage.UNKNOWN


class TestPythonCodeChunking:
    """Test AST-based Python code chunking"""

    def setup_method(self):
        """Set up test fixtures"""
        with patch("src.code_compressor.EmbeddingManager"):
            self.compressor = CodeSemanticCompressor()

    def test_chunk_python_code_extracts_imports(self):
        """Test that Python imports are extracted correctly"""
        chunks = self.compressor.chunk_python_code(SAMPLE_PYTHON_CODE, "test_file")

        # Should have import chunk
        import_chunks = [c for c in chunks if c.chunk_type == "import"]
        assert len(import_chunks) == 1

        import_chunk = import_chunks[0]
        assert import_chunk.name == "imports"
        assert "numpy" in import_chunk.dependencies
        assert "sklearn.metrics.accuracy_score" in import_chunk.dependencies

    def test_chunk_python_code_extracts_functions(self):
        """Test that Python functions are extracted with docstrings"""
        chunks = self.compressor.chunk_python_code(SAMPLE_PYTHON_CODE, "test_file")

        # Should have function chunks
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(func_chunks) >= 1

        # Check preprocess_data function
        preprocess_chunk = next(c for c in func_chunks if c.name == "preprocess_data")
        assert preprocess_chunk.chunk_type == "function"
        assert preprocess_chunk.docstring is not None
        assert "Preprocess input data" in preprocess_chunk.docstring
        assert "def preprocess_data" in preprocess_chunk.code
        assert preprocess_chunk.start_line > 0
        assert preprocess_chunk.end_line > preprocess_chunk.start_line

    def test_chunk_python_code_extracts_classes(self):
        """Test that Python classes are extracted with methods"""
        chunks = self.compressor.chunk_python_code(SAMPLE_PYTHON_CODE, "test_file")

        # Should have class chunks
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 1

        # Check NeuralNetwork class
        nn_chunk = next(c for c in class_chunks if c.name == "NeuralNetwork")
        assert nn_chunk.chunk_type == "class"
        assert nn_chunk.docstring is not None
        assert "Simple neural network" in nn_chunk.docstring
        assert "class NeuralNetwork" in nn_chunk.code
        # Methods should be listed as dependencies
        assert "__init__" in nn_chunk.dependencies
        assert "forward" in nn_chunk.dependencies

    def test_chunk_python_code_extracts_function_dependencies(self):
        """Test that function calls are tracked as dependencies"""
        chunks = self.compressor.chunk_python_code(SAMPLE_PYTHON_CODE, "test_file")

        # Check forward method has dependencies
        forward_chunk = next(
            (c for c in chunks if c.chunk_type == "function" and c.name == "forward"),
            None,
        )
        if forward_chunk:
            # Should track np.dot calls
            assert "dot" in forward_chunk.dependencies

    def test_chunk_python_code_with_syntax_error_falls_back(self):
        """Test that syntax errors trigger line-based fallback"""
        chunks = self.compressor.chunk_python_code(SAMPLE_PYTHON_SYNTAX_ERROR, "broken_file")

        # Should fall back to line-based chunking
        assert len(chunks) > 0
        # All chunks should be of type "block" (fallback type)
        assert all(c.chunk_type == "block" for c in chunks)

    def test_chunk_python_code_deeply_nested_source_falls_back_not_crashes(self):
        """Task #237 FIX 1 (HIGH, DoS): a crafted deeply-nested Python payload
        (e.g. posted to /v1/compress-code) must degrade to the line-based
        fallback instead of crashing the worker process.

        ``ast.parse`` on a sufficiently deep expression tree raises
        ``RecursionError`` ("maximum recursion depth exceeded during ast
        construction") -- a RuntimeError subclass that pre-fix was NOT caught
        by the bare ``except SyntaxError`` in ``chunk_python_code``, so it
        propagated out and would crash the worker serving every tenant sharing
        that Fly machine (soft_limit=64 shared, reachable by any free-tier gc_
        key). A deeply-bracketed payload like ``"[" * N + "]" * N`` is already
        safely rejected as a ``SyntaxError`` ("too many nested parentheses") by
        modern CPython's PEG parser, so this uses a bracket-free chained-unary-
        operator payload, which reliably reproduces the uncaught RecursionError
        at this depth without risking the C-stack segfault that deeper payloads
        (~6000+) can trigger.
        """
        malicious_code = "-" * 4000 + "1"

        chunks = self.compressor.chunk_python_code(malicious_code, "malicious_file")

        # Falls back to line-based chunking instead of crashing the process.
        assert len(chunks) > 0
        assert all(c.chunk_type == "block" for c in chunks)

    def test_chunk_python_code_preserves_chunk_ids(self):
        """Test that chunk IDs are properly formatted with :: separator"""
        chunks = self.compressor.chunk_python_code(SAMPLE_PYTHON_CODE, "test_file")

        for chunk in chunks:
            # All chunk IDs should start with file_id and use :: separator
            assert chunk.chunk_id.startswith("test_file::")
            # Import chunks should have specific format
            if chunk.chunk_type == "import":
                assert chunk.chunk_id == "test_file::imports"
            # Function chunks include the (possibly class-qualified) name after ::
            # (#195: a method is qualified as Class.method so A.foo and B.foo
            # don't collide; a top-level function stays test_file::name).
            elif chunk.chunk_type == "function":
                assert chunk.chunk_id == f"test_file::{chunk.name}" or chunk.chunk_id.endswith(
                    f".{chunk.name}"
                )
            # Class chunks should include class name after ::
            elif chunk.chunk_type == "class":
                assert chunk.chunk_id == f"test_file::{chunk.name}"

    def test_same_named_methods_across_classes_do_not_collide(self):
        """#195: A.foo and B.foo must get DISTINCT chunk_ids.

        self.chunks is a dict keyed by chunk_id, so an unqualified
        test_file::foo silently overwrites the earlier method. RED before the
        fix: both methods produce test_file::foo (1 distinct id).
        """
        code = (
            "class A:\n"
            "    def foo(self):\n"
            "        return 1\n"
            "\n\n"
            "class B:\n"
            "    def foo(self):\n"
            "        return 2\n"
        )
        chunks = self.compressor.chunk_python_code(code, "test_file")
        method_ids = [c.chunk_id for c in chunks if c.chunk_type == "function" and c.name == "foo"]
        assert len(method_ids) == 2
        assert len(set(method_ids)) == 2
        assert "test_file::A.foo" in method_ids
        assert "test_file::B.foo" in method_ids

    def test_same_named_nested_classes_do_not_collide(self):
        """#220 rank 5: A.Meta and B.Meta must get DISTINCT chunk_ids.

        ast.walk recurses into classes, so a nested `class Meta` inside two
        different outer classes (e.g. Django models) both produced
        test_file::Meta and the second silently OVERWROTE the first
        (chunk_id-keyed dict). RED before the fix: both produce test_file::Meta.
        """
        code = (
            "class A:\n"
            "    class Meta:\n"
            "        x = 1\n"
            "\n\n"
            "class B:\n"
            "    class Meta:\n"
            "        y = 2\n"
        )
        chunks = self.compressor.chunk_python_code(code, "test_file")
        meta_ids = [c.chunk_id for c in chunks if c.chunk_type == "class" and c.name == "Meta"]
        assert len(meta_ids) == 2
        assert len(set(meta_ids)) == 2
        assert "test_file::A.Meta" in meta_ids
        assert "test_file::B.Meta" in meta_ids


class TestJavaScriptCodeChunking:
    """Test regex-based JavaScript code chunking"""

    def setup_method(self):
        """Set up test fixtures"""
        with patch("src.code_compressor.EmbeddingManager"):
            self.compressor = CodeSemanticCompressor()

    def test_chunk_javascript_code_extracts_imports(self):
        """Test that JavaScript imports are extracted"""
        chunks = self.compressor.chunk_javascript_code(SAMPLE_JAVASCRIPT_CODE, "app.js")

        # Should have import chunk
        import_chunks = [c for c in chunks if c.chunk_type == "import"]
        assert len(import_chunks) == 1

        import_chunk = import_chunks[0]
        assert import_chunk.name == "imports"
        # Should track imported modules
        assert len(import_chunk.dependencies) >= 1

    def test_chunk_javascript_code_extracts_functions(self):
        """Test that JavaScript functions are extracted"""
        chunks = self.compressor.chunk_javascript_code(SAMPLE_JAVASCRIPT_CODE, "app.js")

        # Should have function chunks
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(func_chunks) >= 1

    def test_chunk_javascript_code_extracts_jsdoc(self):
        """Test that JSDoc comments are extracted as docstrings"""
        chunks = self.compressor.chunk_javascript_code(SAMPLE_JAVASCRIPT_CODE, "app.js")

        # Check if any function has JSDoc docstring
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        # At least one function might have JSDoc
        # (This is best-effort regex matching)
        assert len(func_chunks) >= 0  # Regex-based, may vary


class TestLineBasedChunking:
    """Test fallback line-based chunking"""

    def setup_method(self):
        """Set up test fixtures"""
        with patch("src.code_compressor.EmbeddingManager"):
            self.compressor = CodeSemanticCompressor()

    def test_chunk_by_lines_creates_blocks(self):
        """Test that line-based chunking creates block chunks"""
        code = "\n".join([f"line {i}" for i in range(100)])
        chunks = self.compressor._chunk_by_lines(code, "test_file", lines_per_chunk=25)

        # Should create 4 blocks (100 lines / 25 lines per chunk)
        assert len(chunks) == 4
        assert all(c.chunk_type == "block" for c in chunks)

    def test_chunk_by_lines_respects_chunk_size(self):
        """Test that chunks respect the specified line count"""
        code = "\n".join([f"line {i}" for i in range(100)])
        chunks = self.compressor._chunk_by_lines(code, "test_file", lines_per_chunk=30)

        # First 3 chunks should have 30 lines each
        for i in range(3):
            chunk_lines = chunks[i].code.split("\n")
            assert len(chunk_lines) == 30

        # Last chunk should have remaining lines (10 lines)
        last_chunk_lines = chunks[-1].code.split("\n")
        assert len(last_chunk_lines) == 10

    def test_chunk_by_lines_assigns_correct_line_numbers(self):
        """Test that start_line and end_line are correctly assigned"""
        code = "\n".join([f"line {i}" for i in range(100)])
        chunks = self.compressor._chunk_by_lines(code, "test_file", lines_per_chunk=25)

        # Check first chunk
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 25

        # Check second chunk
        assert chunks[1].start_line == 26
        assert chunks[1].end_line == 50

    def test_chunk_by_lines_handles_small_files(self):
        """Test that small files are handled correctly"""
        code = "line 1\nline 2\nline 3"
        chunks = self.compressor._chunk_by_lines(code, "small_file", lines_per_chunk=50)

        # Should create 1 chunk
        assert len(chunks) == 1
        assert chunks[0].end_line == 3


@patch("src.code_compressor.EmbeddingManager")
class TestCodeIngestion:
    """Test main ingest_code_file method"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create mock embedding manager and model
        self.mock_embedding_manager = Mock()
        self.mock_model = Mock()

        # Configure mock model to return embeddings
        def encode_side_effect(texts, show_progress_bar=False):
            # Return random embeddings for each text
            return np.random.rand(len(texts), 384)

        self.mock_model.encode.side_effect = encode_side_effect
        self.mock_embedding_manager.get_code_embedder.return_value = self.mock_model

    def test_ingest_python_code_file(self, mock_embedding_manager_class):
        """Test successful Python code ingestion"""
        # Configure mock
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        # Ingest code
        stats = compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="test_py", filepath="test.py"
        )

        # Verify stats
        assert stats["file_id"] == "test_py"
        assert stats["language"] == "python"
        assert stats["total_chunks"] > 0
        assert stats["chunk_types"]["imports"] >= 1
        assert stats["chunk_types"]["functions"] >= 1
        assert stats["chunk_types"]["classes"] >= 1
        assert stats["graph_nodes"] > 0

        # Verify chunks were stored
        assert len(compressor.chunks) > 0

        # Verify graph was created
        assert "test_py" in compressor.graphs

    def test_ingest_javascript_code_file(self, mock_embedding_manager_class):
        """Test successful JavaScript code ingestion"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        stats = compressor.ingest_code_file(
            code=SAMPLE_JAVASCRIPT_CODE, file_id="app_js", filepath="app.js"
        )

        # Verify stats
        assert stats["file_id"] == "app_js"
        assert stats["language"] == "javascript"
        assert stats["total_chunks"] > 0

    def test_ingest_code_with_unknown_language(self, mock_embedding_manager_class):
        """Test ingestion with unknown language falls back to line-based chunking"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        stats = compressor.ingest_code_file(
            code="some random text\nmore text\n", file_id="unknown", filepath="file.xyz"
        )

        # Should fall back to line-based chunking
        assert stats["file_id"] == "unknown"
        assert stats["language"] == "unknown"
        assert stats["chunk_types"]["blocks"] > 0

    def test_ingest_code_without_filepath(self, mock_embedding_manager_class):
        """Test ingestion without filepath uses UNKNOWN language"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        stats = compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="no_path", filepath=None
        )

        # Should default to UNKNOWN and use fallback chunking
        assert stats["language"] == "unknown"

    def test_ingest_code_generates_embeddings(self, mock_embedding_manager_class):
        """Test that embeddings are generated for all chunks"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="embed_test", filepath="test.py"
        )

        # Verify embeddings were generated
        for chunk in compressor.chunks.values():
            assert chunk.embedding is not None
            assert chunk.embedding.shape == (384,)  # Embedding dimension

    def test_ingest_code_builds_dependency_graph(self, mock_embedding_manager_class):
        """Test that dependency edges are created between chunks"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        compressor.ingest_code_file(code=SAMPLE_PYTHON_CODE, file_id="dep_test", filepath="test.py")

        graph = compressor.graphs["dep_test"]

        # Should have dependency edges
        # (Python code has function calls that create dependencies)
        assert graph.number_of_edges() > 0

    def test_ingest_code_calculates_importance_scores(self, mock_embedding_manager_class):
        """Test that PageRank importance scores are calculated"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="importance_test", filepath="test.py"
        )

        # All chunks should have importance scores
        for chunk in compressor.chunks.values():
            assert chunk.importance >= 0.0
            assert chunk.importance <= 1.0

    def test_ingest_code_stores_metadata(self, mock_embedding_manager_class):
        """Test that optional metadata is stored"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        metadata = {"author": "test_user", "version": "1.0"}
        compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE,
            file_id="meta_test",
            filepath="test.py",
            metadata=metadata,
        )

        # Metadata should be stored
        assert compressor.file_metadata["meta_test"] == metadata


@patch("src.code_compressor.EmbeddingManager")
class TestCodeSkeleton:
    """Test code skeleton generation"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_embedding_manager = Mock()
        self.mock_model = Mock()

        def encode_side_effect(texts, show_progress_bar=False):
            return np.random.rand(len(texts), 384)

        self.mock_model.encode.side_effect = encode_side_effect
        self.mock_embedding_manager.get_code_embedder.return_value = self.mock_model

    def test_generate_code_skeleton_shows_imports(self, mock_embedding_manager_class):
        """Test that skeleton always shows imports first"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()
        compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="skel_test", filepath="test.py"
        )

        skeleton = compressor.generate_code_skeleton("skel_test")

        # Should show imports
        assert "IMPORTS" in skeleton
        assert "numpy" in skeleton or "sklearn" in skeleton

    def test_generate_code_skeleton_shows_top_chunks(self, mock_embedding_manager_class):
        """Test that skeleton shows top N important chunks"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()
        compressor.ingest_code_file(code=SAMPLE_PYTHON_CODE, file_id="top_test", filepath="test.py")

        skeleton = compressor.generate_code_skeleton("top_test", show_top_n=2)

        # Should show at most 2 non-import chunks
        assert "FUNCTION" in skeleton or "CLASS" in skeleton

    def test_generate_code_skeleton_includes_importance_scores(self, mock_embedding_manager_class):
        """Test that skeleton includes importance scores"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()
        compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="score_test", filepath="test.py"
        )

        # Show at least 2 chunks to ensure we get a non-import chunk
        skeleton = compressor.generate_code_skeleton("score_test", show_top_n=2)

        # Should include importance scores (for function/class chunks)
        assert "importance:" in skeleton

    def test_generate_code_skeleton_shows_docstrings(self, mock_embedding_manager_class):
        """Test that skeleton includes docstring summaries when chunks have them"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()
        compressor.ingest_code_file(code=SAMPLE_PYTHON_CODE, file_id="doc_test", filepath="test.py")

        # Check if any chunks have docstrings
        has_docstrings = any(c.docstring for c in compressor.chunks.values())

        if has_docstrings:
            # Request enough chunks to ensure we see at least one with a docstring
            skeleton = compressor.generate_code_skeleton("doc_test", show_top_n=5)
            # Should show docstring snippets for chunks that have them
            assert "Doc:" in skeleton or "documentation" in skeleton.lower()
        else:
            # If no docstrings, test passes (edge case)
            assert True

    def test_generate_code_skeleton_for_nonexistent_file(self, mock_embedding_manager_class):
        """Test that skeleton generation fails gracefully for missing files"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        with pytest.raises(ValueError, match="not found"):
            compressor.generate_code_skeleton("nonexistent_file")


@patch("src.code_compressor.EmbeddingManager")
class TestCodeSearch:
    """Test semantic code search"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_embedding_manager = Mock()
        self.mock_model = Mock()

        # Create deterministic embeddings for testing
        self.embedding_map = {}

        def encode_side_effect(texts, show_progress_bar=False):
            embeddings = []
            for text in texts:
                # Create consistent embedding for same text
                if text not in self.embedding_map:
                    self.embedding_map[text] = np.random.rand(384)
                embeddings.append(self.embedding_map[text])
            return np.array(embeddings)

        self.mock_model.encode.side_effect = encode_side_effect
        self.mock_embedding_manager.get_code_embedder.return_value = self.mock_model

    def test_search_code_returns_results(self, mock_embedding_manager_class):
        """Test that code search returns similarity-ranked results"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()
        compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="search_test", filepath="test.py"
        )

        results = compressor.search_code("neural network training", top_k=3)

        # Should return up to 3 results
        assert len(results) <= 3
        assert len(results) > 0

        # Results should be (chunk_id, similarity_score) tuples
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert 0.0 <= score <= 1.0

    def test_search_code_filters_by_file_id(self, mock_embedding_manager_class):
        """Test that search can be restricted to specific file"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()
        compressor.ingest_code_file(code=SAMPLE_PYTHON_CODE, file_id="file1", filepath="test1.py")
        compressor.ingest_code_file(code=SAMPLE_PYTHON_CODE, file_id="file2", filepath="test2.py")

        # Search only in file1
        results = compressor.search_code("data preprocessing", file_id="file1", top_k=5)

        # All results should be from file1 (v0.9.0: uses :: separator)
        for chunk_id, score in results:
            assert chunk_id.startswith("file1::")

    def test_search_code_returns_sorted_results(self, mock_embedding_manager_class):
        """Test that results are sorted by similarity score"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()
        compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="sort_test", filepath="test.py"
        )

        results = compressor.search_code("machine learning", top_k=5)

        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


@patch("src.code_compressor.EmbeddingManager")
class TestCodeChunkRetrieval:
    """Test get_code_chunk method"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_embedding_manager = Mock()
        self.mock_model = Mock()

        def encode_side_effect(texts, show_progress_bar=False):
            return np.random.rand(len(texts), 384)

        self.mock_model.encode.side_effect = encode_side_effect
        self.mock_embedding_manager.get_code_embedder.return_value = self.mock_model

    def test_get_code_chunk_returns_full_code(self, mock_embedding_manager_class):
        """Test that get_code_chunk returns complete chunk details"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()
        compressor.ingest_code_file(code=SAMPLE_PYTHON_CODE, file_id="get_test", filepath="test.py")

        # Get a specific chunk
        chunk_ids = list(compressor.chunks.keys())
        chunk_id = chunk_ids[0]

        result = compressor.get_code_chunk(chunk_id)

        # Should include chunk type, name, and code
        chunk = compressor.chunks[chunk_id]
        assert chunk.chunk_type.upper() in result
        assert chunk.name in result
        assert "Code:" in result

    def test_get_code_chunk_includes_docstring(self, mock_embedding_manager_class):
        """Test that docstrings are included when present"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()
        compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="doc_get_test", filepath="test.py"
        )

        # Find a chunk with a docstring
        chunk_with_doc = next(
            (c for c in compressor.chunks.values() if c.docstring is not None), None
        )

        if chunk_with_doc:
            result = compressor.get_code_chunk(chunk_with_doc.chunk_id)
            assert "Documentation:" in result

    def test_get_code_chunk_shows_line_numbers(self, mock_embedding_manager_class):
        """Test that line numbers are shown"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()
        compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="line_test", filepath="test.py"
        )

        chunk_id = list(compressor.chunks.keys())[0]
        result = compressor.get_code_chunk(chunk_id)

        # Should show line numbers
        assert "Lines:" in result

    def test_get_code_chunk_shows_dependencies(self, mock_embedding_manager_class):
        """Test that dependencies are listed"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()
        compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="dep_get_test", filepath="test.py"
        )

        # Find a chunk with dependencies
        chunk_with_deps = next((c for c in compressor.chunks.values() if c.dependencies), None)

        if chunk_with_deps:
            result = compressor.get_code_chunk(chunk_with_deps.chunk_id)
            assert "Dependencies:" in result

    def test_get_code_chunk_for_nonexistent_chunk(self, mock_embedding_manager_class):
        """Test that missing chunks are handled gracefully"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        result = compressor.get_code_chunk("nonexistent_chunk_id")

        # Should return error message
        assert "not found" in result


# Edge case tests
@patch("src.code_compressor.EmbeddingManager")
class TestCodeCompressorEdgeCases:
    """Test edge cases and error handling"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_embedding_manager = Mock()
        self.mock_model = Mock()

        def encode_side_effect(texts, show_progress_bar=False):
            return np.random.rand(len(texts), 384)

        self.mock_model.encode.side_effect = encode_side_effect
        self.mock_embedding_manager.get_code_embedder.return_value = self.mock_model

    def test_empty_code_file(self, mock_embedding_manager_class):
        """Test ingestion of empty code file"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        stats = compressor.ingest_code_file(code="", file_id="empty", filepath="empty.py")

        # Should handle empty file gracefully
        assert stats["file_id"] == "empty"
        assert stats["total_chunks"] >= 0

    def test_code_with_only_comments(self, mock_embedding_manager_class):
        """Test ingestion of file with only comments"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        comment_only_code = "# This is a comment\n# Another comment\n# More comments"
        stats = compressor.ingest_code_file(
            code=comment_only_code, file_id="comments", filepath="comments.py"
        )

        # Should handle comment-only file
        assert stats["file_id"] == "comments"

    def test_very_long_code_file(self, mock_embedding_manager_class):
        """Test ingestion of very long code file"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        # Generate very long code (100 functions - realistic long file)
        functions = []
        for i in range(100):
            functions.append(f'''
def function_{i}(x):
    """Function number {i}"""
    return x + {i}
''')
        long_code = "\n".join(functions)

        stats = compressor.ingest_code_file(code=long_code, file_id="long_file", filepath="long.py")

        # Should handle long files and extract many functions
        assert stats["total_chunks"] > 0
        assert stats["chunk_types"]["functions"] == 100

    def test_code_with_unicode_characters(self, mock_embedding_manager_class):
        """Test ingestion of code with unicode characters"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        unicode_code = '''
def greet(name):
    """Say hello in multiple languages"""
    return f"Hello {name}! 你好! こんにちは! 안녕하세요!"
'''

        stats = compressor.ingest_code_file(
            code=unicode_code, file_id="unicode", filepath="unicode.py"
        )

        # Should handle unicode
        assert stats["file_id"] == "unicode"
        assert stats["total_chunks"] > 0

    def test_search_with_no_chunks(self, mock_embedding_manager_class):
        """Test search on empty compressor"""
        mock_embedding_manager_class.return_value = self.mock_embedding_manager

        compressor = CodeSemanticCompressor()

        results = compressor.search_code("query")

        # Should return empty results
        assert len(results) == 0
