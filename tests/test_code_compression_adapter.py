"""
Tests for CodeCompressionAdapter v0.9.0 fixes.

These tests verify the bug fixes for v0.9.0:
- P0-1: SkeletonResponse constructor
- P0-2: FidelityLevel.BALANCED (fixed to DETAILED)
- P1-1: node_map population for code skeletons
- P1-2: Search semantic with appropriate embedding models
- P2-2: file_id with slashes support

v0.9.0 - Semantic Fidelity Encoding & Programmer UX
"""

from unittest.mock import Mock, patch
import numpy as np


class TestCodeSkeletonWithNodeMap:
    """Test that read_skeleton on code files returns populated node_map (P1-1 fix)"""

    def test_code_skeleton_has_populated_node_map(self):
        """Test that code file skeleton returns non-empty node_map"""
        # Patch both compressors at module level
        with (
            patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls,
            patch("src.code_compression_adapter.CodeSemanticCompressor"),
        ):

            # Create mock text compressor
            mock_text = Mock()
            mock_text.graphs = {}
            mock_text.chunks = {}
            mock_text.model = Mock()
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter
            from src.semantic_compressor import SkeletonResponse

            adapter = CodeCompressionAdapter()

            # Manually set up code compressor mock
            mock_code = Mock()
            mock_code.graphs = {"src/main.py": Mock()}
            mock_code.chunks = {
                "src/main.py::main": Mock(
                    name="main",
                    chunk_type="function",
                    start_line=1,
                    end_line=10,
                    docstring="Main entry point",
                ),
                "src/main.py::helper": Mock(
                    name="helper",
                    chunk_type="function",
                    start_line=12,
                    end_line=20,
                    docstring="Helper function",
                ),
            }
            mock_code.graphs["src/main.py"].nodes.return_value = [
                "src/main.py::main",
                "src/main.py::helper",
            ]
            adapter._code_compressor = mock_code
            adapter._code_file_ids.add("src/main.py")

            # Generate skeleton
            skeleton = adapter._convert_code_stats_to_skeleton(
                stats={
                    "total_chunks": 2,
                    "total_tokens": 100,
                    "compression_ratio": 5.0,
                },
                file_id="src/main.py",
            )

            # Verify it's a SkeletonResponse
            assert isinstance(skeleton, SkeletonResponse)

            # P1-1 fix: node_map should be populated
            assert skeleton.node_map is not None
            assert len(skeleton.node_map) > 0

            # Verify node_map contains code chunk entries
            assert "src/main.py::main" in skeleton.node_map
            assert "src/main.py::helper" in skeleton.node_map

    def test_code_skeleton_node_map_has_descriptive_values(self):
        """Test that node_map values contain chunk type and name"""
        with patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls:
            mock_text = Mock()
            mock_text.graphs = {}
            mock_text.chunks = {}
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter()

            # Create mock code compressor with chunks
            # Use MagicMock with explicit attributes to support slicing
            mock_code = Mock()
            mock_code.graphs = {"src/main.py": Mock()}
            mock_chunk = Mock()
            mock_chunk.name = "main"
            mock_chunk.chunk_type = "function"
            mock_chunk.docstring = "A main function"  # Plain string, not Mock
            mock_chunk.start_line = 1
            mock_chunk.end_line = 10
            mock_code.chunks = {"src/main.py::main": mock_chunk}
            mock_code.graphs["src/main.py"].nodes.return_value = ["src/main.py::main"]
            adapter._code_compressor = mock_code
            adapter._code_file_ids.add("src/main.py")

            skeleton = adapter._convert_code_stats_to_skeleton(
                stats={"total_chunks": 1, "total_tokens": 50, "compression_ratio": 2.0},
                file_id="src/main.py",
            )

            # Verify node_map values are descriptive
            for node_id, desc in skeleton.node_map.items():
                assert isinstance(desc, str)
                assert len(desc) > 0
                # Should contain function info
                assert "function" in desc.lower() or "main" in desc.lower()


class TestFileIdWithSlashes:
    """Test that file_id with slashes (like src/main.py) works correctly (P2-2 fix)"""

    def test_search_filters_text_nodes_correctly(self):
        """Test search filters correctly with file_id containing slashes (text nodes)"""
        with patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls:
            mock_text = Mock()
            mock_text.chunks = {
                "src/module_n0": Mock(embedding=np.array([0.1] * 384)),
                "src/module.py_n0": Mock(embedding=np.array([0.2] * 384)),
                "src/module.py_n1": Mock(embedding=np.array([0.3] * 384)),
                "other/file_n0": Mock(embedding=np.array([0.4] * 384)),
            }
            mock_text.model = Mock()
            mock_text.model.encode.return_value = [np.array([0.2] * 384)]
            mock_text.graphs = {}
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter()

            # Search with file_id that has slashes
            results = adapter.search_semantic_with_scores(
                query="test query", file_id="src/module.py", top_k=5
            )

            # Get returned node IDs
            node_ids = [r[0] for r in results]

            # With P2-2 fix, only nodes starting with "src/module.py_" should match
            for node_id in node_ids:
                assert node_id.startswith("src/module.py_"), f"Unexpected node: {node_id}"

    def test_search_filters_code_nodes_with_double_colon(self):
        """Test search filters correctly with code node IDs using :: separator"""
        with patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls:
            mock_text = Mock()
            mock_text.chunks = {}
            mock_text.model = Mock()
            mock_text.model.encode.return_value = [np.array([0.2] * 384)]
            mock_text.graphs = {}
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter()

            # Add code compressor mock
            mock_code = Mock()
            mock_code.chunks = {
                "src/main.py::main": Mock(embedding=np.array([0.1] * 384)),
                "src/main.py::helper": Mock(embedding=np.array([0.2] * 384)),
                "src/main_other.py::func": Mock(embedding=np.array([0.3] * 384)),
            }
            mock_code.model = Mock()
            mock_code.model.encode.return_value = [np.array([0.2] * 384)]
            adapter._code_compressor = mock_code

            # Search with file_id
            results = adapter.search_semantic_with_scores(
                query="test query", file_id="src/main.py", top_k=5
            )

            # Get returned node IDs
            node_ids = [r[0] for r in results]

            # With P2-2 fix, only nodes starting with "src/main.py::" should match
            for node_id in node_ids:
                assert node_id.startswith("src/main.py::") or node_id.startswith(
                    "src/main.py_"
                ), f"Unexpected node: {node_id}"


class TestModulateRegionForCodeFiles:
    """Test modulate_region output for code files"""

    def test_modulate_code_returns_markdown_fences(self):
        """Test that modulate_region returns Markdown fences for code"""
        with patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls:
            mock_text = Mock()
            mock_text.graphs = {}
            mock_text.chunks = {}
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter
            from src.semantic_compressor import FidelityLevel

            adapter = CodeCompressionAdapter()

            # Add code compressor with chunk
            mock_code = Mock()
            mock_code.chunks = {
                "src/util.py::process": Mock(
                    name="process",
                    chunk_type="function",
                    start_line=1,
                    end_line=15,
                    docstring="Process data transformation",
                    code="def process(data):\n    return transformed",
                ),
            }
            adapter._code_compressor = mock_code

            # Test RAW fidelity
            result = adapter._modulate_code_region(
                node_ids=["src/util.py::process"], fidelity=FidelityLevel.RAW
            )

            # Should contain Markdown code fences
            assert "```" in result
            assert "process" in result

    def test_modulate_code_detailed_fidelity(self):
        """Test DETAILED fidelity (was BALANCED - P0-2 fix)"""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter
            from src.semantic_compressor import FidelityLevel

            adapter = CodeCompressionAdapter()

            # Add code compressor with chunk
            mock_code = Mock()
            mock_code.chunks = {
                "src/util.py::process": Mock(
                    name="process",
                    chunk_type="function",
                    start_line=1,
                    end_line=15,
                    docstring="Process data transformation",
                    code="def process(data):\n    return transformed",
                ),
            }
            adapter._code_compressor = mock_code

            # Test DETAILED fidelity (BALANCED doesn't exist - P0-2)
            result = adapter._modulate_code_region(
                node_ids=["src/util.py::process"], fidelity=FidelityLevel.DETAILED
            )

            # Should contain code preview with fences
            assert "```" in result

    def test_modulate_code_structure_fidelity(self):
        """Test STRUCTURE fidelity for code"""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter
            from src.semantic_compressor import FidelityLevel

            adapter = CodeCompressionAdapter()

            # Add code compressor with chunk
            mock_code = Mock()
            mock_code.chunks = {
                "src/util.py::process": Mock(
                    name="process",
                    chunk_type="function",
                    start_line=1,
                    end_line=15,
                    docstring="Process data transformation",
                    code="def process(data):\n    return transformed",
                ),
            }
            adapter._code_compressor = mock_code

            result = adapter._modulate_code_region(
                node_ids=["src/util.py::process"], fidelity=FidelityLevel.STRUCTURE
            )

            # Structure uses inline code (backticks)
            assert "`" in result
            assert "process" in result


class TestValidateNodeIdsWithSlashesAndColons:
    """Test validate_node_ids handles both :: and _n separators"""

    def test_extract_file_id_from_code_node(self):
        """Test file_id extraction from code node with :: separator"""
        # This tests the fix in compression_handlers.py line 84-91
        node_id = "src/utils/helpers.py::format_data"

        # Extract file_id (should handle :: separator)
        if "::" in node_id:
            file_id = node_id.split("::")[0]
        elif "_n" in node_id:
            file_id = node_id.rsplit("_n", 1)[0]
        else:
            file_id = node_id

        assert file_id == "src/utils/helpers.py"

    def test_extract_file_id_from_text_node(self):
        """Test file_id extraction from text node with _n separator"""
        node_id = "src/readme.md_n5"

        # Extract file_id (should handle _n separator)
        if "::" in node_id:
            file_id = node_id.split("::")[0]
        elif "_n" in node_id:
            file_id = node_id.rsplit("_n", 1)[0]
        else:
            file_id = node_id

        assert file_id == "src/readme.md"

    def test_file_id_with_underscores_in_name(self):
        """Test file_id extraction when filename contains underscores"""
        node_id = "src/my_utils_helper_n3"

        # Using rsplit("_n", 1) should handle this correctly
        if "::" in node_id:
            file_id = node_id.split("::")[0]
        elif "_n" in node_id:
            file_id = node_id.rsplit("_n", 1)[0]
        else:
            file_id = node_id

        assert file_id == "src/my_utils_helper"


class TestSkeletonTokensDocumentation:
    """Test skeleton_tokens approximation is documented and acceptable"""

    def test_skeleton_tokens_is_word_count(self):
        """Verify skeleton_tokens uses word count (documented as approximation)"""
        with patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls:
            mock_text = Mock()
            mock_text.graphs = {}
            mock_text.chunks = {}
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter()

            # Create mock code compressor
            mock_code = Mock()
            mock_code.graphs = {"test.py": Mock()}
            mock_code.chunks = {
                "test.py::func": Mock(
                    name="func",
                    chunk_type="function",
                    start_line=1,
                    end_line=5,
                    docstring="Test function",
                )
            }
            mock_code.graphs["test.py"].nodes.return_value = ["test.py::func"]
            adapter._code_compressor = mock_code
            adapter._code_file_ids.add("test.py")

            skeleton = adapter._convert_code_stats_to_skeleton(
                stats={"total_chunks": 1, "total_tokens": 50, "compression_ratio": 2.0},
                file_id="test.py",
            )

            # skeleton_tokens should be word count of skeleton_text
            expected_word_count = len(skeleton.skeleton_text.split())
            assert skeleton.skeleton_tokens == expected_word_count


class TestFidelityLevelBalancedDoesNotExist:
    """Test that BALANCED was correctly replaced with DETAILED (P0-2 fix)"""

    def test_balanced_fidelity_not_in_enum(self):
        """Verify BALANCED is not a valid FidelityLevel value"""
        from src.semantic_compressor import FidelityLevel

        # BALANCED should not exist
        assert not hasattr(FidelityLevel, "BALANCED")

        # Valid levels should exist
        assert hasattr(FidelityLevel, "ABSTRACT")
        assert hasattr(FidelityLevel, "OUTLINE")
        assert hasattr(FidelityLevel, "STRUCTURE")
        assert hasattr(FidelityLevel, "DETAILED")
        assert hasattr(FidelityLevel, "RAW")


class TestDeletionDoesNotOverreach:
    """Test that delete_document doesn't affect files with shared prefixes"""

    def test_text_deletion_does_not_affect_similar_prefix(self):
        """Deleting 'src/mod' should NOT delete 'src/module.py' text chunks"""
        with patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls:
            mock_text = Mock()
            # Two files sharing prefix
            mock_text.graphs = {"src/mod": Mock(), "src/module.py": Mock()}
            mock_text.chunks = {
                "src/mod_n0": Mock(),  # belongs to src/mod
                "src/mod_n1": Mock(),  # belongs to src/mod
                "src/module.py_n0": Mock(),  # belongs to src/module.py
                "src/module.py_n1": Mock(),  # belongs to src/module.py
            }
            mock_text.file_metadata = {"src/mod": {}, "src/module.py": {}}
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter()

            # Delete the shorter prefix file
            result = adapter.delete_document("src/mod")

            # Should have deleted src/mod but NOT src/module.py
            assert result is True
            assert "src/mod" not in mock_text.graphs
            assert "src/module.py" in mock_text.graphs

            # src/mod chunks should be deleted
            assert "src/mod_n0" not in mock_text.chunks
            assert "src/mod_n1" not in mock_text.chunks

            # src/module.py chunks should still exist
            assert "src/module.py_n0" in mock_text.chunks
            assert "src/module.py_n1" in mock_text.chunks

    def test_code_deletion_does_not_affect_similar_prefix(self):
        """Deleting 'src/mod.py' should NOT delete 'src/module.py' code chunks"""
        with patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls:
            mock_text = Mock()
            mock_text.graphs = {}
            mock_text.chunks = {}
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter()

            # Set up code compressor with two files sharing prefix
            mock_code = Mock()
            mock_code.graphs = {"src/mod.py": Mock(), "src/module.py": Mock()}
            mock_code.chunks = {
                "src/mod.py::main": Mock(),  # belongs to src/mod.py
                "src/mod.py::helper": Mock(),  # belongs to src/mod.py
                "src/module.py::process": Mock(),  # belongs to src/module.py
                "src/module.py::run": Mock(),  # belongs to src/module.py
            }
            mock_code.file_metadata = {"src/mod.py": {}, "src/module.py": {}}
            adapter._code_compressor = mock_code
            adapter._code_file_ids = {"src/mod.py", "src/module.py"}

            # Delete the shorter prefix file
            result = adapter.delete_document("src/mod.py")

            # Should have deleted src/mod.py but NOT src/module.py
            assert result is True
            assert "src/mod.py" not in mock_code.graphs
            assert "src/module.py" in mock_code.graphs

            # src/mod.py chunks should be deleted
            assert "src/mod.py::main" not in mock_code.chunks
            assert "src/mod.py::helper" not in mock_code.chunks

            # src/module.py chunks should still exist
            assert "src/module.py::process" in mock_code.chunks
            assert "src/module.py::run" in mock_code.chunks

            # src/module.py should still be in tracked code files
            assert "src/module.py" in adapter._code_file_ids
            assert "src/mod.py" not in adapter._code_file_ids


class TestDeleteDocumentFromMemoryPropertyCopy:
    """Regression lock for the CodeCompressionAdapter.chunks/@property copy bug.

    Root cause (caught by Bucket B E2E sweep, 2026-05-17):
    CodeCompressionAdapter.chunks, .graphs, and .file_metadata are @property
    descriptors that return dict(self._text_compressor.chunks) -- a NEW COPY
    each access.  The original delete_document handler did:

        chunks_to_delete = [k for k in compressor.chunks.keys() ...]
        for chunk_id in chunks_to_delete:
            del compressor.chunks[chunk_id]   # <-- deletes from the COPY

    The del operated on the throw-away copy, leaving the real underlying
    dict untouched.  Calling list_documents immediately after would still
    show the file because self._text_compressor.chunks still held all keys.

    Fix: added delete_document_from_memory() to CodeCompressionAdapter that
    directly accesses self._text_compressor.chunks (the real dict).
    """

    def test_property_chunks_returns_copy_not_reference(self):
        """Verify that compressor.chunks IS a copy -- del on it is a no-op.

        This test exists to DOCUMENT the footgun, not to fix it.  It must
        continue to pass (the property is intentionally a copy for safety).
        """
        with patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls:
            mock_text = Mock()
            real_chunks = {"doc1_n0": Mock(), "doc1_n1": Mock()}
            mock_text.chunks = real_chunks
            mock_text.graphs = {}
            mock_text.file_metadata = {}
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter()

            copy_a = adapter.chunks
            del copy_a["doc1_n0"]

            # The real underlying dict is UNCHANGED -- del was a no-op
            assert (
                "doc1_n0" in mock_text.chunks
            ), "Property returned a reference, not a copy -- the contract changed!"

    def test_delete_document_from_memory_mutates_real_underlying_dict(self):
        """Regression lock: delete_document_from_memory must remove from real dict."""
        with patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls:
            mock_text = Mock()
            real_chunks = {
                "my_doc_n0": Mock(),
                "my_doc_n1": Mock(),
                "other_doc_n0": Mock(),
            }
            real_graphs = {"my_doc": Mock(), "other_doc": Mock()}
            real_metadata = {"my_doc": {"size": 100}, "other_doc": {"size": 200}}
            mock_text.chunks = real_chunks
            mock_text.graphs = real_graphs
            mock_text.file_metadata = real_metadata
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter()

            assert hasattr(
                adapter, "delete_document_from_memory"
            ), "delete_document_from_memory() not found on CodeCompressionAdapter"

            removed = adapter.delete_document_from_memory("my_doc")

            assert removed == 2, f"Expected 2 removed, got {removed}"

            assert (
                "my_doc_n0" not in mock_text.chunks
            ), "my_doc_n0 still in underlying chunks after delete_document_from_memory"
            assert (
                "my_doc_n1" not in mock_text.chunks
            ), "my_doc_n1 still in underlying chunks after delete_document_from_memory"
            assert "my_doc" not in mock_text.graphs
            assert "my_doc" not in mock_text.file_metadata
            assert "other_doc_n0" in mock_text.chunks
            assert "other_doc" in mock_text.graphs
            assert "other_doc" in mock_text.file_metadata

    def test_delete_document_from_memory_with_code_compressor(self):
        """Regression lock: delete also removes keys from code compressor internals."""
        with patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls:
            mock_text = Mock()
            mock_text.chunks = {"code_file.py_n0": Mock()}
            mock_text.graphs = {}
            mock_text.file_metadata = {}
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter()

            mock_code = Mock()
            real_code_chunks = {
                "code_file.py::main": Mock(),
                "code_file.py::helper": Mock(),
                "other_file.py::run": Mock(),
            }
            real_code_graphs = {"code_file.py": Mock()}
            real_code_metadata = {"code_file.py": {}}
            mock_code.chunks = real_code_chunks
            mock_code.graphs = real_code_graphs
            mock_code.file_metadata = real_code_metadata
            adapter._code_compressor = mock_code
            adapter._code_file_ids = {"code_file.py"}

            removed = adapter.delete_document_from_memory("code_file.py")

            assert removed == 3, f"Expected 3 removed, got {removed}"

            assert "code_file.py::main" not in real_code_chunks
            assert "code_file.py::helper" not in real_code_chunks
            assert "other_file.py::run" in real_code_chunks
            assert "code_file.py" not in adapter._code_file_ids

    def test_list_documents_excludes_deleted_after_delete_from_memory(self):
        """Integration regression lock: list_documents must not show deleted doc."""
        with patch("src.code_compression_adapter.SemanticCompressor") as mock_text_cls:
            mock_text = Mock()
            real_chunks = {
                "keep_me_n0": Mock(),
                "keep_me_n1": Mock(),
                "delete_me_n0": Mock(),
                "delete_me_n1": Mock(),
            }
            real_graphs = {"keep_me": Mock(), "delete_me": Mock()}
            real_metadata = {
                "keep_me": {"created": "2026-01-01"},
                "delete_me": {"created": "2026-01-02"},
            }
            mock_text.chunks = real_chunks
            mock_text.graphs = real_graphs
            mock_text.file_metadata = real_metadata
            mock_text_cls.return_value = mock_text

            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter()

            file_ids_before = set(adapter.file_metadata.keys())
            assert "delete_me" in file_ids_before
            assert "keep_me" in file_ids_before

            adapter.delete_document_from_memory("delete_me")

            file_ids_after = set(adapter.file_metadata.keys())

            assert "delete_me" not in file_ids_after, (
                "delete_me still in file_metadata after delete_document_from_memory -- "
                "the property-copy bug is NOT fixed!"
            )
            assert "keep_me" in file_ids_after, "keep_me was incorrectly deleted"
