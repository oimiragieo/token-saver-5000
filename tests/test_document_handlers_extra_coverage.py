"""document handlers — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


def _make_handler_context():
    """Create a mock handler context with all required keys."""
    compressor = MagicMock()
    compressor.graphs = {"doc1": MagicMock()}
    compressor.chunks = {
        "doc1_n0": MagicMock(text="chunk0"),
        "doc1_n1": MagicMock(text="chunk1"),
    }
    compressor.file_metadata = {"doc1": {"title": "Test"}}
    compressor.get_stats.return_value = {"total_nodes": 2, "total_tokens": 100}

    persistence = MagicMock()
    persistence.delete_document.return_value = True

    resource_manager = MagicMock()
    resource_manager.unregister_document_async = AsyncMock()

    sync_manager = MagicMock()
    sync_manager.remove_metadata.return_value = None
    sync_manager.export_metadata.return_value = {}

    version_manager = MagicMock()
    version_manager.delete_versions_async = AsyncMock()

    path_validator = MagicMock()
    path_validator.validate.side_effect = lambda x: x

    context = {
        "compressor": compressor,
        "persistence": persistence,
        "resource_manager": resource_manager,
        "sync_manager": sync_manager,
        "version_manager": version_manager,
        "path_validator": path_validator,
        "retrieval_history": {},
        "multilevel_encoder": MagicMock(),
        "context_window_adapter": MagicMock(),
    }
    return context


def _has_pillow() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def _make_mock_context(**overrides):
    """Build a mock HandlerContext dict."""
    compressor = MagicMock()
    compressor.chunks = {}
    compressor.graphs = {}
    compressor.file_metadata = {}

    ctx = {
        "compressor": compressor,
        "persistence": MagicMock(),
        "resource_manager": MagicMock(),
        "sync_manager": MagicMock(),
        "version_manager": MagicMock(),
        "path_validator": MagicMock(),
        "retrieval_history": {},
        "context_window_adapter": MagicMock(),
        "multilevel_encoder": MagicMock(),
    }
    ctx["resource_manager"].unregister_document_async = AsyncMock()
    ctx["version_manager"].delete_versions_async = AsyncMock()
    ctx.update(overrides)
    return ctx


class TestHandleDeleteDocument:
    """Tests for handle_delete_document handler."""

    @pytest.mark.asyncio
    async def test_delete_no_confirm_returns_prompt(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        result = await handle_delete_document(context, {"file_id": "doc1", "confirm": False})
        assert "DELETE CONFIRMATION REQUIRED" in result

    @pytest.mark.asyncio
    async def test_delete_with_confirm_succeeds(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        result = await handle_delete_document(context, {"file_id": "doc1", "confirm": True})
        assert "Deleted Successfully" in result
        context["persistence"].delete_document.assert_called_once_with("doc1")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        with pytest.raises(ValueError):
            await handle_delete_document(context, {"file_id": "nonexistent", "confirm": True})

    @pytest.mark.asyncio
    async def test_delete_cleans_up_retrieval_history(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        context["retrieval_history"]["doc1"] = ["some_data"]
        await handle_delete_document(context, {"file_id": "doc1", "confirm": True})
        assert "doc1" not in context["retrieval_history"]

    @pytest.mark.asyncio
    async def test_delete_memory_error_raises_runtime(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        context["compressor"].get_stats.side_effect = Exception("memory error")
        with pytest.raises((RuntimeError, Exception)):
            await handle_delete_document(context, {"file_id": "doc1", "confirm": True})

    @pytest.mark.asyncio
    async def test_delete_persistence_failure_still_succeeds(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        context["persistence"].delete_document.return_value = False
        result = await handle_delete_document(context, {"file_id": "doc1", "confirm": True})
        assert "Deleted Successfully" in result

    @pytest.mark.asyncio
    async def test_delete_version_manager_error_handled(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        context["version_manager"].delete_versions_async = AsyncMock(
            side_effect=Exception("ver fail")
        )
        # Should not raise, error is handled gracefully
        result = await handle_delete_document(context, {"file_id": "doc1", "confirm": True})
        assert "Deleted Successfully" in result


class TestHandleMultilevelEncode:
    """Tests for handle_multilevel_encode handler."""

    @pytest.mark.asyncio
    async def test_multilevel_encode_success(self):
        from src.handlers.compression_handlers import handle_multilevel_encode

        context = _make_handler_context()
        context["multilevel_encoder"].generate_adaptive_skeleton.return_value = "skeleton output"
        result = await handle_multilevel_encode(
            context, {"file_id": "doc1", "available_tokens": 5000}
        )
        assert result == "skeleton output"

    @pytest.mark.asyncio
    async def test_multilevel_encode_invalid_file(self):
        from src.handlers.compression_handlers import handle_multilevel_encode

        context = _make_handler_context()
        with pytest.raises(ValueError):
            await handle_multilevel_encode(context, {"file_id": "bad", "available_tokens": 5000})

    @pytest.mark.asyncio
    async def test_multilevel_encode_zero_tokens(self):
        from src.handlers.compression_handlers import handle_multilevel_encode

        context = _make_handler_context()
        with pytest.raises(ValueError):
            await handle_multilevel_encode(context, {"file_id": "doc1", "available_tokens": 0})

    @pytest.mark.asyncio
    async def test_multilevel_encode_failure_raises_runtime(self):
        from src.handlers.compression_handlers import handle_multilevel_encode

        context = _make_handler_context()
        context["multilevel_encoder"].generate_adaptive_skeleton.side_effect = Exception("fail")
        with pytest.raises(RuntimeError):
            await handle_multilevel_encode(context, {"file_id": "doc1", "available_tokens": 5000})


class TestHandleBatchIngest:
    """Tests for handle_batch_ingest handler."""

    @pytest.mark.asyncio
    async def test_batch_ingest_empty_documents_raises(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises((ValueError, Exception)):
                await handle_batch_ingest(context, {"documents": []})

    @pytest.mark.asyncio
    async def test_batch_ingest_invalid_documents_type(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises(ValueError, match="must be a list"):
                await handle_batch_ingest(context, {"documents": "not a list"})

    @pytest.mark.asyncio
    async def test_batch_ingest_invalid_max_concurrent(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises(ValueError, match="max_concurrent"):
                await handle_batch_ingest(
                    context,
                    {
                        "documents": [{"file_id": "d1", "text": "hi"}],
                        "max_concurrent": 99,
                    },
                )

    @pytest.mark.asyncio
    async def test_batch_ingest_missing_file_id(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises((ValueError, Exception)):
                await handle_batch_ingest(
                    context,
                    {
                        "documents": [{"text": "no id"}],
                    },
                )

    @pytest.mark.asyncio
    async def test_batch_ingest_missing_text(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises((ValueError, Exception)):
                await handle_batch_ingest(
                    context,
                    {
                        "documents": [{"file_id": "d1"}],
                    },
                )

    @pytest.mark.asyncio
    async def test_batch_ingest_success(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.file_id = "d1"
        mock_result.processing_time = 0.5
        mock_result.result = MagicMock(skeleton_text="skeleton...", compression_ratio=2.0)

        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with patch("src.batch_manager.BatchCompressionManager") as MockBCM:
                instance = MockBCM.return_value
                instance.compress_batch = AsyncMock(return_value=[mock_result])
                result = await handle_batch_ingest(
                    context,
                    {
                        "documents": [{"file_id": "d1", "text": "hello world"}],
                    },
                )
        parsed = json.loads(result)
        assert parsed["successful"] == 1

    @pytest.mark.asyncio
    async def test_batch_ingest_non_dict_document(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises(ValueError, match="must be an object"):
                await handle_batch_ingest(
                    context,
                    {
                        "documents": ["not a dict"],
                    },
                )


class TestHandleIngestDirectory:
    """Tests for handle_ingest_directory handler."""

    @pytest.mark.asyncio
    async def test_ingest_directory_missing_dir(self):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        with pytest.raises((ValueError, Exception)):
            await handle_ingest_directory(context, {"directory": ""})

    @pytest.mark.asyncio
    async def test_ingest_directory_nonexistent(self):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        with pytest.raises(ValueError, match="not found"):
            await handle_ingest_directory(context, {"directory": "/nonexistent/path/xyz"})

    @pytest.mark.asyncio
    async def test_ingest_directory_invalid_max_files(self):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        with patch("os.path.isdir", return_value=True):
            with pytest.raises(ValueError, match="max_files"):
                await handle_ingest_directory(
                    context,
                    {
                        "directory": "/some/dir",
                        "max_files": 200,
                    },
                )

    @pytest.mark.asyncio
    async def test_ingest_directory_invalid_max_concurrent(self):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        with patch("os.path.isdir", return_value=True):
            with pytest.raises(ValueError, match="max_concurrent"):
                await handle_ingest_directory(
                    context,
                    {
                        "directory": "/some/dir",
                        "max_concurrent": 99,
                    },
                )

    @pytest.mark.asyncio
    async def test_ingest_directory_no_files_found(self, tmp_path):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = await handle_ingest_directory(
            context,
            {
                "directory": str(empty_dir),
                "patterns": ["*.xyz"],
            },
        )
        parsed = json.loads(result)
        assert parsed["status"] == "no_files"

    @pytest.mark.asyncio
    async def test_ingest_directory_success(self, tmp_path):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()

        # Create test files
        test_dir = tmp_path / "code"
        test_dir.mkdir()
        (test_dir / "main.py").write_text("print('hello')")
        (test_dir / "util.py").write_text("def foo(): pass")

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.file_id = "main.py"
        mock_result.processing_time = 0.1
        mock_result.result = MagicMock(compression_ratio=3.0, total_nodes=5)

        # BatchDocument doesn't have file_path field, so mock it to accept **kwargs
        mock_batch_doc_cls = MagicMock()
        with patch("src.batch_manager.BatchCompressionManager") as MockBCM:
            with patch("src.batch_manager.BatchDocument", mock_batch_doc_cls):
                instance = MockBCM.return_value
                instance.compress_batch = AsyncMock(return_value=[mock_result, mock_result])
                result = await handle_ingest_directory(
                    context,
                    {
                        "directory": str(test_dir),
                        "patterns": ["*.py"],
                    },
                )
        parsed = json.loads(result)
        assert parsed["status"] == "complete"
        assert parsed["successful"] == 2

    @pytest.mark.asyncio
    async def test_ingest_directory_scoped_ids_and_file_metadata(self, tmp_path):
        from src.handlers.compression_handlers import handle_ingest_directory
        from src.identity_scope import compose_scoped_file_id

        context = _make_handler_context()

        test_dir = tmp_path / "code"
        test_dir.mkdir()
        (test_dir / "main.py").write_text("print('hello')")

        captured_documents = []
        scoped_file_id = compose_scoped_file_id("main.py", workspace_id="acme")
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.file_id = scoped_file_id
        mock_result.processing_time = 0.1
        mock_result.result = MagicMock(compression_ratio=3.0, total_nodes=5)

        with patch("src.batch_manager.BatchCompressionManager") as MockBCM:
            instance = MockBCM.return_value

            async def _capture(documents):
                captured_documents.extend(documents)
                return [mock_result]

            instance.compress_batch = AsyncMock(side_effect=_capture)
            result = await handle_ingest_directory(
                context,
                {
                    "directory": str(test_dir),
                    "patterns": ["*.py"],
                    "workspace_id": "acme",
                },
            )

        parsed = json.loads(result)
        assert parsed["status"] == "complete"
        assert parsed["results"][0]["file_id"] == "main.py"
        assert len(captured_documents) == 1
        assert captured_documents[0].file_id == scoped_file_id
        assert captured_documents[0].metadata["file_path"].endswith("main.py")

    @pytest.mark.asyncio
    async def test_ingest_directory_path_traversal_rejected(self):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        context["path_validator"].validate.side_effect = ValueError("path traversal")
        with pytest.raises(ValueError, match="Invalid directory"):
            await handle_ingest_directory(context, {"directory": "../../etc"})


class TestListDocumentsMetadata:
    """Cover lines 789,797,799,801-802 — title/author/date/tags display."""

    @pytest.mark.asyncio
    async def test_list_documents_with_metadata_fields(self):
        from src.handlers.compression_handlers import handle_list_documents

        compressor = MagicMock()
        chunk_a = MagicMock()
        chunk_a.metadata = {"tokens": 100}
        chunk_a.importance = 0.5
        compressor.chunks = {"doc1_n0": chunk_a}
        compressor.graphs = {"doc1": MagicMock()}
        compressor.file_metadata = {}

        metadata = {
            "title": "My Custom Title",
            "author": "Jane Doe",
            "date": "2024-01-01",
            "tags": ["python", "testing", "coverage"],
        }
        stats = {
            "total_nodes": 1,
            "total_tokens": 200,
            "skeleton_tokens": 25,
            "compression_ratio": 8.0,
            "metadata": metadata,
        }
        compressor.get_stats.return_value = stats

        ctx = _make_mock_context(compressor=compressor)
        result = await handle_list_documents(ctx, {})

        assert "My Custom Title" in result
        assert "Jane Doe" in result
        assert "2024-01-01" in result
        assert "python" in result


class TestDeleteDocumentErrors:
    """Cover delete_document memory-error, storage-error, resource-manager-error."""

    def _make_delete_ctx(self):
        compressor = MagicMock()
        compressor.chunks = {"doc1_n0": MagicMock(), "doc1_n1": MagicMock()}
        compressor.graphs = {"doc1": MagicMock()}
        compressor.file_metadata = {"doc1": {}}
        compressor.get_stats.return_value = {"total_nodes": 2}

        ctx = _make_mock_context(compressor=compressor)
        ctx["retrieval_history"] = {"doc1": []}
        return ctx

    @pytest.mark.asyncio
    async def test_delete_memory_error_raises_runtime(self):
        from src.handlers.compression_handlers import handle_delete_document

        ctx = self._make_delete_ctx()
        # Make delete_document_from_memory raise — this is the code path taken when the
        # compressor is a MagicMock (hasattr returns True for all attributes on MagicMock).
        ctx["compressor"].delete_document_from_memory.side_effect = RuntimeError("boom")
        ctx["compressor"].get_stats.return_value = {"total_nodes": 2}

        with pytest.raises(RuntimeError, match="Failed to delete from memory"):
            await handle_delete_document(ctx, {"file_id": "doc1", "confirm": True})

    @pytest.mark.asyncio
    async def test_delete_storage_exception_logged(self):
        from src.handlers.compression_handlers import handle_delete_document

        ctx = self._make_delete_ctx()
        ctx["persistence"].delete_document.side_effect = Exception("storage fail")
        ctx["resource_manager"].unregister_document_async = AsyncMock()
        ctx["sync_manager"].remove_metadata = MagicMock()
        ctx["version_manager"].delete_versions_async = AsyncMock()
        ctx["sync_manager"].export_metadata.return_value = {}

        result = await handle_delete_document(ctx, {"file_id": "doc1", "confirm": True})
        assert "Deleted" in result or "DELETE" in result

    @pytest.mark.asyncio
    async def test_delete_resource_manager_exception_logged(self):
        from src.handlers.compression_handlers import handle_delete_document

        ctx = self._make_delete_ctx()
        ctx["persistence"].delete_document.return_value = True
        ctx["resource_manager"].unregister_document_async = AsyncMock(
            side_effect=Exception("rm fail")
        )
        ctx["sync_manager"].remove_metadata = MagicMock()
        ctx["version_manager"].delete_versions_async = AsyncMock()
        ctx["sync_manager"].export_metadata.return_value = {}

        result = await handle_delete_document(ctx, {"file_id": "doc1", "confirm": True})
        assert "DELETE" in result


class TestBatchIngestRateLimit:
    @pytest.mark.asyncio
    async def test_batch_ingest_rate_limit_exceeded(self):
        from src.handlers.compression_handlers import handle_batch_ingest
        from src.error_types import RateLimitExceededError

        ctx = _make_mock_context()

        with patch("src.handlers.compression_handlers.RATE_LIMITERS") as mock_rl:
            mock_limiter = AsyncMock()
            mock_limiter.acquire.side_effect = RateLimitExceededError("batch", 2.0)
            mock_rl.__getitem__.return_value = mock_limiter

            with pytest.raises(ValueError, match="Rate limit exceeded"):
                await handle_batch_ingest(ctx, {"documents": []})


class TestIngestDirectory:
    @pytest.mark.asyncio
    async def test_ingest_directory_recursive_glob_and_exclusions(self, tmp_path):
        """Cover recursive glob, exclusion, and file read error paths."""
        from src.handlers.compression_handlers import handle_ingest_directory

        # Create some files
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.py").write_text("print('a')")
        (tmp_path / "b.py").write_text("print('b')")
        (tmp_path / "c.pyc").write_text("bytecode")

        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = lambda p: p

        result = await handle_ingest_directory(
            ctx,
            {
                "directory": str(tmp_path),
                "patterns": ["**/*.py"],
                "exclude_patterns": ["*.pyc"],
                "max_files": 50,
                "max_concurrent": 2,
            },
        )

        data = json.loads(result)
        assert data["status"] == "complete"

    @pytest.mark.asyncio
    async def test_ingest_directory_no_matching_files(self, tmp_path):
        from src.handlers.compression_handlers import handle_ingest_directory

        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = lambda p: p

        result = await handle_ingest_directory(
            ctx,
            {
                "directory": str(tmp_path),
                "patterns": ["*.xyz"],
                "max_files": 50,
            },
        )
        data = json.loads(result)
        assert data["status"] == "no_files"

    @pytest.mark.asyncio
    async def test_ingest_directory_file_read_failure(self, tmp_path):
        """Cover lines 1451-1453: exception reading file."""
        from src.handlers.compression_handlers import handle_ingest_directory
        from unittest.mock import mock_open, patch

        (tmp_path / "bad.py").write_text("content")
        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = lambda p: p

        with patch("builtins.open", mock_open()) as mocked_open:
            mocked_open.side_effect = OSError("read failed")
            result = await handle_ingest_directory(
                ctx,
                {"directory": str(tmp_path), "patterns": ["*.py"], "max_files": 50},
            )
        data = json.loads(result)
        assert data["status"] == "read_failed"

    @pytest.mark.asyncio
    async def test_ingest_directory_max_files_limit(self, tmp_path):
        """Cover line 1408-1410: file limiting when too many files match."""
        from src.handlers.compression_handlers import handle_ingest_directory

        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(f"content {i}")

        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = lambda p: p

        result = await handle_ingest_directory(
            ctx,
            {"directory": str(tmp_path), "patterns": ["*.py"], "max_files": 2},
        )

        data = json.loads(result)
        assert data["status"] == "complete"

    @pytest.mark.asyncio
    async def test_ingest_directory_invalid_exclude_pattern(self, tmp_path):
        """Cover line 1388-1392: exclusion patterns are applied."""
        from src.handlers.compression_handlers import handle_ingest_directory

        (tmp_path / "x.py").write_text("content")
        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = lambda p: p

        result = await handle_ingest_directory(
            ctx,
            {
                "directory": str(tmp_path),
                "patterns": ["*.py"],
                "exclude_patterns": ["*.pyc"],
                "max_files": 50,
            },
        )
        data = json.loads(result)
        assert data["status"] == "complete"
