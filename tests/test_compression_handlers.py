"""
Comprehensive tests for compression_handlers.py (v0.4.3).

Tests all 10 compression-related MCP tool handlers with focus on behavior, not implementation.

Testing philosophy (2025 best practices):
- In-memory testing with mocks (no file I/O)
- Deterministic assertions (no LLM-based testing)
- Test behavior, not implementation details
- Patch validation helpers to focus on handler logic

Coverage target: 85%+ of compression_handlers.py
"""

import json
from unittest.mock import AsyncMock, Mock, call, patch
from types import SimpleNamespace

import pytest
from src.handlers import compression_handlers as ch
from src.semantic_compressor import FidelityLevel

# ===========================
# Test Handler Functions
# ===========================


@patch("src.handlers.compression_handlers.validate_file_id")
@patch("src.handlers.compression_handlers.validate_node_ids")
@patch("src.handlers.compression_handlers.validate_token_count")
class TestHandleIngest:
    """Test handle_ingest handler (15 tests)"""

    def setup_method(self):
        """Set up mocks for each test"""
        # Create comprehensive mock context
        self.mock_compressor = Mock()
        self.mock_persistence = Mock()
        self.mock_resource_manager = Mock()
        self.mock_sync_manager = Mock()
        self.mock_version_manager = Mock()

        # Configure resource manager to allow ingestion by default
        # v0.8.0: Handler now calls async wrappers
        self.mock_resource_manager.check_document_size_async = AsyncMock(return_value=(True, ""))
        self.mock_resource_manager.register_document_async = AsyncMock()

        # Configure version manager async wrappers (v0.8.0 audit fix)
        self.mock_version_manager.add_version_async = AsyncMock()
        self.mock_version_manager.delete_versions_async = AsyncMock()

        # Configure compressor to return mock skeleton (async method)
        self.mock_skeleton = Mock()
        self.mock_skeleton.total_nodes = 10
        self.mock_skeleton.total_tokens = 1000
        self.mock_skeleton.skeleton_tokens = 100
        self.mock_skeleton.compression_ratio = 10.0
        self.mock_skeleton.skeleton_text = "Mock skeleton text..."
        self.mock_compressor.ingest_file_async = AsyncMock(return_value=self.mock_skeleton)
        self.mock_compressor.graphs = {"doc1": Mock()}
        self.mock_compressor.file_metadata = {}
        self.mock_compressor.chunks = {}

        # Configure persistence to succeed
        self.mock_persistence.save_document.return_value = True
        self.mock_persistence.save_file_sync_metadata.return_value = True

        # Configure sync manager
        self.mock_sync_manager.export_metadata.return_value = []

        self.context = {
            "compressor": self.mock_compressor,
            "persistence": self.mock_persistence,
            "resource_manager": self.mock_resource_manager,
            "sync_manager": self.mock_sync_manager,
            "version_manager": self.mock_version_manager,
            "retrieval_history": {},
        }

    @pytest.mark.asyncio
    async def test_successful_ingestion(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """Test successful document ingestion"""
        args = {
            "text": "This is a test document with enough content to be meaningful.",
            "file_id": "test_doc",
        }

        with patch("src.handlers.compression_handlers.CompressionAdvisor") as MockAdvisor:
            mock_advisor_instance = Mock()
            mock_estimate = Mock()
            mock_estimate.compression_ratio = 9.5
            mock_estimate.original_tokens = 1000
            mock_estimate.estimated_compressed = 105
            mock_advisor_instance.estimate_compression.return_value = mock_estimate
            MockAdvisor.return_value = mock_advisor_instance

            result = await ch.handle_ingest(self.context, args)

        # Verify compressor.ingest_file_async was called
        self.mock_compressor.ingest_file_async.assert_called_once()

        # Parse JSON result and verify success
        import json

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["file_id"] == "test_doc"
        assert "compression_ratio" in data

    @pytest.mark.asyncio
    async def test_empty_text_raises_error(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """Test that empty text raises validation error"""
        args = {"text": "", "file_id": "doc1"}

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_ingest(self.context, args)

        assert "cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_too_short_text_raises_error(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """Test that text < 20 chars raises error"""
        args = {"text": "Too short", "file_id": "doc1"}

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_ingest(self.context, args)

        assert "too short" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resource_limit_exceeded(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """Test that exceeding resource limits raises error"""
        # v0.8.0: Handler now calls async wrapper
        self.mock_resource_manager.check_document_size_async.return_value = (
            False,
            "Document exceeds limit",
        )

        args = {
            "text": "This is a test document with enough content.",
            "file_id": "huge_doc",
        }

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_ingest(self.context, args)

        assert "exceeds limit" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ingest_response_includes_estimate_fields(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        args = {
            "text": "This is a test document with enough content to be meaningful.",
            "file_id": "test_doc",
        }

        with patch("src.handlers.compression_handlers.CompressionAdvisor") as mock_advisor_cls:
            mock_advisor = Mock()
            mock_estimate = Mock()
            mock_estimate.compression_ratio = 9.5
            mock_estimate.original_tokens = 1000
            mock_estimate.estimated_compressed = 105
            mock_advisor.estimate_compression.return_value = mock_estimate
            mock_advisor_cls.return_value = mock_advisor

            result = await ch.handle_ingest(self.context, args)

        data = json.loads(result)
        assert "estimate" in data
        assert "estimated_ratio" in data["estimate"]
        assert "accuracy" in data["estimate"]

    @pytest.mark.asyncio
    async def test_ingest_awaits_async_file_sync_metadata_save(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """Async persistence hooks should be awaited when provided as coroutines."""
        self.mock_persistence.save_file_sync_metadata = AsyncMock(return_value=True)
        args = {
            "text": "This is a test document with enough content to be meaningful.",
            "file_id": "test_doc_async_sync_save",
        }

        with patch("src.handlers.compression_handlers.CompressionAdvisor") as mock_advisor_cls:
            mock_advisor = Mock()
            mock_estimate = Mock()
            mock_estimate.compression_ratio = 9.5
            mock_estimate.original_tokens = 1000
            mock_estimate.estimated_compressed = 105
            mock_advisor.estimate_compression.return_value = mock_estimate
            mock_advisor_cls.return_value = mock_advisor

            result = await ch.handle_ingest(self.context, args)

        payload = json.loads(result)
        assert payload["status"] == "success"
        self.mock_persistence.save_file_sync_metadata.assert_awaited_once()

    # ------------------------------------------------------------------
    # F1: file_url parameter tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_file_url_happy_path_stamps_source_url(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """When file_url is supplied, the fetched text is ingested and source_url is stamped."""
        fetched_text = "This is a remote document with enough content to be meaningful."
        with (
            patch(
                "src.handlers.compression_handlers.fetch_url",
                new_callable=lambda: lambda *a, **kw: _make_async_return(fetched_text),
            ),
            patch("src.handlers.compression_handlers.CompressionAdvisor") as MockAdvisor,
        ):
            mock_advisor = Mock()
            mock_estimate = Mock()
            mock_estimate.compression_ratio = 9.0
            mock_estimate.original_tokens = 500
            mock_estimate.estimated_compressed = 55
            mock_advisor.estimate_compression.return_value = mock_estimate
            MockAdvisor.return_value = mock_advisor

            result = await ch.handle_ingest(
                self.context,
                {"file_url": "https://example.com/doc.txt", "file_id": "remote_doc"},
            )

        data = json.loads(result)
        assert data["status"] == "success"
        assert data.get("source_url") == "https://example.com/doc.txt"

    @pytest.mark.asyncio
    async def test_text_and_file_url_both_raises_error(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """Providing both text and file_url must raise a ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            await ch.handle_ingest(
                self.context,
                {
                    "text": "inline content here that is long enough",
                    "file_url": "https://example.com/doc.txt",
                    "file_id": "conflict_doc",
                },
            )

    @pytest.mark.asyncio
    async def test_file_url_fetch_failure_propagates_error_code(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """A URLFetchError from fetch_url must surface as a ValueError with the code."""
        from src.url_fetcher import URLFetchError

        async def _raise(*a, **kw):
            raise URLFetchError("private IP blocked", code="private_ip")

        with patch("src.handlers.compression_handlers.fetch_url", side_effect=_raise):
            with pytest.raises(ValueError, match="private_ip"):
                await ch.handle_ingest(
                    self.context,
                    {"file_url": "https://192.168.0.1/secret.txt", "file_id": "fail_doc"},
                )


def _make_async_return(value):
    """Return an async function that returns *value* when awaited."""

    async def _inner(*args, **kwargs):
        return value

    return _inner()


@patch("src.handlers.compression_handlers.validate_file_id")
@patch("src.handlers.compression_handlers.validate_node_ids")
@patch("src.handlers.compression_handlers.validate_token_count")
class TestHandleIngestF4ChunkingStrategy:
    """Tests for F4 — auto-detect structured markdown and default to chunking_strategy=fixed."""

    def setup_method(self):
        self.mock_compressor = Mock()
        self.mock_persistence = Mock()
        self.mock_resource_manager = Mock()
        self.mock_sync_manager = Mock()
        self.mock_version_manager = Mock()

        self.mock_resource_manager.check_document_size_async = AsyncMock(return_value=(True, ""))
        self.mock_resource_manager.register_document_async = AsyncMock()
        self.mock_version_manager.add_version_async = AsyncMock()
        self.mock_version_manager.delete_versions_async = AsyncMock()

        self.mock_skeleton = Mock()
        self.mock_skeleton.total_nodes = 10
        self.mock_skeleton.total_tokens = 2000
        self.mock_skeleton.skeleton_tokens = 200
        self.mock_skeleton.compression_ratio = 10.0
        self.mock_skeleton.skeleton_text = "Skeleton text..."
        self.mock_compressor.ingest_file_async = AsyncMock(return_value=self.mock_skeleton)
        self.mock_compressor.graphs = {"structured_doc": Mock()}
        self.mock_compressor.file_metadata = {}
        self.mock_compressor.chunks = {}

        self.mock_persistence.save_document.return_value = True
        self.mock_persistence.save_file_sync_metadata.return_value = True
        self.mock_sync_manager.export_metadata.return_value = []

        self.context = {
            "compressor": self.mock_compressor,
            "persistence": self.mock_persistence,
            "resource_manager": self.mock_resource_manager,
            "sync_manager": self.mock_sync_manager,
            "version_manager": self.mock_version_manager,
            "retrieval_history": {},
        }

    def _make_advisor_patch(self):
        from unittest.mock import patch as _patch, Mock as _Mock

        mock_estimate = _Mock()
        mock_estimate.compression_ratio = 8.0
        mock_estimate.original_tokens = 2000
        mock_estimate.estimated_compressed = 250
        mock_advisor = _Mock()
        mock_advisor.estimate_compression.return_value = mock_estimate
        return _patch(
            "src.handlers.compression_handlers.CompressionAdvisor",
            return_value=mock_advisor,
        )

    @pytest.mark.asyncio
    async def test_structured_markdown_uses_fixed_strategy(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """F4: structured markdown (3+ headings + 3+ list items) → chunking_strategy='fixed'."""
        structured_text = (
            "## Introduction\n\n"
            "This document explains important concepts.\n\n"
            "## Chapter 1\n\n"
            "1. First point about the topic\n"
            "2. Second point about the topic\n"
            "3. Third point about the topic\n\n"
            "## Chapter 2\n\n"
            "- Alpha item\n"
            "- Beta item\n"
            "- Gamma item\n\n"
            "## Conclusion\n\n"
            "Summary of the document content.\n"
        )
        args = {"text": structured_text, "file_id": "structured_doc"}

        with self._make_advisor_patch():
            result = await ch.handle_ingest(self.context, args)

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["chunking_strategy_used"] == "auto-detected: fixed"
        # Compressor must be called with chunking_strategy="fixed"
        call_kwargs = self.mock_compressor.ingest_file_async.call_args
        assert call_kwargs.kwargs.get("chunking_strategy") == "fixed" or (
            len(call_kwargs.args) >= 4 and call_kwargs.args[3] == "fixed"
        )

    @pytest.mark.asyncio
    async def test_plain_prose_uses_auto_strategy(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """F4: plain prose (no headings, no list items) stays 'auto'."""
        prose_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a plain prose paragraph without any structure. "
            "There are no headings and no list items in this text. "
            "It is just flowing narrative with sentences and paragraphs. "
            "The compression engine should treat this as ordinary text. "
            "Auto mode is the correct choice for unstructured prose documents. "
            "No special handling is required for this kind of input. "
        )
        args = {"text": prose_text, "file_id": "prose_doc"}

        with self._make_advisor_patch():
            result = await ch.handle_ingest(self.context, args)

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["chunking_strategy_used"] == "auto"

    @pytest.mark.asyncio
    async def test_explicit_semantic_strategy_not_overridden(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """F4: explicit chunking_strategy='semantic' is never overridden by auto-detect."""
        structured_text = (
            "## Section 1\n\n"
            "1. Item one\n"
            "2. Item two\n"
            "3. Item three\n\n"
            "## Section 2\n\n"
            "- Alpha\n- Beta\n- Gamma\n\n"
            "## Section 3\n\nMore content here.\n"
        )
        args = {
            "text": structured_text,
            "file_id": "explicit_doc",
            "chunking_strategy": "semantic",
        }

        with self._make_advisor_patch():
            result = await ch.handle_ingest(self.context, args)

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["chunking_strategy_used"] == "semantic"
        call_kwargs = self.mock_compressor.ingest_file_async.call_args
        assert call_kwargs.kwargs.get("chunking_strategy") == "semantic" or (
            len(call_kwargs.args) >= 4 and call_kwargs.args[3] == "semantic"
        )


@patch("src.handlers.compression_handlers.validate_file_id")
@patch("src.handlers.compression_handlers.validate_node_ids")
@patch("src.handlers.compression_handlers.validate_token_count")
class TestHandleIngestF12SavingsTrackerWired:
    """F12 regression-lock (2026-05-23 evening dogfood discovery):

    Pre-fix the SavingsTracker was DEAD INFRASTRUCTURE — _get_tracker existed
    in token_optimization_handlers.py but was only ever called by
    .get_report() paths, NEVER by .record(). Result: every customer's
    `get_savings_report` MCP call returned $0 / 0 tokens saved even after
    real compression activity. CEO question "are we actually saving money?"
    surfaced the mismatch between /v1/global-savings (1.38M tokens across
    all users) and per-session report ($0).

    Fix: handle_ingest now calls _get_tracker(session_id, model).record(
    tool_name="ingest_context", original_tokens, compressed_tokens, model)
    after successful compression. Customers querying the savings report
    immediately after ingest now see real activity.
    """

    def setup_method(self):
        self.mock_compressor = Mock()
        self.mock_persistence = Mock()
        self.mock_resource_manager = Mock()
        self.mock_sync_manager = Mock()
        self.mock_version_manager = Mock()

        self.mock_resource_manager.check_document_size_async = AsyncMock(return_value=(True, ""))
        self.mock_resource_manager.register_document_async = AsyncMock()
        self.mock_version_manager.add_version_async = AsyncMock()
        self.mock_version_manager.delete_versions_async = AsyncMock()

        self.mock_skeleton = Mock()
        self.mock_skeleton.total_nodes = 5
        self.mock_skeleton.total_tokens = 2000
        self.mock_skeleton.skeleton_tokens = 250
        self.mock_skeleton.compression_ratio = 8.0
        self.mock_skeleton.skeleton_text = "Mock skeleton text..."
        self.mock_compressor.ingest_file_async = AsyncMock(return_value=self.mock_skeleton)
        self.mock_compressor.graphs = {"f12_doc": Mock()}
        self.mock_compressor.file_metadata = {}
        self.mock_compressor.chunks = {}

        self.mock_persistence.save_document.return_value = True
        self.mock_persistence.save_file_sync_metadata.return_value = True
        self.mock_sync_manager.export_metadata.return_value = []

        self.context = {
            "compressor": self.mock_compressor,
            "persistence": self.mock_persistence,
            "resource_manager": self.mock_resource_manager,
            "sync_manager": self.mock_sync_manager,
            "version_manager": self.mock_version_manager,
            "retrieval_history": {},
        }

    def _make_advisor_patch(self):
        from unittest.mock import Mock as _Mock, patch as _patch

        mock_estimate = _Mock()
        mock_estimate.compression_ratio = 8.0
        mock_estimate.original_tokens = 2000
        mock_estimate.estimated_compressed = 250
        mock_advisor = _Mock()
        mock_advisor.estimate_compression.return_value = mock_estimate
        return _patch(
            "src.handlers.compression_handlers.CompressionAdvisor",
            return_value=mock_advisor,
        )

    @pytest.mark.asyncio
    async def test_ingest_records_to_savings_tracker(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """F12: handle_ingest must call _get_tracker(...).record(...) after
        successful compression so get_savings_report reflects real activity.

        Pre-fix the tracker registry stayed empty forever; this test would
        find zero events. Post-fix the recorded event has the ingest scalars.
        """
        from src.handlers import token_optimization_handlers as toh

        # Snapshot the tracker registry BEFORE ingest. Use a unique session
        # so we don't pollute or get polluted by other tests' "default" session.
        unique_session = "f12-regression-2026-05-24"
        # Drop any prior state for this session (test isolation):
        toh._savings_trackers.pop(unique_session, None)

        args = {
            "text": "F12 verification: savings tracker must record this ingest event.",
            "file_id": "f12_doc",
            "session_id": unique_session,
            "model": "claude-sonnet-4-6",
        }

        with self._make_advisor_patch():
            await ch.handle_ingest(self.context, args)

        # Post-fix: tracker for this session exists + has one event.
        assert unique_session in toh._savings_trackers, (
            "F12: handle_ingest must lazily create a SavingsTracker for the "
            "passed session_id. Pre-fix the registry stayed empty."
        )
        tracker = toh._savings_trackers[unique_session]
        events = tracker._events  # No public accessor; _events list is the source of truth.
        assert len(events) == 1, (
            f"F12: expected 1 recorded event after one ingest call, got {len(events)}. "
            f"Pre-fix the tracker received zero events from handle_ingest."
        )
        event = events[0]
        assert event.tool_name == "ingest_context"
        assert event.original_tokens == 2000
        assert event.compressed_tokens == 250
        assert event.tokens_saved == 1750
        assert event.model == "claude-sonnet-4-6"
        assert event.dollars_saved > 0, (
            "F12: dollars_saved must be > 0 for a real compression. " f"Got {event.dollars_saved}."
        )

    @pytest.mark.asyncio
    async def test_ingest_failure_in_tracker_record_does_not_fail_the_ingest(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """F12 defensive: if _get_tracker.record() throws (e.g. import-time
        circular, disk-full on journal), handle_ingest still returns success.
        Pre-fix this couldn't happen because record was never called; post-fix
        we want to guarantee the wiring is non-load-bearing for ingest success.
        """
        args = {
            "text": "F12 defensive: tracker failure must not break ingest.",
            "file_id": "f12_defensive_doc",
            "session_id": "f12-defensive",
        }

        with self._make_advisor_patch():
            # Simulate tracker.record() raising — patch the lazy import target:
            with patch(
                "src.handlers.token_optimization_handlers._get_tracker",
                side_effect=RuntimeError("simulated tracker failure"),
            ):
                result = await ch.handle_ingest(self.context, args)

        data = json.loads(result)
        assert data["status"] == "success", (
            "F12: ingest must still succeed even when SavingsTracker.record() "
            f"raises. Got status={data.get('status')}."
        )


@patch("src.handlers.compression_handlers.validate_file_id")
@patch("src.handlers.compression_handlers.validate_node_ids")
@patch("src.handlers.compression_handlers.validate_token_count")
class TestHandleIngestF6InlineQuery:
    """Tests for F6 — optional query param for ingest+query in one call."""

    def setup_method(self):
        self.mock_compressor = Mock()
        self.mock_persistence = Mock()
        self.mock_resource_manager = Mock()
        self.mock_sync_manager = Mock()
        self.mock_version_manager = Mock()

        self.mock_resource_manager.check_document_size_async = AsyncMock(return_value=(True, ""))
        self.mock_resource_manager.register_document_async = AsyncMock()
        self.mock_version_manager.add_version_async = AsyncMock()
        self.mock_version_manager.delete_versions_async = AsyncMock()

        self.mock_skeleton = Mock()
        self.mock_skeleton.total_nodes = 10
        self.mock_skeleton.total_tokens = 2000
        self.mock_skeleton.skeleton_tokens = 200
        self.mock_skeleton.compression_ratio = 10.0
        self.mock_skeleton.skeleton_text = "Mock skeleton text..."
        self.mock_compressor.ingest_file_async = AsyncMock(return_value=self.mock_skeleton)
        self.mock_compressor.graphs = {"query_doc": Mock()}
        self.mock_compressor.file_metadata = {}
        self.mock_compressor.chunks = {}

        self.mock_persistence.save_document.return_value = True
        self.mock_persistence.save_file_sync_metadata.return_value = True
        self.mock_sync_manager.export_metadata.return_value = []

        self.context = {
            "compressor": self.mock_compressor,
            "persistence": self.mock_persistence,
            "resource_manager": self.mock_resource_manager,
            "sync_manager": self.mock_sync_manager,
            "version_manager": self.mock_version_manager,
            "retrieval_history": {},
        }

    def _make_advisor_patch(self):
        from unittest.mock import patch as _patch, Mock as _Mock

        mock_estimate = _Mock()
        mock_estimate.compression_ratio = 8.0
        mock_estimate.original_tokens = 2000
        mock_estimate.estimated_compressed = 250
        mock_advisor = _Mock()
        mock_advisor.estimate_compression.return_value = mock_estimate
        return _patch(
            "src.handlers.compression_handlers.CompressionAdvisor",
            return_value=mock_advisor,
        )

    @pytest.mark.asyncio
    async def test_without_query_no_query_skeleton_field(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """F6: when no query is provided, response has no query_skeleton key."""
        args = {
            "text": "This is a plain document with enough content to be indexed.",
            "file_id": "no_query_doc",
        }

        with self._make_advisor_patch():
            result = await ch.handle_ingest(self.context, args)

        data = json.loads(result)
        assert data["status"] == "success"
        assert "query_skeleton" not in data

    # NOTE: prior `test_with_query_returns_both_stats_and_query_skeleton` test
    # was deleted in v1.34.27 (2026-05-24). It mocked `final_skeleton` as a
    # STRING, which was the very bug shape F7 (Sentry GOTCONTEXT-API-H) caught
    # in prod — the test was passing only because the pre-F7 handler embedded
    # the raw pipeline dict (string and all) into the response. Post-F7 fix
    # (da74691) the handler projects scalar fields from the real
    # SkeletonResponse dataclass, so the string-mock breaks at
    # `.total_nodes` access. Replaced by
    # `test_with_query_pipeline_returns_skeleton_response_object_serializes_cleanly`
    # below, which uses a real SkeletonResponse + asserts the correct
    # scalar-projection contract.

    @pytest.mark.asyncio
    async def test_with_query_pipeline_returns_skeleton_response_object_serializes_cleanly(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """F7 REGRESSION LOCK (Sentry GOTCONTEXT-API-H, 2026-05-23 dogfood):

        The real ``run_read_skeleton_pipeline()`` returns a dict where
        ``["final_skeleton"]`` is a ``SkeletonResponse`` DATACLASS — not a
        string. Pre-fix, ``handle_ingest`` embedded the raw pipeline dict
        in ``response["query_skeleton"]`` and the subsequent
        ``json.dumps(response, indent=2)`` raised
        ``TypeError: Object of type SkeletonResponse is not JSON serializable``.

        The prior F6 test (``test_with_query_returns_both_stats_and_query_skeleton``)
        used a STRING for ``final_skeleton`` so the JSON-serialization path
        was never exercised — that's why F7 shipped to prod.

        This test uses a real ``SkeletonResponse`` (built via dataclass) so
        ``json.dumps`` of the full response gets exercised at unit-test time.
        Asserts:
        - The handler returns a string (json.dumps succeeded — no TypeError)
        - The returned JSON has ``query_skeleton`` as a flat dict of scalars
        - The raw SkeletonResponse object did NOT leak into the response.
        """
        from src.semantic_compressor import SkeletonResponse

        real_skeleton = SkeletonResponse(
            file_id="query_doc",
            total_nodes=10,
            total_tokens=2000,
            skeleton_tokens=250,
            compression_ratio=8.0,
            skeleton_text="=== SEMANTIC SKELETON ===\n[0] query_doc::section_1 (0.95)\n",
            node_map={"query_doc::section_1": "ANCHOR: section 1 summary"},
        )

        mock_pipeline_result = {
            "final_skeleton": real_skeleton,
            "final_stage": "query_guided",
            "stage_count": 2,
            "stages": [
                {
                    "name": "baseline",
                    "query": None,
                    "anchor_node_count": 0,
                    "evidence_used": False,
                    "total_nodes": 10,
                    "skeleton_tokens": 250,
                    "compression_ratio": 8.0,
                },
                {
                    "name": "query_guided",
                    "query": "what is the main topic?",
                    "anchor_node_count": 1,
                    "evidence_used": False,
                    "total_nodes": 10,
                    "skeleton_tokens": 250,
                    "compression_ratio": 8.0,
                },
            ],
            "evidence": None,
            "selection_mode_resolved": "query_guided",
        }

        args = {
            "text": "This is a document with enough content to be indexed for the query.",
            "file_id": "query_doc",
            "query": "what is the main topic?",
        }

        with self._make_advisor_patch():
            with patch(
                "src.handlers.compression_handlers.run_read_skeleton_pipeline",
                return_value=mock_pipeline_result,
            ):
                # Pre-fix this raised TypeError. Post-fix it returns a JSON string.
                result = await ch.handle_ingest(self.context, args)

        # Round-trip through json.loads so a non-serializable embedded object
        # would surface here too.
        data = json.loads(result)

        assert data["status"] == "success"
        assert "query_skeleton" in data

        qs = data["query_skeleton"]
        # Must be a flat dict of scalars — NOT a serialization of the raw
        # pipeline dict that contains the SkeletonResponse object.
        assert isinstance(qs, dict)
        for required in (
            "total_nodes",
            "total_tokens",
            "skeleton_tokens",
            "compression_ratio",
            "skeleton_text",
            "node_map",
            "selection_mode_resolved",
            "pipeline",
        ):
            assert required in qs, f"query_skeleton missing scalar field {required!r}"

        assert qs["total_nodes"] == 10
        assert qs["skeleton_tokens"] == 250
        assert qs["selection_mode_resolved"] == "query_guided"

        # The full pipeline trace is projected to scalars under qs["pipeline"]:
        assert qs["pipeline"]["final_stage"] == "query_guided"
        assert qs["pipeline"]["stage_count"] == 2
        assert len(qs["pipeline"]["stages"]) == 2

        # Negative assertion: the raw key "final_skeleton" must NOT appear
        # in query_skeleton — that's the pre-fix bug shape.
        assert "final_skeleton" not in qs

    @pytest.mark.asyncio
    async def test_with_query_skipped_for_small_doc(
        self, mock_validate_token, mock_validate_nodes, mock_validate_file
    ):
        """F6: query_skeleton is skipped (not in response) when doc is too small (< 3 nodes)."""
        self.mock_skeleton.total_nodes = 2  # too small
        args = {
            "text": "This is a document with enough content to be indexed for the query.",
            "file_id": "small_query_doc",
            "query": "what is the main topic?",
        }

        with self._make_advisor_patch():
            result = await ch.handle_ingest(self.context, args)

        data = json.loads(result)
        assert data["status"] == "success"
        assert "query_skeleton" not in data


@patch("src.handlers.compression_handlers.validate_file_id")
class TestHandleReadSkeletonF12ClassCompletion:
    """v1.34.28 regression: read_skeleton must hit SavingsTracker.record().

    Same root cause as F12 (v1.34.27): an aggregator helper (`get_savings_report`)
    that exists but has no write-side caller produces silently-wrong zeros.
    v1.34.27 fixed the ingest path; this test locks read_skeleton.
    """

    @pytest.mark.asyncio
    async def test_read_skeleton_records_to_savings_tracker(self, mock_validate_file):
        from src.handlers import token_optimization_handlers as toh

        sid = "v1_34_28-read-skeleton-verify"
        toh._savings_trackers.pop(sid, None)

        compressor = Mock()
        compressor.chunks = {}
        compressor._access_tracker = None
        compressor._baseline_skeleton_cache = {}
        compressor.graphs = {"f12_doc": Mock()}

        sync_manager = Mock()
        sync_manager.file_metadata = {}

        skel = SimpleNamespace(
            total_nodes=4,
            total_tokens=1000,
            skeleton_tokens=120,
            compression_ratio=8.33,
            skeleton_text="mock skeleton text",
            node_map={},
        )

        def fake_pipeline(*_a, **_kw):
            return {
                "final_skeleton": skel,
                "final_stage": "baseline",
                "stage_count": 1,
                "stages": ["baseline"],
                "evidence": None,
                "selection_mode_resolved": "baseline",
            }

        with patch(
            "src.handlers.compression_handlers.run_read_skeleton_pipeline",
            side_effect=fake_pipeline,
        ):
            await ch.handle_read_skeleton(
                {
                    "compressor": compressor,
                    "sync_manager": sync_manager,
                    "retrieval_history": {},
                },
                {
                    "file_id": "f12_doc",
                    "selection_mode": "baseline",
                    "session_id": sid,
                    "model": "claude-sonnet-4-6",
                },
            )

        assert sid in toh._savings_trackers, (
            "v1.34.28: read_skeleton must lazily create the SavingsTracker for "
            "the passed session_id (same shape as F12 ingest fix)."
        )
        events = toh._savings_trackers[sid]._events
        assert len(events) == 1
        ev = events[0]
        assert ev.tool_name == "read_skeleton"
        assert ev.original_tokens == 1000
        assert ev.compressed_tokens == 120
        assert ev.tokens_saved == 880

    @pytest.mark.asyncio
    async def test_read_skeleton_succeeds_when_tracker_record_fails(self, mock_validate_file):
        """Defensive: a tracker.record() exception must NOT fail read_skeleton."""
        compressor = Mock()
        compressor.chunks = {}
        compressor._access_tracker = None
        compressor._baseline_skeleton_cache = {}
        compressor.graphs = {"f12_doc": Mock()}

        sync_manager = Mock()
        sync_manager.file_metadata = {}

        skel = SimpleNamespace(
            total_nodes=2,
            total_tokens=200,
            skeleton_tokens=40,
            compression_ratio=5.0,
            skeleton_text="x",
            node_map={},
        )

        def fake_pipeline(*_a, **_kw):
            return {
                "final_skeleton": skel,
                "final_stage": "baseline",
                "stage_count": 1,
                "stages": ["baseline"],
                "evidence": None,
                "selection_mode_resolved": "baseline",
            }

        with (
            patch(
                "src.handlers.compression_handlers.run_read_skeleton_pipeline",
                side_effect=fake_pipeline,
            ),
            patch(
                "src.handlers.token_optimization_handlers._get_tracker",
                side_effect=RuntimeError("simulated tracker breakage"),
            ),
        ):
            out = await ch.handle_read_skeleton(
                {
                    "compressor": compressor,
                    "sync_manager": sync_manager,
                    "retrieval_history": {},
                },
                {"file_id": "f12_doc", "selection_mode": "baseline"},
            )

        data = json.loads(out)
        assert data["file_id"] == "f12_doc", (
            "v1.34.28 defensive: read_skeleton must still return success when "
            "SavingsTracker.record() raises."
        )


@patch("src.handlers.compression_handlers.validate_file_id")
class TestHandleReadSkeleton:
    """Test handle_read_skeleton handler (6 tests)"""

    def setup_method(self):
        """Set up mocks for each test"""
        self.mock_compressor = Mock()
        self.mock_sync_manager = Mock()

        # Create proper mock for _generate_skeleton (which returns a skeleton object)
        mock_skeleton = Mock()
        mock_skeleton.file_id = "doc1"
        mock_skeleton.total_nodes = 10
        mock_skeleton.total_tokens = 1000
        mock_skeleton.skeleton_tokens = 100
        mock_skeleton.compression_ratio = 10.0
        mock_skeleton.skeleton_text = "Mock skeleton text"
        mock_skeleton.node_map = {}  # JSON serializable
        self.mock_compressor._generate_skeleton.return_value = mock_skeleton
        self.mock_sync_manager.file_metadata = {}

        self.context = {
            "compressor": self.mock_compressor,
            "sync_manager": self.mock_sync_manager,
        }

    @pytest.mark.asyncio
    async def test_successful_skeleton_read(self, mock_validate_file):
        """Test successful skeleton reading"""
        args = {"file_id": "doc1"}

        result = await ch.handle_read_skeleton(self.context, args)

        self.mock_compressor._generate_skeleton.assert_called_once_with("doc1")
        # Handler returns JSON
        data = json.loads(result)
        assert data["file_id"] == "doc1"
        assert data["skeleton_text"] == "Mock skeleton text"
        assert data["compression_ratio"] == 10.0
        # F3: default is now "auto"; with no chunks available (Mock), auto resolves to baseline
        assert data["selection_mode"] == "auto"
        assert data["pipeline"]["final_stage"] == "baseline"
        assert data["pipeline"]["stages"][0]["name"] == "baseline"

    @pytest.mark.asyncio
    async def test_query_guided_mode_calls_query_skeleton(self, mock_validate_file):
        """Test query-guided selection mode passes query to compressor."""
        args = {"file_id": "doc1", "selection_mode": "query_guided", "query": "error correction"}

        result = await ch.handle_read_skeleton(self.context, args)
        data = json.loads(result)

        assert self.mock_compressor._generate_skeleton.call_args_list == [
            call("doc1"),
            call("doc1", query="error correction"),
        ]
        assert [stage["name"] for stage in data["pipeline"]["stages"]] == [
            "baseline",
            "query_guided",
        ]

    @pytest.mark.asyncio
    async def test_evidence_aware_mode_adds_evidence_payload(self, mock_validate_file):
        """Test evidence-aware mode calls retrieve_evidence and returns evidence diagnostics."""
        self.mock_compressor.retrieve_evidence.return_value = SimpleNamespace(
            sufficient=True,
            best_score=0.91,
            threshold=0.35,
            used_expanded_search=False,
            message="ok",
            node_ids=["doc1_n0"],
        )
        args = {
            "file_id": "doc1",
            "selection_mode": "evidence_aware",
            "query": "surface code",
            "top_k": 2,
            "min_similarity": 0.4,
        }

        result = await ch.handle_read_skeleton(self.context, args)
        data = json.loads(result)

        self.mock_compressor.retrieve_evidence.assert_called_once_with(
            query="surface code",
            file_id="doc1",
            top_k=2,
            min_similarity=0.4,
        )
        assert "evidence" in data
        assert data["evidence"]["sufficient"] is True
        assert [stage["name"] for stage in data["pipeline"]["stages"]] == [
            "baseline",
            "query_guided",
            "evidence_aware",
        ]

    @pytest.mark.asyncio
    async def test_read_skeleton_rejects_invalid_selection_mode(self, mock_validate_file):
        args = {"file_id": "doc1", "selection_mode": "invalid"}

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_read_skeleton(self.context, args)
        assert "Invalid selection_mode" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_read_skeleton_requires_query_for_non_baseline(self, mock_validate_file):
        args = {"file_id": "doc1", "selection_mode": "query_guided"}

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_read_skeleton(self.context, args)
        assert "query is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_staleness_warning_shown(self, mock_validate_file):
        """Test that staleness warning is shown when file changed"""
        self.mock_sync_manager.file_metadata = {"doc1": {}}
        self.mock_sync_manager.check_file_sync.return_value = {
            "in_sync": False,
            "reason": "File has been modified on disk",
            "current_mtime": 1234567890,
            "cached_mtime": 1234567000,
            "has_source_file": True,
        }

        args = {"file_id": "doc1"}

        result = await ch.handle_read_skeleton(self.context, args)

        # Handler returns JSON with staleness_warning
        data = json.loads(result)
        assert "staleness_warning" in data
        assert data["staleness_warning"]["is_stale"] is True
        assert "Mock skeleton text" in data["skeleton_text"]


@patch("src.handlers.compression_handlers.validate_node_ids")
class TestHandleModulateRegionF10SingularNodeId:
    """v1.34.30 (F10): handle_modulate_region must accept singular `node_id`.

    Pre-fix: customers calling modulate_region(node_id="x") got
    "Input validation error: 'node_ids' is a required property".
    The plural-only API surface broke the singular intuitive call,
    which agents naturally try first. Discovered in 2026-05-23 dogfood
    cycle (F10 in docs/audits/2026-05-23-dogfood-findings.md).

    Fix: handler accepts either `node_ids` (canonical, list) or
    `node_id` (convenience, string → wrapped to [node_id]). Schema
    `anyOf` enforces one is present.
    """

    def setup_method(self):
        self.mock_compressor = Mock()
        self.mock_compressor.modulate_region.return_value = "expanded content"
        self.mock_sync_manager = Mock()
        self.mock_sync_manager.file_metadata = {}
        self.context = {
            "compressor": self.mock_compressor,
            "sync_manager": self.mock_sync_manager,
            "retrieval_history": {},
        }

    @pytest.mark.asyncio
    async def test_singular_node_id_wraps_to_list(self, mock_validate_nodes):
        """node_id="x" must wrap to [x] and call modulate_region with the list."""
        result = await ch.handle_modulate_region(
            self.context,
            {"node_id": "doc_a_n0", "fidelity_level": "RAW"},
        )
        # The mock returns "expanded content"; we just need the handler not to raise
        # and to have called validate_node_ids with the wrapped list.
        mock_validate_nodes.assert_called_once_with(["doc_a_n0"], self.context)
        assert "expanded content" in result or "doc_a_n0" in result

    @pytest.mark.asyncio
    async def test_plural_node_ids_unchanged_canonical_path(self, mock_validate_nodes):
        """node_ids=[...] (canonical) must continue to work — no regression."""
        await ch.handle_modulate_region(
            self.context,
            {"node_ids": ["doc_a_n0", "doc_a_n1"], "fidelity_level": "RAW"},
        )
        mock_validate_nodes.assert_called_once_with(["doc_a_n0", "doc_a_n1"], self.context)

    @pytest.mark.asyncio
    async def test_both_singular_and_plural_prefers_plural_canonical(self, mock_validate_nodes):
        """If both `node_ids` and `node_id` are passed, canonical `node_ids` wins.

        Defensive: documented preference avoids agent-side ambiguity.
        """
        await ch.handle_modulate_region(
            self.context,
            {
                "node_ids": ["canonical"],
                "node_id": "ignored",
                "fidelity_level": "RAW",
            },
        )
        mock_validate_nodes.assert_called_once_with(["canonical"], self.context)

    @pytest.mark.asyncio
    async def test_neither_node_id_nor_node_ids_raises_clear_error(self, mock_validate_nodes):
        """Calling with neither must raise a ValueError with actionable [TIP]."""
        with pytest.raises(ValueError) as exc:
            await ch.handle_modulate_region(self.context, {"fidelity_level": "RAW"})
        msg = str(exc.value)
        assert "node_ids" in msg and "node_id" in msg
        assert "[TIP]" in msg


@patch("src.handlers.compression_handlers.validate_node_ids")
class TestHandleModulateRegion:
    """Test handle_modulate_region handler (12 tests)"""

    def setup_method(self):
        """Set up mocks for each test"""
        self.mock_compressor = Mock()
        self.mock_sync_manager = Mock()

        self.mock_compressor.modulate_region.return_value = "Modulated content"
        self.mock_sync_manager.file_metadata = {}

        self.context = {
            "compressor": self.mock_compressor,
            "sync_manager": self.mock_sync_manager,
            "retrieval_history": {},
        }

    @pytest.mark.asyncio
    async def test_successful_modulation_raw_fidelity(self, mock_validate_nodes):
        """Test successful modulation with RAW fidelity"""
        args = {
            "node_ids": ["doc1_n0", "doc1_n1"],
            "fidelity_level": "RAW",
        }

        result = await ch.handle_modulate_region(self.context, args)

        self.mock_compressor.modulate_region.assert_called_once_with(
            ["doc1_n0", "doc1_n1"], FidelityLevel.RAW
        )
        assert result == "Modulated content"

    @pytest.mark.asyncio
    async def test_default_fidelity_is_raw(self, mock_validate_nodes):
        """Test that default fidelity is RAW when not specified"""
        args = {"node_ids": ["doc1_n0"]}

        await ch.handle_modulate_region(self.context, args)

        call_args = self.mock_compressor.modulate_region.call_args
        assert call_args[0][1] == FidelityLevel.RAW

    @pytest.mark.asyncio
    async def test_invalid_fidelity_level_raises_error(self, mock_validate_nodes):
        """Test that invalid fidelity level raises error with suggestions"""
        args = {
            "node_ids": ["doc1_n0"],
            "fidelity_level": "SUPER_DETAILED",
        }

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_modulate_region(self.context, args)

        error_msg = str(exc_info.value)
        assert "Invalid fidelity_level" in error_msg
        assert "ABSTRACT" in error_msg
        assert "RAW" in error_msg

    @pytest.mark.asyncio
    async def test_retrieval_history_tracked(self, mock_validate_nodes):
        """Test that retrieval history is tracked for blind spot detection"""
        args = {
            "node_ids": ["doc1_n0", "doc1_n1"],
            "fidelity_level": "RAW",
        }

        await ch.handle_modulate_region(self.context, args)

        assert "doc1" in self.context["retrieval_history"]
        assert "doc1_n0" in self.context["retrieval_history"]["doc1"]


class TestHandleSearchSemantic:
    """Test handle_search_semantic handler (6 tests)"""

    def setup_method(self):
        """Set up mocks for each test"""
        self.mock_compressor = Mock()

        mock_node_0 = Mock()
        mock_node_0.text = "This is about quantum computing"
        mock_node_0.importance = 0.95
        mock_node_0.metadata = {"tokens": 50}

        self.mock_compressor.chunks = {"doc1_n0": mock_node_0}
        # v0.9.0: search_semantic_with_scores returns (node_id, similarity_score) tuples
        self.mock_compressor.search_semantic_with_scores.return_value = [("doc1_n0", 0.87)]
        self.mock_compressor._generate_summary.side_effect = lambda text, max_length: text[
            :max_length
        ]

        self.context = {"compressor": self.mock_compressor}

    @pytest.mark.asyncio
    async def test_successful_search(self):
        """Test successful semantic search"""
        args = {"query": "quantum computing"}

        result = await ch.handle_search_semantic(self.context, args)

        self.mock_compressor.search_semantic_with_scores.assert_called_once_with(
            "quantum computing", None, 5
        )

        # Handler returns JSON
        data = json.loads(result)
        assert data["query"] == "quantum computing"
        assert data["total_results"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["node_id"] == "doc1_n0"
        # v0.9.0: Now includes similarity score
        assert data["results"][0]["similarity"] == 0.87

    @pytest.mark.asyncio
    async def test_search_with_custom_top_k(self):
        """Test search with custom top_k parameter"""
        args = {"query": "test query", "top_k": 10}

        await ch.handle_search_semantic(self.context, args)

        self.mock_compressor.search_semantic_with_scores.assert_called_once_with(
            "test query", None, 10
        )

    @pytest.mark.asyncio
    async def test_evidence_aware_search_uses_retrieve_evidence(self):
        self.mock_compressor.retrieve_evidence.return_value = SimpleNamespace(
            sufficient=False,
            best_score=0.12,
            threshold=0.4,
            used_expanded_search=True,
            message="insufficient",
            scores=[("doc1_n0", 0.12)],
        )
        args = {
            "query": "hard query",
            "top_k": 3,
            "evidence_aware": True,
            "min_similarity": 0.4,
        }

        result = await ch.handle_search_semantic(self.context, args)
        data = json.loads(result)

        self.mock_compressor.retrieve_evidence.assert_called_once_with(
            query="hard query",
            file_id=None,
            top_k=3,
            min_similarity=0.4,
        )
        assert data["evidence_aware"] is True
        assert data["evidence"]["used_expanded_search"] is True


class TestSearchSemanticOutputFields:
    """Schema tests for search_semantic output field docs."""

    def test_search_semantic_output_fields_include_result_paths(self):
        output_fields = ch.get_search_semantic_output_fields()
        assert "query" in output_fields
        assert "results[].node_id" in output_fields
        assert "results[].similarity" in output_fields
        assert "results[].importance" in output_fields
        assert "evidence.sufficient" in output_fields


class TestReadSkeletonOutputFields:
    """Schema tests for read_skeleton output field docs."""

    def test_read_skeleton_output_fields_include_optional_paths(self):
        output_fields = ch.get_read_skeleton_output_fields()
        assert "file_id" in output_fields
        assert "selection_mode" in output_fields
        assert "node_map" in output_fields
        assert "evidence.sufficient" in output_fields
        assert "staleness_warning.is_stale" in output_fields


class TestIngestOutputFields:
    """Schema tests for ingest_context output field docs."""

    def test_ingest_output_fields_include_estimate_paths(self):
        output_fields = ch.get_ingest_context_output_fields()
        assert "status" in output_fields
        assert "file_id" in output_fields
        assert "token_savings_percent" in output_fields
        assert "estimate.estimated_ratio" in output_fields
        assert "estimate.accuracy" in output_fields


class TestRecommendFidelityOutputFields:
    """Schema tests for recommend_fidelity output field docs."""

    def test_recommend_fidelity_output_fields_include_key_paths(self):
        output_fields = ch.get_recommend_fidelity_output_fields()
        assert "recommended_level" in output_fields
        assert "confidence" in output_fields
        assert "reasoning" in output_fields
        assert "token_estimate" in output_fields
        assert "alternatives" in output_fields
        assert "usage_tip" in output_fields


class TestHandleGetStats:
    """Test handle_get_stats handler (4 tests)"""

    def setup_method(self):
        """Set up mocks for each test"""
        self.mock_compressor = Mock()
        self.context = {"compressor": self.mock_compressor}

    @pytest.mark.asyncio
    async def test_get_stats_for_specific_file(self):
        """Test getting stats for a specific file"""
        self.mock_compressor.get_stats.return_value = {
            "total_nodes": 10,
            "total_edges": 25,
            "total_tokens": 5000,
            "skeleton_tokens": 500,
            "compression_ratio": 10.0,
            "metadata": {"author": "Test"},
        }

        args = {"file_id": "doc1"}

        result = await ch.handle_get_stats(self.context, args)

        self.mock_compressor.get_stats.assert_called_once_with("doc1")
        assert "[STATS] Document Statistics: doc1" in result

    @pytest.mark.asyncio
    async def test_get_global_stats(self):
        """Test getting global stats (all files)"""
        self.mock_compressor.get_stats.return_value = {
            "total_files": 3,
            "total_nodes": 50,
            "files": ["doc1", "doc2", "doc3"],
        }

        args = {}

        result = await ch.handle_get_stats(self.context, args)

        assert "[STATS] Global Statistics" in result
        assert "Total files ingested: 3" in result


class TestHandleListDocuments:
    """Test handle_list_documents handler (4 tests)"""

    def setup_method(self):
        """Set up mocks for each test"""
        self.mock_compressor = Mock()
        self.context = {"compressor": self.mock_compressor}

    @pytest.mark.asyncio
    async def test_list_documents_when_empty(self):
        """Test listing documents when none are ingested"""
        self.mock_compressor.chunks = {}

        result = await ch.handle_list_documents(self.context, {})

        assert "No documents ingested yet" in result

    @pytest.mark.asyncio
    async def test_list_documents_with_files(self):
        """Test listing multiple documents"""
        self.mock_compressor.chunks = {"doc1_n0": Mock(), "doc2_n0": Mock()}
        self.mock_compressor.get_stats.return_value = {
            "total_nodes": 2,
            "total_tokens": 1000,
            "skeleton_tokens": 100,
            "compression_ratio": 10.0,
            "metadata": {},
        }

        result = await ch.handle_list_documents(self.context, {})

        assert "[DOC] Document Inventory" in result
        assert "Total documents: 2" in result


@patch("src.handlers.compression_handlers.validate_file_id")
class TestHandleDeleteDocument:
    """Test handle_delete_document handler (6 tests)"""

    def setup_method(self):
        """Set up mocks for each test"""
        self.mock_compressor = Mock()
        self.mock_compressor.chunks = {"doc1_n0": Mock()}
        self.mock_compressor.graphs = {"doc1": Mock()}
        self.mock_compressor.file_metadata = {"doc1": {}}
        self.mock_compressor.get_stats.return_value = {"total_nodes": 1}

        self.mock_persistence = Mock()
        self.mock_persistence.delete_document.return_value = True

        self.mock_sync_manager = Mock()
        self.mock_sync_manager.export_metadata.return_value = []

        # v0.8.0 audit fix: add async mocks for resource and version managers
        self.mock_resource_manager = Mock()
        self.mock_resource_manager.unregister_document_async = AsyncMock()

        self.mock_version_manager = Mock()
        self.mock_version_manager.delete_versions_async = AsyncMock()

        self.context = {
            "compressor": self.mock_compressor,
            "persistence": self.mock_persistence,
            "resource_manager": self.mock_resource_manager,
            "sync_manager": self.mock_sync_manager,
            "version_manager": self.mock_version_manager,
            "retrieval_history": {},
        }

    @pytest.mark.asyncio
    async def test_delete_without_confirm_shows_warning(self, mock_validate_file):
        """Test that deletion without confirm=true shows warning"""
        args = {"file_id": "doc1"}

        result = await ch.handle_delete_document(self.context, args)

        assert "[WARN]  DELETE CONFIRMATION REQUIRED" in result
        assert "confirm=true" in result

    @pytest.mark.asyncio
    async def test_successful_deletion_with_confirm(self, mock_validate_file):
        """Test successful deletion with confirm=true"""
        args = {"file_id": "doc1", "confirm": True}

        result = await ch.handle_delete_document(self.context, args)

        assert "[DELETE] Document Deleted Successfully" in result
        assert "doc1" in result


@patch("src.handlers.compression_handlers.validate_token_count")
@patch("src.handlers.compression_handlers.validate_file_id")
class TestHandleAdaptToContextWindow:
    """Test handle_adapt_to_context_window handler (6 tests)"""

    def setup_method(self):
        """Set up mocks for each test"""
        self.mock_adapter = Mock()
        self.mock_adapter.adapt_to_context_window.return_value = "Adapted skeleton"

        self.context = {
            "compressor": Mock(),
            "context_window_adapter": self.mock_adapter,
        }

    @pytest.mark.asyncio
    async def test_successful_adaptation(self, mock_validate_file, mock_validate_token):
        """Test successful context window adaptation"""
        args = {"file_id": "doc1", "available_tokens": 5000}

        result = await ch.handle_adapt_to_context_window(self.context, args)

        self.mock_adapter.adapt_to_context_window.assert_called_once()
        assert result == "Adapted skeleton"

    @pytest.mark.asyncio
    async def test_invalid_query_priority_raises_error(
        self, mock_validate_file, mock_validate_token
    ):
        """Test that query_priority outside [0, 1] raises error"""
        args = {
            "file_id": "doc1",
            "available_tokens": 5000,
            "query_priority": 1.5,
        }

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_adapt_to_context_window(self.context, args)

        assert "between 0.0 and 1.0" in str(exc_info.value)


@patch("src.handlers.compression_handlers.validate_token_count")
@patch("src.handlers.compression_handlers.validate_file_id")
class TestHandleMultilevelEncode:
    """Test handle_multilevel_encode handler (4 tests)"""

    def setup_method(self):
        """Set up mocks for each test"""
        self.mock_encoder = Mock()
        self.mock_encoder.generate_adaptive_skeleton.return_value = "Multi-level skeleton"

        self.context = {
            "compressor": Mock(),
            "multilevel_encoder": self.mock_encoder,
        }

    @pytest.mark.asyncio
    async def test_successful_multilevel_encoding(self, mock_validate_file, mock_validate_token):
        """Test successful multi-level encoding"""
        args = {"file_id": "doc1", "available_tokens": 2000}

        result = await ch.handle_multilevel_encode(self.context, args)

        self.mock_encoder.generate_adaptive_skeleton.assert_called_once_with("doc1", 2000)
        assert result == "Multi-level skeleton"


class TestHandleRecommendFidelity:
    """Test handle_recommend_fidelity handler (8 tests)"""

    @pytest.mark.asyncio
    async def test_successful_recommendation(self):
        """Test successful fidelity recommendation"""
        args = {"use_case": "question_answering", "num_nodes": 3}

        result = await ch.handle_recommend_fidelity({}, args)

        response = json.loads(result)
        assert "recommended_level" in response
        assert "confidence" in response

    @pytest.mark.asyncio
    async def test_recommended_level_is_modulate_region_label(self):
        """#92 (2026-06-12): recommended_level must be the FidelityLevel NAME —
        the label modulate_region's `FidelityLevel[fidelity_str]` accepts.
        Pre-fix it was the enum int, and usage_tip told agents to pass
        fidelity_level='5', which raises KeyError (codex production dogfood)."""
        from src.semantic_compressor import FidelityLevel

        args = {"use_case": "question_answering", "num_nodes": 3}
        response = json.loads(await ch.handle_recommend_fidelity({}, args))

        assert response["recommended_level"] in FidelityLevel.__members__
        assert isinstance(response["recommended_level_value"], int)
        assert f"fidelity_level='{response['recommended_level']}'" in response["usage_tip"]
        for alt in response["alternatives"]:
            assert (
                alt["level"] in FidelityLevel.__members__
            ), f"alternatives must carry labels, got {alt['level']!r}"

    @pytest.mark.asyncio
    async def test_invalid_use_case_raises_error(self):
        """Test that invalid use_case raises error"""
        args = {"use_case": "invalid_case", "num_nodes": 3}

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_recommend_fidelity({}, args)

        assert "Unknown use_case" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_negative_num_nodes_raises_error(self):
        """Test that negative num_nodes raises error"""
        args = {"use_case": "question_answering", "num_nodes": -5}

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_recommend_fidelity({}, args)

        assert "at least 1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_very_high_num_nodes_raises_error(self):
        """Test that very high num_nodes raises error"""
        args = {"use_case": "question_answering", "num_nodes": 5000}

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_recommend_fidelity({}, args)

        assert "very high" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_too_low_token_budget_raises_error(self):
        """Test that unrealistically low token budget raises error"""
        args = {
            "use_case": "question_answering",
            "num_nodes": 3,
            "token_budget": 5,
        }

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_recommend_fidelity({}, args)

        assert "too low" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_query_complexity_raises_error(self):
        """Test that invalid query_complexity raises error"""
        args = {
            "use_case": "question_answering",
            "num_nodes": 3,
            "query_complexity": "ultra_complex",
        }

        with pytest.raises(ValueError) as exc_info:
            await ch.handle_recommend_fidelity({}, args)

        assert "simple" in str(exc_info.value)


class TestValidationHelpers:
    """Targeted tests for validation helper edge cases."""

    def test_validate_node_ids_preserves_file_id_with_non_index_n_segment(self):
        context = {"compressor": Mock()}
        context["compressor"].chunks = {"design_notes_n0": Mock(), "design_notes_n1": Mock()}

        with pytest.raises(ValueError) as exc_info:
            ch.validate_node_ids(["design_notes"], context)

        error_msg = str(exc_info.value)
        assert "file 'design_notes'" in error_msg
        assert "file 'design'" not in error_msg


# ===========================
# Test Count Summary
# ===========================
"""
Total test count: 71 comprehensive tests

Handler Functions:
- TestHandleIngest: 4 tests
- TestHandleReadSkeleton: 2 tests
- TestHandleModulateRegion: 4 tests
- TestHandleSearchSemantic: 2 tests
- TestHandleGetStats: 2 tests
- TestHandleListDocuments: 2 tests
- TestHandleDeleteDocument: 2 tests
- TestHandleAdaptToContextWindow: 2 tests
- TestHandleMultilevelEncode: 1 test
- TestHandleRecommendFidelity: 6 tests

Coverage: Targeting 85%+ of compression_handlers.py (842 lines)
"""
