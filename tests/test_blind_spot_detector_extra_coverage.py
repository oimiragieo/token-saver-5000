"""blind spot detector — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np


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
        "ace_framework": MagicMock(),
        "focus_manager": MagicMock(),
    }
    ctx["resource_manager"].unregister_document_async = AsyncMock()
    ctx["version_manager"].delete_versions_async = AsyncMock()
    ctx.update(overrides)
    return ctx


def _make_semantic_node(text="test", importance=0.5, embedding=None):
    node = MagicMock()
    node.text = text
    node.importance = importance
    node.embedding = embedding if embedding is not None else np.random.rand(384).astype(np.float32)
    node.metadata = {"tokens": 10, "position": 0, "entities": []}
    return node


def _make_code_chunk(
    name="func", chunk_type="function", code="def f(): pass", docstring="", start_line=1, end_line=5
):
    chunk = MagicMock()
    chunk.name = name
    chunk.chunk_type = chunk_type
    chunk.code = code
    chunk.docstring = docstring
    chunk.start_line = start_line
    chunk.end_line = end_line
    return chunk


class TestBlindSpotDetector:
    """Cover blind spot detector urgency and detection paths."""

    def _make_detector(self):
        from src.blind_spot_detector import BlindSpotDetector

        compressor = MagicMock()
        compressor.model.encode.return_value = [np.random.rand(384)]
        detector = BlindSpotDetector(compressor)
        return detector, compressor

    def test_urgency_levels(self):
        """Cover lines 90-95 - all urgency levels."""
        detector, _ = self._make_detector()
        # critical: score >= 0.6
        level, score = detector._calculate_urgency(0.8, 0.8)
        assert level == "critical"
        # high: score >= 0.4
        level, score = detector._calculate_urgency(0.65, 0.65)
        assert level == "high"
        # medium: score >= 0.25
        level, score = detector._calculate_urgency(0.5, 0.55)
        assert level == "medium"
        # low: score < 0.25
        level, score = detector._calculate_urgency(0.3, 0.3)
        assert level == "low"

    def test_analyze_response_with_retrieved_relevant(self):
        """Cover lines 145, 149 - relevant content was retrieved."""
        import networkx as nx

        detector, compressor = self._make_detector()
        graph = nx.Graph()
        graph.add_node("doc_n0")
        graph.add_node("doc_n1")
        compressor.graphs = {"doc": graph}

        node0 = _make_semantic_node("relevant text", importance=0.3)
        node1 = _make_semantic_node("other text", importance=0.2)
        compressor.chunks = {"doc_n0": node0, "doc_n1": node1}

        # High similarity with retrieved node
        with patch("src.blind_spot_detector.cosine_similarity", return_value=[[0.85]]):
            report = detector.analyze_response("response text", "doc", ["doc_n0", "doc_n1"])
            assert report.total_blind_spots == 0

    def test_analyze_response_high_spots(self):
        """Cover lines 187-193 - high urgency blind spots."""
        import networkx as nx

        detector, compressor = self._make_detector()
        graph = nx.Graph()
        for i in range(5):
            graph.add_node(f"doc_n{i}")
        compressor.graphs = {"doc": graph}

        nodes = {}
        for i in range(5):
            nodes[f"doc_n{i}"] = _make_semantic_node(f"Node {i} content", importance=0.7)
        compressor.chunks = nodes
        compressor._generate_summary.return_value = "summary"

        sim_values = [[[0.75]]] * 5
        with patch("src.blind_spot_detector.cosine_similarity", side_effect=sim_values):
            report = detector.analyze_response("query", "doc", [])
            assert report.total_blind_spots > 0

    def test_validate_response_critical(self):
        """Cover line 316 - validate finds critical blind spots."""
        detector, compressor = self._make_detector()
        mock_report = MagicMock()
        mock_report.critical_blind_spots = 2
        mock_report.auto_inject = ["n0"]
        with patch.object(detector, "analyze_response", return_value=mock_report):
            valid, msg = detector.validate_response_fidelity("response", "doc", [])
            assert not valid
            assert "incomplete" in msg

    def test_hallucination_detector_no_graph(self):
        """Cover line 356 - hallucination detector with no graph."""
        from src.blind_spot_detector import HaloEffectDetector

        compressor = MagicMock()
        compressor.model.encode.return_value = [np.random.rand(384)]
        compressor.graphs = {}
        detector = HaloEffectDetector(compressor)
        is_hall, claims = detector.detect_hallucination("response", "doc")
        assert is_hall is False
        assert claims == []
