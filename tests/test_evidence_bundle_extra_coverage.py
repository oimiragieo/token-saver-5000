"""evidence bundle — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest


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


class TestEvidenceBundle:
    def test_contract_check_from_dict(self):
        from src.evidence_bundle import ContractCheck, ContractStatus

        data = {"name": "test", "status": "passed", "message": "ok"}
        check = ContractCheck.from_dict(data)
        assert check.status == ContractStatus.PASSED

    def test_contract_result_add_error(self):
        from src.evidence_bundle import ContractResult

        cr = ContractResult()
        cr.add_error("check1", "something broke")
        assert not cr.overall_passed
        assert cr.failed_count == 0  # error != failed

    def test_contract_result_roundtrip(self):
        from src.evidence_bundle import ContractResult

        cr = ContractResult()
        cr.add_check("test1", True, "ok")
        cr.add_check("test2", False, "bad")
        d = cr.to_dict()
        cr2 = ContractResult.from_dict(d)
        assert cr2.passed_count == 1
        assert cr2.failed_count == 1

    def test_quality_metrics_roundtrip(self):
        from src.evidence_bundle import QualityMetrics

        qm = QualityMetrics(ssim_score=0.9, custom_metrics={"extra": 1.0})
        d = qm.to_dict()
        qm2 = QualityMetrics.from_dict(d)
        assert qm2.ssim_score == 0.9


class TestEvidenceBundle_boost4b:
    """Cover evidence bundle edge cases."""

    def test_quality_metrics_to_dict_partial(self):
        """Cover lines 162, 166, 168 - partial metrics."""
        from src.evidence_bundle import QualityMetrics

        metrics = QualityMetrics(
            ssim_score=None,
            embedding_similarity=0.9,
            compression_ratio=None,
            token_reduction=0.5,
            structure_score=None,
            custom_metrics={"my_metric": 0.7},
        )
        d = metrics.to_dict()
        assert "embedding_similarity" in d
        assert "token_reduction" in d
        assert "custom_metrics" in d
        assert "ssim_score" not in d

    def test_compression_achieved_zero_input(self):
        """Cover line 261 - zero input tokens."""
        from src.evidence_bundle import EvidenceBundle

        bundle = MagicMock(spec=EvidenceBundle)
        bundle.input_token_count = 0
        bundle.output_token_count = 50
        # Call the property directly
        result = EvidenceBundle.compression_achieved.fget(bundle)
        assert result == 0.0

    def test_store_chain_integrity_violation(self):
        """Cover line 429 - chain integrity violation."""
        from src.evidence_bundle import EvidenceStore, EvidenceBundle

        store = EvidenceStore()
        b1 = MagicMock(spec=EvidenceBundle)
        b1.bundle_hash = "hash1"
        b1.previous_bundle_hash = None
        b1._compute_hash = MagicMock(return_value="hash1")
        store._bundles = [b1]

        b2 = MagicMock(spec=EvidenceBundle)
        b2.previous_bundle_hash = "wrong_hash"
        with pytest.raises(ValueError, match="Chain integrity"):
            store.append(b2)

    def test_verify_chain_with_broken_link(self):
        """Cover lines 457, 473-478."""
        from src.evidence_bundle import EvidenceStore

        store = EvidenceStore()
        b1 = MagicMock()
        b1.verify_integrity.return_value = True
        b1.bundle_hash = "hash1"
        b1.bundle_id = "b1"
        b1.previous_bundle_hash = None

        b2 = MagicMock()
        b2.verify_integrity.return_value = True
        b2.bundle_hash = "hash2"
        b2.bundle_id = "b2"
        b2.previous_bundle_hash = "wrong_hash"

        store._bundles = [b1, b2]
        valid, errors = store.verify_chain()
        assert not valid
        assert len(errors) > 0

    def test_get_by_time_range(self):
        """Cover lines 473-478 - time range filtering."""
        from src.evidence_bundle import EvidenceStore

        store = EvidenceStore()
        b1 = MagicMock()
        b1.timestamp = 100.0
        b2 = MagicMock()
        b2.timestamp = 200.0
        b3 = MagicMock()
        b3.timestamp = 300.0
        store._bundles = [b1, b2, b3]

        result = store.get_by_time_range(start_time=150.0, end_time=250.0)
        assert len(result) == 1
        assert result[0].timestamp == 200.0

    def test_store_save_and_load(self, tmp_path):
        """Cover lines 526, 536, 547-550."""
        from src.evidence_bundle import EvidenceStore

        store = EvidenceStore(storage_path=tmp_path / "evidence.json")
        store._bundles = []
        store._save()
        assert (tmp_path / "evidence.json").exists()

    def test_store_load_failure(self, tmp_path):
        """Cover lines 547-550 - load failure."""
        from src.evidence_bundle import EvidenceStore

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("invalid json{{{")
        store = EvidenceStore(storage_path=bad_file)
        store._load()
        assert store._bundles == []
        assert store._chain_valid is False

    def test_clear_with_storage(self, tmp_path):
        """Cover line 557 - clear with storage path."""
        from src.evidence_bundle import EvidenceStore

        store_file = tmp_path / "evidence.json"
        store_file.write_text("{}")
        store = EvidenceStore(storage_path=store_file)
        store._bundles = [MagicMock()]
        store.clear()
        assert store._bundles == []
        assert not store_file.exists()
