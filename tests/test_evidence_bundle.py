"""
Tests for Evidence Bundle System

Comprehensive tests for:
- EvidenceBundle creation and integrity
- ContractResult and ContractCheck
- EvidenceStore with chain verification
- Serialization/deserialization
"""

import json
import pytest
import tempfile
import time
from pathlib import Path

from src.evidence_bundle import (
    EvidenceBundle,
    EvidenceStore,
    ContractResult,
    ContractCheck,
    ContractStatus,
    QualityMetrics,
    get_evidence_store,
    reset_evidence_store,
)


class TestContractCheck:
    """Tests for ContractCheck dataclass"""

    def test_create_passed_check(self):
        """Test creating a passed check"""
        check = ContractCheck(
            name="test_check",
            status=ContractStatus.PASSED,
            message="All good"
        )
        assert check.name == "test_check"
        assert check.status == ContractStatus.PASSED
        assert check.message == "All good"

    def test_create_failed_check(self):
        """Test creating a failed check"""
        check = ContractCheck(
            name="validation",
            status=ContractStatus.FAILED,
            message="Validation failed",
            details={"field": "input", "reason": "empty"}
        )
        assert check.status == ContractStatus.FAILED
        assert check.details["field"] == "input"

    def test_to_dict(self):
        """Test serialization to dict"""
        check = ContractCheck(
            name="test",
            status=ContractStatus.PASSED
        )
        d = check.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "passed"

    def test_from_dict(self):
        """Test deserialization from dict"""
        data = {
            "name": "test",
            "status": "failed",
            "message": "Error"
        }
        check = ContractCheck.from_dict(data)
        assert check.name == "test"
        assert check.status == ContractStatus.FAILED


class TestContractResult:
    """Tests for ContractResult"""

    def test_empty_result(self):
        """Test empty contract result"""
        result = ContractResult()
        assert result.overall_passed
        assert result.passed_count == 0
        assert result.failed_count == 0

    def test_add_passing_check(self):
        """Test adding passing checks"""
        result = ContractResult()
        result.add_check("check1", True, "Passed")
        result.add_check("check2", True)

        assert result.overall_passed
        assert result.passed_count == 2
        assert result.failed_count == 0

    def test_add_failing_check(self):
        """Test adding failing check sets overall to failed"""
        result = ContractResult()
        result.add_check("check1", True)
        result.add_check("check2", False, "Failed!")

        assert not result.overall_passed
        assert result.passed_count == 1
        assert result.failed_count == 1

    def test_add_error_check(self):
        """Test adding error check"""
        result = ContractResult()
        result.add_error("error_check", "Something went wrong")

        assert not result.overall_passed
        assert result.checks[0].status == ContractStatus.ERROR

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict"""
        result = ContractResult()
        result.add_check("check1", True)
        result.add_check("check2", False, "Failed")

        d = result.to_dict()
        restored = ContractResult.from_dict(d)

        assert restored.overall_passed == result.overall_passed
        assert len(restored.checks) == len(result.checks)


class TestQualityMetrics:
    """Tests for QualityMetrics"""

    def test_empty_metrics(self):
        """Test empty metrics"""
        metrics = QualityMetrics()
        assert metrics.ssim_score is None
        assert metrics.compression_ratio is None

    def test_partial_metrics(self):
        """Test partial metrics"""
        metrics = QualityMetrics(
            ssim_score=0.92,
            compression_ratio=5.5
        )
        assert metrics.ssim_score == 0.92
        assert metrics.embedding_similarity is None

    def test_custom_metrics(self):
        """Test custom metrics dict"""
        metrics = QualityMetrics(
            custom_metrics={"my_metric": 0.75}
        )
        assert metrics.custom_metrics["my_metric"] == 0.75

    def test_to_dict_filters_none(self):
        """Test that to_dict filters None values"""
        metrics = QualityMetrics(ssim_score=0.9)
        d = metrics.to_dict()
        assert "ssim_score" in d
        assert "embedding_similarity" not in d

    def test_from_dict(self):
        """Test from_dict reconstruction"""
        data = {
            "ssim_score": 0.88,
            "compression_ratio": 7.5,
            "custom_metrics": {"test": 1.0}
        }
        metrics = QualityMetrics.from_dict(data)
        assert metrics.ssim_score == 0.88
        assert metrics.compression_ratio == 7.5
        assert metrics.custom_metrics["test"] == 1.0


class TestEvidenceBundle:
    """Tests for EvidenceBundle"""

    def test_create_bundle(self):
        """Test creating evidence bundle"""
        bundle = EvidenceBundle.create(
            operation="ingest",
            input_data="Hello world",
            output_data="skeleton",
            input_token_count=2,
            output_token_count=1,
            parameters={"fidelity": "BALANCED"}
        )

        assert bundle.operation == "ingest"
        assert bundle.input_token_count == 2
        assert bundle.output_token_count == 1
        assert bundle.bundle_hash is not None
        assert len(bundle.bundle_hash) == 64  # SHA-256

    def test_hash_determinism(self):
        """Test that same inputs produce same hash"""
        params = {
            "operation": "test",
            "input_data": "same input",
            "output_data": "same output",
            "input_token_count": 10,
            "output_token_count": 5,
            "parameters": {"key": "value"}
        }

        bundle1 = EvidenceBundle.create(**params)
        # Create another with same params (but different timestamp)
        # The hash should still be based on content
        time.sleep(0.01)
        bundle2 = EvidenceBundle.create(**params)

        # Note: timestamps differ, so hashes will differ
        # But the hash computation itself is deterministic
        assert bundle1.input_hash == bundle2.input_hash
        assert bundle1.output_hash == bundle2.output_hash

    def test_integrity_verification(self):
        """Test bundle integrity check"""
        bundle = EvidenceBundle.create(
            operation="test",
            input_data="data",
            output_data="result",
            input_token_count=1,
            output_token_count=1,
            parameters={}
        )

        # Should pass integrity check
        assert bundle.verify_integrity()

        # Tamper with content
        original_hash = bundle.bundle_hash
        bundle.input_token_count = 999

        # Integrity check should fail (hash doesn't match)
        assert not bundle.verify_integrity()

    def test_compression_achieved(self):
        """Test compression ratio calculation"""
        bundle = EvidenceBundle.create(
            operation="compress",
            input_data="a" * 1000,
            output_data="summary",
            input_token_count=100,
            output_token_count=10,
            parameters={}
        )

        assert bundle.compression_achieved == 10.0

    def test_contracts_satisfied(self):
        """Test contracts satisfied property"""
        pre = ContractResult()
        pre.add_check("pre1", True)

        post = ContractResult()
        post.add_check("post1", True)

        bundle = EvidenceBundle.create(
            operation="test",
            input_data="in",
            output_data="out",
            input_token_count=1,
            output_token_count=1,
            parameters={},
            preconditions=pre,
            postconditions=post
        )

        assert bundle.contracts_satisfied

        # Now with failed contract
        post2 = ContractResult()
        post2.add_check("post1", False)

        bundle2 = EvidenceBundle.create(
            operation="test",
            input_data="in",
            output_data="out",
            input_token_count=1,
            output_token_count=1,
            parameters={},
            preconditions=pre,
            postconditions=post2
        )

        assert not bundle2.contracts_satisfied

    def test_serialization_roundtrip(self):
        """Test full serialization roundtrip"""
        pre = ContractResult()
        pre.add_check("check1", True)

        post = ContractResult()
        post.add_check("check2", True)

        metrics = QualityMetrics(ssim_score=0.9, compression_ratio=5.0)

        bundle = EvidenceBundle.create(
            operation="test",
            input_data="input text here",
            output_data="output skeleton",
            input_token_count=100,
            output_token_count=20,
            parameters={"fidelity": "BALANCED", "threshold": 0.75},
            quality_metrics=metrics,
            preconditions=pre,
            postconditions=post,
            metadata={"source": "test"}
        )

        # Serialize
        d = bundle.to_dict()
        json_str = json.dumps(d)

        # Deserialize
        d2 = json.loads(json_str)
        restored = EvidenceBundle.from_dict(d2)

        # Verify
        assert restored.operation == bundle.operation
        assert restored.input_hash == bundle.input_hash
        assert restored.bundle_hash == bundle.bundle_hash
        assert restored.quality_metrics.ssim_score == 0.9
        assert restored.metadata["source"] == "test"


class TestEvidenceStore:
    """Tests for EvidenceStore"""

    def test_empty_store(self):
        """Test empty store"""
        store = EvidenceStore()
        assert len(store) == 0
        assert store.chain_valid

    def test_append_bundle(self):
        """Test appending bundles"""
        store = EvidenceStore()

        bundle1 = EvidenceBundle.create(
            operation="op1",
            input_data="in1",
            output_data="out1",
            input_token_count=1,
            output_token_count=1,
            parameters={}
        )

        store.append(bundle1)
        assert len(store) == 1

    def test_chain_linking(self):
        """Test that bundles are chain-linked"""
        store = EvidenceStore()

        bundle1 = EvidenceBundle.create(
            operation="op1",
            input_data="in1",
            output_data="out1",
            input_token_count=1,
            output_token_count=1,
            parameters={}
        )
        store.append(bundle1)

        bundle2 = EvidenceBundle.create(
            operation="op2",
            input_data="in2",
            output_data="out2",
            input_token_count=2,
            output_token_count=1,
            parameters={}
        )
        store.append(bundle2)

        # Bundle 2 should link to bundle 1
        assert bundle2.previous_bundle_hash == bundle1.bundle_hash

    def test_chain_verification(self):
        """Test chain verification"""
        store = EvidenceStore()

        for i in range(5):
            bundle = EvidenceBundle.create(
                operation=f"op{i}",
                input_data=f"in{i}",
                output_data=f"out{i}",
                input_token_count=i+1,
                output_token_count=1,
                parameters={}
            )
            store.append(bundle)

        is_valid, errors = store.verify_chain()
        assert is_valid
        assert len(errors) == 0

    def test_tampered_chain_detection(self):
        """Test detection of tampered chain"""
        store = EvidenceStore()

        bundle1 = EvidenceBundle.create(
            operation="op1",
            input_data="in1",
            output_data="out1",
            input_token_count=1,
            output_token_count=1,
            parameters={}
        )
        store.append(bundle1)

        bundle2 = EvidenceBundle.create(
            operation="op2",
            input_data="in2",
            output_data="out2",
            input_token_count=2,
            output_token_count=1,
            parameters={}
        )
        store.append(bundle2)

        # Tamper with first bundle
        store._bundles[0].input_token_count = 999

        is_valid, errors = store.verify_chain()
        assert not is_valid
        assert len(errors) > 0

    def test_get_by_operation(self):
        """Test filtering by operation"""
        store = EvidenceStore()

        for op in ["ingest", "ingest", "modulate", "search"]:
            bundle = EvidenceBundle.create(
                operation=op,
                input_data="data",
                output_data="result",
                input_token_count=1,
                output_token_count=1,
                parameters={}
            )
            store.append(bundle)

        ingest_bundles = store.get_by_operation("ingest")
        assert len(ingest_bundles) == 2

    def test_get_failed_contracts(self):
        """Test getting bundles with failed contracts"""
        store = EvidenceStore()

        # Good bundle
        good_post = ContractResult()
        good_post.add_check("check", True)

        good = EvidenceBundle.create(
            operation="good",
            input_data="in",
            output_data="out",
            input_token_count=1,
            output_token_count=1,
            parameters={},
            postconditions=good_post
        )
        store.append(good)

        # Bad bundle
        bad_post = ContractResult()
        bad_post.add_check("check", False)

        bad = EvidenceBundle.create(
            operation="bad",
            input_data="in",
            output_data="out",
            input_token_count=1,
            output_token_count=1,
            parameters={},
            postconditions=bad_post
        )
        store.append(bad)

        failed = store.get_failed_contracts()
        assert len(failed) == 1
        assert failed[0].operation == "bad"

    def test_statistics(self):
        """Test statistics generation"""
        store = EvidenceStore()

        for i in range(10):
            bundle = EvidenceBundle.create(
                operation="ingest" if i < 7 else "search",
                input_data="data" * (i + 1),
                output_data="result",
                input_token_count=(i + 1) * 10,
                output_token_count=10,
                parameters={}
            )
            store.append(bundle)

        stats = store.get_statistics()
        assert stats["total_bundles"] == 10
        assert "ingest" in stats["operations"]
        assert stats["operations"]["ingest"]["count"] == 7

    def test_persistence(self):
        """Test persistence to disk"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evidence.json"

            # Create and populate store
            store1 = EvidenceStore(storage_path=path)
            for i in range(3):
                bundle = EvidenceBundle.create(
                    operation=f"op{i}",
                    input_data=f"in{i}",
                    output_data=f"out{i}",
                    input_token_count=i+1,
                    output_token_count=1,
                    parameters={}
                )
                store1.append(bundle)

            # Load from disk
            store2 = EvidenceStore(storage_path=path)
            assert len(store2) == 3

            # Verify chain
            is_valid, errors = store2.verify_chain()
            assert is_valid

    def test_iteration(self):
        """Test iterating over store"""
        store = EvidenceStore()

        for i in range(3):
            bundle = EvidenceBundle.create(
                operation=f"op{i}",
                input_data=f"in{i}",
                output_data=f"out{i}",
                input_token_count=1,
                output_token_count=1,
                parameters={}
            )
            store.append(bundle)

        operations = [b.operation for b in store]
        assert operations == ["op0", "op1", "op2"]

    def test_indexing(self):
        """Test indexing store"""
        store = EvidenceStore()

        bundle = EvidenceBundle.create(
            operation="test",
            input_data="in",
            output_data="out",
            input_token_count=1,
            output_token_count=1,
            parameters={}
        )
        store.append(bundle)

        assert store[0].operation == "test"


class TestGlobalStore:
    """Tests for global store singleton"""

    def setup_method(self):
        """Reset global store before each test"""
        reset_evidence_store()

    def test_get_global_store(self):
        """Test getting global store"""
        store1 = get_evidence_store()
        store2 = get_evidence_store()
        assert store1 is store2

    def test_reset_global_store(self):
        """Test resetting global store"""
        store1 = get_evidence_store()
        bundle = EvidenceBundle.create(
            operation="test",
            input_data="in",
            output_data="out",
            input_token_count=1,
            output_token_count=1,
            parameters={}
        )
        store1.append(bundle)

        reset_evidence_store()

        store2 = get_evidence_store()
        assert len(store2) == 0
