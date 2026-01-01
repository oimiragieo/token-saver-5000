"""
Tests for Compression Verifier

Tests for:
- CompressionContract preconditions and postconditions
- CompressionVerifier verification flow
- BatchVerifier for multiple operations
- ContractViolation tracking
"""

import pytest

from src.compression_verifier import (
    CompressionContract,
    CompressionVerifier,
    BatchVerifier,
    VerificationResult,
    ContractViolation,
    ContractType,
)
from src.evidence_bundle import (
    EvidenceBundle,
    ContractResult,
    QualityMetrics,
)


class TestCompressionContractPreconditions:
    """Tests for precondition checking"""

    def test_valid_input(self):
        """Test valid input passes all preconditions"""
        result = CompressionContract.check_preconditions(
            document="This is a valid document with multiple words.",
            fidelity_level="BALANCED"
        )

        assert result.overall_passed
        assert result.passed_count >= 5

    def test_empty_document(self):
        """Test empty document fails"""
        result = CompressionContract.check_preconditions(
            document="",
            fidelity_level="BALANCED"
        )

        assert not result.overall_passed
        # Should fail non_empty_document check
        failed_checks = [c for c in result.checks if c.status.value == "failed"]
        assert any("empty" in c.name.lower() for c in failed_checks)

    def test_invalid_fidelity(self):
        """Test invalid fidelity level fails"""
        result = CompressionContract.check_preconditions(
            document="Valid document",
            fidelity_level="INVALID_LEVEL"
        )

        assert not result.overall_passed
        failed_checks = [c for c in result.checks if c.status.value == "failed"]
        assert any("fidelity" in c.name.lower() for c in failed_checks)

    def test_null_bytes(self):
        """Test document with null bytes fails"""
        result = CompressionContract.check_preconditions(
            document="Document with\x00null byte",
            fidelity_level="BALANCED"
        )

        assert not result.overall_passed

    def test_unsafe_file_path(self):
        """Test unsafe file path detection"""
        result = CompressionContract.check_preconditions(
            document="Valid document",
            fidelity_level="BALANCED",
            file_path="../../../etc/passwd"
        )

        assert not result.overall_passed

    def test_valid_fidelity_levels(self):
        """Test all valid fidelity levels"""
        for level in ["ABSTRACT", "OUTLINE", "STRUCTURE", "DETAILED", "RAW", "BALANCED"]:
            result = CompressionContract.check_preconditions(
                document="Test document",
                fidelity_level=level
            )
            assert result.overall_passed, f"Failed for fidelity level: {level}"


class TestCompressionContractPostconditions:
    """Tests for postcondition checking"""

    def test_valid_output(self):
        """Test valid output passes postconditions"""
        result = CompressionContract.check_postconditions(
            skeleton_text="Compressed skeleton output",
            node_map={"node1": "description"},
            original_tokens=100,
            skeleton_tokens=20,
            fidelity_level="BALANCED",
            compression_ratio=5.0
        )

        assert result.overall_passed

    def test_empty_skeleton(self):
        """Test empty skeleton fails"""
        result = CompressionContract.check_postconditions(
            skeleton_text="",
            node_map={"node1": "desc"},
            original_tokens=100,
            skeleton_tokens=0,
            fidelity_level="BALANCED",
            compression_ratio=0.0
        )

        assert not result.overall_passed

    def test_skeleton_token_limit(self):
        """Test skeleton exceeding token limit"""
        result = CompressionContract.check_postconditions(
            skeleton_text="x" * 1000,
            node_map={"node1": "desc"},
            original_tokens=100,
            skeleton_tokens=60000,  # Over 50k limit
            fidelity_level="BALANCED",
            compression_ratio=0.5
        )

        assert not result.overall_passed

    def test_low_compression_ratio(self):
        """Test low compression ratio fails"""
        result = CompressionContract.check_postconditions(
            skeleton_text="Almost same as original",
            node_map={"node1": "desc"},
            original_tokens=100,
            skeleton_tokens=90,
            fidelity_level="ABSTRACT",  # Expects high compression
            compression_ratio=1.1
        )

        # ABSTRACT expects 10x, achieved 1.1x
        assert not result.overall_passed

    def test_raw_fidelity_no_compression(self):
        """Test RAW fidelity doesn't require compression"""
        result = CompressionContract.check_postconditions(
            skeleton_text="Full original text here",
            node_map={},
            original_tokens=100,
            skeleton_tokens=100,  # No reduction
            fidelity_level="RAW",
            compression_ratio=1.0
        )

        # RAW mode should pass even with no compression
        assert result.overall_passed


class TestCompressionContractInvariants:
    """Tests for composition invariants"""

    def test_valid_composition(self):
        """Test valid graph composition"""
        result = CompressionContract.check_composition_invariants(
            node_ids=["n1", "n2", "n3"],
            edges=[("n1", "n2"), ("n2", "n3")],
            skeleton_text="[n1] connects to [n2]"
        )

        assert result.overall_passed

    def test_edge_to_missing_node(self):
        """Test edge pointing to non-existent node"""
        result = CompressionContract.check_composition_invariants(
            node_ids=["n1", "n2"],
            edges=[("n1", "n3")],  # n3 doesn't exist
            skeleton_text=""
        )

        assert not result.overall_passed

    def test_duplicate_edges(self):
        """Test duplicate edge detection"""
        result = CompressionContract.check_composition_invariants(
            node_ids=["n1", "n2"],
            edges=[("n1", "n2"), ("n1", "n2")],  # Duplicate
            skeleton_text=""
        )

        assert not result.overall_passed


class TestCompressionVerifier:
    """Tests for CompressionVerifier"""

    def test_verify_valid_operation(self):
        """Test verifying valid compression operation"""
        verifier = CompressionVerifier()

        result = verifier.verify_compression_operation(
            document="This is a test document with multiple sentences. "
                     "It contains enough content to compress meaningfully.",
            skeleton_text="Test document with sentences.",
            node_map={"node1": "test content"},
            original_tokens=20,
            skeleton_tokens=5,
            fidelity_level="BALANCED",
            compression_ratio=4.0
        )

        assert result.verified
        assert result.all_contracts_passed

    def test_verify_invalid_operation(self):
        """Test verifying invalid operation"""
        verifier = CompressionVerifier()

        result = verifier.verify_compression_operation(
            document="",  # Empty - fails precondition
            skeleton_text="output",
            node_map={},
            original_tokens=0,
            skeleton_tokens=1,
            fidelity_level="BALANCED",
            compression_ratio=0.0
        )

        assert not result.verified
        assert len(result.violations) > 0

    def test_verify_bundle(self):
        """Test verifying evidence bundle"""
        verifier = CompressionVerifier()

        # Create bundle with passing contracts
        pre = ContractResult()
        pre.add_check("check1", True)

        post = ContractResult()
        post.add_check("check2", True)

        bundle = EvidenceBundle.create(
            operation="ingest",
            input_data="test document",
            output_data="skeleton",
            input_token_count=10,
            output_token_count=5,
            parameters={"fidelity": "BALANCED"},
            preconditions=pre,
            postconditions=post
        )

        result = verifier.verify_bundle(bundle)

        assert result.verified

    def test_verify_tampered_bundle(self):
        """Test detecting tampered bundle"""
        verifier = CompressionVerifier()

        bundle = EvidenceBundle.create(
            operation="ingest",
            input_data="test",
            output_data="out",
            input_token_count=5,
            output_token_count=2,
            parameters={}
        )

        # Tamper with bundle
        bundle.input_token_count = 999
        # Hash no longer matches

        result = verifier.verify_bundle(bundle)

        assert not result.verified
        assert any(v.contract_name == "bundle_integrity" for v in result.violations)

    def test_statistics(self):
        """Test verification statistics"""
        verifier = CompressionVerifier()

        # Run some verifications
        for i in range(5):
            verifier.verify_compression_operation(
                document=f"Document {i}",
                skeleton_text=f"Doc {i}",
                node_map={},
                original_tokens=10,
                skeleton_tokens=5,
                fidelity_level="BALANCED",
                compression_ratio=2.0
            )

        stats = verifier.get_statistics()
        assert stats["total_verifications"] == 5

    def test_reset_statistics(self):
        """Test resetting statistics"""
        verifier = CompressionVerifier()

        verifier.verify_compression_operation(
            document="Test",
            skeleton_text="T",
            node_map={},
            original_tokens=5,
            skeleton_tokens=1,
            fidelity_level="BALANCED",
            compression_ratio=5.0
        )

        verifier.reset_statistics()
        stats = verifier.get_statistics()

        assert stats["total_verifications"] == 0


class TestBatchVerifier:
    """Tests for BatchVerifier"""

    def test_verify_multiple_bundles(self):
        """Test verifying multiple bundles"""
        batch_verifier = BatchVerifier()

        bundles = []
        for i in range(5):
            pre = ContractResult()
            pre.add_check("check", True)
            post = ContractResult()
            post.add_check("check", True)

            bundle = EvidenceBundle.create(
                operation=f"op{i}",
                input_data=f"input{i}",
                output_data=f"output{i}",
                input_token_count=10,
                output_token_count=5,
                parameters={},
                preconditions=pre,
                postconditions=post
            )
            bundles.append(bundle)

        results, stats = batch_verifier.verify_bundles(bundles)

        assert len(results) == 5
        assert stats["count"] == 5
        assert stats["passed"] == 5

    def test_aggregate_statistics(self):
        """Test statistics aggregation"""
        batch_verifier = BatchVerifier()

        # Mix of passing and failing
        bundles = []
        for i in range(4):
            pre = ContractResult()
            pre.add_check("check", True)

            post = ContractResult()
            post.add_check("check", i < 3)  # Last one fails

            bundle = EvidenceBundle.create(
                operation="op",
                input_data="in",
                output_data="out",
                input_token_count=10,
                output_token_count=5,
                parameters={},
                preconditions=pre,
                postconditions=post
            )
            bundles.append(bundle)

        results, stats = batch_verifier.verify_bundles(bundles)

        assert stats["passed"] == 3
        assert stats["failed"] == 1
        assert stats["pass_rate"] == 0.75


class TestVerificationResult:
    """Tests for VerificationResult"""

    def test_to_dict(self):
        """Test serialization"""
        result = VerificationResult(
            verified=True,
            preconditions=ContractResult(),
            postconditions=ContractResult()
        )

        d = result.to_dict()
        assert d["verified"] is True
        assert "all_contracts_passed" in d
        assert "preconditions" in d

    def test_all_contracts_passed(self):
        """Test all_contracts_passed property"""
        pre = ContractResult()
        pre.add_check("check", True)

        post = ContractResult()
        post.add_check("check", True)

        result = VerificationResult(
            verified=True,
            preconditions=pre,
            postconditions=post
        )

        assert result.all_contracts_passed

    def test_violations_tracking(self):
        """Test violation tracking"""
        result = VerificationResult(verified=False)
        result.violations.append(ContractViolation(
            contract_name="test",
            contract_type=ContractType.PRECONDITION,
            expected="pass",
            actual="fail"
        ))

        assert len(result.violations) == 1
        assert result.violations[0].contract_type == ContractType.PRECONDITION


class TestContractViolation:
    """Tests for ContractViolation"""

    def test_to_dict(self):
        """Test serialization"""
        violation = ContractViolation(
            contract_name="empty_check",
            contract_type=ContractType.PRECONDITION,
            expected="non-empty",
            actual="empty",
            severity="error",
            remediation="Provide valid input"
        )

        d = violation.to_dict()
        assert d["contract_name"] == "empty_check"
        assert d["contract_type"] == "precondition"
        assert d["severity"] == "error"
        assert d["remediation"] == "Provide valid input"
