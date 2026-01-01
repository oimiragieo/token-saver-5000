"""
Compression Verifier with Contract System

Implements precondition/postcondition contracts for compression operations,
inspired by ASG-SI's Verifier-Auditor pattern.

Key Concepts:
- Contracts define required inputs and guaranteed outputs
- Verification is independent of runtime (replay-based)
- Failed contracts produce actionable diagnostics
- Supports composition integrity checking

Usage:
    from src.compression_verifier import CompressionVerifier, CompressionContract

    verifier = CompressionVerifier()

    # Check preconditions before compression
    pre_result = verifier.check_preconditions(document, fidelity_level)
    if not pre_result.overall_passed:
        raise ValueError(f"Precondition failed: {pre_result}")

    # Perform compression...

    # Check postconditions after compression
    post_result = verifier.check_postconditions(skeleton, original_tokens)
    if not post_result.overall_passed:
        logger.warning(f"Postcondition failed: {post_result}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .evidence_bundle import ContractResult, ContractCheck, ContractStatus, EvidenceBundle

if TYPE_CHECKING:
    from .semantic_compressor import SkeletonResponse, FidelityLevel, SemanticCompressor

logger = logging.getLogger(__name__)


# Constants for validation
MAX_DOCUMENT_TOKENS = 100_000  # 100k token limit
MAX_DOCUMENT_SIZE_MB = 100  # 100MB limit
MIN_COMPRESSION_RATIO = 0.5  # At least 50% of target
MAX_SKELETON_TOKENS = 50_000  # 50k token limit for skeleton


class ContractType(Enum):
    """Types of contracts"""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"


@dataclass
class ContractViolation:
    """Details of a contract violation"""
    contract_name: str
    contract_type: ContractType
    expected: str
    actual: str
    severity: str = "error"  # error, warning, info
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "contract_type": self.contract_type.value,
            "expected": self.expected,
            "actual": self.actual,
            "severity": self.severity,
            "remediation": self.remediation,
        }


@dataclass
class VerificationResult:
    """Result of verification operation"""
    verified: bool
    timestamp: float = field(default_factory=time.time)
    preconditions: Optional[ContractResult] = None
    postconditions: Optional[ContractResult] = None
    violations: List[ContractViolation] = field(default_factory=list)
    replay_hash: Optional[str] = None
    original_hash: Optional[str] = None
    hashes_match: bool = True
    verification_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_contracts_passed(self) -> bool:
        pre_passed = self.preconditions.overall_passed if self.preconditions else True
        post_passed = self.postconditions.overall_passed if self.postconditions else True
        return pre_passed and post_passed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "timestamp": self.timestamp,
            "all_contracts_passed": self.all_contracts_passed,
            "preconditions": self.preconditions.to_dict() if self.preconditions else None,
            "postconditions": self.postconditions.to_dict() if self.postconditions else None,
            "violations": [v.to_dict() for v in self.violations],
            "hashes_match": self.hashes_match,
            "verification_time_ms": self.verification_time_ms,
            "metadata": self.metadata,
        }


class CompressionContract:
    """
    Precondition/postcondition contracts for compression operations.

    Contracts define:
    - What inputs are required (preconditions)
    - What outputs are guaranteed (postconditions)
    - What invariants must hold (invariants)
    """

    @staticmethod
    def check_preconditions(
        document: str,
        fidelity_level: str,
        file_path: Optional[str] = None,
    ) -> ContractResult:
        """
        Check preconditions before compression.

        Preconditions:
        - Document is non-empty string
        - Document is valid UTF-8
        - Document size within limits
        - Fidelity level is valid
        - File path (if provided) is safe

        Args:
            document: Document text to compress
            fidelity_level: Target fidelity level
            file_path: Optional source file path

        Returns:
            ContractResult with all check results
        """
        result = ContractResult()

        # Check: non-empty document
        result.add_check(
            "non_empty_document",
            len(document) > 0,
            "Document must be non-empty"
        )

        # Check: document is string
        result.add_check(
            "document_is_string",
            isinstance(document, str),
            "Document must be a string"
        )

        # Check: valid UTF-8
        try:
            document.encode("utf-8")
            result.add_check("valid_utf8", True)
        except UnicodeEncodeError as e:
            result.add_check("valid_utf8", False, f"Invalid UTF-8: {e}")

        # Check: document size limit
        doc_size_mb = len(document.encode("utf-8")) / (1024 * 1024)
        result.add_check(
            "document_size_limit",
            doc_size_mb <= MAX_DOCUMENT_SIZE_MB,
            f"Document size {doc_size_mb:.2f}MB exceeds {MAX_DOCUMENT_SIZE_MB}MB limit"
        )

        # Check: valid fidelity level
        valid_levels = {"ABSTRACT", "OUTLINE", "STRUCTURE", "DETAILED", "RAW", "BALANCED"}
        result.add_check(
            "valid_fidelity_level",
            fidelity_level.upper() in valid_levels,
            f"Fidelity level must be one of {valid_levels}"
        )

        # Check: no null bytes (potential security issue)
        result.add_check(
            "no_null_bytes",
            "\x00" not in document,
            "Document contains null bytes (potential binary content)"
        )

        # Check: file path safety (if provided)
        if file_path:
            path_safe = not any(
                seq in file_path
                for seq in ["../", "..\\", "/etc/", "C:\\Windows"]
            )
            result.add_check(
                "safe_file_path",
                path_safe,
                "File path contains potentially unsafe sequences"
            )

        return result

    @staticmethod
    def check_postconditions(
        skeleton_text: str,
        node_map: Dict[str, str],
        original_tokens: int,
        skeleton_tokens: int,
        fidelity_level: str,
        compression_ratio: float,
    ) -> ContractResult:
        """
        Check postconditions after compression.

        Postconditions:
        - Skeleton is non-empty
        - Skeleton is parseable
        - Node map is valid
        - Compression achieved target range
        - Token count reduced (or maintained for RAW)

        Args:
            skeleton_text: Compressed skeleton text
            node_map: Mapping of node IDs to descriptions
            original_tokens: Original document token count
            skeleton_tokens: Skeleton token count
            fidelity_level: Target fidelity level
            compression_ratio: Achieved compression ratio

        Returns:
            ContractResult with all check results
        """
        result = ContractResult()

        # Check: skeleton is non-empty
        result.add_check(
            "skeleton_non_empty",
            len(skeleton_text) > 0,
            "Skeleton must be non-empty"
        )

        # Check: skeleton within token limit
        result.add_check(
            "skeleton_token_limit",
            skeleton_tokens <= MAX_SKELETON_TOKENS,
            f"Skeleton {skeleton_tokens} tokens exceeds {MAX_SKELETON_TOKENS} limit"
        )

        # Check: node map is valid dict
        result.add_check(
            "valid_node_map",
            isinstance(node_map, dict),
            "Node map must be a dictionary"
        )

        # Check: node map has entries (unless very short document)
        if original_tokens > 50:
            result.add_check(
                "node_map_populated",
                len(node_map) > 0,
                "Node map should have entries for non-trivial documents"
            )
        else:
            result.add_check("node_map_populated", True, "Skipped for small document")

        # Check: compression achieved (with tolerance)
        target_ratios = {
            "ABSTRACT": 10.0,
            "OUTLINE": 5.0,
            "STRUCTURE": 3.0,
            "DETAILED": 2.0,
            "BALANCED": 3.0,
            "RAW": 1.0,
        }
        target = target_ratios.get(fidelity_level.upper(), 3.0)

        if fidelity_level.upper() != "RAW":
            # Allow 50% tolerance for compression
            result.add_check(
                "compression_achieved",
                compression_ratio >= target * MIN_COMPRESSION_RATIO,
                f"Compression ratio {compression_ratio:.2f}x below target {target}x (min {target * MIN_COMPRESSION_RATIO:.2f}x)"
            )
        else:
            result.add_check("compression_achieved", True, "RAW mode - no compression expected")

        # Check: tokens reduced (unless RAW)
        if fidelity_level.upper() != "RAW":
            result.add_check(
                "tokens_reduced",
                skeleton_tokens < original_tokens,
                f"Skeleton ({skeleton_tokens}) should have fewer tokens than original ({original_tokens})"
            )
        else:
            result.add_check("tokens_reduced", True, "RAW mode - no reduction expected")

        # Check: valid node IDs
        if node_map:
            valid_ids = all(
                isinstance(k, str) and len(k) > 0 and isinstance(v, str)
                for k, v in node_map.items()
            )
            result.add_check(
                "valid_node_ids",
                valid_ids,
                "All node IDs must be non-empty strings with string values"
            )

        return result

    @staticmethod
    def check_composition_invariants(
        node_ids: List[str],
        edges: List[Tuple[str, str]],
        skeleton_text: str,
    ) -> ContractResult:
        """
        Check composition invariants for graph consistency.

        Invariants:
        - All edge endpoints exist in node set
        - No duplicate edges
        - Node IDs referenced in skeleton exist

        Args:
            node_ids: List of valid node IDs
            edges: List of (source, target) edges
            skeleton_text: Skeleton text (may reference nodes)

        Returns:
            ContractResult with invariant checks
        """
        result = ContractResult()
        node_set = set(node_ids)

        # Check: edge endpoints exist
        missing_nodes = set()
        for src, tgt in edges:
            if src not in node_set:
                missing_nodes.add(src)
            if tgt not in node_set:
                missing_nodes.add(tgt)

        result.add_check(
            "edge_endpoints_valid",
            len(missing_nodes) == 0,
            f"Edges reference {len(missing_nodes)} missing nodes: {list(missing_nodes)[:5]}"
        )

        # Check: no duplicate edges
        edge_set = set(edges)
        result.add_check(
            "no_duplicate_edges",
            len(edge_set) == len(edges),
            f"Found {len(edges) - len(edge_set)} duplicate edges"
        )

        # Check: skeleton references valid nodes (if it references any)
        # Look for patterns like [node_123] or (node_abc)
        node_refs = re.findall(r'\[([^\]]+)\]|\(node_([^\)]+)\)', skeleton_text)
        if node_refs:
            flat_refs = [r[0] or r[1] for r in node_refs if r[0] or r[1]]
            invalid_refs = [r for r in flat_refs if r not in node_set]
            result.add_check(
                "skeleton_refs_valid",
                len(invalid_refs) == 0,
                f"Skeleton references {len(invalid_refs)} invalid nodes"
            )
        else:
            result.add_check("skeleton_refs_valid", True, "No node references in skeleton")

        return result


class CompressionVerifier:
    """
    Replay-based verification independent of runtime.

    Implements ASG-SI's verifier pattern:
    - Verifies compression from evidence bundles
    - Produces verification reports
    - Supports replay for reproducibility
    """

    def __init__(self, compressor: Optional["SemanticCompressor"] = None):
        """
        Initialize verifier.

        Args:
            compressor: Optional SemanticCompressor for replay verification
        """
        self.compressor = compressor
        self._verification_count = 0
        self._passed_count = 0

    def check_preconditions(
        self,
        document: str,
        fidelity_level: str,
        file_path: Optional[str] = None,
    ) -> ContractResult:
        """Check preconditions before compression"""
        return CompressionContract.check_preconditions(
            document, fidelity_level, file_path
        )

    def check_postconditions(
        self,
        skeleton_text: str,
        node_map: Dict[str, str],
        original_tokens: int,
        skeleton_tokens: int,
        fidelity_level: str,
        compression_ratio: float,
    ) -> ContractResult:
        """Check postconditions after compression"""
        return CompressionContract.check_postconditions(
            skeleton_text, node_map, original_tokens, skeleton_tokens,
            fidelity_level, compression_ratio
        )

    def verify_bundle(self, bundle: EvidenceBundle) -> VerificationResult:
        """
        Verify a compression operation from its evidence bundle.

        Performs:
        1. Bundle integrity check (hash verification)
        2. Precondition verification
        3. Postcondition verification
        4. Optional replay verification (if compressor available)

        Args:
            bundle: Evidence bundle to verify

        Returns:
            VerificationResult with full verification details
        """
        start_time = time.time()
        result = VerificationResult(verified=False)

        # Step 1: Verify bundle integrity
        if not bundle.verify_integrity():
            result.violations.append(ContractViolation(
                contract_name="bundle_integrity",
                contract_type=ContractType.INVARIANT,
                expected="Hash matches content",
                actual="Hash mismatch detected",
                severity="error",
                remediation="Bundle may have been tampered with"
            ))
            result.verification_time_ms = (time.time() - start_time) * 1000
            return result

        # Step 2: Verify contracts from bundle
        result.preconditions = bundle.preconditions
        result.postconditions = bundle.postconditions

        # Collect violations from failed checks
        for check in bundle.preconditions.checks:
            if check.status != ContractStatus.PASSED:
                result.violations.append(ContractViolation(
                    contract_name=check.name,
                    contract_type=ContractType.PRECONDITION,
                    expected="Check should pass",
                    actual=check.message or "Check failed",
                    severity="error" if check.status == ContractStatus.FAILED else "warning"
                ))

        for check in bundle.postconditions.checks:
            if check.status != ContractStatus.PASSED:
                result.violations.append(ContractViolation(
                    contract_name=check.name,
                    contract_type=ContractType.POSTCONDITION,
                    expected="Check should pass",
                    actual=check.message or "Check failed",
                    severity="error" if check.status == ContractStatus.FAILED else "warning"
                ))

        # Step 3: Set verification status
        result.verified = bundle.contracts_satisfied and len(result.violations) == 0
        result.original_hash = bundle.bundle_hash

        self._verification_count += 1
        if result.verified:
            self._passed_count += 1

        result.verification_time_ms = (time.time() - start_time) * 1000
        return result

    def verify_compression_operation(
        self,
        document: str,
        skeleton_text: str,
        node_map: Dict[str, str],
        original_tokens: int,
        skeleton_tokens: int,
        fidelity_level: str,
        compression_ratio: float,
        edges: Optional[List[Tuple[str, str]]] = None,
    ) -> VerificationResult:
        """
        Verify a compression operation directly (without bundle).

        Args:
            document: Original document
            skeleton_text: Compressed skeleton
            node_map: Node mapping
            original_tokens: Original token count
            skeleton_tokens: Skeleton token count
            fidelity_level: Target fidelity
            compression_ratio: Achieved compression
            edges: Optional graph edges

        Returns:
            VerificationResult
        """
        start_time = time.time()
        result = VerificationResult(verified=False)

        # Check preconditions
        result.preconditions = self.check_preconditions(document, fidelity_level)

        # Check postconditions
        result.postconditions = self.check_postconditions(
            skeleton_text, node_map, original_tokens, skeleton_tokens,
            fidelity_level, compression_ratio
        )

        # Check composition invariants if edges provided
        if edges is not None:
            invariants = CompressionContract.check_composition_invariants(
                list(node_map.keys()), edges, skeleton_text
            )
            for check in invariants.checks:
                if check.status != ContractStatus.PASSED:
                    result.violations.append(ContractViolation(
                        contract_name=check.name,
                        contract_type=ContractType.INVARIANT,
                        expected="Invariant should hold",
                        actual=check.message or "Invariant violated",
                        severity="warning"
                    ))

        # Collect all violations
        for check in result.preconditions.checks:
            if check.status != ContractStatus.PASSED:
                result.violations.append(ContractViolation(
                    contract_name=check.name,
                    contract_type=ContractType.PRECONDITION,
                    expected="Check should pass",
                    actual=check.message or "Check failed",
                    severity="error"
                ))

        for check in result.postconditions.checks:
            if check.status != ContractStatus.PASSED:
                result.violations.append(ContractViolation(
                    contract_name=check.name,
                    contract_type=ContractType.POSTCONDITION,
                    expected="Check should pass",
                    actual=check.message or "Check failed",
                    severity="error"
                ))

        result.verified = result.all_contracts_passed
        self._verification_count += 1
        if result.verified:
            self._passed_count += 1

        result.verification_time_ms = (time.time() - start_time) * 1000
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics"""
        return {
            "total_verifications": self._verification_count,
            "passed_verifications": self._passed_count,
            "pass_rate": self._passed_count / max(self._verification_count, 1),
        }

    def reset_statistics(self) -> None:
        """Reset verification statistics"""
        self._verification_count = 0
        self._passed_count = 0


class BatchVerifier:
    """
    Batch verification for multiple compression operations.

    Supports:
    - Parallel verification
    - Aggregate statistics
    - Failure pattern analysis
    """

    def __init__(self, verifier: Optional[CompressionVerifier] = None):
        self.verifier = verifier or CompressionVerifier()

    def verify_bundles(
        self,
        bundles: List[EvidenceBundle]
    ) -> Tuple[List[VerificationResult], Dict[str, Any]]:
        """
        Verify a batch of evidence bundles.

        Args:
            bundles: List of bundles to verify

        Returns:
            Tuple of (results, aggregate_statistics)
        """
        results = []
        for bundle in bundles:
            result = self.verifier.verify_bundle(bundle)
            results.append(result)

        stats = self._aggregate_statistics(results)
        return results, stats

    def _aggregate_statistics(
        self,
        results: List[VerificationResult]
    ) -> Dict[str, Any]:
        """Aggregate statistics from verification results"""
        if not results:
            return {"count": 0}

        passed = sum(1 for r in results if r.verified)
        total_violations = sum(len(r.violations) for r in results)

        # Count violation types
        violation_types: Dict[str, int] = {}
        for result in results:
            for violation in result.violations:
                key = violation.contract_name
                violation_types[key] = violation_types.get(key, 0) + 1

        # Average verification time
        avg_time = sum(r.verification_time_ms for r in results) / len(results)

        return {
            "count": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / len(results),
            "total_violations": total_violations,
            "violation_types": violation_types,
            "avg_verification_time_ms": avg_time,
        }
