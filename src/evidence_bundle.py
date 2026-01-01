"""
Evidence Bundle System for Audited Compression Operations

Implements tamper-evident audit trails for semantic compression operations,
inspired by ASG-SI (Audited Skill-Graph Self-Improvement) paper.

Key Concepts from ASG-SI:
- "Improvement should be reconstructible from inspectable artifacts"
- Evidence bundles use cryptographic hashing for tamper detection
- Each operation produces a verifiable audit record

Features:
- EvidenceBundle: Immutable record of compression operation
- EvidenceStore: Append-only storage with integrity verification
- ContractResult: Precondition/postcondition check results
- Cryptographic chaining for tamper evidence

Usage:
    from src.evidence_bundle import EvidenceBundle, EvidenceStore

    store = EvidenceStore()
    bundle = EvidenceBundle.create(
        operation="ingest",
        input_data=document,
        output_data=skeleton,
        parameters={"fidelity": "BALANCED"},
        quality_metrics={"ssim": 0.92}
    )
    store.append(bundle)

    # Later verification
    is_valid = store.verify_chain()
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ContractStatus(Enum):
    """Status of a contract check"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ContractCheck:
    """Individual contract check result"""
    name: str
    status: ContractStatus
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContractCheck":
        return cls(
            name=data["name"],
            status=ContractStatus(data["status"]),
            message=data.get("message"),
            details=data.get("details")
        )


@dataclass
class ContractResult:
    """
    Result of contract verification (preconditions or postconditions).

    Attributes:
        checks: List of individual check results
        overall_passed: Whether all required checks passed
        timestamp: When verification was performed
    """
    checks: List[ContractCheck] = field(default_factory=list)
    overall_passed: bool = True
    timestamp: float = field(default_factory=time.time)

    def add_check(self, name: str, passed: bool, message: Optional[str] = None) -> None:
        """Add a check result"""
        status = ContractStatus.PASSED if passed else ContractStatus.FAILED
        self.checks.append(ContractCheck(name=name, status=status, message=message))
        if not passed:
            self.overall_passed = False

    def add_error(self, name: str, error: str) -> None:
        """Add an error check"""
        self.checks.append(ContractCheck(
            name=name,
            status=ContractStatus.ERROR,
            message=error
        ))
        self.overall_passed = False

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.status == ContractStatus.PASSED)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if c.status == ContractStatus.FAILED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "overall_passed": self.overall_passed,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContractResult":
        result = cls(
            overall_passed=data["overall_passed"],
            timestamp=data.get("timestamp", time.time())
        )
        result.checks = [ContractCheck.from_dict(c) for c in data.get("checks", [])]
        return result


@dataclass
class QualityMetrics:
    """
    Quality metrics for compression operation.

    Tracks multiple dimensions of compression quality:
    - ssim_score: Semantic SSIM (structure preservation)
    - embedding_similarity: Cosine similarity of embeddings
    - compression_ratio: Achieved compression ratio
    - token_reduction: Percentage of tokens saved
    - structure_score: Graph connectivity preservation
    """
    ssim_score: Optional[float] = None
    embedding_similarity: Optional[float] = None
    compression_ratio: Optional[float] = None
    token_reduction: Optional[float] = None
    structure_score: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.ssim_score is not None:
            result["ssim_score"] = self.ssim_score
        if self.embedding_similarity is not None:
            result["embedding_similarity"] = self.embedding_similarity
        if self.compression_ratio is not None:
            result["compression_ratio"] = self.compression_ratio
        if self.token_reduction is not None:
            result["token_reduction"] = self.token_reduction
        if self.structure_score is not None:
            result["structure_score"] = self.structure_score
        if self.custom_metrics:
            result["custom_metrics"] = self.custom_metrics
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityMetrics":
        return cls(
            ssim_score=data.get("ssim_score"),
            embedding_similarity=data.get("embedding_similarity"),
            compression_ratio=data.get("compression_ratio"),
            token_reduction=data.get("token_reduction"),
            structure_score=data.get("structure_score"),
            custom_metrics=data.get("custom_metrics", {})
        )


@dataclass
class EvidenceBundle:
    """
    Tamper-evident audit record for a compression operation.

    Implements the ASG-SI concept of evidence bundles that make
    improvement "reconstructible from inspectable artifacts."

    Attributes:
        bundle_id: Unique identifier for this operation
        operation: Type of operation (ingest, modulate, search, etc.)
        timestamp: Unix timestamp of operation
        input_hash: SHA-256 hash of input data (for privacy)
        input_token_count: Original token count
        output_hash: SHA-256 hash of output data
        output_token_count: Compressed token count
        parameters: Operation parameters (fidelity, thresholds, etc.)
        quality_metrics: Quality measurements
        preconditions: Precondition check results
        postconditions: Postcondition check results
        verifier_version: Version of verifier for reproducibility
        previous_bundle_hash: Hash of previous bundle (chain integrity)
        bundle_hash: Hash of this bundle (computed on creation)
    """
    bundle_id: str
    operation: str
    timestamp: float
    input_hash: str
    input_token_count: int
    output_hash: str
    output_token_count: int
    parameters: Dict[str, Any]
    quality_metrics: QualityMetrics
    preconditions: ContractResult
    postconditions: ContractResult
    verifier_version: str = "1.0.0"
    previous_bundle_hash: Optional[str] = None
    bundle_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute bundle hash if not provided"""
        if self.bundle_hash is None:
            self.bundle_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of bundle contents"""
        # Deterministic serialization for consistent hashing
        data = {
            "bundle_id": self.bundle_id,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "input_token_count": self.input_token_count,
            "output_hash": self.output_hash,
            "output_token_count": self.output_token_count,
            "parameters": self.parameters,
            "quality_metrics": self.quality_metrics.to_dict(),
            "preconditions": self.preconditions.to_dict(),
            "postconditions": self.postconditions.to_dict(),
            "verifier_version": self.verifier_version,
            "previous_bundle_hash": self.previous_bundle_hash,
        }
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify bundle hash matches contents"""
        expected = self._compute_hash()
        return self.bundle_hash == expected

    @property
    def compression_achieved(self) -> float:
        """Calculate compression ratio"""
        if self.input_token_count == 0:
            return 0.0
        return self.input_token_count / max(self.output_token_count, 1)

    @property
    def contracts_satisfied(self) -> bool:
        """Check if all contracts passed"""
        return self.preconditions.overall_passed and self.postconditions.overall_passed

    @classmethod
    def create(
        cls,
        operation: str,
        input_data: Union[str, bytes],
        output_data: Union[str, bytes],
        input_token_count: int,
        output_token_count: int,
        parameters: Dict[str, Any],
        quality_metrics: Optional[QualityMetrics] = None,
        preconditions: Optional[ContractResult] = None,
        postconditions: Optional[ContractResult] = None,
        previous_bundle_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvidenceBundle":
        """
        Factory method to create an EvidenceBundle.

        Args:
            operation: Type of operation
            input_data: Raw input (hashed for privacy)
            output_data: Raw output (hashed for privacy)
            input_token_count: Original token count
            output_token_count: Compressed token count
            parameters: Operation parameters
            quality_metrics: Quality measurements
            preconditions: Precondition results
            postconditions: Postcondition results
            previous_bundle_hash: For chain integrity
            metadata: Additional metadata

        Returns:
            EvidenceBundle with computed hashes
        """
        # Hash inputs for privacy
        if isinstance(input_data, str):
            input_data = input_data.encode()
        if isinstance(output_data, str):
            output_data = output_data.encode()

        input_hash = hashlib.sha256(input_data).hexdigest()
        output_hash = hashlib.sha256(output_data).hexdigest()

        return cls(
            bundle_id=str(uuid.uuid4()),
            operation=operation,
            timestamp=time.time(),
            input_hash=input_hash,
            input_token_count=input_token_count,
            output_hash=output_hash,
            output_token_count=output_token_count,
            parameters=parameters,
            quality_metrics=quality_metrics or QualityMetrics(),
            preconditions=preconditions or ContractResult(),
            postconditions=postconditions or ContractResult(),
            previous_bundle_hash=previous_bundle_hash,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "bundle_id": self.bundle_id,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "input_token_count": self.input_token_count,
            "output_hash": self.output_hash,
            "output_token_count": self.output_token_count,
            "parameters": self.parameters,
            "quality_metrics": self.quality_metrics.to_dict(),
            "preconditions": self.preconditions.to_dict(),
            "postconditions": self.postconditions.to_dict(),
            "verifier_version": self.verifier_version,
            "previous_bundle_hash": self.previous_bundle_hash,
            "bundle_hash": self.bundle_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceBundle":
        """Deserialize from dictionary"""
        return cls(
            bundle_id=data["bundle_id"],
            operation=data["operation"],
            timestamp=data["timestamp"],
            input_hash=data["input_hash"],
            input_token_count=data["input_token_count"],
            output_hash=data["output_hash"],
            output_token_count=data["output_token_count"],
            parameters=data["parameters"],
            quality_metrics=QualityMetrics.from_dict(data.get("quality_metrics", {})),
            preconditions=ContractResult.from_dict(data.get("preconditions", {})),
            postconditions=ContractResult.from_dict(data.get("postconditions", {})),
            verifier_version=data.get("verifier_version", "1.0.0"),
            previous_bundle_hash=data.get("previous_bundle_hash"),
            bundle_hash=data.get("bundle_hash"),
            metadata=data.get("metadata", {}),
        )


class EvidenceStore:
    """
    Append-only storage for evidence bundles with integrity verification.

    Implements blockchain-style chaining for tamper detection.
    Each bundle's hash depends on the previous bundle's hash.

    Features:
    - Append-only storage (immutable history)
    - Chain integrity verification
    - Persistence to disk (JSON format)
    - Query by operation type, time range, etc.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize evidence store.

        Args:
            storage_path: Optional path for persistent storage
        """
        self._bundles: List[EvidenceBundle] = []
        self._storage_path = storage_path
        self._chain_valid = True

        if storage_path and storage_path.exists():
            self._load()

    @property
    def chain_valid(self) -> bool:
        """Check if chain integrity is valid"""
        return self._chain_valid

    def __len__(self) -> int:
        return len(self._bundles)

    def __iter__(self):
        return iter(self._bundles)

    def __getitem__(self, index: int) -> EvidenceBundle:
        return self._bundles[index]

    def append(self, bundle: EvidenceBundle) -> None:
        """
        Append bundle to store with chain linking.

        Args:
            bundle: Bundle to append

        Raises:
            ValueError: If bundle already has conflicting chain link
        """
        # Link to previous bundle
        if self._bundles:
            expected_prev = self._bundles[-1].bundle_hash
            if bundle.previous_bundle_hash is None:
                bundle.previous_bundle_hash = expected_prev
                bundle.bundle_hash = bundle._compute_hash()
            elif bundle.previous_bundle_hash != expected_prev:
                raise ValueError(
                    f"Chain integrity violation: expected previous_hash={expected_prev}, "
                    f"got {bundle.previous_bundle_hash}"
                )

        self._bundles.append(bundle)

        if self._storage_path:
            self._save()

    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify integrity of entire chain.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for i, bundle in enumerate(self._bundles):
            # Verify bundle integrity
            if not bundle.verify_integrity():
                errors.append(f"Bundle {i} ({bundle.bundle_id}): hash mismatch")

            # Verify chain link
            if i > 0:
                expected_prev = self._bundles[i - 1].bundle_hash
                if bundle.previous_bundle_hash != expected_prev:
                    errors.append(
                        f"Bundle {i} ({bundle.bundle_id}): chain link broken "
                        f"(expected {expected_prev[:16]}..., got {bundle.previous_bundle_hash[:16] if bundle.previous_bundle_hash else 'None'}...)"
                    )

        self._chain_valid = len(errors) == 0
        return self._chain_valid, errors

    def get_by_operation(self, operation: str) -> List[EvidenceBundle]:
        """Get all bundles for a specific operation type"""
        return [b for b in self._bundles if b.operation == operation]

    def get_by_time_range(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[EvidenceBundle]:
        """Get bundles within a time range"""
        result = self._bundles
        if start_time is not None:
            result = [b for b in result if b.timestamp >= start_time]
        if end_time is not None:
            result = [b for b in result if b.timestamp <= end_time]
        return result

    def get_failed_contracts(self) -> List[EvidenceBundle]:
        """Get all bundles with failed contracts"""
        return [b for b in self._bundles if not b.contracts_satisfied]

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self._bundles:
            return {
                "total_bundles": 0,
                "chain_valid": True,
                "operations": {},
            }

        operations = {}
        total_compression = 0.0
        contracts_passed = 0

        for bundle in self._bundles:
            op = bundle.operation
            if op not in operations:
                operations[op] = {"count": 0, "avg_compression": 0.0}
            operations[op]["count"] += 1
            total_compression += bundle.compression_achieved
            if bundle.contracts_satisfied:
                contracts_passed += 1

        for op in operations:
            count = operations[op]["count"]
            op_bundles = [b for b in self._bundles if b.operation == op]
            operations[op]["avg_compression"] = sum(
                b.compression_achieved for b in op_bundles
            ) / count

        return {
            "total_bundles": len(self._bundles),
            "chain_valid": self._chain_valid,
            "operations": operations,
            "avg_compression": total_compression / len(self._bundles),
            "contracts_pass_rate": contracts_passed / len(self._bundles),
            "first_bundle_time": self._bundles[0].timestamp,
            "last_bundle_time": self._bundles[-1].timestamp,
        }

    def _save(self) -> None:
        """Save to disk"""
        if not self._storage_path:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0.0",
            "bundles": [b.to_dict() for b in self._bundles]
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load from disk"""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)

            self._bundles = [
                EvidenceBundle.from_dict(b) for b in data.get("bundles", [])
            ]

            # Verify chain on load
            self.verify_chain()

        except Exception as e:
            logger.error(f"Failed to load evidence store: {e}")
            self._bundles = []
            self._chain_valid = False

    def clear(self) -> None:
        """Clear all bundles (for testing only)"""
        self._bundles = []
        self._chain_valid = True
        if self._storage_path and self._storage_path.exists():
            self._storage_path.unlink()


# Singleton instance for global access
_global_store: Optional[EvidenceStore] = None


def get_evidence_store(storage_path: Optional[Path] = None) -> EvidenceStore:
    """Get or create the global evidence store"""
    global _global_store
    if _global_store is None:
        _global_store = EvidenceStore(storage_path)
    return _global_store


def reset_evidence_store() -> None:
    """Reset global evidence store (for testing)"""
    global _global_store
    _global_store = None
