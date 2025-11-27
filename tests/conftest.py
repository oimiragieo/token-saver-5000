"""
Shared pytest fixtures for Token Saver 5000 test suite.

This module provides reusable fixtures for all test files, including:
- SemanticCompressor instances
- Handler contexts with all managers
- Sample documents and test data
- Temporary file helpers
- Async event loop configuration

Added in v0.7.0 Week 3-4 (Comprehensive Testing Suite).
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest

from src.semantic_compressor import SemanticCompressor
from src.resource_manager import ResourceManager
from src.file_sync_manager import FileSyncManager
from src.version_manager import VersionManager
from src.persistence import PersistenceManager
from src.afm import FocusManager
from src.batch_manager import BatchDocument
from src.path_validator import PathValidator


# ===========================
# Event Loop Configuration
# ===========================


@pytest.fixture(scope="session")
def event_loop_policy():
    """Configure event loop policy for async tests."""
    return asyncio.get_event_loop_policy()


# ===========================
# Core Component Fixtures
# ===========================


@pytest.fixture
def compressor():
    """Create a fresh SemanticCompressor instance for testing."""
    return SemanticCompressor()


@pytest.fixture
def afm():
    """Create a fresh FocusManager instance for testing."""
    return FocusManager()


@pytest.fixture
def resource_manager():
    """Create a ResourceManager instance for testing."""
    return ResourceManager()


@pytest.fixture
def file_sync_manager():
    """Create a FileSyncManager instance for testing."""
    return FileSyncManager()


@pytest.fixture
def version_manager():
    """Create a VersionManager instance for testing."""
    return VersionManager()


@pytest.fixture
def persistence_manager():
    """Create a PersistenceManager instance for testing."""
    return PersistenceManager()


@pytest.fixture
def path_validator():
    """Create a PathValidator instance for testing."""
    return PathValidator()


# ===========================
# Handler Context Fixture
# ===========================


@pytest.fixture
def handler_context(
    compressor,
    afm,
    resource_manager,
    file_sync_manager,
    version_manager,
    persistence_manager,
    path_validator,
):
    """
    Create a complete handler context for testing MCP handlers.

    This fixture provides all managers and state needed for handler testing.
    """
    return {
        "compressor": compressor,
        "afm": afm,
        "sync_manager": file_sync_manager,
        "version_manager": version_manager,
        "ace_contexts": {},
        "resource_manager": resource_manager,
        "persistence": persistence_manager,
        "retrieval_history": {},
        "path_validator": path_validator,
    }


# ===========================
# Sample Data Fixtures
# ===========================


@pytest.fixture
def sample_text_short():
    """Short text sample for basic testing."""
    return "Machine learning models use neural networks for pattern recognition."


@pytest.fixture
def sample_text_medium():
    """Medium-length text sample for compression testing."""
    return """
    Quantum computing represents a paradigm shift in computation. Unlike classical
    computers that use bits (0 or 1), quantum computers use qubits that can exist
    in superposition states. This enables parallel computation across multiple
    states simultaneously. Quantum entanglement allows qubits to be correlated
    in ways impossible for classical bits, enabling powerful quantum algorithms
    like Shor's factoring algorithm and Grover's search algorithm.
    """


@pytest.fixture
def sample_text_large():
    """Large text sample for performance and load testing."""
    return (
        """
    Deep learning has revolutionized artificial intelligence across multiple domains.
    Convolutional neural networks excel at computer vision tasks, processing images
    through hierarchical feature extraction. Recurrent neural networks and transformers
    handle sequential data, enabling breakthroughs in natural language processing.

    The transformer architecture, introduced in "Attention is All You Need", relies
    on self-attention mechanisms to process sequences in parallel. This parallelization
    enables training on massive datasets with billions of parameters. Models like
    GPT-4, Claude, and PaLM demonstrate emergent capabilities including reasoning,
    code generation, and multi-modal understanding.

    Training these large language models requires distributed computing across
    thousands of GPUs or TPUs. Techniques like gradient checkpointing, mixed
    precision training, and model parallelism make training feasible. The resulting
    models exhibit few-shot learning capabilities, adapting to new tasks with
    minimal examples.

    Semantic compression aims to preserve meaning while reducing token count.
    Graph-based methods use PageRank to identify important concepts. Embedding
    similarity captures semantic relationships. Adaptive fidelity allows trading
    detail for efficiency. This enables efficient AI interactions by reducing
    context window usage while maintaining comprehension.
    """
        * 5
    )  # Repeat 5x for ~2000 tokens


@pytest.fixture
def sample_code():
    """Sample Python code for code compression testing."""
    return '''
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]

def quicksort(arr: list) -> list:
    """Sort array using quicksort algorithm."""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)
'''


@pytest.fixture
def sample_documents():
    """Sample documents for batch testing."""
    return [
        BatchDocument(
            file_id="doc1",
            text="Quantum computing uses qubits for parallel computation through superposition.",
            metadata={"topic": "quantum", "year": 2024},
        ),
        BatchDocument(
            file_id="doc2",
            text="Machine learning algorithms learn patterns from data without explicit programming.",
            metadata={"topic": "ml", "year": 2024},
        ),
        BatchDocument(
            file_id="doc3",
            text="Blockchain provides distributed consensus through cryptographic verification.",
            metadata={"topic": "blockchain", "year": 2024},
        ),
        BatchDocument(
            file_id="doc4",
            text="Neural networks consist of layers of interconnected nodes with weighted connections.",
            metadata={"topic": "neural-nets", "year": 2024},
        ),
    ]


@pytest.fixture
def sample_dialogue():
    """Sample dialogue messages for AFM testing."""
    return [
        {"role": "user", "content": "I have a severe peanut allergy."},
        {
            "role": "assistant",
            "content": "I've noted your peanut allergy. I'll keep this in mind for any food recommendations.",
        },
        {"role": "user", "content": "What's a good recipe for dinner?"},
        {
            "role": "assistant",
            "content": "I'll avoid peanuts. How about grilled chicken with vegetables?",
        },
    ]


# ===========================
# Temporary File Fixtures
# ===========================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file with sample content."""
    file_path = temp_dir / "test_document.txt"
    file_path.write_text(
        "This is a test document for file-based operations. "
        "It contains multiple sentences with semantic meaning. "
        "The semantic compressor should process this content correctly."
    )
    yield file_path
    # Cleanup handled by temp_dir fixture


@pytest.fixture
def temp_code_file(temp_dir):
    """Create a temporary Python file with sample code."""
    file_path = temp_dir / "test_code.py"
    file_path.write_text(
        '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

if __name__ == "__main__":
    hello_world()
    print(add(2, 3))
'''
    )
    yield file_path


# ===========================
# Performance Testing Fixtures
# ===========================


@pytest.fixture
def performance_documents():
    """Generate large set of documents for performance testing."""
    documents = []
    topics = ["quantum", "ml", "blockchain", "neural-nets", "crypto"]

    for i in range(100):
        topic = topics[i % len(topics)]
        documents.append(
            BatchDocument(
                file_id=f"perf_doc_{i}",
                text=f"Document {i} about {topic}. " * 20,  # ~400 chars each
                metadata={"topic": topic, "index": i},
            )
        )

    return documents


@pytest.fixture
def large_document():
    """Generate a very large document for load testing."""
    paragraphs = [
        "Quantum computing represents a paradigm shift in computational power.",
        "Machine learning algorithms extract patterns from massive datasets.",
        "Blockchain technology enables trustless distributed systems.",
        "Neural networks mimic biological brain structures for learning.",
        "Cryptographic protocols ensure data security and privacy.",
    ]

    # Generate ~10k token document
    return " ".join(paragraphs * 200)


# ===========================
# Chaos Engineering Fixtures
# ===========================


@pytest.fixture
def mock_disk_full(monkeypatch):
    """Mock disk full error for chaos testing."""
    original_open = open

    def failing_open(*args, **kwargs):
        # Simulate disk full on write operations
        if "w" in args[1] if len(args) > 1 else kwargs.get("mode", "r"):
            raise OSError(28, "No space left on device")  # ENOSPC
        return original_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)
    yield
    # monkeypatch automatically restores original


@pytest.fixture
def mock_network_partition(monkeypatch):
    """Mock network partition for chaos testing."""
    import asyncio

    async def failing_sleep(*args, **kwargs):
        raise asyncio.TimeoutError("Network partition simulated")

    # This is a simplified mock - real tests will need more sophisticated mocking
    yield failing_sleep


@pytest.fixture
def mock_model_crash(monkeypatch):
    """Mock model crash during embedding generation."""

    def failing_encode(*args, **kwargs):
        raise RuntimeError("Model crashed: CUDA out of memory")

    yield failing_encode


# ===========================
# Assertion Helpers
# ===========================


def assert_valid_skeleton(skeleton: Dict[str, Any]):
    """Assert that a skeleton has valid structure."""
    assert "file_id" in skeleton
    assert "original_size" in skeleton
    assert "compressed_size" in skeleton
    assert "compression_ratio" in skeleton
    assert "nodes" in skeleton
    assert isinstance(skeleton["nodes"], list)


def assert_valid_embedding(embedding):
    """Assert that an embedding has valid shape and values."""
    import numpy as np

    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1  # 1D vector
    assert embedding.shape[0] > 0  # Non-empty
    assert not np.isnan(embedding).any()  # No NaN values
    assert not np.isinf(embedding).any()  # No inf values


# Export assertion helpers for use in tests
pytest.assert_valid_skeleton = assert_valid_skeleton
pytest.assert_valid_embedding = assert_valid_embedding
