"""
Type definitions for Token Saver 5000 MCP Server.

This module contains TypedDict definitions for complex data structures,
following 2025 Python best practices (PEP 589, PEP 705, PEP 728).

TypedDict Benefits:
- Better IDE autocomplete and type checking
- Clear documentation of expected dictionary structure
- Mypy/Pyright validation of dictionary access
- No runtime overhead (type hints only)
"""

from typing import TypedDict, Callable, Any, TYPE_CHECKING
from typing_extensions import ReadOnly  # For read-only fields (Python 3.13+)

# Import types for type annotations
from src.semantic_compressor import SemanticCompressor
from src.blind_spot_detector import BlindSpotDetector, HaloEffectDetector
from src.afm import FocusManager  # AFM dialogue manager
from src.adaptive_rate_allocator import ContextWindowAdapter, MultiLevelSemanticEncoder
from src.persistence import PersistenceManager
from src.resource_manager import ResourceManager
from src.file_sync_manager import FileSyncManager
from src.version_manager import VersionManager
from src.ace_framework import ACEFramework
from src.path_validator import PathValidator

# Avoid circular import: server.py imports HandlerContext from this module
if TYPE_CHECKING:
    from src.server import ACEContextManager


class HandlerContext(TypedDict, total=True):
    """
    Context dictionary passed to all MCP tool handlers.

    This TypedDict defines the complete structure of the context dictionary
    that is built in SemanticModulatorServer._build_context() and passed
    to every handler via route_tool_call().

    Attributes:
        compressor: Main semantic compression engine
        blind_spot_detector: Detects missing context in responses
        halo_detector: Detects hallucinations in responses
        context_window_adapter: JSCCM-inspired context window adaptation
        multilevel_encoder: JSCCM-inspired multi-level encoding
        focus_manager: AFM (Adaptive Focus Memory) dialogue manager
        persistence: Document persistence (ChromaDB + JSON fallback)
        resource_manager: Resource limits enforcement (100MB/doc, 1GB total)
        sync_manager: File sync staleness detection (mtime + MD5)
        version_manager: Version history with diffs
        path_validator: Path validation (prevents CWE-22 path traversal)
        ace_framework: ACE (Agentic Context Engineering) framework
        ace_contexts: ACE context manager with LRU eviction
        validate_file_id: Validation helper for file IDs
        validate_node_ids: Validation helper for node IDs
        validate_token_count: Validation helper for token counts
        save_file_sync_metadata: Helper to save file sync metadata

    Example:
        ```python
        def handle_ingest(args: Dict[str, Any], context: HandlerContext) -> str:
            # Type-safe access with IDE autocomplete
            compressor = context["compressor"]
            resource_manager = context["resource_manager"]
            # Mypy will catch typos like context["comprsesor"]
            ...
        ```

    Notes:
        - All fields are required (total=True)
        - Callable fields are validation helpers from server
        - This enables full IDE support and type checking
        - No runtime overhead (type hints only)
    """

    # Core compression components
    compressor: ReadOnly[SemanticCompressor]
    blind_spot_detector: ReadOnly[BlindSpotDetector]
    halo_detector: ReadOnly[HaloEffectDetector]

    # JSCCM components
    context_window_adapter: ReadOnly[ContextWindowAdapter]
    multilevel_encoder: ReadOnly[MultiLevelSemanticEncoder]

    # AFM component
    focus_manager: ReadOnly[FocusManager]

    # Persistence and resources
    persistence: ReadOnly[PersistenceManager]
    resource_manager: ReadOnly[ResourceManager]

    # File sync and versioning
    sync_manager: ReadOnly[FileSyncManager]
    version_manager: ReadOnly[VersionManager]
    path_validator: ReadOnly[PathValidator]  # Security: Prevents path traversal attacks

    # ACE Framework
    ace_framework: ReadOnly[ACEFramework]
    ace_contexts: ReadOnly["ACEContextManager"]  # From src.server (TYPE_CHECKING import)

    # Validation helpers (callables from server)
    validate_file_id: ReadOnly[Callable[[str, bool], None]]
    validate_node_ids: ReadOnly[Callable[[list[str]], None]]
    validate_token_count: ReadOnly[Callable[[int, str], None]]
    save_file_sync_metadata: ReadOnly[Callable[[str, str | None], None]]


class ToolArguments(TypedDict, total=False):
    """
    Common tool arguments dictionary.

    This is a base TypedDict for common arguments passed to tools.
    Individual tools can extend this or define their own specific TypedDict.

    Attributes:
        text: Document text content
        file_id: Document identifier
        file_path: Optional file path for sync tracking
        metadata: Optional metadata dictionary
        node_ids: List of node IDs to retrieve
        fidelity: Fidelity level (RAW, DETAILED, STRUCTURE, OUTLINE, ABSTRACT)
        query: Search query string
        top_k: Number of results to return

    Notes:
        - All fields are optional (total=False)
        - Specific handlers may require certain fields
        - This provides baseline type hints for common patterns
    """

    # Document ingestion
    text: str
    file_id: str
    file_path: str | None
    metadata: dict[str, Any] | None

    # Node retrieval
    node_ids: list[str]
    fidelity: str

    # Search
    query: str
    top_k: int

    # Context window adaptation
    target_tokens: int
    query_priority: float
    metadata_priority: float

    # AFM
    user_message: str
    system_preamble: str | None
    context_budget: int
    include_preamble: bool

    # ACE Framework
    context_id: str
    bullet_points: list[str]
    playbook_topic: str


# Type aliases for clarity
FileID = str
NodeID = str
ChunkID = str
ContextID = str
MessageID = str

# Handler function type
HandlerFunction = Callable[[dict[str, Any], HandlerContext], str]
