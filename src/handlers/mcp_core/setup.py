"""setup_mcp_tools: concatenate every schema-list module, sort, filter by profile.

Moved from mcp_core.py (N2 slice 2). Import order below mirrors the original
file's own tool grouping order for reviewability -- order does not affect
behavior since the result is sorted by name immediately after concatenation.
"""

from typing import List

from mcp.types import Tool

from ._profile import _tools_for_profile
from .schemas_misc import DOCS_TOOLS, DETECTION_TOOLS, RESOURCE_TOOLS, HELP_TOOLS, CONNECTOR_TOOLS
from .schemas_compression import COMPRESSION_TOOLS
from .schemas_afm_temporal import AFM_TOOLS, TEMPORAL_TOOLS
from .schemas_filesync_bundle import FILESYNC_TOOLS, BUNDLE_TOOLS
from .schemas_prompts_ace import ACE_TOOLS, PROMPT_TOOLS
from .schemas_multimodal_viz import VISUALIZATION_TOOLS, MULTIMODAL_TOOLS
from .schemas_experimental import EXPERIMENTAL_TOOLS
from .schemas_model_experiment import MODEL_TOOLS, EXPERIMENT_TOOLS
from .schemas_memory import MEMORY_TOOLS
from .schemas_token_optimization import TOKEN_OPTIMIZATION_TOOLS


def setup_mcp_tools(profile: str = "full") -> List[Tool]:
    """
    Define all MCP tools available in the Semantic Modulator server.

    Returns:
        List of Tool objects with complete schemas (name, description, inputSchema)

    Tool Categories:
    - Document Compression (9): ingest, read_skeleton, modulate_region, search, stats, list, delete, adapt, multilevel
    - Batch Processing (1): batch_ingest_documents
    - Directory Ingestion (1): ingest_directory
    - Graph Visualization (4): export_graph_json, visualize_graph_html, export_graph_graphml, explain_compression_decision
    - Fidelity Advisor (1): recommend_fidelity
    - Detection (2): check_blind_spots, detect_hallucination
    - AFM Dialogue (6): add_message, build_context, get_stats, clear, export, import
    - File Sync (4): check_sync, diff, refresh, version_history
    - Resource Management (3): check_health, check_environment, should_compress
    - Help & Documentation (1): tool_help
    - ACE Framework (7): ace_generate, ace_reflect, ace_curate, ace_grow, ace_refine, ace_get_playbook, ace_execute_cycle
    - Experimental (9): toon_encode, toon_decode, scar_compress, scar_get_stats, multimodal_ingest,
                        verify_compression, calculate_reward, get_evidence_stats, generate_synthetic_tests
    """
    all_tools = (
        DOCS_TOOLS
        + COMPRESSION_TOOLS
        + DETECTION_TOOLS
        + AFM_TOOLS
        + FILESYNC_TOOLS
        + RESOURCE_TOOLS
        + HELP_TOOLS
        + ACE_TOOLS
        + VISUALIZATION_TOOLS
        + EXPERIMENTAL_TOOLS
        + MULTIMODAL_TOOLS
        + BUNDLE_TOOLS
        + MODEL_TOOLS
        + PROMPT_TOOLS
        + MEMORY_TOOLS
        + EXPERIMENT_TOOLS
        + CONNECTOR_TOOLS
        + TEMPORAL_TOOLS
        + TOKEN_OPTIMIZATION_TOOLS
    )
    all_tools.sort(key=lambda t: t.name)
    return _tools_for_profile(all_tools, profile)
