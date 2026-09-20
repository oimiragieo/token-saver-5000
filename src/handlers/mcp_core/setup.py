"""MCP schema setup generated from the typed engine tool registry."""


from mcp.types import Tool

from .registry import registered_tools


def setup_mcp_tools(profile: str = "full") -> list[Tool]:
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
    return [entry.schema for entry in registered_tools(profile)]
