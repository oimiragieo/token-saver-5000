"""route_tool_call: dispatch table + validation/logging wrapper.

Moved verbatim from mcp_core.py (N2 slice 2). Imports every handler-module
alias directly (same relative-import pattern the original file used) rather
than importing from the schemas_*.py files, since the router dict only needs
the handler functions, not the Tool schemas.
"""

from typing import Any, Dict

from .. import compression_handlers as ch
from .. import afm_handlers as afm
from .. import bundle_handlers as bh
from .. import file_sync_handlers as fs
from .. import resource_handlers as rh
from .. import detection_handlers as dh
from .. import ace_handlers as ace
from .. import visualization_handlers as vh
from .. import help_handlers as hh
from .. import connector_handlers as coh
from .. import docs_handlers as doch
from .. import model_handlers as moh
from .. import multimodal_handlers as mmh
from .. import temporal_handlers as th
from .. import experimental_handlers as exp
from .. import experiment_handlers as eh
from .. import memory_handlers as mh
from .. import prompt_handlers as ph
from .. import token_optimization_handlers as toh

from ...structured_logging import get_logger
from ._profile import _enabled_tool_names, _normalize_tool_profile

logger = get_logger("semantic-modulator")


async def route_tool_call(
    name: str, args: Dict[str, Any], context: Dict[str, Any], tool_profile: str = "full"
) -> str:
    """
    Route MCP tool calls to appropriate handler functions.

    This function automatically logs:
    - Request ID (correlation)
    - Tool name
    - Execution duration (milliseconds)
    - Memory delta (MB)
    - Success/failure status
    - Error details (if failed)

    Args:
        name: Tool name (e.g., "ingest_context", "afm_add_message")
        args: Tool arguments as dictionary
        context: Server context with all necessary components:
            - compressor: SemanticCompressor instance
            - blind_spot_detector: BlindSpotDetector instance
            - halo_detector: HaloEffectDetector instance
            - context_window_adapter: ContextWindowAdapter instance
            - multilevel_encoder: MultiLevelSemanticEncoder instance
            - focus_manager: FocusManager instance
            - persistence: PersistenceManager instance
            - resource_manager: ResourceManager instance
            - sync_manager: FileSyncManager instance
            - version_manager: VersionManager instance
            - validate_file_id: Validation function
            - validate_node_ids: Validation function
            - validate_token_count: Validation function
            - save_file_sync_metadata: Save function

    Returns:
        Handler result as formatted string

    Raises:
        ValueError: If tool name is unknown
    """
    # Define routing table mapping tool names to handler functions
    router = {
        "gc_read_doc": doch.handle_gc_read_doc,
        "gc_search_docs": doch.handle_gc_search_docs,
        # Document Compression (9 tools)
        "ingest_context": ch.handle_ingest,
        "read_skeleton": ch.handle_read_skeleton,
        "modulate_region": ch.handle_modulate_region,
        "search_semantic": ch.handle_search_semantic,
        "get_stats": ch.handle_get_stats,
        "list_documents": ch.handle_list_documents,
        "delete_document": ch.handle_delete_document,
        "adapt_to_context_window": ch.handle_adapt_to_context_window,
        "multilevel_encode": ch.handle_multilevel_encode,
        # Batch Processing (1 tool)
        "batch_ingest_documents": ch.handle_batch_ingest,
        # Directory Ingestion (1 tool)
        "ingest_directory": ch.handle_ingest_directory,
        # Fidelity Advisor (1 tool)
        "recommend_fidelity": ch.handle_recommend_fidelity,
        # Detection (2 tools)
        "check_blind_spots": dh.handle_check_blind_spots,
        "detect_hallucination": dh.handle_detect_hallucination,
        # AFM Dialogue (6 tools)
        "afm_add_message": afm.handle_afm_add_message,
        "afm_build_context": afm.handle_afm_build_context,
        "afm_get_stats": afm.handle_afm_get_stats,
        "afm_clear_history": afm.handle_afm_clear_history,
        "afm_export_history": afm.handle_afm_export_history,
        "afm_import_history": afm.handle_afm_import_history,
        # File Sync (4 tools)
        "check_file_sync": fs.handle_check_file_sync,
        "diff_cached_file": fs.handle_diff_cached_file,
        "refresh_document": fs.handle_refresh_document,
        "get_version_history": fs.handle_get_version_history,
        # Resource Management (3 tools)
        "check_resource_health": rh.handle_check_resource_health,
        "check_environment": rh.handle_check_environment,
        "should_compress": rh.handle_should_compress,
        # Help & Documentation (1 tool)
        "tool_help": hh.handle_tool_help,
        # Graph Visualization (4 tools)
        "export_graph_json": vh.handle_export_graph_json,
        "visualize_graph_html": vh.handle_visualize_graph_html,
        "export_graph_graphml": vh.handle_export_graph_graphml,
        "explain_compression_decision": vh.handle_explain_compression_decision,
        # ACE Framework (7 tools) - now using HandlerContext like all other handlers
        "ace_generate": ace.handle_ace_generate,
        "ace_reflect": ace.handle_ace_reflect,
        "ace_curate": ace.handle_ace_curate,
        "ace_grow_context": ace.handle_ace_grow_context,
        "ace_refine_context": ace.handle_ace_refine_context,
        "ace_get_playbook": ace.handle_ace_get_playbook,
        "ace_execute_cycle": ace.handle_ace_execute_cycle,
        # Experimental (5 tools) - NOT production-ready
        "toon_encode": exp.handle_toon_encode,
        "toon_decode": exp.handle_toon_decode,
        "scar_compress": exp.handle_scar_compress,
        "scar_get_stats": exp.handle_scar_get_stats,
        "multimodal_ingest": exp.handle_multimodal_ingest,
        "ingest_multimodal": mmh.handle_ingest_multimodal,
        "search_multimodal": mmh.handle_search_multimodal,
        "create_handoff_bundle": bh.handle_create_handoff_bundle,
        "list_handoff_bundles": bh.handle_list_handoff_bundles,
        "get_handoff_bundle": bh.handle_get_handoff_bundle,
        "replay_handoff_bundle": bh.handle_replay_handoff_bundle,
        "get_provider_profile": moh.handle_get_provider_profile,
        "estimate_model_cost": moh.handle_estimate_model_cost,
        "optimize_for_model": moh.handle_optimize_for_model,
        "assess_cache_compatibility": moh.handle_assess_cache_compatibility,
        "capture_cache_telemetry": moh.handle_capture_cache_telemetry,
        "diagnose_cache_miss": moh.handle_diagnose_cache_miss,
        # ASG-SI (4 tools) - Experimental self-improvement framework
        "verify_compression": exp.handle_verify_compression,
        "calculate_reward": exp.handle_calculate_reward,
        "get_evidence_stats": exp.handle_get_evidence_stats,
        "generate_synthetic_tests": exp.handle_generate_synthetic_tests,
        # New tools: diff re-ingestion, cross-doc dedup, presets
        "diff_reingest": ch.handle_diff_reingest,
        "find_duplicates": ch.handle_find_duplicates,
        "get_compression_presets": ch.handle_get_presets,
        "create_prompt_template": ph.handle_create_prompt_template,
        "update_prompt_template": ph.handle_update_prompt_template,
        "list_prompt_templates": ph.handle_list_prompt_templates,
        "get_prompt_template": ph.handle_get_prompt_template,
        "deploy_prompt_version": ph.handle_deploy_prompt_version,
        "compare_prompt_versions": ph.handle_compare_prompt_versions,
        "render_prompt_template": ph.handle_render_prompt_template,
        "list_prefix_collisions": ph.handle_list_prefix_collisions,
        "audit_prompt_cacheability": ph.handle_audit_prompt_cacheability,
        "add_memory": mh.handle_add_memory,
        "search_memory": mh.handle_search_memory,
        "list_memories": mh.handle_list_memories,
        "delete_memory": mh.handle_delete_memory,
        "summarize_user_memory": mh.handle_summarize_user_memory,
        "get_user_profile": mh.handle_get_user_profile,
        # Knowledge Management (Phase 1-4)
        "ingest_transcript": mh.handle_ingest_transcript,
        "compile_knowledge": mh.handle_compile_knowledge,
        "get_knowledge_index": mh.handle_get_knowledge_index,
        "lint_knowledge": mh.handle_lint_knowledge,
        "search_memory_index": mh.handle_search_memory_index,
        "create_dataset": eh.handle_create_dataset,
        "list_datasets": eh.handle_list_datasets,
        "run_experiment": eh.handle_run_experiment,
        "get_experiment_run": eh.handle_get_experiment_run,
        "compare_experiment_runs": eh.handle_compare_experiment_runs,
        "list_connector_types": coh.handle_list_connector_types,
        "create_connector_feed": coh.handle_create_connector_feed,
        "list_connector_feeds": coh.handle_list_connector_feeds,
        "get_connector_feed": coh.handle_get_connector_feed,
        "sync_connector_feed": coh.handle_sync_connector_feed,
        "get_context_block": th.handle_get_context_block,
        "search_timeline": th.handle_search_timeline,
        "list_fact_history": th.handle_list_fact_history,
        "invalidate_fact": th.handle_invalidate_fact,
        "check_context_budget": ch.handle_check_context_budget,
        # Phase 5: Research-based features (2025 papers)
        "prune_by_relevance": ch.handle_prune_by_relevance,
        "get_multi_level_skeleton": ch.handle_multi_level_skeleton,
        "evict_stale": ch.handle_evict_stale,
        "advise_context": ch.handle_advise_context,
        "get_compression_insights": ch.handle_get_compression_insights,
        "generate_rewrite_prompt": ch.handle_generate_rewrite_prompt,
        # Token Optimization (v0.11.0)
        "estimate_tokens": toh.handle_estimate_tokens,
        "configure_for_client": toh.handle_configure_for_client,
        "set_compression_profile": toh.handle_set_compression_profile,
        "get_compression_profile": toh.handle_get_compression_profile,
        # arXiv paper techniques (v0.12.0)
        "compress_meta_tokens": toh.handle_compress_meta_tokens,
        "recommend_compression": toh.handle_recommend_compression,
        # Session Journal (v0.13.0)
        "recover_session": toh.handle_recover_session,
        # Tensor-Grep Integration (v0.13.0)
        "compress_codebase": ch.handle_compress_codebase,
        "search_code": ch.handle_search_code,
        # CLI Output Optimizer
        "filter_cli_output": toh.handle_filter_cli_output,
        # Savings Tracker (v0.14.0)
        "get_savings_report": toh.handle_get_savings_report,
        "get_savings_inline": toh.handle_get_savings_inline,
        # Cache Strategy Advisor
        "advise_cache_strategy": toh.handle_advise_cache_strategy,
        # Structural Summary + Dead Code Detector (v0.15.0)
        "generate_structural_summary": toh.handle_generate_structural_summary,
        "detect_dead_code": toh.handle_detect_dead_code,
        # Tee/Recovery (v0.16.0)
        "get_original_output": toh.handle_get_original_output,
        "list_tee_entries": toh.handle_list_tee_entries,
        "tee_store_stats": toh.handle_tee_store_stats,
        # Discover Savings (v0.16.0)
        "discover_savings": toh.handle_discover_savings,
        # ROI Calculator (v0.16.0)
        "calculate_roi": toh.handle_calculate_roi,
        # Budget Monitor (v0.16.0)
        "check_budget": toh.handle_check_budget,
        # Team Export (v0.16.0)
        "export_team_data": toh.handle_export_team_data,
    }

    enabled_tools = _enabled_tool_names(set(router.keys()), tool_profile)

    # Lookup handler
    if name not in router:
        available_tools = ", ".join(sorted(enabled_tools))
        raise ValueError(
            f"Unknown tool: '{name}'\n\n"
            f"Available tools ({len(enabled_tools)}):\n{available_tools}\n\n"
            f"[TIP] Tip: Use list_tools() to see all available tools with descriptions"
        )
    if name not in enabled_tools:
        available_tools = ", ".join(sorted(enabled_tools))
        raise ValueError(
            f"Tool '{name}' is not enabled in profile '{_normalize_tool_profile(tool_profile)}'.\n\n"
            f"Available tools ({len(enabled_tools)}):\n{available_tools}\n\n"
            f"[TIP] Tip: Use profile 'full' to enable advanced tools"
        )

    # Pre-execute input validation
    from ...validation_hooks import validate_tool_input

    validation_errors = validate_tool_input(name, args)
    if validation_errors:
        import json

        return json.dumps(
            {
                "error": "Input validation failed",
                "validation_errors": validation_errors,
            },
            indent=2,
        )

    # Route to handler (async)
    handler = router[name]
    handler_module = getattr(handler, "__module__", "unknown")
    handler_name = getattr(handler, "__name__", "unknown")
    logger.info(
        "tool_routing", tool_name=name, handler_module=handler_module, handler_function=handler_name
    )

    return await handler(context, args)
