# Folder guide: `src/handlers/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../../CLAUDE.md).

## Contents

### Python modules

#### `__init__.py`

Handler Modules for Semantic Modulator MCP Server

_No top-level classes or functions (may re-export only)._

#### `ace_handlers.py`

ACE (Agentic Context Engineering) MCP Tool Handlers

| Kind | Name |
|------|------|
| `def` | `_get_or_create_context` |
| `def` | `_calculate_avg_confidence` |
| `def` | `_build_success_response` |
| `def` | `_build_error_response` |
| `def` | `_add_bullet_to_context` |
| `def` | `_update_bullets_performance` |
| `def` | `_filter_and_serialize_bullets` |
| `async def` | `handle_ace_generate` |
| `async def` | `handle_ace_reflect` |
| `async def` | `handle_ace_curate` |
| `async def` | `handle_ace_grow_context` |
| `async def` | `handle_ace_refine_context` |
| `async def` | `handle_ace_get_playbook` |
| `async def` | `handle_ace_execute_cycle` |

#### `afm_handlers.py`

AFM (Adaptive Focus Memory) Handler Functions

| Kind | Name |
|------|------|
| `async def` | `handle_afm_add_message` |
| `async def` | `handle_afm_build_context` |
| `async def` | `handle_afm_get_stats` |
| `async def` | `handle_afm_clear_history` |
| `async def` | `handle_afm_export_history` |
| `async def` | `handle_afm_import_history` |

#### `bundle_handlers.py`

Stable handlers for structured handoff bundles.

| Kind | Name |
|------|------|
| `def` | `_scope_kwargs` |
| `def` | `_scoped_file_id` |
| `def` | `_registry` |
| `def` | `get_bundle_output_fields` |
| `async def` | `handle_create_handoff_bundle` |
| `async def` | `handle_list_handoff_bundles` |
| `async def` | `handle_get_handoff_bundle` |
| `async def` | `handle_replay_handoff_bundle` |

#### `compress_manifest.py`

MCP tool manifest compression handler (v1.8.0 A2b).

| Kind | Name |
|------|------|
| `def` | `_estimate_tokens` |
| `def` | `_compress_description` |
| `def` | `handle_compress_manifest` |

#### `compression_handlers.py`

Compression-related MCP tool handlers.

| Kind | Name |
|------|------|
| `def` | `_is_structured_markdown` |
| `def` | `_resolve_chunking_strategy` |
| `def` | `_scope_kwargs` |
| `def` | `_scoped_file_id` |
| `def` | `_scope_filtered_results` |
| `def` | `_scope_filtered_file_ids` |
| `def` | `_has_scope_args` |
| `def` | `_scope_filtered_duplicates` |
| `def` | `_compressor_temporal_graph` |
| `def` | `_temporal_graph` |
| `def` | `_temporal_excluded_node_ids` |
| `def` | `_temporal_filter_search_results` |
| `def` | `_scoped_global_stats` |
| `def` | `_scope_label` |
| `async def` | `_resolve_awaitable` |
| `async def` | `_call_explicit_optional_method` |
| `def` | `_generate_skeleton_with_optional_filters` |
| `def` | `_flatten_output_fields` |
| `def` | `get_search_semantic_output_fields` |
| `def` | `get_read_skeleton_output_fields` |
| `def` | `get_ingest_context_output_fields` |
| `def` | `get_recommend_fidelity_output_fields` |
| `def` | `validate_file_id` |
| `def` | `validate_node_ids` |
| `def` | `validate_token_count` |
| `async def` | `handle_ingest` |
| `async def` | `handle_read_skeleton` |
| `async def` | `handle_modulate_region` |
| `async def` | `handle_search_semantic` |
| `async def` | `handle_get_stats` |
| `async def` | `handle_list_documents` |
| `async def` | `handle_delete_document` |
| `async def` | `handle_adapt_to_context_window` |
| `async def` | `handle_multilevel_encode` |
| `async def` | `handle_recommend_fidelity` |
| `async def` | `handle_batch_ingest` |
| `async def` | `handle_ingest_directory` |
| `async def` | `handle_diff_reingest` |
| `async def` | `handle_find_duplicates` |
| `async def` | `handle_get_presets` |
| `async def` | `handle_check_context_budget` |
| `async def` | `handle_prune_by_relevance` |
| `async def` | `handle_multi_level_skeleton` |
| `async def` | `handle_evict_stale` |
| `async def` | `handle_advise_context` |
| `async def` | `handle_get_compression_insights` |
| `async def` | `handle_generate_rewrite_prompt` |
| `async def` | `handle_compress_codebase` |
| `async def` | `handle_search_code` |

#### `connector_handlers.py`

Handlers for managed connector feeds.

| Kind | Name |
|------|------|
| `def` | `_registry` |
| `def` | `_required_string` |
| `def` | `_scope_args` |
| `def` | `get_connector_output_fields` |
| `async def` | `handle_list_connector_types` |
| `async def` | `handle_create_connector_feed` |
| `async def` | `handle_list_connector_feeds` |
| `async def` | `handle_get_connector_feed` |
| `async def` | `handle_sync_connector_feed` |

#### `detection_handlers.py`

Detection Handler Functions

| Kind | Name |
|------|------|
| `def` | `_scoped_file_id` |
| `def` | `_validate_file_id` |
| `async def` | `handle_check_blind_spots` |
| `async def` | `handle_detect_hallucination` |

#### `docs_handlers.py`

GotContext documentation MCP handlers.

| Kind | Name |
|------|------|
| `class` | `DocChunk` |
| `def` | `_slugify` |
| `def` | `_tokenize` |
| `def` | `_count_tokens` |
| `def` | `_trim_words` |
| `def` | `_docs_root` |
| `def` | `_read_llms_txt` |
| `def` | `_extract_first_url` |
| `def` | `_parse_llms_txt` |
| `def` | `_build_doc_maps` |
| `def` | `_idf` |
| `def` | `_term_freq_with_stemming` |
| `def` | `_bm25_scores` |
| `def` | `_normalize_top_k` |
| `def` | `_normalize_slug_or_url` |
| `def` | `_excerpt_around_anchor` |
| `async def` | `_fetch_url_markdown` |
| `async def` | `handle_gc_search_docs` |
| `async def` | `handle_gc_read_doc` |

#### `experiment_handlers.py`

Handlers for datasets and tracked experiment runs.

| Kind | Name |
|------|------|
| `def` | `_dataset_registry` |
| `def` | `_experiment_tracker` |
| `def` | `_record_metrics` |
| `def` | `_required_string` |
| `def` | `_optional_string` |
| `def` | `get_dataset_output_fields` |
| `def` | `get_experiment_output_fields` |
| `async def` | `handle_create_dataset` |
| `async def` | `handle_list_datasets` |
| `async def` | `handle_run_experiment` |
| `async def` | `handle_get_experiment_run` |
| `async def` | `handle_compare_experiment_runs` |

#### `experimental_handlers.py`

Experimental handlers for MCP tools.

| Kind | Name |
|------|------|
| `def` | `_scoped_doc_id` |
| `def` | `_get_toon_serializer` |
| `def` | `_get_multimodal_compressor` |
| `def` | `_get_scar_compressor` |
| `async def` | `handle_toon_encode` |
| `async def` | `handle_toon_decode` |
| `async def` | `handle_scar_compress` |
| `async def` | `handle_scar_get_stats` |
| `async def` | `handle_multimodal_ingest` |
| `def` | `_get_compression_verifier` |
| `def` | `_get_reward_calculator` |
| `def` | `_get_evidence_store` |
| `def` | `_get_experience_synthesizer` |
| `async def` | `handle_verify_compression` |
| `async def` | `handle_calculate_reward` |
| `async def` | `handle_get_evidence_stats` |
| `async def` | `handle_generate_synthetic_tests` |

#### `file_sync_handlers.py`

File Sync Handler Functions

| Kind | Name |
|------|------|
| `def` | `_scope_kwargs` |
| `async def` | `handle_check_file_sync` |
| `async def` | `handle_diff_cached_file` |
| `async def` | `handle_refresh_document` |
| `async def` | `handle_get_version_history` |

#### `help_handlers.py`

Help Handler Module

| Kind | Name |
|------|------|
| `def` | `_infer_tool_category` |
| `def` | `_placeholder_value` |
| `def` | `_schema_parameters` |
| `def` | `_schema_example_args` |
| `def` | `_auto_related_tools` |
| `def` | `get_tool_help_registry` |
| `async def` | `handle_tool_help` |

#### `mcp_core.py`

MCP Core Routing Module

| Kind | Name |
|------|------|
| `def` | `_normalize_tool_profile` |
| `def` | `_enabled_tool_names` |
| `def` | `_tools_for_profile` |
| `def` | `setup_mcp_tools` |
| `async def` | `route_tool_call` |

#### `memory_handlers.py`

Handlers for explicit memory, personalization, and knowledge management APIs.

| Kind | Name |
|------|------|
| `def` | `_memory_api` |
| `def` | `_scope_args` |
| `def` | `_required_string` |
| `def` | `_optional_string` |
| `def` | `get_memory_output_fields` |
| `def` | `get_user_profile_output_fields` |
| `async def` | `handle_add_memory` |
| `async def` | `handle_search_memory` |
| `async def` | `handle_list_memories` |
| `async def` | `handle_delete_memory` |
| `async def` | `handle_summarize_user_memory` |
| `async def` | `handle_get_user_profile` |
| `async def` | `handle_ingest_transcript` |
| `async def` | `handle_compile_knowledge` |
| `async def` | `handle_get_knowledge_index` |
| `async def` | `handle_lint_knowledge` |
| `async def` | `handle_search_memory_index` |

#### `model_handlers.py`

Handlers for provider-aware model optimization surfaces.

| Kind | Name |
|------|------|
| `def` | `get_model_output_fields` |
| `def` | `get_cache_compatibility_output_fields` |
| `def` | `get_cache_telemetry_output_fields` |
| `def` | `get_cache_diagnostic_output_fields` |
| `async def` | `handle_get_provider_profile` |
| `async def` | `handle_estimate_model_cost` |
| `async def` | `handle_optimize_for_model` |
| `async def` | `handle_assess_cache_compatibility` |
| `async def` | `handle_capture_cache_telemetry` |
| `async def` | `handle_diagnose_cache_miss` |

#### `multimodal_handlers.py`

Stable handlers for production multimodal ingestion.

| Kind | Name |
|------|------|
| `def` | `_service` |
| `def` | `_scoped_doc_id` |
| `def` | `get_multimodal_output_fields` |
| `async def` | `handle_ingest_multimodal` |
| `async def` | `handle_search_multimodal` |

#### `prompt_handlers.py`

Handlers for managed prompt template registry operations.

| Kind | Name |
|------|------|
| `def` | `_registry` |
| `def` | `_record_metrics` |
| `def` | `_required_string` |
| `def` | `_optional_string` |
| `def` | `_optional_variables` |
| `def` | `get_prompt_output_fields` |
| `def` | `get_prompt_deploy_output_fields` |
| `def` | `get_prompt_compare_output_fields` |
| `def` | `get_prompt_audit_output_fields` |
| `def` | `get_prompt_render_output_fields` |
| `def` | `get_prefix_collision_output_fields` |
| `async def` | `handle_create_prompt_template` |
| `async def` | `handle_update_prompt_template` |
| `async def` | `handle_list_prompt_templates` |
| `async def` | `handle_get_prompt_template` |
| `async def` | `handle_deploy_prompt_version` |
| `async def` | `handle_compare_prompt_versions` |
| `async def` | `handle_audit_prompt_cacheability` |
| `async def` | `handle_render_prompt_template` |
| `async def` | `handle_list_prefix_collisions` |

#### `resource_handlers.py`

Resource Management Handler Functions

| Kind | Name |
|------|------|
| `def` | `_flatten_output_fields` |
| `def` | `get_check_environment_output_fields` |
| `def` | `is_binary_content` |
| `async def` | `handle_check_resource_health` |
| `def` | `create_progress_bar` |
| `async def` | `handle_check_environment` |
| `async def` | `handle_should_compress` |

#### `temporal_handlers.py`

Handlers for temporal context, timelines, and fact lifecycle management.

| Kind | Name |
|------|------|
| `def` | `_scope_kwargs` |
| `def` | `_scoped_file_id` |
| `def` | `_temporal_graph` |
| `def` | `get_temporal_output_fields` |
| `async def` | `handle_get_context_block` |
| `async def` | `handle_search_timeline` |
| `async def` | `handle_list_fact_history` |
| `async def` | `handle_invalidate_fact` |

#### `token_optimization_handlers.py`

Token Optimization Handlers (v0.11.0)

| Kind | Name |
|------|------|
| `def` | `_get_journal` |
| `def` | `get_session_config_store` |
| `def` | `get_profile_manager` |
| `async def` | `handle_estimate_tokens` |
| `async def` | `handle_configure_for_client` |
| `async def` | `handle_set_compression_profile` |
| `async def` | `handle_get_compression_profile` |
| `async def` | `handle_compress_meta_tokens` |
| `async def` | `handle_recommend_compression` |
| `async def` | `handle_recover_session` |
| `async def` | `handle_filter_cli_output` |
| `def` | `_get_tracker` |
| `async def` | `handle_get_savings_report` |
| `async def` | `handle_advise_cache_strategy` |
| `async def` | `handle_generate_structural_summary` |
| `async def` | `handle_detect_dead_code` |
| `async def` | `handle_get_savings_inline` |
| `def` | `_get_tee_store` |
| `async def` | `handle_get_original_output` |
| `async def` | `handle_list_tee_entries` |
| `async def` | `handle_tee_store_stats` |
| `async def` | `handle_discover_savings` |
| `async def` | `handle_calculate_roi` |
| `async def` | `handle_check_budget` |
| `async def` | `handle_export_team_data` |

#### `visualization_handlers.py`

Visualization Handlers for Graph Visualization MCP Tools (v0.6.0)

| Kind | Name |
|------|------|
| `def` | `_scoped_file_id` |
| `async def` | `handle_export_graph_json` |
| `async def` | `handle_visualize_graph_html` |
| `async def` | `handle_export_graph_graphml` |
| `async def` | `handle_explain_compression_decision` |

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
