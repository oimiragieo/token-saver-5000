# Folder guide: `tests/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../CLAUDE.md).

## Contents

### Python modules

#### `__init__.py`

Test suite for Semantic Modulator

_No top-level classes or functions (may re-export only)._

#### `conftest.py`

Shared pytest fixtures for Token Saver 5000 test suite.

| Kind | Name |
|------|------|
| `def` | `_reset_shared_state` |
| `def` | `compressor` |
| `def` | `afm` |
| `def` | `ace_framework` |
| `def` | `resource_manager` |
| `def` | `file_sync_manager` |
| `def` | `version_manager` |
| `def` | `persistence_manager` |
| `def` | `path_validator` |
| `def` | `handler_context` |
| `def` | `sample_text_short` |
| `def` | `sample_text_medium` |
| `def` | `sample_text_large` |
| `def` | `sample_code` |
| `def` | `sample_documents` |
| `def` | `sample_dialogue` |
| `def` | `temp_dir` |
| `def` | `temp_file` |
| `def` | `temp_code_file` |
| `def` | `performance_documents` |
| `def` | `large_document` |
| `def` | `mock_disk_full` |
| `def` | `mock_network_partition` |
| `def` | `mock_model_crash` |
| `def` | `parity_corpus` |
| `def` | `characterization_context_factory` |
| `def` | `assert_valid_skeleton` |
| `def` | `assert_valid_embedding` |

#### `f11_fixture_harness.py`

Reusable A-vs-C comparison harness for the #250 multi-node fixture corpus.

| Kind | Name |
|------|------|
| `def` | `probe_model_load` |
| `def` | `set_ranker_path` |
| `def` | `node_section_marker` |
| `class` | `PerQueryResult` |
| `def` | `_rank_metrics` |
| `def` | `run_query` |
| `def` | `run_fixture` |
| `def` | `compare_paths` |
| `class` | `ClassSummary` |
| `def` | `per_class_summary` |
| `def` | `format_report` |
| `def` | `main` |
| `class` | `ThreeWayClassSummary` |
| `def` | `per_class_summary_gac` |
| `def` | `format_report_gac` |
| `def` | `main_gac` |

#### `test_420_bare_prefix_collision_sweep.py`

#420 sweep of remaining bare-prefix node-id collision sites.

| Kind | Name |
|------|------|
| `class` | `_Node` |
| `async def` | `test_handle_ingest_persists_only_own_chunks_not_prefix_sibling` |
| `async def` | `test_handle_diff_reingest_persists_only_own_chunks_not_prefix_sibling` |
| `def` | `test_resolve_anchored_node_ids_excludes_prefix_sibling` |
| `def` | `_plain_semantic_compressor` |
| `def` | `_doc_graph` |
| `async def` | `test_delete_confirmation_count_excludes_prefix_sibling` |
| `async def` | `test_delete_document_removes_only_own_chunks_from_plain_compressor` |
| `async def` | `test_connector_sync_persists_only_own_chunks_not_prefix_sibling` |
| `async def` | `test_refresh_document_persists_only_own_chunks_not_prefix_sibling` |
| `class` | `_FakeCodeModel` |
| `def` | `_code_chunk` |
| `def` | `test_search_code_file_id_filter_excludes_prefix_sibling` |
| `class` | `_FakeScarModel` |
| `def` | `_scar_node` |
| `def` | `test_search_with_alignment_file_id_filter_excludes_prefix_sibling` |

#### `test_ace_context_manager_module.py`

Contract tests for ACE context manager module extraction.

| Kind | Name |
|------|------|
| `def` | `test_ace_context_manager_module_exports_class` |
| `def` | `test_server_reexports_same_ace_context_manager_class` |

#### `test_ace_framework.py`

Unit tests for ACE Framework

| Kind | Name |
|------|------|
| `def` | `embedding_manager` |
| `def` | `sample_bullet` |
| `def` | `sample_context` |
| `def` | `test_bullet_creation` |
| `def` | `test_bullet_update_performance_success` |
| `def` | `test_bullet_update_performance_failure` |
| `def` | `test_bullet_success_rate_calculation` |
| `def` | `test_bullet_confidence_bounds` |
| `def` | `test_bullet_serialization` |
| `def` | `test_context_creation` |
| `def` | `test_context_add_bullet` |
| `def` | `test_context_update_bullet` |
| `def` | `test_context_remove_bullet` |
| `def` | `test_context_get_bullets_by_type` |
| `def` | `test_context_get_top_bullets` |
| `def` | `test_context_performance_stats` |
| `def` | `test_context_serialization` |
| `def` | `test_generator_creation` |
| `def` | `test_generator_trajectory_generation` |
| `def` | `test_generator_empty_context` |
| `def` | `test_generator_finds_relevant_bullets` |
| `def` | `test_reflector_creation` |
| `def` | `test_reflector_success_insights` |
| `def` | `test_reflector_failure_insights` |
| `def` | `test_curator_creation` |
| `def` | `test_curator_integrate_new_insight` |
| `def` | `test_curator_skips_insight_missing_bullet_type` |
| `def` | `test_curator_skips_insight_with_invalid_bullet_type` |
| `def` | `test_curator_integrates_valid_insight_despite_malformed_sibling` |
| `def` | `test_curator_update_similar_insight` |
| `def` | `test_curator_deduplication` |
| `def` | `test_curator_pruning` |
| `def` | `test_curator_max_bullets_limit` |
| `def` | `test_framework_creation` |
| `def` | `test_framework_create_initial_context` |
| `def` | `test_framework_execute_ace_cycle_success` |
| `def` | `test_framework_execute_ace_cycle_failure` |
| `def` | `test_framework_iterative_improvement` |
| `def` | `test_full_ace_workflow` |
| `def` | `test_deduplication_prevents_bloat` |

#### `test_ace_handlers.py`

Comprehensive tests for ace_handlers.py

| Kind | Name |
|------|------|
| `def` | `mock_ace_context` |
| `def` | `mock_ace_framework` |
| `def` | `mock_context` |
| `class` | `TestHelperFunctions` |
| `class` | `TestHandleAceGenerate` |
| `class` | `TestHandleAceReflect` |
| `class` | `TestHandleAceCurate` |
| `class` | `TestHandleAceGrowContext` |
| `class` | `TestHandleAceRefineContext` |
| `class` | `TestHandleAceGetPlaybook` |
| `class` | `TestHandleAceExecuteCycle` |

#### `test_adaptive_compression.py`

Tests for adaptive compression ratio and cost telemetry features.

| Kind | Name |
|------|------|
| `class` | `TestComputeAdaptiveRatio` |
| `class` | `TestAdaptiveRatioIntegration` |
| `class` | `TestComputeCostSavings` |
| `class` | `TestTokenSavingsTelemetry` |
| `class` | `TestHandlerCostSavingsIntegration` |

#### `test_adaptive_rate_allocator_coverage.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `class` | `_Chunk` |
| `class` | `_DummyCompressor` |
| `def` | `test_adaptive_allocator_complexity_and_selection` |
| `def` | `test_context_window_adapter_and_multilevel_encoder` |

#### `test_afm.py`

Tests for Adaptive Focus Memory (AFM) module

| Kind | Name |
|------|------|
| `class` | `TestTokenCounter` |
| `class` | `TestMessage` |
| `class` | `TestImportanceClassifier` |
| `class` | `TestHeuristicCompressor` |
| `class` | `TestFocusManager` |
| `class` | `TestRecencyWeighting` |
| `class` | `TestScoring` |

#### `test_afm_handlers.py`

Comprehensive tests for afm_handlers.py

| Kind | Name |
|------|------|
| `class` | `MockPackingStats` |
| `def` | `mock_context` |
| `def` | `mock_message` |
| `class` | `TestHandleAfmAddMessage` |
| `class` | `TestHandleAfmBuildContext` |
| `class` | `TestHandleAfmGetStats` |
| `class` | `TestHandleAfmClearHistory` |
| `class` | `TestHandleAfmExportHistory` |
| `class` | `TestHandleAfmImportHistory` |

#### `test_anchor_comi_survival.py`

Audit 2026-06-24 CRITICAL regression lock.

| Kind | Name |
|------|------|
| `def` | `_build_sectioned_doc` |
| `def` | `_extract_anchor_ids` |
| `def` | `test_comi_coarse_pass_never_drops_explicit_anchor` |

#### `test_assessment_handlers.py`

Tests for should_compress assessment handler (v0.9.2+).

| Kind | Name |
|------|------|
| `class` | `TestShouldCompressExtensionDetection` |
| `class` | `TestShouldCompressContentSniffing` |
| `class` | `TestShouldCompressTokenEstimation` |
| `class` | `TestShouldCompressPathSecurity` |
| `class` | `TestShouldCompressEdgeCases` |

#### `test_async_operations.py`

Tests for async operations in semantic compressor and handlers.

| Kind | Name |
|------|------|
| `def` | `compressor` |
| `def` | `handler_context` |
| `class` | `TestAsyncEncoding` |
| `class` | `TestConcurrentOperations` |
| `class` | `TestHandlerAsyncConversion` |
| `class` | `TestTimeoutPrevention` |
| `class` | `TestAsyncErrorHandling` |
| `class` | `TestPerformanceCharacteristics` |

#### `test_audit_194_async_chunking.py`

Audit #194 regression lock: async def functions/methods must be chunked.

| Kind | Name |
|------|------|
| `def` | `_bare` |
| `def` | `test_top_level_async_def_is_chunked` |
| `def` | `test_async_method_counted_in_class_methods` |

#### `test_audit_compression_correctness.py`

Enterprise audit — compression-engine correctness regression locks.

| Kind | Name |
|------|------|
| `class` | `_FakeEmbedder` |
| `def` | `fake_embedder` |
| `def` | `_make_compressor_with_chunks` |
| `class` | `TestRrfSufficiencyReachable` |
| `class` | `TestMigDoesNotMutateImportance` |
| `class` | `TestFileIdPrefixIsolation` |
| `class` | `TestTfidfSilentFallbackRaises` |
| `class` | `TestEmbeddingManagerSingletonTier` |
| `class` | `TestScoreTypeUnderPathC` |
| `class` | `TestGetStatsNoSkeletonSideEffect` |
| `class` | `TestBm25PrefixMatch` |

#### `test_audit_followups_134.py`

Audit follow-up #134 regression locks (2026-06-24).

| Kind | Name |
|------|------|
| `def` | `_node` |
| `def` | `test_find_duplicates_same_file_code_nodes_not_compared` |
| `def` | `test_max_dense_cosine_nan_query_returns_zero` |
| `def` | `test_select_skeleton_ordered_nan_query_falls_back_to_importance` |

#### `test_batch_processing.py`

Tests for batch processing functionality (v0.6.0)

| Kind | Name |
|------|------|
| `def` | `compressor` |
| `def` | `sample_documents` |
| `def` | `handler_context` |
| `class` | `TestBasicBatchIngestion` |
| `class` | `TestConcurrentOperations` |
| `class` | `TestErrorIsolation` |
| `class` | `TestProgressTracking` |
| `class` | `TestRetryFunctionality` |
| `class` | `TestUtilityFunctions` |
| `class` | `TestMCPToolIntegration` |
| `class` | `TestPerformanceCharacteristics` |

#### `test_benchmark_guard.py`

Tests for benchmark guard threshold checks.

| Kind | Name |
|------|------|
| `def` | `_write_json` |
| `def` | `test_evaluate_report_against_thresholds_passes_with_good_metrics` |
| `def` | `test_evaluate_report_against_thresholds_detects_regression` |
| `def` | `test_check_benchmark_guard_script_passes_and_fails` |
| `def` | `test_check_benchmark_guard_writes_summary_markdown` |
| `def` | `test_check_benchmark_guard_strict_case_set` |

#### `test_benchmark_harness.py`

Tests for fixed-corpus benchmark harness.

| Kind | Name |
|------|------|
| `def` | `test_load_benchmark_cases_has_expected_fixture` |
| `def` | `test_filter_cases_selects_subset` |
| `def` | `test_run_benchmark_cases_meets_golden_thresholds` |
| `def` | `test_run_benchmark_cases_query_guided_mode` |
| `def` | `test_summary_to_dict_and_write_output` |
| `def` | `test_run_benchmark_cases_includes_quality_metrics_when_query_present` |
| `def` | `test_summary_to_dict_contains_quality_metric_fields` |
| `def` | `test_run_method_comparison_includes_extractive_baseline` |

#### `test_blind_spot_detector_coverage.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `class` | `_Node` |
| `class` | `_Model` |
| `class` | `_Compressor` |
| `def` | `test_analyze_response_and_wrappers` |
| `def` | `test_halo_effect_detector_modes_and_missing_file` |

#### `test_bootstrap.py`

Contract tests for app bootstrap entrypoints.

| Kind | Name |
|------|------|
| `def` | `test_bootstrap_create_server_returns_server_instance` |
| `async def` | `test_bootstrap_async_main_uses_lifespan_and_run` |
| `def` | `test_bootstrap_main_delegates_to_run_function` |

#### `test_bugfixes.py`

Tests for bug fixes discovered during deep audit.

| Kind | Name |
|------|------|
| `class` | `TestComputeAdaptiveRatioValidation` |
| `class` | `TestComputeCostSavingsNegative` |
| `class` | `TestFindDuplicatesNullGuard` |
| `class` | `TestDiffReingestNullGuard` |
| `class` | `TestHandleIngestProtection` |
| `class` | `TestAtomicPersistence` |
| `class` | `TestNewMCPHandlers` |
| `class` | `TestToolRegistration` |
| `class` | `TestFindDuplicatesTimeout` |
| `class` | `TestFindDuplicatesTenantScoping` |
| `class` | `TestHandleBatchIngestTenantScoping` |
| `class` | `TestValidationHooksDestructive` |
| `class` | `TestMetricsWiring` |

#### `test_bundle_golden_outputs.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `class` | `_FakeTracker` |
| `class` | `_FakeReplay` |
| `class` | `_FakeTemporal` |
| `class` | `_FakeCompressor` |
| `def` | `test_distilled_bundle_outputs_are_stable` |

#### `test_bundle_handlers.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `class` | `_FakeTracker` |
| `class` | `_FakeReplay` |
| `class` | `_FakeTemporal` |
| `class` | `_FakeCompressor` |
| `def` | `reset_registry` |
| `def` | `handler_context` |
| `async def` | `test_create_and_fetch_handoff_bundle` |
| `async def` | `test_list_and_replay_handoff_bundle` |

#### `test_cache_compatibility.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_assess_cache_compatibility_for_gemini_cli_accepts_stats_visibility` |
| `def` | `test_assess_cache_compatibility_for_codex_flags_missing_visibility` |

#### `test_cache_comprehensive.py`

Comprehensive Cache Tests (v1.0.0 - Phase 1)

| Kind | Name |
|------|------|
| `def` | `temp_cache_file` |
| `def` | `sample_embeddings` |
| `class` | `TestTTLExpiration` |
| `class` | `TestPersistenceErrorHandling` |
| `class` | `TestEdgeCasesAndBoundaries` |
| `class` | `TestSingletonAndMemory` |
| `class` | `TestAdditionalCoverage` |

#### `test_cache_diagnostics.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `setup_function` |
| `def` | `test_diagnose_cache_miss_detects_framework_uuid_injection` |
| `def` | `test_diagnose_cache_miss_detects_timestamp_in_prefix` |
| `def` | `test_diagnose_cache_miss_reports_provider_eviction_when_prefix_matches` |
| `def` | `test_extract_section_order_reads_rendered_prompt_headers` |
| `def` | `test_detect_section_interleaving_reports_added_stable_sections` |
| `def` | `test_diagnose_cache_miss_detects_section_interleaving_before_uuid_injection` |
| `def` | `test_diagnose_cache_miss_detects_whitespace_only_semantic_drift` |
| `def` | `test_diagnose_cache_miss_detects_json_serialization_semantic_drift` |
| `def` | `test_diagnose_cache_miss_detects_partial_cache_reuse_underperformance` |
| `async def` | `test_diagnose_cache_miss_handler_returns_actionable_diagnostic` |
| `async def` | `test_route_tool_call_dispatches_diagnose_cache_miss` |

#### `test_cache_optimization.py`

Tests for Phase 1 cache optimization: cache-stable ordering and provider format hints.

| Kind | Name |
|------|------|
| `def` | `test_cache_stable_ordering_puts_status_first` |
| `def` | `test_cache_stable_ordering_volatile_last` |
| `def` | `test_cache_stable_codex_mirrors_at_tail` |
| `def` | `test_cache_stable_unknown_provider_noop` |
| `def` | `test_cache_stable_preserves_all_values` |
| `def` | `test_cache_stable_empty_dict` |
| `def` | `test_cache_stable_google_no_stable_summary` |
| `def` | `test_cache_stable_openai_no_stable_summary_when_no_status_or_file_id` |
| `def` | `test_format_response_with_provider_applies_ordering` |
| `def` | `test_format_response_provider_none_no_reordering` |
| `def` | `test_format_response_provider_does_not_break_truncation` |
| `def` | `test_format_response_provider_backward_compat` |
| `def` | `test_detect_provider_claude` |
| `def` | `test_detect_provider_haiku_prefix` |
| `def` | `test_detect_provider_gemini` |
| `def` | `test_detect_provider_gpt` |
| `def` | `test_detect_provider_codex` |
| `def` | `test_detect_provider_o3` |
| `def` | `test_detect_provider_o1` |
| `def` | `test_detect_provider_o4` |
| `def` | `test_detect_provider_unknown` |
| `def` | `test_client_config_has_provider_field` |
| `def` | `test_client_config_claude_format_hints` |
| `def` | `test_client_config_gemini_format_hints` |
| `def` | `test_client_config_codex_format_hints` |
| `def` | `test_client_config_gpt_format_hints` |
| `def` | `test_client_config_unknown_format_hints` |
| `def` | `test_client_config_cache_stable_ordering_default_true` |
| `def` | `test_client_config_backward_compatible` |

#### `test_cache_telemetry.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_summarize_anthropic_cache_hit_with_warning_free_result` |
| `def` | `test_summarize_openai_cached_tokens_from_nested_usage_details` |
| `def` | `test_summarize_gemini_cache_miss_emits_warning_when_hit_expected` |
| `def` | `test_summarize_gemini_usage_metadata_snake_case_payload` |
| `def` | `test_summarize_gemini_cli_stats_payload_with_camel_case_totals` |
| `def` | `test_summarize_missing_provider_field_raises_clear_error` |
| `def` | `test_normalize_cache_telemetry_flattens_validation_and_session_state` |
| `async def` | `test_capture_cache_telemetry_handler_normalizes_result` |

#### `test_cache_telemetry_contracts.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_capture_cache_telemetry_tool_schema_contract` |
| `def` | `test_diagnose_cache_miss_tool_schema_contract` |
| `async def` | `test_capture_cache_telemetry_help_documents_normalized_fields` |
| `async def` | `test_diagnose_cache_miss_help_documents_output_fields` |

#### `test_chaos_engineering.py`

Chaos Engineering Tests for Token Saver 5000 v0.7.0

| Kind | Name |
|------|------|
| `class` | `TestDiskFailures` |
| `class` | `TestModelCrashes` |
| `class` | `TestNetworkIssues` |
| `class` | `TestDataCorruption` |

#### `test_ci_workflows.py`

Tests for CI workflow definitions related to skill scripts.

| Kind | Name |
|------|------|
| `def` | `test_ci_workflow_exists_and_runs_canonical_validation` |
| `def` | `test_legacy_test_workflow_is_manual_deprecated_shim` |
| `def` | `test_legacy_lint_workflow_is_manual_deprecated_shim` |
| `def` | `test_skill_ci_workflow_exists` |
| `def` | `test_skill_ci_workflow_has_path_filters_and_commands` |
| `def` | `test_benchmark_guard_workflow_exists_and_has_regression_gate` |
| `def` | `test_mcp_profile_guard_workflow_exists_and_checks_core_contract` |

#### `test_claude_folder_guides.py`

Contracts for per-folder claude.md guides (determinism, line endings).

| Kind | Name |
|------|------|
| `def` | `_iter_claude_guides` |
| `def` | `test_claude_folder_guides_have_no_crlf` |
| `def` | `test_gitattributes_sets_lf_for_claude_guides` |

#### `test_cli_benchmark.py`

Comprehensive tests for the cli_benchmark package.

| Kind | Name |
|------|------|
| `class` | `TestLoadManifest` |
| `class` | `TestCorpusFiles` |
| `class` | `TestLoadCorpus` |
| `class` | `TestBuildPrompt` |
| `class` | `TestParseJsonOutput` |
| `class` | `TestParseClaudeResult` |
| `class` | `TestParseGeminiResult` |
| `class` | `TestIsAvailable` |
| `class` | `TestRunPromptDryRun` |
| `class` | `TestPricing` |
| `class` | `TestCLIResult` |
| `class` | `TestComparisonResult` |
| `class` | `TestBenchmarkReport` |
| `class` | `TestAggregateRepeats` |
| `class` | `TestProjectScaffold` |
| `class` | `TestRunBenchmarkDryRun` |

#### `test_cli_output_optimizer.py`

Tests for CLIOutputOptimizer (TDD — written before implementation).

| Kind | Name |
|------|------|
| `class` | `TestDetectCommand` |
| `class` | `TestAnsiStrip` |
| `class` | `TestGitDiffStats` |
| `class` | `TestGitStatusCompact` |
| `class` | `TestFailureFocus` |
| `class` | `TestInstallSummary` |
| `class` | `TestLintGroup` |
| `class` | `TestJsonStructure` |
| `class` | `TestLogDedup` |
| `class` | `TestProgressStrip` |
| `class` | `TestTreeCompress` |
| `class` | `TestFilterIntegration` |

#### `test_client_config.py`

Tests for src/client_config.py (ClientConfig class, SessionConfigStore).

| Kind | Name |
|------|------|
| `def` | `test_configure_for_known_claude_model` |
| `def` | `test_configure_for_known_gpt_model` |
| `def` | `test_configure_for_unknown_model_uses_default` |
| `def` | `test_configure_with_explicit_context_window` |
| `def` | `test_explicit_window_overrides_model_lookup` |
| `def` | `test_skeleton_ratio_is_in_valid_range` |
| `def` | `test_skeleton_ratio_scales_with_window_size` |
| `def` | `test_1m_window_ratio` |
| `def` | `test_50k_window_ratio` |
| `def` | `test_get_recommended_ratio` |
| `def` | `test_invalid_context_window_zero_rejected` |
| `def` | `test_invalid_context_window_negative_rejected` |
| `def` | `test_model_database_has_defaults` |
| `def` | `test_model_database_default_is_100k` |
| `def` | `test_session_config_persists` |
| `def` | `test_session_config_isolated` |
| `def` | `test_get_config_returns_default_when_unset` |
| `def` | `test_session_config_overwrite` |
| `def` | `test_session_store_isolated_across_instances` |

#### `test_code_compression_adapter.py`

Tests for CodeCompressionAdapter v0.9.0 fixes.

| Kind | Name |
|------|------|
| `class` | `TestCodeSkeletonWithNodeMap` |
| `class` | `TestFileIdWithSlashes` |
| `class` | `TestModulateRegionForCodeFiles` |
| `class` | `TestValidateNodeIdsWithSlashesAndColons` |
| `class` | `TestSkeletonTokensDocumentation` |
| `class` | `TestFidelityLevelBalancedDoesNotExist` |
| `class` | `TestDeletionDoesNotOverreach` |
| `class` | `TestDeleteDocumentFromMemoryPropertyCopy` |

#### `test_code_compressor.py`

Comprehensive tests for code_compressor.py (AST-based code compression)

| Kind | Name |
|------|------|
| `class` | `TestCodeLanguageDetection` |
| `class` | `TestPythonCodeChunking` |
| `class` | `TestJavaScriptCodeChunking` |
| `class` | `TestLineBasedChunking` |
| `class` | `TestCodeIngestion` |
| `class` | `TestCodeSkeleton` |
| `class` | `TestCodeSearch` |
| `class` | `TestCodeChunkRetrieval` |
| `class` | `TestCodeCompressorEdgeCases` |

#### `test_code_compressor_no_stdout_print.py`

Regression lock: code-ingest path must never write to stdout (2026-07-06).

| Kind | Name |
|------|------|
| `class` | `TestIngestCodeFileNeverPrintsToStdout` |
| `def` | `test_no_bare_print_calls_left_in_ingest_code_file` |

#### `test_code_dependency_edges.py`

Task #236 rank11 — dependency-edge building must be indexed, not quadratic.

| Kind | Name |
|------|------|
| `def` | `_chunk` |
| `def` | `_naive_reference_edges` |
| `class` | `_IterCountingList` |
| `class` | `TestDependencyEdgeParity` |
| `class` | `TestDependencyEdgeComplexity` |
| `class` | `TestIngestIntegrationModelFree` |

#### `test_code_graph_oom_fix.py`

Block-wise code-graph edge-building OOM fix (task #236 rank11).

| Kind | Name |
|------|------|
| `def` | `_emb` |
| `def` | `_reference_edges` |
| `def` | `_blockwise_edges` |
| `class` | `TestCodeBlockWiseEdgeEquivalence` |
| `class` | `TestCodeGraphBuildingMemoryBound` |

#### `test_codex_enhancements.py`

Tests for Codex CLI token optimization enhancements (v0.11.0).

| Kind | Name |
|------|------|
| `class` | `TestParseCodexJsonl` |
| `class` | `TestCodexModelsInDatabase` |
| `class` | `TestCodexCompressionTriggers` |
| `class` | `TestCodexPricing` |
| `class` | `TestCodexIsAvailable` |
| `class` | `TestCodexBuildCommand` |
| `class` | `TestCodexDryRun` |

#### `test_compress_manifest_handler.py`

Tests for the gc_compress_manifest handler (v1.8.0 A2b).

| Kind | Name |
|------|------|
| `def` | `_call` |
| `class` | `TestCompressManifestHandlerRoundTrip` |
| `class` | `TestCompressManifestHandlerSavings` |
| `class` | `TestCompressManifestHandlerEdgeCases` |

#### `test_compression_advisor_honesty.py`

#92 (2026-06-12): estimated_ratio honesty locks.

| Kind | Name |
|------|------|
| `def` | `_prose` |
| `class` | `TestEstimateMirrorsAdaptiveCurve` |

#### `test_compression_handlers.py`

Comprehensive tests for compression_handlers.py (v0.4.3).

| Kind | Name |
|------|------|
| `class` | `TestHandleIngest` |
| `def` | `_make_async_return` |
| `class` | `TestHandleIngestF4ChunkingStrategy` |
| `class` | `TestHandleIngestF12SavingsTrackerWired` |
| `class` | `TestHandleIngestF6InlineQuery` |
| `class` | `TestHandleReadSkeletonF12ClassCompletion` |
| `class` | `TestHandleReadSkeleton` |
| `class` | `TestHandleModulateRegionF10SingularNodeId` |
| `class` | `TestHandleModulateRegion` |
| `class` | `TestHandleSearchSemantic` |
| `class` | `TestSearchSemanticOutputFields` |
| `class` | `TestReadSkeletonOutputFields` |
| `class` | `TestIngestOutputFields` |
| `class` | `TestRecommendFidelityOutputFields` |
| `class` | `TestHandleGetStats` |
| `class` | `TestHandleListDocuments` |
| `class` | `TestHandleDeleteDocument` |
| `class` | `TestHandleAdaptToContextWindow` |
| `class` | `TestHandleMultilevelEncode` |
| `class` | `TestHandleRecommendFidelity` |
| `class` | `TestValidationHelpers` |

#### `test_compression_pipeline.py`

Tests for the composable read-skeleton compression pipeline.

| Kind | Name |
|------|------|
| `def` | `_skeleton` |
| `def` | `test_read_skeleton_pipeline_runs_baseline_then_evidence_stages` |
| `def` | `_make_compressor_for_auto` |
| `def` | `test_auto_mode_resolves_to_evidence_aware_for_structured_doc` |
| `def` | `test_auto_mode_resolves_to_baseline_for_plain_prose` |
| `def` | `test_auto_mode_resolves_to_baseline_when_raw_text_is_none` |
| `def` | `test_auto_mode_synthesises_h1_as_query_when_no_query_supplied` |
| `def` | `test_auto_mode_uses_explicit_query_when_supplied` |
| `def` | `test_selection_mode_resolved_present_in_baseline_path` |
| `def` | `test_selection_mode_resolved_present_in_query_guided_path` |
| `def` | `test_resolve_auto_mode_detects_structured_doc` |
| `def` | `test_resolve_auto_mode_falls_back_for_plain_prose` |
| `def` | `test_resolve_auto_mode_returns_baseline_for_empty_text` |
| `def` | `test_extract_h1_query_returns_heading_text` |
| `def` | `test_extract_h1_query_returns_none_when_no_h1` |
| `def` | `test_auto_mode_with_caller_query_on_prose_resolves_to_query_guided` |
| `def` | `test_auto_mode_with_caller_query_on_structured_doc_stays_evidence_aware` |

#### `test_compression_profiles.py`

Tests for src/compression_profiles.py module.

| Kind | Name |
|------|------|
| `def` | `test_minimal_profile_values` |
| `def` | `test_summary_profile_values` |
| `def` | `test_balanced_profile_values` |
| `def` | `test_detailed_profile_values` |
| `def` | `test_full_profile_values` |
| `def` | `test_get_profile_by_name` |
| `def` | `test_unknown_profile_raises` |
| `def` | `test_unknown_profile_error_mentions_valid_names` |
| `def` | `test_default_profile_is_balanced` |
| `def` | `test_default_profile_matches_constants` |
| `def` | `test_list_profiles` |
| `def` | `test_list_profiles_no_duplicates` |
| `def` | `test_all_skeleton_ratios_in_range` |
| `def` | `test_all_chunk_sizes_positive` |
| `def` | `test_skeleton_ratios_increase_with_profile_intensity` |
| `def` | `test_profile_manager_set_and_get` |
| `def` | `test_profile_manager_isolated_sessions` |
| `def` | `test_profile_manager_default_when_unset` |
| `def` | `test_profile_manager_overwrite` |
| `def` | `test_profile_manager_rejects_unknown_profile` |
| `def` | `test_profile_manager_isolated_across_instances` |
| `def` | `test_apply_profile_to_params` |
| `def` | `test_explicit_params_override_profile` |
| `def` | `test_apply_profile_partial_override` |
| `def` | `test_apply_profile_does_not_mutate_params` |
| `def` | `test_apply_profile_returns_new_dict` |
| `def` | `test_compression_profile_has_description` |
| `def` | `test_compression_profile_name_matches_key` |

#### `test_compression_rewards.py`

Tests for Compression Rewards System

| Kind | Name |
|------|------|
| `class` | `TestSchemaValidationResult` |
| `class` | `TestSemanticPreservationResult` |
| `class` | `TestFidelityAdherenceResult` |
| `class` | `TestCompositionIntegrityResult` |
| `class` | `TestMemoryDisciplineResult` |
| `class` | `TestCompressionReward` |
| `class` | `TestCompressionRewardCalculator` |
| `class` | `TestProgressiveRewardShaper` |
| `class` | `TestFidelityConstants` |

#### `test_compression_verifier.py`

Tests for Compression Verifier

| Kind | Name |
|------|------|
| `class` | `TestCompressionContractPreconditions` |
| `class` | `TestCompressionContractPostconditions` |
| `class` | `TestCompressionContractInvariants` |
| `class` | `TestCompressionVerifier` |
| `class` | `TestBatchVerifier` |
| `class` | `TestVerificationResult` |
| `class` | `TestContractViolation` |

#### `test_connector_handlers.py`

Tests for managed connector feed handlers.

| Kind | Name |
|------|------|
| `def` | `connector_context` |
| `async def` | `test_create_list_get_and_sync_connector_feed` |

#### `test_connector_registry.py`

Tests for connector registry and managed feeds.

| Kind | Name |
|------|------|
| `def` | `setup_function` |
| `def` | `test_lists_supported_connector_types` |
| `def` | `test_create_and_update_feed_sync_state` |

#### `test_context_blocks.py`

Tests for lifecycle-aware context block assembly.

| Kind | Name |
|------|------|
| `def` | `test_build_context_block_summarizes_access_and_compression` |

#### `test_context_service.py`

Contract tests for app-layer server context service.

| Kind | Name |
|------|------|
| `class` | `_Node` |
| `def` | `test_context_service_build_context_includes_contract_keys` |
| `def` | `test_context_service_contract_key_constants_align_with_expected_context_map` |
| `def` | `test_context_service_contract_key_mismatch_message_format` |
| `def` | `test_context_service_validate_context_map_rejects_extra_keys_with_exact_message` |
| `def` | `test_context_service_build_context_uses_validate_context_map_class_dispatch` |
| `def` | `test_context_service_validate_file_id_not_found_has_helpful_message` |
| `def` | `test_context_service_extract_file_id` |

#### `test_coverage_boost.py`

Coverage boost tests for uncovered code paths.

| Kind | Name |
|------|------|
| `class` | `TestHandleVisualizeGraphHtml` |
| `class` | `TestHandleExportGraphGraphml` |
| `class` | `TestHandleExplainCompressionDecision` |
| `class` | `TestHandleVerifyCompression` |
| `class` | `TestHandleCalculateReward` |
| `class` | `TestHandleGenerateSyntheticTests` |
| `def` | `_make_mock_compressor` |
| `class` | `TestGraphVisualizerExplainDecision` |
| `class` | `TestGraphVisualizerHtml` |
| `class` | `TestGraphVisualizerExportGraphml` |
| `class` | `TestGraphVisualizerRenderAscii` |
| `class` | `TestStressTestCompression` |
| `class` | `TestStressTestAFM` |
| `class` | `TestValidateBoundaryCases` |
| `class` | `TestCompressionPresets` |
| `class` | `TestValidationHooks` |

#### `test_coverage_boost2.py`

Coverage boost tests for persistence, compression handlers, code_compression_adapter,

| Kind | Name |
|------|------|
| `class` | `TestSaveGraphDataSafe` |
| `class` | `TestLoadGraphDataSafe` |
| `class` | `TestChromaDBPaths` |
| `class` | `TestAfmHistory` |
| `class` | `TestClearAll` |
| `class` | `TestChromaDBInitFailure` |
| `def` | `_make_handler_context` |
| `class` | `TestHandleDeleteDocument` |
| `class` | `TestHandleMultilevelEncode` |
| `class` | `TestHandleBatchIngest` |
| `class` | `TestHandleIngestDirectory` |
| `class` | `TestCodeCompressionAdapter` |
| `class` | `TestEmbeddingManager` |
| `class` | `TestONNXEmbeddingManager` |

#### `test_coverage_boost3.py`

Coverage boost tests - Round 3.

| Kind | Name |
|------|------|
| `def` | `_has_pillow` |
| `def` | `_make_mock_context` |
| `class` | `TestListDocumentsMetadata` |
| `class` | `TestDeleteDocumentErrors` |
| `class` | `TestAdaptToContextWindowError` |
| `class` | `TestMultilevelEncodeError` |
| `class` | `TestRecommendFidelityValidation` |
| `class` | `TestBatchIngestRateLimit` |
| `class` | `TestIngestDirectory` |
| `class` | `TestPersistenceChromaDB` |
| `class` | `TestPersistenceLoadGraphSafe` |
| `class` | `TestPersistenceAFMHistory` |
| `class` | `TestAFMComponents` |
| `class` | `TestONNXEmbeddings` |
| `class` | `TestGraphVisualizer` |
| `class` | `TestEmbeddingManager` |
| `class` | `TestVersionManager` |
| `class` | `TestResourceManager` |
| `class` | `TestBatchManager` |
| `class` | `TestFileSyncManager` |
| `class` | `TestObservability` |
| `class` | `TestMetrics` |
| `class` | `TestHealth` |
| `class` | `TestErrorTypes` |
| `class` | `TestErrorHelpers` |
| `class` | `TestCompressionRewards` |
| `class` | `TestEvidenceBundle` |
| `class` | `TestBenchmarkGuard` |
| `class` | `TestScarCompressor` |
| `class` | `TestResourceHandlers` |
| `class` | `TestAdaptiveRateAllocator` |
| `class` | `TestMultimodalCompressor` |
| `class` | `TestRateLimiter` |
| `class` | `TestPersistenceLoadDocJSON` |

#### `test_coverage_boost4.py`

Coverage boost tests - Round 4.

| Kind | Name |
|------|------|
| `def` | `_has_pillow` |
| `def` | `_make_mock_context` |
| `def` | `_make_semantic_node` |
| `def` | `_make_code_chunk` |
| `class` | `TestCodeCompressionAdapterProperties` |
| `class` | `TestCodeCompressionAdapterSkeleton` |
| `class` | `TestCodeCompressionAdapterCodeNodes` |
| `class` | `TestExperimentalHandlers` |
| `class` | `TestGraphVisualizer` |
| `class` | `TestResourceHandlersDiagnostics` |
| `class` | `TestEmbeddingsImports` |
| `class` | `TestMultimodalCompressor` |
| `class` | `TestSCAREnhancedCompressor` |
| `class` | `TestONNXEmbeddings` |
| `class` | `TestObservability` |
| `class` | `TestEvidenceBundle` |
| `class` | `TestCompressionHandlersValidation` |
| `class` | `TestCompressionHandlersBatch` |
| `class` | `TestPersistence` |
| `class` | `TestAdaptiveRateAllocator` |
| `class` | `TestStructuredLogging` |
| `class` | `TestHealth` |
| `class` | `TestACEHandlersRateLimit` |
| `class` | `TestBlindSpotDetector` |
| `class` | `TestAFMHandlersRateLimit` |
| `class` | `TestCompressionHandlersModulate` |

#### `test_cross_encoder_scorer.py`

Tests for CrossEncoderScorer (#187 rerank model-wire).

| Kind | Name |
|------|------|
| `def` | `test_empty_documents_returns_empty_without_loading` |
| `def` | `test_usable_as_injected_scorer_when_rerank_disabled_no_load` |
| `def` | `_model_cached` |
| `def` | `test_real_onnx_cross_encoder_ranks_relevant_above_irrelevant` |

#### `test_cujs.py`

Tests for Critical User Journey (CUJ) baselines.

| Kind | Name |
|------|------|
| `def` | `cuj1` |
| `def` | `test_cuj_1_passes` |
| `def` | `test_cuj_1_finds_relevant_files` |
| `def` | `test_cuj_1_compresses_codebase` |
| `def` | `test_cuj_1_search_beats_naive` |
| `def` | `test_cuj_1_savings_above_50pct` |
| `def` | `test_cuj_1_has_configure_step` |
| `def` | `cuj2` |
| `def` | `test_cuj_2_passes` |
| `def` | `test_cuj_2_compresses_document` |
| `def` | `test_cuj_2_result_has_steps` |
| `def` | `test_cuj_2_savings_positive` |
| `def` | `test_cuj_2_output_tokens_reduced` |
| `def` | `cuj3` |
| `def` | `test_cuj_3_passes` |
| `def` | `test_cuj_3_git_diff_filtered` |
| `def` | `test_cuj_3_pytest_focuses_failures` |
| `def` | `test_cuj_3_npm_strips_progress` |
| `def` | `test_cuj_3_all_three_steps_present` |
| `def` | `test_cuj_3_savings_positive` |
| `def` | `cuj4` |
| `def` | `test_cuj_4_passes` |
| `def` | `test_cuj_4_search_finds_cache` |
| `def` | `test_cuj_4_beats_compress_all` |
| `def` | `test_cuj_4_three_strategies_compared` |
| `def` | `test_cuj_4_savings_above_90pct` |
| `def` | `cuj5` |
| `def` | `test_cuj_5_passes` |
| `def` | `test_cuj_5_recovery_has_files` |
| `def` | `test_cuj_5_recovery_has_config` |
| `def` | `test_cuj_5_recovery_compact` |
| `def` | `test_cuj_5_savings_vs_reingest` |
| `def` | `cuj6` |
| `def` | `test_cuj_6_passes` |
| `def` | `test_cuj_6_dollars_saved_positive` |
| `def` | `test_cuj_6_roi_computed` |
| `def` | `test_cuj_6_breakeven_reasonable` |
| `def` | `test_cuj_6_monthly_savings_exceeds_pro_plan` |
| `def` | `test_cuj_6_compression_ratio_above_10x` |
| `def` | `cuj13` |
| `def` | `test_cuj_13_passes` |
| `def` | `test_cuj_13_ingests_transcripts` |
| `def` | `test_cuj_13_compiles_articles` |
| `def` | `test_cuj_13_deduplicates` |
| `def` | `test_cuj_13_lints_knowledge` |
| `def` | `test_cuj_13_searches_compiled_index` |
| `def` | `test_cuj_13_has_all_four_steps` |
| `def` | `test_cuj_13_savings_positive` |
| `def` | `full_baseline` |
| `def` | `test_all_cujs_pass` |
| `def` | `test_baseline_has_thirteen_journeys` |
| `def` | `test_baseline_summary_populated` |
| `def` | `test_baseline_output_json_serializable` |
| `def` | `test_git_diff_fixture_has_sufficient_lines` |
| `def` | `test_pytest_fixture_has_failures` |
| `def` | `test_npm_fixture_has_warning` |

#### `test_dataset_registry.py`

Tests for named experiment datasets.

| Kind | Name |
|------|------|
| `def` | `setup_function` |
| `def` | `test_seeded_benchmark_dataset_exists` |
| `def` | `test_create_and_get_inline_dataset` |

#### `test_dead_code_detector.py`

Tests for dead_code_detector.py -- import-graph-based dead file detection.

| Kind | Name |
|------|------|
| `def` | `test_extract_imports_from_statement` |
| `def` | `test_extract_from_imports` |
| `def` | `test_is_entry_point_server` |
| `def` | `test_is_entry_point_init` |
| `def` | `test_is_never_dead_test_prefix` |
| `def` | `test_is_never_dead_conftest` |
| `def` | `test_detect_report_fields` |
| `def` | `test_detect_test_files_live` |
| `def` | `test_detect_entry_points_live` |
| `def` | `test_detect_tokens_saved_non_negative` |
| `def` | `test_detect_tokens_saved_positive_when_dead` |
| `def` | `test_detect_empty_directory` |
| `def` | `test_detect_single_file_is_live` |
| `def` | `test_detect_imported_files_live` |
| `def` | `test_detect_unimported_file_is_dead` |
| `def` | `test_detect_handles_missing_files` |
| `def` | `test_detect_all_entry_patterns_live` |

#### `test_dead_code_detector_perf.py`

`detect_dead_files` must not be quadratic in realpath syscalls.

| Kind | Name |
|------|------|
| `def` | `_write_pkg` |
| `def` | `test_resolve_calls_do_not_grow_quadratically` |
| `def` | `test_the_import_graph_is_still_correct` |
| `def` | `test_a_broken_symlink_does_not_abort_the_scan` |

#### `test_detection_handlers.py`

Comprehensive tests for detection_handlers.py

| Kind | Name |
|------|------|
| `class` | `MockBlindSpot` |
| `class` | `MockBlindSpotReport` |
| `def` | `mock_context` |
| `def` | `mock_rate_limiter` |
| `class` | `TestHandleCheckBlindSpots` |
| `class` | `TestHandleDetectHallucination` |
| `class` | `TestDetectionHandlersIntegration` |

#### `test_docs_contracts.py`

Documentation and help-surface contract tests for launch readiness.

| Kind | Name |
|------|------|
| `def` | `test_help_registry_covers_all_registered_mcp_tools` |
| `async def` | `test_tool_help_list_reports_real_tool_count` |
| `async def` | `test_formerly_missing_tools_return_help` |

#### `test_docs_handlers.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `docs_fixture` |
| `async def` | `test_search_returns_results_above_threshold` |
| `async def` | `test_search_top_k_respected` |
| `async def` | `test_read_doc_by_slug_returns_markdown` |
| `async def` | `test_read_doc_unindexed_slug_returns_suggestions_not_noisy_blob` |
| `async def` | `test_read_doc_truncates_long_responses` |
| `async def` | `test_session_dedup_does_not_repeat_urls` |
| `def` | `test_docs_root_returns_none_when_local_path_missing` |
| `def` | `test_read_llms_txt_falls_back_to_live_url_when_local_missing` |
| `def` | `test_read_llms_txt_returns_empty_on_url_fetch_failure` |
| `async def` | `test_search_stemming_matches_morphological_variants` |
| `async def` | `test_search_short_query_requires_exact_match` |
| `def` | `test_term_freq_with_stemming_short_term_exact_only` |
| `def` | `test_term_freq_with_stemming_long_term_prefix_match` |
| `def` | `test_term_freq_with_stemming_python_does_not_match_pythonic` |
| `async def` | `test_read_doc_enforces_character_cap_on_pathological_output` |

#### `test_e2e_scenarios.py`

End-to-End Scenario Tests for Token Saver 5000 v0.7.0.

| Kind | Name |
|------|------|
| `class` | `TestResearchPaperWorkflow` |
| `class` | `TestCodebaseWorkflow` |
| `class` | `TestDialogueWorkflow` |

#### `test_edge_cases.py`

Comprehensive Edge Case and Error Handling Tests

| Kind | Name |
|------|------|
| `class` | `TestInputValidation` |
| `class` | `TestResourceLimits` |
| `class` | `TestAFMEdgeCases` |
| `class` | `TestDataIntegrity` |
| `class` | `TestConcurrency` |
| `class` | `TestValidationHelpers` |
| `class` | `TestErrorRecovery` |
| `class` | `TestSemanticCompressorAdvanced` |
| `class` | `TestAFMAdvanced` |
| `class` | `TestFileSyncEdgeCases` |
| `class` | `TestVersionManagerEdgeCases` |

#### `test_edgecase_hardening_212.py`

Task #212 edge-case hardening (2026-07-12 gotcontext.ai engine audit).

| Kind | Name |
|------|------|
| `def` | `_bare_semantic_compressor` |
| `def` | `test_dense_cjk_oversized_paragraph_splits_into_multiple_chunks` |
| `def` | `test_dense_cjk_semantic_strategy_also_splits` |
| `def` | `test_ascii_oversized_paragraph_regression_unchanged` |
| `def` | `test_find_intra_duplicates_respects_max_nodes_cap` |
| `def` | `test_intra_dedup_bounded_time_on_large_near_duplicate_corpus` |
| `def` | `test_minified_single_line_js_multiple_functions_bounded` |
| `def` | `test_rust_generic_signature_never_bisected_mid_line` |
| `def` | `test_go_generic_signature_never_bisected_mid_line` |
| `def` | `test_split_json_records_preserves_ambiguous_leaf_types_byte_identical` |
| `def` | `test_detect_structured_content_whitespace_variants_return_none` |
| `def` | `manual_case6_needle_in_long_table_survives_at_any_position` |
| `def` | `manual_case12_fact_at_midpoint_vs_boundary_recall_parity` |

#### `test_embeddings_hf_import.py`

Regression tests for src/embeddings.py import hygiene.

| Kind | Name |
|------|------|
| `def` | `_make_fake_hf_tqdm` |
| `def` | `test_enable_progress_bars_imported_from_tqdm_submodule` |
| `def` | `test_enable_progress_bars_import_failure_is_swallowed` |

#### `test_embeddings_offline_recursion.py`

Regression tests for the offline/degraded embedding infinite-recursion bug.

| Kind | Name |
|------|------|
| `def` | `_isolated_singleton` |
| `def` | `_force_no_sentence_transformers` |
| `def` | `test_offline_no_onnx_raises_clean_runtime_error_without_recursion` |
| `def` | `test_offline_onnx_model_unloadable_does_not_recurse` |
| `def` | `test_offline_no_onnx_request_onnx_tier_also_bounded` |
| `def` | `test_adapter_raises_immediately_when_onnx_unavailable` |
| `def` | `test_adapter_exposes_get_sentence_embedding_dimension` |

#### `test_enterprise_layout.py`

Contract tests for enterprise package layout scaffolding.

| Kind | Name |
|------|------|
| `def` | `test_semantic_modulator_package_exposes_version` |
| `def` | `test_mcp_registry_wrapper_matches_legacy_setup` |
| `async def` | `test_mcp_router_wrapper_exposes_route_tool_call` |
| `def` | `test_bootstrap_wrapper_exposes_server_factory` |
| `def` | `test_server_uses_enterprise_tooling_gateway` |

#### `test_estimate_accuracy_relative.py`

`estimate_accuracy` must grade RELATIVE error, not absolute ratio points.

| Kind | Name |
|------|------|
| `class` | `TestLiveDogfoodCases` |
| `class` | `TestScaleInvariance` |
| `class` | `TestBandsAreOrdered` |
| `class` | `TestDegenerateInputs` |
| `class` | `TestCodexReviewFindings` |

#### `test_estimate_positive_ratio.py`

#142 regression: the ingest estimate ratio must stay POSITIVE for high skeleton

| Kind | Name |
|------|------|
| `def` | `test_estimate_ratio_stays_positive_for_high_skeleton_ratio` |
| `def` | `test_estimate_designed_band_boost_preserved` |
| `def` | `test_medium_doc_never_predicts_expansion` |

#### `test_evidence_bundle.py`

Tests for Evidence Bundle System

| Kind | Name |
|------|------|
| `class` | `TestContractCheck` |
| `class` | `TestContractResult` |
| `class` | `TestQualityMetrics` |
| `class` | `TestEvidenceBundle` |
| `class` | `TestEvidenceStore` |
| `class` | `TestGlobalStore` |

#### `test_experience_synthesis.py`

Tests for Experience Synthesis Module

| Kind | Name |
|------|------|
| `class` | `TestSyntheticDocument` |
| `class` | `TestStressTestResult` |
| `class` | `TestExperienceSynthesizerDocuments` |
| `class` | `TestExperienceSynthesizerDialogues` |
| `class` | `TestExperienceSynthesizerACE` |
| `class` | `TestBoundaryTestSuite` |
| `class` | `TestRandomGeneration` |
| `class` | `TestTestCategories` |
| `class` | `TestStressTestInfrastructure` |
| `class` | `TestMixedLanguageDocument` |
| `class` | `TestDeepNestedStructure` |

#### `test_experiment_golden_reports.py`

Golden regression tests for experiment report shapes.

| Kind | Name |
|------|------|
| `def` | `setup_function` |
| `def` | `test_experiment_run_shape_is_stable` |

#### `test_experiment_handlers.py`

Tests for dataset and experiment MCP handlers.

| Kind | Name |
|------|------|
| `def` | `experiment_context` |
| `async def` | `test_create_list_run_get_and_compare_experiment` |
| `async def` | `test_experiment_tools_are_registered_and_routable` |

#### `test_experiment_tracker.py`

Tests for tracked experiment runs.

| Kind | Name |
|------|------|
| `def` | `setup_function` |
| `def` | `test_run_and_compare_experiments` |
| `def` | `test_get_unknown_run_raises` |

#### `test_experimental_handlers.py`

Tests for experimental handlers.

| Kind | Name |
|------|------|
| `def` | `mock_handler_context` |
| `def` | `mock_handler_context_no_validator` |
| `class` | `TestTOONEncode` |
| `class` | `TestTOONDecode` |
| `class` | `TestSCARCompress` |
| `class` | `TestSCARGetStats` |
| `class` | `TestMultimodalIngest` |
| `class` | `TestExperimentalHandlerRegistry` |
| `class` | `TestMCPCoreIntegration` |
| `class` | `TestOptionalDependencies` |

#### `test_extractive_anchor.py`

Model-free unit tests for `_extractive_anchor_content` (world-class audit #1).

| Kind | Name |
|------|------|
| `def` | `_bare` |
| `def` | `test_empty_text_or_zero_budget_returns_empty` |
| `def` | `test_keeps_multiple_sentences_up_to_budget` |
| `def` | `test_at_least_one_sentence_even_if_over_budget` |
| `def` | `test_strips_markdown_noise` |
| `def` | `test_budget_truncates_before_second_sentence` |
| `def` | `test_deterministic_source_order_preserved` |
| `def` | `test_strips_bold_italic_emphasis` |

#### `test_extractive_baseline.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_extractive_compressor_prioritizes_query_relevant_sentences` |

#### `test_f11_fixture_corpus.py`

Regression locks + measurement smoke for the #250 F11 multi-node fixture

| Kind | Name |
|------|------|
| `class` | `TestParaphraseQueriesHaveZeroContentOverlap` |
| `class` | `TestLexicalTrapConstructionIsValid` |
| `class` | `TestQueryClassCoverage` |
| `class` | `TestChunkCountEngagesF11` |
| `class` | `TestHarnessRunsEndToEndOnBothPaths` |

#### `test_f11_gated_fusion.py`

F11 Path G — gated fusion (design memo idea #1) — TDD unit tests.

| Kind | Name |
|------|------|
| `class` | `TestQueryHasLexicalShape` |
| `class` | `TestBlocker1QuotedPhraseSignalIsDoubleQuoteOnly` |
| `class` | `TestBm25Top1IsDiscriminative` |
| `class` | `TestBlocker2GateUsesPrefixStemLikeBm25Scorer` |
| `class` | `TestGateShouldFuseGCombinator` |
| `class` | `_FakeEmbedder` |
| `def` | `fake_embedder` |
| `def` | `_make_compressor_with_chunks` |
| `def` | `_set_path` |
| `class` | `TestPathGMatchesPathAWhenGateClosed` |
| `class` | `TestPathGMatchesPathCWhenGateOpen` |
| `class` | `TestPathAAndPathCRemainUntouched` |
| `class` | `TestBlocker3ScoreTypeLabelReflectsWhatRan` |

#### `test_f11_header_aware_chunking.py`

tests/test_f11_header_aware_chunking.py

| Kind | Name |
|------|------|
| `def` | `compressor` |
| `def` | `test_header_section_surfaces_at_top_with_matching_query` |
| `def` | `test_structured_gate_triggers_on_headings_without_list_items` |
| `def` | `test_oversized_section_retains_heading_prefix` |
| `def` | `test_list_structured_doc_still_splits` |
| `def` | `test_heading_metadata_stored_on_nodes` |

#### `test_f11_path_c.py`

F11 Path C — BM25 + Reciprocal Rank Fusion hybrid retrieval tests.

| Kind | Name |
|------|------|
| `class` | `_FakeEmbedder` |
| `def` | `fake_embedder` |
| `def` | `_make_compressor_with_chunks` |
| `class` | `TestBM25Utils` |
| `class` | `TestBM25ScoresForNodes` |
| `class` | `TestIdfPollutionRegression` |
| `class` | `TestRRFFuse` |
| `class` | `TestSearchSemanticWithScoresPathDispatch` |
| `class` | `TestScoreTypeInHandlerResponse` |

#### `test_file_sync.py`

Test File Sync and Version Management Systems

| Kind | Name |
|------|------|
| `class` | `TestFileSyncManager` |
| `class` | `TestVersionManager` |
| `class` | `TestACEContextManager` |
| `class` | `TestIntegration` |
| `class` | `TestThreadSafety` |

#### `test_file_sync_handlers.py`

Comprehensive Tests for File Sync Handler Functions

| Kind | Name |
|------|------|
| `class` | `TestHandleCheckFileSync` |
| `class` | `TestHandleDiffCachedFile` |
| `class` | `TestHandleRefreshDocument` |
| `class` | `TestHandleGetVersionHistory` |
| `class` | `TestFileSyncHandlersIntegration` |

#### `test_filter_rules.py`

Tests for the custom filter rules DSL engine.

| Kind | Name |
|------|------|
| `class` | `TestFilterRule` |
| `class` | `TestFilterRulePipeline` |
| `class` | `TestFilterRuleSet` |
| `class` | `TestFilterRuleEngineLoad` |
| `class` | `TestFilterRuleEngineApply` |
| `class` | `TestFilterRuleEngineVerify` |
| `class` | `TestConfigPaths` |

#### `test_functional.py`

Functional Tests for Semantic Modulator

| Kind | Name |
|------|------|
| `class` | `TestBasicFunctionality` |
| `class` | `TestBlindSpotDetection` |
| `class` | `TestSCARFunctionality` |
| `class` | `TestEdgeCases` |
| `def` | `run_all_tests` |

#### `test_gemini_enhancements.py`

Tests for Gemini CLI token optimization enhancements (v0.12.0).

| Kind | Name |
|------|------|
| `def` | `estimator` |
| `def` | `test_gemini_estimate_ascii_only` |
| `def` | `test_gemini_estimate_non_ascii` |
| `def` | `test_gemini_estimate_mixed` |
| `def` | `test_gemini_estimate_long_text_fallback` |
| `def` | `test_gemini_estimate_exactly_100k_uses_per_char_formula` |
| `def` | `test_gemini_estimate_empty` |
| `def` | `test_gemini_estimate_single_ascii` |
| `def` | `test_gemini_estimate_four_ascii` |
| `def` | `test_estimate_all_includes_gemini` |
| `def` | `test_estimate_all_gemini_value_matches_estimate_gemini` |
| `def` | `test_estimate_all_gemini_is_non_negative` |
| `def` | `test_proportional_truncation_preserves_head_and_tail` |
| `def` | `test_proportional_truncation_sets_truncated_true` |
| `def` | `test_proportional_truncation_no_continuation_token` |
| `def` | `test_proportional_small_response_unchanged` |
| `def` | `test_default_strategy_is_paginate` |
| `def` | `test_head_strategy_keeps_first_n_chars` |
| `def` | `test_head_strategy_small_response_passes_through` |
| `def` | `test_proportional_response_contains_truncation_prefix` |
| `def` | `test_header_added_when_tool_name_provided` |
| `def` | `test_header_under_200_chars` |
| `def` | `test_header_is_first_key` |
| `def` | `test_header_contains_tool_name` |
| `def` | `test_header_contains_token_saver_brand` |
| `def` | `test_no_header_when_tool_name_omitted` |
| `def` | `test_header_format_includes_pipe_separators` |
| `def` | `test_urgency_normal_no_change` |
| `def` | `test_urgency_compact_caps_ratio` |
| `def` | `test_urgency_compact_caps_fidelity` |
| `def` | `test_urgency_compact_preserves_already_compact_ratio` |
| `def` | `test_urgency_compact_preserves_already_compact_fidelity` |
| `def` | `test_urgency_emergency_forces_minimal` |
| `def` | `test_urgency_emergency_overrides_already_minimal` |
| `def` | `test_urgency_unknown_raises_error` |
| `def` | `test_urgency_does_not_mutate_input` |
| `def` | `test_urgency_returns_new_dict` |
| `def` | `test_gemini_2_5_pro_in_database` |
| `def` | `test_gemini_2_5_flash_in_database` |
| `def` | `test_gemini_3_1_pro_in_database` |
| `def` | `test_gemini_3_1_flash_in_database` |
| `def` | `test_gemini_3_1_flash_lite_in_database` |
| `def` | `test_gemini_context_window_is_1m` |
| `def` | `test_existing_models_context_windows_unchanged` |
| `def` | `test_compression_triggers_dict_exists` |
| `def` | `test_gemini_compression_trigger_is_50pct` |
| `def` | `test_claude_compression_trigger_is_93pct` |
| `def` | `test_gpt4o_compression_trigger_is_1` |
| `def` | `test_default_compression_trigger` |
| `def` | `test_compression_trigger_stored_in_config` |
| `def` | `test_from_model_reads_compression_trigger` |
| `def` | `test_from_model_claude_compression_trigger` |
| `def` | `test_from_model_unknown_model_gets_default_trigger` |
| `def` | `test_gemini_model_gets_lower_ratio_than_claude` |
| `def` | `test_get_recommended_ratio_with_compression_trigger` |
| `def` | `test_get_recommended_ratio_default_trigger_is_0_80` |
| `def` | `test_compression_trigger_does_not_break_ratio_range` |

#### `test_generic_conservative_strategy.py`

Tests for the #137 generic conservative fallback strategy in

| Kind | Name |
|------|------|
| `def` | `_retry_storm_lines` |
| `def` | `_distinct_batch_lines` |
| `def` | `_python_traceback_lines` |
| `def` | `build_retry_storm_fixture` |
| `class` | `TestStrategyMapAndFilterWiring` |
| `def` | `_py_traceback` |
| `class` | `TestLogOutputGenericFallback137c` |
| `class` | `TestCrResolve` |
| `class` | `TestProgressNoiseDrop` |
| `class` | `TestExactDupCollapse` |
| `class` | `TestMaskedNearDupCollapse` |
| `class` | `TestDistinctNumericData` |
| `class` | `TestStructuredContentBypass` |
| `class` | `TestLengthCeiling` |
| `class` | `TestStackFrameElision` |
| `class` | `TestBlankRunCollapse` |
| `class` | `TestTimestampOnlyLineDrop` |
| `class` | `TestRetryStormFixture` |
| `class` | `TestUniqueProseHonestFloor` |

#### `test_github_connector.py`

Tests for GitHub connector normalization.

| Kind | Name |
|------|------|
| `def` | `test_collects_repo_files_into_documents` |

#### `test_graph_oom_fix.py`

Tests for the block-wise graph edge-building OOM fix.

| Kind | Name |
|------|------|
| `def` | `_make_l2_normalised` |
| `def` | `_reference_edges` |
| `def` | `_blockwise_edges` |
| `class` | `TestBlockWiseEdgeEquivalence` |
| `class` | `TestGraphBuildingMemoryBound` |

#### `test_gtm_benchmarks.py`

GTM claim reproduction suite.

| Kind | Name |
|------|------|
| `class` | `TestDocumentCompressionClaims` |
| `class` | `TestMeasuredCompressionRegressionLocks` |
| `class` | `TestCLIOutputOptimizerClaims` |
| `class` | `TestSchemaCompressionClaims` |
| `class` | `TestROIClaims` |
| `class` | `TestToolCountClaims` |
| `class` | `TestFeatureExistence` |

#### `test_handler_chunk_selection_boundary.py`

Handler-side chunk selection must be boundary-safe (#420).

| Kind | Name |
|------|------|
| `class` | `_Node` |
| `def` | `_corpus` |
| `class` | `TestPrefixSiblingIsolation` |
| `class` | `TestCodeNodeIdsAlsoIsolate` |
| `class` | `TestOrdering` |
| `class` | `TestEmptyAndMissing` |

#### `test_handler_encode_offload.py`

Receipts for roadmap A3: the read_skeleton + search_semantic MCP handlers run

| Kind | Name |
|------|------|
| `def` | `_loop_thread_id` |
| `class` | `_RecordingCompressor` |
| `def` | `_search_context` |
| `class` | `TestSearchSemanticOffload` |
| `class` | `TestReadSkeletonOffload` |

#### `test_health.py`

Tests for health checks and diagnostics.

| Kind | Name |
|------|------|
| `class` | `TestBasicHealthChecks` |
| `class` | `TestComponentHealth` |
| `class` | `TestHealthStatus` |
| `class` | `TestCaching` |
| `class` | `TestPerformanceMetrics` |
| `class` | `TestDiagnostics` |
| `class` | `TestComponentHealthDataclass` |
| `class` | `TestHealthStatusEnum` |
| `class` | `TestEdgeCases` |

#### `test_help_handlers.py`

Tests for help handler documentation registry and responses.

| Kind | Name |
|------|------|
| `async def` | `test_read_skeleton_help_includes_query_guided_params` |
| `async def` | `test_search_semantic_help_includes_evidence_aware_params` |
| `async def` | `test_modulate_region_help_uses_fidelity_level_parameter_name` |
| `async def` | `test_cache_compatibility_help_includes_visibility_controls` |
| `async def` | `test_tool_list_response_contains_total_tools` |
| `async def` | `test_check_environment_help_mentions_tool_profile_diagnostics` |
| `async def` | `test_check_environment_help_output_fields_match_runtime_keys` |
| `async def` | `test_check_environment_help_output_fields_match_canonical_schema` |
| `async def` | `test_search_semantic_help_output_fields_match_canonical_schema` |
| `async def` | `test_search_semantic_help_output_fields_cover_runtime_keys` |
| `async def` | `test_read_skeleton_help_output_fields_match_canonical_schema` |
| `async def` | `test_read_skeleton_help_output_fields_cover_runtime_keys` |
| `async def` | `test_ingest_context_help_output_fields_match_canonical_schema` |
| `async def` | `test_ingest_context_help_output_fields_cover_runtime_keys` |
| `async def` | `test_recommend_fidelity_help_output_fields_match_canonical_schema` |
| `async def` | `test_recommend_fidelity_help_output_fields_cover_runtime_keys` |
| `async def` | `test_high_value_tools_have_help_entries` |
| `async def` | `test_tool_list_response_mentions_real_workflow_tools` |

#### `test_history_compaction.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_history_compaction_preserves_recent_turns_at_tail` |

#### `test_http_server.py`

Tests for HTTP Server Module

| Kind | Name |
|------|------|
| `class` | `TestHTTPEndpoints` |
| `class` | `TestHTTPIntegration` |
| `class` | `TestHTTPConfiguration` |
| `class` | `TestHTTPLifecycle` |
| `def` | `test_get_http_config_structure` |
| `def` | `test_is_http_enabled_function_exists` |
| `async def` | `test_create_app_returns_application` |
| `async def` | `test_handler_functions_are_async` |
| `async def` | `test_handlers_accept_request_parameter` |
| `class` | `TestHTTPEdgeCases` |

#### `test_identifier_preservation.py`

Tests for engine identifier-preservation + the proxy ResponseInterceptor wiring.

| Kind | Name |
|------|------|
| `class` | `TestExtractCriticalIdentifiers` |
| `class` | `TestApplyIdentifierGuard` |
| `class` | `TestInterceptorIdentifierPreservation` |
| `class` | `TestHardeningBounds` |
| `class` | `TestNumericLiteralOrderingFix` |

#### `test_integration_workflows.py`

Integration Workflow Tests for Token Saver 5000 v0.7.0

| Kind | Name |
|------|------|
| `class` | `TestBasicWorkflows` |
| `class` | `TestFileSyncWorkflows` |
| `class` | `TestVersionHistory` |
| `class` | `TestACEWorkflows` |
| `class` | `TestAFMWorkflows` |
| `class` | `TestBatchProcessingIntegration` |
| `class` | `TestCrossFeatureIntegration` |

#### `test_js_nested_brace_chunking.py`

#212(b) (backlog master-plan Wave 1, 2026-07-12): the JS/TS chunker captured a

| Kind | Name |
|------|------|
| `def` | `_cc` |
| `def` | `_body` |
| `def` | `test_nested_if_block_keeps_tail` |
| `def` | `test_object_literal_braces_preserved` |
| `def` | `test_brace_inside_string_does_not_close_early` |
| `def` | `test_brace_inside_comment_does_not_close_early` |
| `def` | `test_unbalanced_function_does_not_crash` |
| `def` | `test_two_top_level_functions_both_captured_full` |
| `def` | `test_simple_function_unchanged_regression` |
| `def` | `test_regex_char_class_brace_not_structural` |
| `def` | `test_regex_looking_like_block_comment_not_dropped` |
| `def` | `test_return_regex_with_brace_not_truncated` |
| `def` | `test_division_not_misread_as_regex` |

#### `test_knowledge_compiler.py`

Tests for knowledge compiler (src/knowledge_compiler.py).

| Kind | Name |
|------|------|
| `def` | `reset_memory` |
| `def` | `_populate_memories` |
| `class` | `TestConceptArticle` |
| `class` | `TestKnowledgeCompiler` |

#### `test_knowledge_handlers.py`

Tests for knowledge management MCP handlers (ingest_transcript, compile, lint, index search).

| Kind | Name |
|------|------|
| `def` | `memory_context` |
| `async def` | `test_ingest_transcript_handler` |
| `async def` | `test_ingest_transcript_mode_decisions` |
| `async def` | `test_ingest_transcript_empty_text_rejected` |
| `async def` | `test_ingest_transcript_invalid_mode_rejected` |
| `async def` | `test_ingest_transcript_with_scoping` |
| `async def` | `test_compile_knowledge_handler` |
| `async def` | `test_compile_knowledge_empty` |
| `async def` | `test_compile_knowledge_write_files` |
| `async def` | `test_get_knowledge_index_handler` |
| `async def` | `test_lint_knowledge_handler` |
| `async def` | `test_lint_knowledge_invalid_stale_days` |
| `async def` | `test_search_memory_index_handler` |
| `async def` | `test_search_memory_index_no_match` |
| `async def` | `test_search_memory_index_requires_query` |
| `async def` | `test_knowledge_tools_registered_in_router` |

#### `test_knowledge_lint.py`

Tests for knowledge lint (src/knowledge_lint.py).

| Kind | Name |
|------|------|
| `def` | `reset_memory` |
| `def` | `_utc_iso` |
| `def` | `_make_memory` |
| `class` | `TestLintFinding` |
| `class` | `TestLintReport` |
| `class` | `TestStaleCheck` |
| `class` | `TestDuplicateCheck` |
| `class` | `TestContradictionCheck` |
| `class` | `TestACEDecayCheck` |
| `class` | `TestOrphanCheck` |
| `class` | `TestLintFromAPI` |

#### `test_launch_readiness.py`

Launch-readiness documentation tests for Phase 10.

| Kind | Name |
|------|------|
| `def` | `_read` |
| `def` | `test_readme_mentions_multi_tenant_and_web_api_positioning` |
| `def` | `test_saas_multi_tenant_guide_exists_and_covers_scope_fields` |
| `def` | `test_workflow_orchestration_guide_exists_and_covers_core_sequences` |
| `def` | `test_claude_workspace_strategy_is_consistent_with_gitignore` |
| `def` | `test_competitor_analysis_submodules_are_declared_and_documented` |

#### `test_lifecycle_service.py`

Contract tests for app-layer server lifecycle service.

| Kind | Name |
|------|------|
| `def` | `test_lifecycle_startup_invokes_load_steps_in_order` |
| `def` | `test_lifecycle_shutdown_calls_save_and_never_suppresses_exceptions` |
| `def` | `test_lifecycle_service_request_contracts_declared` |
| `def` | `test_lifecycle_service_validate_startup_request_map_rejects_extra_key` |

#### `test_mcp_context_surfaces.py`

Tests for MCP prompt and resource surfaces.

| Kind | Name |
|------|------|
| `class` | `Tooling` |
| `def` | `test_get_prompt_returns_prompt_messages` |
| `def` | `test_list_prompts_returns_supported_prompt_names` |
| `def` | `test_list_resources_includes_catalog_and_status` |
| `def` | `test_list_resource_templates_includes_tool_help` |
| `async def` | `test_read_resource_returns_tool_catalog_json` |
| `async def` | `test_read_resource_returns_tool_help_template_payload` |
| `async def` | `test_read_resource_returns_installation_status` |
| `async def` | `test_read_resource_returns_install_modes_with_guided_setup_command` |

#### `test_mcp_contracts_extended.py`

Phase 0 MCP contract tests for core stable runtime surfaces.

| Kind | Name |
|------|------|
| `def` | `_make_mock_skeleton` |
| `def` | `test_core_stable_schema_contracts_cover_expected_fields` |
| `def` | `test_file_sync_tools_accept_scope_fields` |
| `def` | `test_secondary_document_tools_accept_scope_fields` |
| `def` | `test_model_optimization_tools_expose_expected_contracts` |
| `async def` | `test_ingest_runtime_output_matches_canonical_contract` |
| `async def` | `test_read_and_search_runtime_outputs_match_canonical_contracts` |

#### `test_mcp_packaging.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_pyproject_declares_mcp_console_scripts` |
| `def` | `test_claude_desktop_config_example_uses_token_saver_command` |
| `def` | `test_claude_code_config_example_uses_token_saver_command` |
| `def` | `test_merge_mcp_server_adds_token_saver_entry` |
| `def` | `test_remove_mcp_server_preserves_other_servers` |
| `def` | `test_remove_mcp_server_cleans_empty_mcp_servers_block` |
| `def` | `test_install_mcp_writes_token_saver_entry` |
| `def` | `test_uninstall_mcp_removes_token_saver_and_preserves_other_servers` |
| `def` | `test_build_server_config_sets_pythonpath` |
| `def` | `test_build_server_config_portable_uses_workspace_folder` |
| `def` | `test_detect_project_mcp_config_path_uses_dot_claude` |
| `def` | `test_render_mcp_config_returns_full_mcp_payload` |
| `def` | `test_render_mcp_config_supports_custom_server_key` |
| `def` | `test_render_mcp_config_portable_uses_workspace_folder` |
| `def` | `test_install_project_mcp_writes_repo_scoped_config` |
| `def` | `test_uninstall_project_mcp_deletes_empty_config_file` |
| `def` | `test_install_project_mcp_preserves_existing_servers` |
| `def` | `test_install_portable_project_mcp_uses_workspace_folder` |
| `def` | `test_install_portable_project_mcp_preserves_existing_servers` |
| `def` | `test_inspect_mcp_installation_reports_missing_configs` |
| `def` | `test_inspect_mcp_installation_reports_portable_project_config` |
| `def` | `test_inspect_mcp_installation_reports_desktop_config` |
| `def` | `test_resolve_project_root_for_cli_defaults_to_cwd` |
| `def` | `test_recommend_setup_target_prefers_portable_project_for_repo_like_directory` |
| `def` | `test_recommend_setup_target_defaults_to_desktop_for_plain_directory` |
| `def` | `test_build_status_report_recommends_follow_up_command` |

#### `test_mcp_routing.py`

Unit tests for MCP Core Routing (mcp_core.py)

| Kind | Name |
|------|------|
| `class` | `TestSetupMCPTools` |
| `class` | `TestRouteToolCall` |
| `class` | `TestRouterIntegrity` |

#### `test_mcp_tooling_gateway.py`

Contract tests for MCP tooling gateway in enterprise app layer.

| Kind | Name |
|------|------|
| `def` | `test_gateway_reexports_supported_profiles` |
| `def` | `test_gateway_resolve_profile_falls_back_to_full` |
| `async def` | `test_gateway_route_tool_call_delegates` |
| `def` | `test_gateway_profile_state_contract_declared` |
| `def` | `test_gateway_validate_profile_state_map_rejects_extra_key` |
| `def` | `test_gateway_set_profile_state_uses_validate_profile_state_map_class_dispatch` |

#### `test_media_metadata.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_detect_media_kind_recognizes_supported_types` |
| `def` | `test_inspect_media_path_reports_file_metadata` |
| `def` | `test_inspect_media_path_requires_existing_file` |
| `def` | `test_summarize_payload_bytes_counts_text_and_binary` |

#### `test_memory_contracts.py`

Contract tests for explicit memory MCP tools.

| Kind | Name |
|------|------|
| `def` | `setup_function` |
| `def` | `test_memory_tools_are_registered_in_mcp_core` |
| `def` | `test_knowledge_tools_are_registered_in_mcp_core` |
| `async def` | `test_memory_tools_have_help_entries` |
| `async def` | `test_router_dispatches_memory_tools` |
| `async def` | `test_router_dispatches_knowledge_tools` |

#### `test_memory_golden_profiles.py`

Golden regression tests for explicit memory profile synthesis.

| Kind | Name |
|------|------|
| `def` | `test_profile_output_shape_is_stable` |

#### `test_memory_handlers.py`

Tests for explicit memory MCP handlers.

| Kind | Name |
|------|------|
| `def` | `memory_context` |
| `async def` | `test_add_search_list_and_delete_memory` |
| `async def` | `test_summarize_and_profile_handlers` |

#### `test_memory_optimization.py`

Tests for Memory Optimization (v0.6.0)

| Kind | Name |
|------|------|
| `class` | `TestONNXEmbeddings` |
| `class` | `TestTFIDFEmbeddings` |
| `class` | `TestLRUEmbeddingCache` |
| `class` | `TestEmbeddingTierSwitching` |

#### `test_memory_service.py`

Tests for explicit memory service behavior.

| Kind | Name |
|------|------|
| `def` | `setup_function` |
| `def` | `test_add_list_search_and_delete_memory` |
| `def` | `test_scope_isolation_matches_phase_one_tenancy_rules` |
| `def` | `test_user_summary_and_profile_are_derived_from_explicit_memories` |

#### `test_meta_tokens.py`

Tests for src/meta_tokens.py — lossless meta-token compression (arXiv 2506.00307).

| Kind | Name |
|------|------|
| `def` | `make_compressor` |
| `class` | `TestFindRepeatedSubsequences` |
| `class` | `TestRankBySavings` |
| `class` | `TestCompress` |
| `class` | `TestDecompress` |
| `class` | `TestMetaTokenEntry` |

#### `test_metrics.py`

Comprehensive Metrics Tests (v1.0.0 - v0.6.1)

| Kind | Name |
|------|------|
| `def` | `metrics` |
| `class` | `TestBasicMetrics` |
| `class` | `TestCardinalityControl` |
| `class` | `TestSingleton` |
| `class` | `TestPrometheusOutput` |
| `class` | `TestGracefulDegradation` |
| `class` | `TestReset` |
| `class` | `TestEdgeCases` |

#### `test_mig_lambda_routing.py`

B1 (modernization roadmap 2026-06-08): COMI/MIG lambda_redundancy routing.

| Kind | Name |
|------|------|
| `def` | `_build_compressor_with_doc` |
| `class` | `TestPresetExposesLambdaRedundancy` |
| `class` | `TestGenerateSkeletonUsesMigLambdaOnQuery` |
| `class` | `TestA2VectorizedPathPreserved` |

#### `test_mig_scoring.py`

Tests for MIG (Marginal Information Gain) scoring in src/token_refiner.py.

| Kind | Name |
|------|------|
| `class` | `TestMIGConfig` |
| `class` | `TestMIGScorer` |
| `class` | `TestTokenRefinerMIG` |

#### `test_mmr_vectorization_equivalence.py`

Output-equivalence + speedup receipts for the vectorized MMR skeleton selector

| Kind | Name |
|------|------|
| `def` | `_reference_select_skeleton_nodes` |
| `class` | `_FakeModel` |
| `def` | `_make_nodes` |
| `def` | `_random_unit` |
| `def` | `_build_compressor` |
| `def` | `_params` |
| `class` | `TestMMRSelectionEquivalence` |
| `class` | `TestMMRSpeedup` |

#### `test_model_accounting_hypothesis.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_cost_savings_never_negative_for_expansion` |

#### `test_model_handlers.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `async def` | `test_get_provider_profile_handler` |
| `async def` | `test_estimate_model_cost_handler` |
| `async def` | `test_optimize_for_model_handler` |
| `async def` | `test_assess_cache_compatibility_handler` |
| `async def` | `test_capture_cache_telemetry_handler_warns_on_expected_cache_miss` |
| `async def` | `test_capture_cache_telemetry_handler_emits_observability_payload` |
| `async def` | `test_capture_cache_telemetry_handler_validates_prompt_expectation` |
| `async def` | `test_capture_cache_telemetry_handler_surfaces_stale_expectation_warning` |
| `async def` | `test_capture_cache_telemetry_handler_surfaces_stale_sibling_coherence` |
| `async def` | `test_capture_cache_telemetry_handler_adds_diagnostic_with_actual_prefix` |
| `async def` | `test_capture_cache_telemetry_handler_surfaces_section_interleaving` |
| `async def` | `test_capture_cache_telemetry_handler_surfaces_cache_creation_churn` |
| `async def` | `test_diagnose_cache_miss_handler_requires_known_prompt_id` |
| `async def` | `test_capture_cache_telemetry_handler_records_session_metrics` |
| `async def` | `test_capture_cache_telemetry_handler_records_openai_session_cached_tokens` |
| `async def` | `test_capture_cache_telemetry_handler_surfaces_prefix_siblings` |
| `async def` | `test_capture_cache_telemetry_handler_surfaces_cache_health_degradation` |
| `async def` | `test_capture_cache_telemetry_handler_surfaces_prefix_integrity_on_cache_hit` |
| `async def` | `test_capture_cache_telemetry_handler_surfaces_prefix_drift_trend` |
| `async def` | `test_capture_cache_telemetry_handler_surfaces_cross_label_cache_skew` |

#### `test_model_optimizer.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_optimize_for_high_cost_model_biases_toward_compaction` |
| `def` | `test_optimize_for_large_context_model_can_keep_more_detail` |
| `def` | `test_optimize_for_model_includes_prompt_structure_guidance` |
| `def` | `test_optimize_for_codex_family_includes_routing_stickiness_guidance` |
| `def` | `test_optimize_for_model_includes_cache_threshold_guidance` |
| `def` | `test_advise_cache_threshold_reports_gap_when_below_minimum` |
| `def` | `test_build_prompt_cache_key_is_stable_for_same_inputs` |
| `def` | `test_summarize_provider_cache_usage_uses_provider_profile_costs` |

#### `test_multi_agent_setup.py`

Tests for multi-agent MCP configuration (install/uninstall/inspect).

| Kind | Name |
|------|------|
| `class` | `TestDetectAgentConfigPath` |
| `class` | `TestInstallAgentConfig` |
| `class` | `TestUninstallAgentConfig` |
| `class` | `TestInspectAllAgents` |
| `class` | `TestFormatAllAgentsReport` |
| `class` | `TestBuildAgentsMdSection` |

#### `test_multimodal_compressor_coverage.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `class` | `_DummyEncoder` |
| `class` | `_DummyManager` |
| `def` | `test_multimodal_node_defaults` |
| `def` | `test_ingest_search_get_content_and_summary` |
| `def` | `test_search_image_query_returns_empty_when_encoder_absent` |

#### `test_multimodal_handlers.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `class` | `FakeMultimodalService` |
| `async def` | `test_ingest_multimodal_uses_scoped_doc_id_and_returns_visible_doc_id` |
| `async def` | `test_search_multimodal_returns_visible_doc_id` |
| `async def` | `test_search_multimodal_requires_image_path_for_image_queries` |

#### `test_multimodal_ingestion.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_build_multimodal_request_normalizes_supported_payloads` |
| `def` | `test_build_multimodal_request_rejects_video_payloads` |
| `def` | `test_build_multimodal_request_requires_audio_transcripts` |
| `def` | `test_build_multimodal_request_requires_supported_content` |
| `def` | `test_estimate_multimodal_payload_bytes_counts_binary_and_text` |

#### `test_node_identity.py`

Unit tests for shared node identity parsing helpers.

| Kind | Name |
|------|------|
| `def` | `test_extract_file_id_from_text_node_suffix` |
| `def` | `test_extract_file_id_from_code_node` |
| `def` | `test_extract_file_id_ignores_non_index_n_segments` |
| `def` | `test_collect_file_ids_supports_mixed_node_formats` |

#### `test_nonlatin_and_code_entity_edgecases.py`

Dogfood 2026-07-12 (#212 non-English + code edge cases): two perceived-quality

| Kind | Name |
|------|------|
| `def` | `_c` |
| `def` | `test_nonlatin_summary_strips_markdown_heading` |
| `def` | `test_latin_summary_still_strips_heading_regression` |
| `def` | `test_summary_strips_blockquote_markers` |
| `def` | `test_summary_keeps_midline_greater_than_comparison` |
| `def` | `test_punctuation_only_fragment_still_skipped` |
| `def` | `test_extractive_anchor_keeps_nonlatin_sentence` |
| `def` | `test_mid_content_heading_marker_stripped_all_langs` |
| `def` | `test_hash_not_at_line_start_preserved` |
| `def` | `test_code_syntax_entity_recovers_clean_identifier` |
| `def` | `test_legit_entities_with_dots_slashes_underscores_preserved` |
| `def` | `test_apostrophe_entity_not_truncated` |
| `def` | `test_cjk_summary_selects_first_sentence_not_whole_paragraph` |
| `def` | `test_japanese_summary_selects_first_sentence` |
| `def` | `test_fullwidth_question_exclamation_terminators_split` |
| `def` | `test_cjk_split_does_not_regress_latin_first_sentence` |
| `def` | `test_extractive_anchor_respects_cjk_sentence_boundary` |

#### `test_observability.py`

Tests for OpenTelemetry distributed tracing.

| Kind | Name |
|------|------|
| `def` | `reset_singleton` |
| `def` | `mock_tracer` |
| `def` | `observability_enabled` |
| `def` | `observability_disabled` |
| `class` | `TestBasicTracing` |
| `class` | `TestAsyncContextPropagation` |
| `class` | `TestExceptionHandling` |
| `class` | `TestTraceContext` |
| `class` | `TestConfiguration` |
| `class` | `TestOTLPExport` |
| `class` | `TestGracefulDegradation` |
| `class` | `TestNoOpSpan` |
| `class` | `TestAdvancedFeatures` |
| `class` | `TestIntegration` |
| `class` | `TestPerformance` |

#### `test_observability_contracts.py`

Phase 0 observability validation for handler-side metrics and warnings.

| Kind | Name |
|------|------|
| `def` | `_make_ingest_context` |
| `async def` | `test_handle_ingest_logs_metrics_warning_on_metrics_failure` |
| `async def` | `test_handle_ingest_logs_persistence_error_but_returns_success` |
| `async def` | `test_handle_search_semantic_logs_access_tracking_warning` |
| `def` | `test_observability_defaults_to_local_only_export_in_development` |

#### `test_observability_coverage.py`

Coverage tests for handler-level observability spans.

| Kind | Name |
|------|------|
| `class` | `_FakeTracker` |
| `class` | `_FakeReplay` |
| `class` | `_FakeTemporal` |
| `class` | `_FakeCompressor` |
| `def` | `memory_context` |
| `def` | `connector_context` |
| `def` | `reset_bundle_registry` |
| `def` | `_observe_mock` |
| `async def` | `test_memory_handlers_emit_trace` |
| `async def` | `test_connector_handlers_emit_trace` |
| `async def` | `test_bundle_handlers_emit_trace` |
| `async def` | `test_model_handlers_emit_trace` |

#### `test_onnx_torch_free_path.py`

The ONNX embedder must serve without torch, and must not change its vectors.

| Kind | Name |
|------|------|
| `def` | `_model_is_cached` |
| `class` | `_StubSession` |
| `class` | `_StubTokenizer` |
| `def` | `_manager_with_stubs` |
| `def` | `test_raw_path_honours_the_per_model_pooling_selector` |
| `def` | `test_token_type_ids_are_sent_only_when_the_session_declares_them` |
| `def` | `test_an_all_padding_row_does_not_become_nan` |
| `def` | `test_a_cold_model_routes_to_the_export_path` |
| `def` | `test_a_real_encode_does_not_load_torch` |

#### `test_opencode_enhancements.py`

Tests for OpenCode enhancements: model database, cache strategy advisor, pricing, providers.

| Kind | Name |
|------|------|
| `class` | `TestModelDatabase` |
| `class` | `TestCacheStrategyAdvisor` |
| `class` | `TestPricing` |
| `class` | `TestBenchmarkProvider` |
| `class` | `TestMCPCacheStrategyTool` |

#### `test_output_determinism.py`

Cross-process output determinism regression gate.

| Kind | Name |
|------|------|
| `def` | `test_extract_key_entities_first_occurrence_deterministic` |
| `def` | `test_extract_key_entities_dedup_preserves_order` |
| `def` | `test_skeleton_deterministic_same_process` |
| `def` | `_run_worker` |
| `def` | `test_skeleton_deterministic_across_hashseed` |

#### `test_parity_characterization.py`

Phase 0 characterization tests for core MCP compression flows.

| Kind | Name |
|------|------|
| `async def` | `test_core_compression_flow_characterization` |
| `async def` | `test_query_guided_read_uses_baseline_cache_prefix` |

#### `test_parity_e2e.py`

Phase 10 cross-surface launch-readiness tests.

| Kind | Name |
|------|------|
| `def` | `platform_context` |
| `async def` | `test_platform_workflow_spans_memory_prompt_experiment_connector_and_model` |
| `async def` | `test_workspace_scopes_isolate_memory_and_connector_outputs` |

#### `test_path_validator.py`

Comprehensive tests for PathValidator (path traversal prevention).

| Kind | Name |
|------|------|
| `def` | `temp_project_dir` |
| `def` | `validator_with_temp_dir` |
| `class` | `TestPathTraversalPrevention` |
| `class` | `TestAbsolutePathValidation` |
| `class` | `TestSymlinkResolution` |
| `class` | `TestRelativePathHandling` |
| `class` | `TestInputValidation` |
| `class` | `TestDefaultBehavior` |
| `class` | `TestHelperMethods` |
| `class` | `TestMultipleAllowedDirectories` |
| `class` | `TestIntegration` |

#### `test_performance.py`

Performance tests for Token Saver 5000 v0.7.0.

| Kind | Name |
|------|------|
| `class` | `TestThroughputPerformance` |
| `class` | `TestLatencyPerformance` |
| `class` | `TestMemoryUsagePerformance` |
| `class` | `TestCacheEffectiveness` |
| `class` | `TestBurstCapacity` |

#### `test_persistence_comprehensive.py`

Comprehensive Persistence Tests (v1.0.0 - Phase 1)

| Kind | Name |
|------|------|
| `def` | `temp_storage_dir` |
| `def` | `persistence_manager` |
| `def` | `sample_document_data` |
| `class` | `TestDocumentPersistence` |
| `class` | `TestAFMHistoryPersistence` |
| `class` | `TestFileSyncMetadataPersistence` |
| `class` | `TestErrorHandlingAndRecovery` |
| `class` | `TestConcurrentAccess` |
| `class` | `TestStorageStatsAndUtilities` |
| `class` | `TestF8AtomicWriteParentDirAutoCreate` |

#### `test_persistence_orchestration_service.py`

Contract tests for app-layer persistence orchestration service.

| Kind | Name |
|------|------|
| `def` | `test_load_persisted_documents_restores_graph_and_registers_resource` |
| `def` | `test_load_file_sync_metadata_imports_when_present` |
| `def` | `test_save_file_sync_metadata_warns_when_not_saved` |
| `def` | `test_persistence_service_contract_keysets_declared` |
| `def` | `test_persistence_service_validate_load_sync_request_map_rejects_extra_key` |

#### `test_personalization.py`

Tests for personalization synthesis from explicit memories.

| Kind | Name |
|------|------|
| `def` | `test_build_user_profile_extracts_preferences_and_topics` |
| `def` | `test_summarize_user_memories_returns_compact_summary_view` |

#### `test_phase3_bcd.py`

Tests for ROI calculator, budget monitor, and team export (Phase 3B-3D).

| Kind | Name |
|------|------|
| `async def` | `test_calculate_roi_defaults` |
| `async def` | `test_calculate_roi_custom_params` |
| `async def` | `test_calculate_roi_unknown_model_uses_default` |
| `async def` | `test_calculate_roi_zero_compression` |
| `async def` | `test_calculate_roi_full_compression` |
| `async def` | `test_calculate_roi_comparison_format` |
| `class` | `TestBudgetLimit` |
| `class` | `TestTokenBudgetMonitor` |
| `class` | `TestCreateBudgetMonitor` |
| `async def` | `test_check_budget_handler_no_limits` |
| `async def` | `test_check_budget_handler_with_recording` |
| `class` | `TestTeamMemberStats` |
| `class` | `TestTeamExporter` |
| `async def` | `test_export_team_data_json` |
| `async def` | `test_export_team_data_csv` |
| `async def` | `test_export_team_data_prometheus` |
| `async def` | `test_export_team_data_empty` |

#### `test_phase4_features.py`

Tests for Phase 4 features: Citation tags, auto ratio exposure, validation hooks,

| Kind | Name |
|------|------|
| `class` | `TestCitationTags` |
| `class` | `TestAutoRatioMCPExposure` |
| `class` | `TestInputValidationHooks` |
| `class` | `TestCompressionPresets` |
| `class` | `TestTokenThresholdMonitor` |
| `class` | `TestMemoryClassification` |
| `class` | `TestStreamingCompression` |
| `class` | `TestDiffAwareReingestion` |
| `class` | `TestMultimodalProduction` |
| `class` | `TestSCARTrainingPipeline` |
| `class` | `TestCrossDocumentDedup` |

#### `test_phase5_features.py`

Tests for Phase 5 features - 11 new capabilities based on 2025 research papers.

| Kind | Name |
|------|------|
| `class` | `TestAttentionPruning` |
| `class` | `TestSemanticChunking` |
| `class` | `TestIntraDocDedup` |
| `class` | `TestContextDecay` |
| `class` | `TestFidelityScoring` |
| `class` | `TestMultiLevelSkeleton` |
| `class` | `TestQueryAdaptiveCompression` |
| `class` | `TestContextAdvisor` |
| `class` | `TestCompressionReplay` |
| `class` | `TestGenerativeRewrite` |
| `class` | `TestPhase5MCPWiring` |

#### `test_prepare_raw_chunks.py`

#190 unit 2: SemanticCompressor._prepare_raw_chunks routes a structured JSON

| Kind | Name |
|------|------|
| `def` | `_bare_compressor` |
| `def` | `test_json_array_uses_record_chunking_not_text_chunker` |
| `def` | `test_prose_falls_through_to_text_chunker` |
| `def` | `test_json_array_kind_but_unsplittable_falls_back` |
| `def` | `test_jsonl_uses_record_chunking_not_text_chunker` |
| `def` | `test_jsonl_kind_but_unsplittable_falls_back` |
| `def` | `test_csv_uses_record_chunking_header_as_own_node` |
| `def` | `test_csv_multiline_quoted_falls_through` |

#### `test_progress_service.py`

Contract tests for app-layer progress rendering service.

| Kind | Name |
|------|------|
| `def` | `test_progress_service_render_empty_bar` |
| `def` | `test_progress_service_render_warning_and_crit_states` |
| `def` | `test_progress_service_request_contract_declared` |
| `def` | `test_progress_service_validate_progress_request_map_rejects_extra_key` |

#### `test_prompt_cache_audit.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_audit_prompt_cacheability_accepts_canonical_order` |
| `def` | `test_audit_prompt_cacheability_flags_out_of_order_query` |
| `def` | `test_audit_prompt_cacheability_flags_volatile_ids_in_stable_prefix` |
| `async def` | `test_audit_prompt_cacheability_handler_returns_structured_result` |
| `def` | `test_evaluate_prompt_stability_returns_prefix_fingerprint` |
| `def` | `test_evaluate_prompt_stability_flags_volatile_content_for_enforcement` |

#### `test_prompt_cache_audit_contracts.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_prompt_cache_audit_tool_is_registered` |
| `async def` | `test_prompt_cache_audit_help_is_documented` |

#### `test_prompt_cache_validation.py`

Tests for prompt cache expectation tracking and validation.

| Kind | Name |
|------|------|
| `def` | `setup_function` |
| `def` | `test_record_expectation_marks_cache_friendly_prompt_as_expected_hit` |
| `def` | `test_validate_provider_response_warns_on_unexpected_cache_miss` |
| `def` | `test_invalidate_expectations_for_redeployed_label_marks_expectation_stale` |
| `def` | `test_validate_provider_response_reports_stale_expectation_details` |
| `def` | `test_validate_provider_response_reports_stale_colliding_sibling` |
| `def` | `test_validate_provider_response_reports_clean_sibling_coherence` |
| `def` | `test_validate_provider_response_reports_section_interleaving_drift` |
| `def` | `test_validate_provider_response_reports_semantic_equivalence_drift` |
| `def` | `test_validate_provider_response_reports_partial_cache_reuse_underperformance` |
| `def` | `test_validate_provider_response_reports_repeated_cache_creation_churn` |
| `def` | `test_validate_provider_response_reports_missing_expectation` |
| `def` | `test_record_session_telemetry_aggregates_cache_metrics` |
| `def` | `test_record_session_telemetry_uses_openai_cached_tokens_field` |
| `def` | `test_record_session_telemetry_uses_gemini_cached_content_token_count` |
| `def` | `test_record_session_telemetry_adds_degradation_warning_for_repeated_misses` |
| `def` | `test_record_expectation_tracks_cross_template_prefix_collisions` |
| `def` | `test_record_session_telemetry_establishes_label_cache_health_baseline` |
| `def` | `test_record_session_telemetry_flags_cache_health_degradation` |
| `def` | `test_record_session_telemetry_does_not_flag_small_cache_health_variance` |
| `def` | `test_validate_provider_response_reports_prefix_integrity_match` |
| `def` | `test_validate_provider_response_reports_prefix_integrity_drift_even_on_cache_hit` |
| `def` | `test_validate_provider_response_tracks_prefix_drift_frequency` |
| `def` | `test_record_session_telemetry_detects_cross_label_cache_skew_for_shared_prefix` |
| `def` | `test_record_session_telemetry_does_not_flag_cross_label_skew_for_different_prefixes` |

#### `test_prompt_caching.py`

Tests for prompt caching optimization improvements.

| Kind | Name |
|------|------|
| `class` | `TestACEBulletDisplayDict` |
| `class` | `TestACEContextDisplayDict` |
| `class` | `TestACEHandlersCacheOptimized` |
| `class` | `TestSkeletonQueryPlacement` |

#### `test_prompt_contracts.py`

Contract tests for prompt registry MCP tools and help entries.

| Kind | Name |
|------|------|
| `def` | `setup_function` |
| `def` | `test_prompt_tools_are_registered_in_mcp_core` |
| `async def` | `test_prompt_tools_have_help_entries` |
| `async def` | `test_router_dispatches_prompt_registry_tools` |

#### `test_prompt_handlers.py`

Tests for prompt registry MCP handlers.

| Kind | Name |
|------|------|
| `def` | `prompt_context` |
| `async def` | `test_create_and_list_prompt_templates` |
| `async def` | `test_update_deploy_get_and_compare_prompt_templates` |
| `async def` | `test_deploy_prompt_version_handler_reports_prefix_change` |
| `async def` | `test_deploy_prompt_version_handler_reports_prefix_collisions` |
| `async def` | `test_deploy_prompt_version_handler_rejects_unacknowledged_prefix_change` |
| `async def` | `test_audit_prompt_cacheability_handler_flags_ordering_issues` |
| `async def` | `test_render_prompt_template_handler_returns_stability_guard` |
| `async def` | `test_render_prompt_template_handler_can_enforce_stability` |
| `async def` | `test_render_prompt_template_handler_returns_rendered_sections` |

#### `test_prompt_registry.py`

Tests for prompt registry service behavior.

| Kind | Name |
|------|------|
| `def` | `setup_function` |
| `def` | `test_seeded_templates_exist_from_compression_presets` |
| `def` | `test_create_update_get_deploy_prompt_template` |
| `def` | `test_update_template_reports_authoring_stable_prefix_analysis` |
| `def` | `test_create_template_reports_initial_stable_prefix_analysis` |
| `def` | `test_compare_versions_returns_changed_fields_and_diff` |
| `def` | `test_compare_versions_identifies_volatile_only_changes` |
| `def` | `test_render_prompt_template_builds_cache_friendly_sections` |
| `def` | `test_render_prompt_template_rejects_missing_variables` |
| `def` | `test_deploy_version_reports_prefix_change_when_repointing_label` |
| `def` | `test_deploy_version_rejects_prefix_change_without_explicit_acknowledgement` |

#### `test_prompt_versioning_golden.py`

Golden-ish regression tests for prompt version comparison output.

| Kind | Name |
|------|------|
| `def` | `setup_function` |
| `def` | `test_prompt_version_comparison_has_stable_shape` |

#### `test_provider_profiles.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_get_provider_profile_returns_known_model` |
| `def` | `test_provider_profile_supports_alias_lookup` |
| `def` | `test_provider_profile_supports_gemini_3_alias_lookup` |
| `def` | `test_provider_profile_supports_codex_alias_lookup` |
| `def` | `test_provider_profile_exposes_cache_threshold_guidance` |
| `def` | `test_unknown_model_raises_value_error` |
| `def` | `test_list_provider_profiles_includes_multiple_providers` |
| `def` | `test_gemini_flash_is_cheaper_than_pro` |
| `def` | `test_gemini_auto_priced_between_pro_and_flash` |
| `def` | `test_gemini_flash_alias_resolves` |
| `def` | `test_gpt5_mini_cheaper_than_full` |
| `def` | `test_gpt41_mini_cheaper_than_full` |
| `def` | `test_mini_aliases_resolve` |
| `def` | `test_opus_4_7_is_registered` |
| `def` | `test_opus_4_7_aliases_resolve` |
| `def` | `test_opus_4_6_still_resolves_with_hyphen_alias` |
| `def` | `test_gpt_5_6_family_registered` |
| `def` | `test_gpt_5_6_alias_resolves_to_sol` |
| `def` | `test_minimum_cacheable_tokens_matches_verified_provider_docs` |
| `def` | `test_claude_mythos_5_minimum_cacheable_tokens_is_unverified_placeholder` |

#### `test_proxy_cli.py`

Tests for the MCP proxy CLI with savings tracking.

| Kind | Name |
|------|------|
| `class` | `TestProxyCLIDryRun` |
| `class` | `TestProxySavingsTracking` |
| `class` | `TestProxySessionMetrics` |
| `class` | `TestInterceptionStatsProperties` |

#### `test_proxy_core.py`

Tests for the MCP proxy core components.

| Kind | Name |
|------|------|
| `class` | `TestResponseInterceptor` |
| `class` | `TestToolIndex` |
| `class` | `TestSchemaCompressor` |
| `class` | `TestProxyConfig` |
| `class` | `TestProxyServer` |

#### `test_quality_gate.py`

Deterministic, INDEPENDENT-oracle compression quality gate (Phase 0).

| Kind | Name |
|------|------|
| `def` | `_sections` |
| `def` | `_build_correct_skeleton` |
| `class` | `TestAnswerabilityGrader` |
| `class` | `TestByteIdentityGrader` |
| `class` | `TestSourceOrderGrader` |
| `def` | `_make_compressor_with_raw_chunks` |
| `class` | `TestModulateRegionRoundtripGrader` |
| `class` | `TestGateItselfIsNotBroken` |
| `def` | `_probe_model_load` |
| `class` | `TestRealCompressorIntegration` |
| `class` | `TestRealCompressorSourceOrderEndToEnd` |
| `class` | `TestBidirectionalCompressorEvaluation` |

#### `test_quality_predictor.py`

Tests for src/quality_predictor.py — PoC compression quality prediction.

| Kind | Name |
|------|------|
| `def` | `make_predictor` |
| `class` | `TestPredictQuality` |
| `class` | `TestExtractEntities` |
| `class` | `TestComputeEntityRetention` |
| `class` | `TestComputeCoverage` |
| `class` | `TestSimulateCompression` |
| `class` | `TestPredictAllProfiles` |
| `class` | `TestAutoSelectProfile` |

#### `test_query_adaptive_budget_floor.py`

Regression lock: query-adaptive skeleton budget must never collapse below

| Kind | Name |
|------|------|
| `def` | `_build_ten_topic_doc` |
| `def` | `_anchor_count` |
| `class` | `TestQueryAdaptiveBudgetNeverCollapsesBelowFloor` |

#### `test_query_guided_compression.py`

TDD coverage for query-guided and evidence-aware compression paths.

| Kind | Name |
|------|------|
| `def` | `_extract_anchor_ids` |
| `def` | `test_read_skeleton_query_guided_adds_selection_metadata` |
| `def` | `test_query_guided_header_skeleton_count_matches_rendered_anchors` |
| `def` | `test_query_guided_anchors_include_top_semantic_hit` |
| `def` | `test_retrieve_evidence_detects_insufficient_query_signal` |
| `def` | `test_read_skeleton_evidence_aware_includes_status_and_skeleton` |

#### `test_read_skeleton_auto_fidelity.py`

Regression locks for the read_skeleton production-default fidelity fix.

| Kind | Name |
|------|------|
| `def` | `_build_multi_node_doc` |
| `def` | `_count_anchor_nodes` |
| `def` | `_count_hidden_nodes` |
| `class` | `TestProductionReadSkeletonIsFaithful` |
| `class` | `TestExplicitAggressiveModePreserved` |
| `class` | `TestHiddenLineCarriesSummary` |
| `class` | `TestAdaptiveCurveHasNoCliff` |

#### `test_reliability.py`

Comprehensive Tests for Reliability Infrastructure

| Kind | Name |
|------|------|
| `async def` | `test_timeout_manager_successful_operation` |
| `async def` | `test_timeout_manager_timeout_exceeded` |
| `async def` | `test_timeout_manager_configured_timeout` |
| `async def` | `test_timeout_manager_explicit_timeout_override` |
| `async def` | `test_circuit_breaker_closed_state` |
| `async def` | `test_circuit_breaker_opens_after_threshold` |
| `async def` | `test_circuit_breaker_rejects_when_open` |
| `async def` | `test_circuit_breaker_half_open_after_timeout` |
| `async def` | `test_circuit_breaker_reopens_on_half_open_failure` |
| `async def` | `test_circuit_breaker_reset` |
| `async def` | `test_retry_policy_success_no_retry` |
| `async def` | `test_retry_policy_success_after_retries` |
| `async def` | `test_retry_policy_exhausted` |
| `async def` | `test_retry_policy_exponential_backoff` |
| `async def` | `test_retry_policy_non_retryable_exception` |
| `async def` | `test_rate_limiter_within_capacity` |
| `async def` | `test_rate_limiter_burst_capacity` |
| `async def` | `test_rate_limiter_refill` |
| `async def` | `test_rate_limiter_blocking_wait` |
| `async def` | `test_rate_limiter_non_blocking_reject` |
| `async def` | `test_rate_limiter_statistics` |
| `def` | `test_get_rate_limiter` |
| `def` | `test_get_rate_limiter_not_configured` |
| `def` | `test_configure_rate_limiter` |
| `async def` | `test_timeout_with_retry_integration` |
| `async def` | `test_circuit_breaker_with_retry_integration` |

#### `test_rerank_wire.py`

#187 rerank wire: the rerank stage reorders / fails safe (model-free).

| Kind | Name |
|------|------|
| `class` | `_Chunk` |
| `def` | `_bare` |
| `def` | `test_rerank_pool_reorders_by_injected_scorer` |
| `def` | `test_rerank_pool_is_fail_safe_on_scorer_error` |
| `def` | `test_rerank_pool_single_candidate_is_noop` |

#### `test_reranker_gate.py`

Model-free unit tests for the recall-gated rerank stage (#187).

| Kind | Name |
|------|------|
| `def` | `_run` |
| `def` | `_ids` |
| `def` | `test_disabled_is_noop` |
| `def` | `test_fewer_than_two_candidates_is_noop` |
| `def` | `test_empty_query_is_noop` |
| `def` | `test_reorders_by_rerank_score` |
| `def` | `test_pool_truncation_keeps_tail_in_place` |
| `def` | `test_confidence_skip_fires_on_separated_top1` |
| `def` | `test_confidence_skip_does_not_fire_on_close_scores` |
| `def` | `test_scorer_wrong_length_fails_safe` |
| `def` | `test_scorer_receives_query_and_texts` |
| `def` | `test_ties_preserve_retrieval_order` |

#### `test_resource_handlers.py`

Comprehensive tests for resource_handlers.py

| Kind | Name |
|------|------|
| `def` | `mock_context` |
| `def` | `mock_context_with_validator` |
| `def` | `get_context_for_tempfile` |
| `class` | `TestCreateProgressBar` |
| `class` | `TestHandleCheckResourceHealth` |
| `class` | `TestHandleCheckEnvironment` |
| `class` | `TestHandleShouldCompress` |
| `class` | `TestIsBinaryContent` |
| `class` | `TestBinaryFileDetection` |

#### `test_resource_manager_coverage.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_document_limit_checks_and_register_unregister` |
| `def` | `test_health_summary_stats_and_cleanup_recommendation` |
| `def` | `test_memory_health_paths_with_monkeypatched_psutil` |
| `async def` | `test_async_wrappers_delegate` |

#### `test_response_formatter.py`

Tests for src/response_formatter.py (ResponseFormatter class).

| Kind | Name |
|------|------|
| `def` | `formatter` |
| `def` | `small_formatter` |
| `def` | `_make_response` |
| `def` | `test_format_response_under_limit_passes_through` |
| `def` | `test_format_response_preserves_original_data_keys` |
| `def` | `test_token_estimates_included_in_every_response` |
| `def` | `test_token_estimates_have_numeric_values` |
| `def` | `test_truncated_flag_false_for_small_response` |
| `def` | `test_truncated_flag_set_when_paginated` |
| `def` | `test_format_response_at_soft_limit_strips_metadata` |
| `def` | `test_metadata_keys_stripped_when_serialized_size_exceeds_soft_limit` |
| `def` | `test_format_response_at_hard_limit_paginates` |
| `def` | `test_paginated_response_contains_preview` |
| `def` | `test_continuation_token_is_none_when_not_paginated` |
| `def` | `test_pagination_continuation_token_is_deterministic` |
| `def` | `test_different_data_produces_different_tokens` |
| `def` | `test_custom_limits_via_constructor` |
| `def` | `test_default_limits_match_constants` |
| `def` | `test_empty_response_handling` |
| `def` | `test_nested_json_response_measurement` |
| `def` | `test_format_response_does_not_mutate_input` |
| `def` | `test_response_with_list_value` |

#### `test_router_binding.py`

Contract tests for app-layer MCP router binding helper.

| Kind | Name |
|------|------|
| `class` | `FakeServer` |
| `def` | `test_router_binding_registers_list_and_call_handlers` |
| `async def` | `test_router_binding_call_handler_wraps_result_and_errors` |
| `def` | `test_router_binding_request_contract_declared` |
| `def` | `test_router_binding_validate_bind_request_map_rejects_extra_key` |

#### `test_runtime_service.py`

Contract tests for app-layer runtime execution service.

| Kind | Name |
|------|------|
| `class` | `_FakeCtx` |
| `async def` | `test_runtime_service_runs_server_with_stdio_streams` |
| `async def` | `test_runtime_service_logs_startup_event` |
| `def` | `test_runtime_service_run_request_contract_declared` |
| `def` | `test_runtime_service_contract_key_mismatch_message_format` |
| `def` | `test_runtime_service_validate_run_request_map_rejects_extra_key` |
| `async def` | `test_runtime_service_run_uses_validate_run_request_map_class_dispatch` |

#### `test_s3_connector.py`

Tests for S3 connector normalization.

| Kind | Name |
|------|------|
| `def` | `test_collects_s3_objects_into_documents` |

#### `test_savings_dashboard.py`

Tests for the SavingsDashboard cross-session aggregation.

| Kind | Name |
|------|------|
| `def` | `tmp_storage` |
| `def` | `_create_session_db` |
| `class` | `TestSavingsDashboard` |
| `class` | `TestFormatters` |
| `class` | `TestArgParser` |

#### `test_savings_discover.py`

Tests for the savings discovery module.

| Kind | Name |
|------|------|
| `class` | `TestSavingsOpportunity` |
| `class` | `TestDiscoveryReport` |
| `class` | `TestAnalyzeText` |
| `class` | `TestAnalyzeFile` |
| `class` | `TestScanDirectory` |
| `class` | `TestAnalyzeItems` |
| `class` | `TestFormatReport` |
| `class` | `TestConstants` |

#### `test_savings_tracker.py`

Tests for the SavingsTracker module.

| Kind | Name |
|------|------|
| `def` | `_make_tracker` |
| `class` | `TestRecord` |
| `class` | `TestGetReportEmpty` |
| `class` | `TestGetReportSingleEvent` |
| `class` | `TestGetReportMultipleEvents` |
| `class` | `TestROICalculations` |
| `class` | `TestGetInlineSummary` |
| `class` | `TestModelPricing` |
| `class` | `TestJournalPersistence` |
| `class` | `TestJSONSerialisable` |

#### `test_scar_alignment_scores.py`

Targeted tests for SCAR alignment score behavior.

| Kind | Name |
|------|------|
| `def` | `test_search_with_alignment_scores_are_normalized` |
| `def` | `test_search_with_alignment_rejects_invalid_weight` |

#### `test_schema_stability.py`

Tests for schema stability in src/handlers/mcp_core.py.

| Kind | Name |
|------|------|
| `def` | `all_tools` |
| `def` | `tool_names` |
| `def` | `test_tool_schemas_are_alphabetically_ordered` |
| `def` | `test_schema_output_is_deterministic` |
| `def` | `test_schema_hash_stable` |
| `def` | `test_tool_descriptions_contain_no_dynamic_content` |
| `def` | `test_no_description_contains_runtime_value` |
| `def` | `test_all_tools_have_non_empty_description` |
| `def` | `test_all_tools_have_input_schema` |
| `def` | `test_all_tools_have_unique_names` |
| `def` | `test_core_stable_profile_contains_required_tools` |
| `def` | `test_core_stable_is_subset_of_full` |
| `def` | `test_full_profile_has_more_tools_than_core_stable` |

#### `test_search_compress_pipeline.py`

Tests for the search-then-compress pipeline.

| Kind | Name |
|------|------|
| `def` | `_corpus_exists` |
| `def` | `test_search_finds_auth_files` |
| `def` | `test_search_finds_database_files` |
| `def` | `test_search_finds_api_files` |
| `def` | `test_search_compress_result_fields` |
| `def` | `test_search_compress_fewer_than_naive` |
| `def` | `test_search_compress_fewer_than_compress_all` |
| `def` | `test_search_vs_naive_savings_positive` |
| `def` | `test_search_vs_compress_all_savings_nonneg` |
| `def` | `test_max_files_limit` |
| `def` | `test_stages_populated` |
| `def` | `test_search_method_is_string` |
| `def` | `test_empty_directory` |
| `def` | `test_nonexistent_directory` |
| `def` | `test_glob_fallback_works` |
| `def` | `test_compression_ratio_reasonable` |
| `def` | `test_files_scanned_matches_directory` |
| `def` | `test_run_benchmark_function` |

#### `test_segment_compression_cache.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_segment_cache_reuses_cached_result_for_same_query_and_method` |

#### `test_selection_strategy_toggle.py`

TDD tests for H1 of v1.11.0: selection_strategy parameter on read_skeleton/_generate_skeleton.

| Kind | Name |
|------|------|
| `def` | `_build_compressor_with_doc` |
| `class` | `TestSelectionStrategyToggle` |

#### `test_semantic_compressor_unit.py`

Unit Tests for SemanticCompressor

| Kind | Name |
|------|------|
| `class` | `TestSemanticNodeDataclass` |
| `class` | `TestTokenCounting` |
| `class` | `TestTextChunking` |
| `class` | `TestEntityExtraction` |
| `class` | `TestSummaryGeneration` |
| `class` | `TestGraphBuilding` |
| `class` | `TestPageRankImportance` |
| `class` | `TestSkeletonGeneration` |
| `class` | `TestFidelityModulation` |
| `class` | `TestSemanticSearch` |
| `class` | `TestInputValidation` |
| `class` | `TestGetStats` |
| `class` | `TestCompressorConfiguration` |
| `class` | `TestEdgeCasesAndRobustness` |
| `class` | `TestIntegrationScenarios` |
| `class` | `TestPageRankCaching` |
| `def` | `run_coverage_report` |

#### `test_semantic_fidelity.py`

Compression Validation Tests (formerly "Semantic Fidelity Benchmark Tests").

| Kind | Name |
|------|------|
| `def` | `sample_document` |
| `def` | `compressor` |
| `def` | `count_tokens` |
| `class` | `TestCompressionValidation` |

#### `test_server_aliases.py`

Contract tests for server alias mapping helpers.

| Kind | Name |
|------|------|
| `def` | `test_build_server_class_overrides_collects_required_aliases` |
| `def` | `test_build_server_class_overrides_missing_alias_raises_helpful_error` |
| `def` | `test_allowed_factory_override_keys_include_server_and_app_keys` |
| `def` | `test_validate_override_keys_rejects_unknown_entries` |
| `def` | `test_validate_override_keys_accepts_none_and_known_entries` |

#### `test_server_factory_service.py`

Contract tests for app-layer server factory service.

| Kind | Name |
|------|------|
| `def` | `test_factory_build_creates_core_components_and_services` |
| `def` | `test_factory_build_uses_cwd_and_home_for_path_validator` |
| `def` | `test_factory_build_default_delegates_to_build_with_production_wiring` |
| `def` | `test_factory_build_default_rejects_unknown_override_keys` |
| `def` | `test_resolve_class_overrides_merges_known_keys_and_defaults` |
| `def` | `test_default_class_map_covers_allowed_override_keys` |
| `def` | `test_build_default_uses_resolve_class_overrides_result_for_build_wiring` |
| `def` | `test_build_kwargs_from_resolved_classes_maps_expected_build_arguments` |
| `def` | `test_build_default_uses_build_kwargs_helper_for_build_invocation` |
| `def` | `test_build_helpers_expose_expected_default_configs` |
| `def` | `test_build_uses_helper_configs_for_constructor_kwargs` |
| `def` | `test_build_class_dispatch_uses_subclass_helper_overrides` |
| `def` | `test_build_logging_helpers_expose_expected_payloads` |
| `def` | `test_build_class_dispatch_uses_subclass_logging_helper_overrides` |
| `def` | `test_build_service_layer_wires_adapter_dependencies` |
| `def` | `test_build_delegates_service_layer_construction_through_helper` |
| `def` | `test_build_core_runtime_layer_wires_foundational_dependencies` |
| `def` | `test_build_delegates_core_runtime_construction_through_helper` |
| `def` | `test_factory_typed_artifact_contracts_are_declared` |
| `def` | `test_factory_class_map_contract_declared_and_aligned_with_overrides` |
| `def` | `test_build_default_rejects_default_class_map_drift_before_override_merge` |
| `def` | `test_factory_build_kwargs_contract_declared_and_aligned_with_helper_output` |
| `def` | `test_build_default_rejects_build_kwargs_drift_before_build_call` |
| `def` | `test_validate_factory_contracts_runs_validation_pipeline_in_order` |
| `def` | `test_build_default_delegates_to_validate_factory_contracts` |
| `def` | `test_build_default_from_validation_delegates_to_build` |
| `def` | `test_build_default_delegates_to_build_default_from_validation` |
| `def` | `test_factory_build_default_request_contract_declared` |
| `def` | `test_factory_build_request_contract_declared` |
| `def` | `test_build_request_from_default_validation_merges_runtime_and_wiring` |
| `def` | `test_build_default_from_validation_delegates_to_build_from_request` |
| `def` | `test_validate_build_request_map_contract_declared_and_aligned` |
| `def` | `test_build_from_request_rejects_request_drift_before_build_call` |
| `def` | `test_validate_build_default_request_map_contract_declared_and_aligned` |
| `def` | `test_build_default_rejects_default_request_drift_before_factory_validation` |
| `def` | `test_validate_factory_validation_result_contract_declared_and_aligned` |
| `def` | `test_build_default_from_validation_rejects_validation_drift_before_request_merge` |
| `def` | `test_validate_default_build_inputs_runs_request_then_factory_validation` |
| `def` | `test_build_default_delegates_to_validate_default_build_inputs` |
| `def` | `test_validate_default_build_inputs_map_contract_declared_and_aligned` |
| `def` | `test_build_default_rejects_default_build_inputs_drift_before_dispatch` |
| `def` | `test_validate_default_build_inputs_map_validates_nested_payloads_in_order` |
| `def` | `test_build_default_rejects_default_build_inputs_nested_request_drift_before_dispatch` |
| `def` | `test_validate_default_build_inputs_map_uses_class_dispatch_for_nested_validators` |
| `def` | `test_build_default_from_validation_rejects_request_drift_before_validation_merge` |
| `def` | `_full_build_kwargs` |
| `def` | `test_factory_contract_key_constants_align_with_typeddict_schemas` |
| `def` | `test_contract_key_mismatch_message_uses_canonical_format` |
| `def` | `test_validate_build_default_request_map_rejects_extra_keys_with_exact_message` |
| `def` | `test_validate_factory_validation_result_map_rejects_extra_keys_with_exact_message` |
| `def` | `test_validate_default_build_inputs_map_rejects_extra_keys_with_exact_message` |
| `def` | `test_validate_build_request_map_rejects_extra_keys_with_exact_message` |
| `def` | `test_default_build_happy_path_runs_full_validated_chain` |
| `def` | `test_validate_default_class_map_uses_class_dispatch_for_contract_keys` |
| `def` | `test_validate_build_kwargs_map_uses_class_dispatch_for_contract_keys` |
| `def` | `test_validate_build_default_request_map_uses_class_dispatch_for_contract_keys` |
| `def` | `test_validate_factory_validation_result_map_uses_class_dispatch_for_contract_keys` |

#### `test_server_lifecycle.py`

Tests for server lifespan management (v0.4.4)

| Kind | Name |
|------|------|
| `class` | `TestServerLifecycle` |
| `async def` | `test_server_initialization_sequence` |

#### `test_server_runtime_wrapper.py`

Compatibility tests for server runtime delegation wrapper.

| Kind | Name |
|------|------|
| `async def` | `test_server_run_delegates_to_runtime_service` |

#### `test_server_service_adapter.py`

Contract tests for app-layer server service adapter.

| Kind | Name |
|------|------|
| `def` | `test_adapter_delegates_persistence_load_calls_with_expected_args` |
| `def` | `test_adapter_delegates_context_and_progress_helpers` |

#### `test_server_unit.py`

Unit tests for SemanticModulatorServer (src/server.py)

| Kind | Name |
|------|------|
| `class` | `TestACEContextManager` |
| `class` | `TestSemanticModulatorServerInitialization` |
| `class` | `TestLoadPersistedDocuments` |
| `class` | `TestLoadFileSyncMetadata` |
| `class` | `TestSaveFileSyncMetadata` |
| `class` | `TestBuildContext` |
| `class` | `TestValidateFileId` |
| `class` | `TestValidateNodeIds` |
| `class` | `TestValidateTokenCount` |
| `class` | `TestCreateProgressBar` |

#### `test_server_wiring_contract.py`

Integration contract tests for server wiring from factory output.

| Kind | Name |
|------|------|
| `def` | `test_server_constructor_relies_on_factory_for_runtime_and_adapter` |

#### `test_session_continuity.py`

Tests for session continuity — validates skeleton survival across context compaction.

| Kind | Name |
|------|------|
| `def` | `_make_large_document` |
| `class` | `TestSkeletonSurvivesJsonRoundTrip` |
| `class` | `TestSkeletonSurvivesTruncation` |
| `class` | `TestSkeletonSelfContained` |
| `class` | `TestCompressionPreservesKeyInformation` |

#### `test_session_journal.py`

Tests for SessionJournal (src/session_journal.py).

| Kind | Name |
|------|------|
| `def` | `_make_journal` |
| `class` | `TestWriteEvent` |
| `class` | `TestRecover` |
| `class` | `TestSessionIsolation` |
| `class` | `TestPersistence` |
| `class` | `TestWALMode` |

#### `test_skeleton_ratio_knob.py`

Regression lock: `ingest_context`'s `skeleton_ratio` param must actually

| Kind | Name |
|------|------|
| `def` | `_multi_section_doc` |
| `def` | `_count_anchors` |
| `class` | `TestSkeletonRatioValidation` |
| `class` | `TestSkeletonRatioTakesEffect` |

#### `test_skill_external_adapters.py`

Tests for portable skill external adapter normalization.

| Kind | Name |
|------|------|
| `class` | `_LangChainDoc` |
| `class` | `_LlamaNode` |
| `def` | `test_normalize_langchain_like_objects` |
| `def` | `test_normalize_llamaindex_like_objects` |
| `def` | `test_infer_adapter_for_json_shapes` |
| `def` | `test_adapt_input_to_text_auto_and_metadata` |

#### `test_skill_pipeline.py`

Unit tests for skill-level composable compression pipeline.

| Kind | Name |
|------|------|
| `def` | `test_pipeline_runs_stages_in_order` |
| `async def` | `test_pipeline_async_supports_sync_and_async_stages` |
| `def` | `test_pipeline_error_contains_stage_name` |

#### `test_skill_scripts.py`

Smoke and end-to-end tests for Claude skill utility scripts.

| Kind | Name |
|------|------|
| `def` | `_run` |
| `def` | `_run_skill` |
| `def` | `test_profile_tokens_help` |
| `def` | `test_compress_context_help` |
| `def` | `test_validate_evidence_help` |
| `def` | `test_skill_workflow_help` |
| `def` | `test_profile_tokens_json_output` |
| `def` | `test_compress_context_evidence_aware_output` |
| `def` | `test_validate_evidence_exit_codes` |
| `def` | `test_skill_workflow_end_to_end` |
| `def` | `test_portable_skill_compress_context_includes_explainability_fields` |
| `def` | `test_portable_skill_workflow_includes_explainability_fields` |
| `def` | `test_skill_benchmark_toon_vs_json_guard_contract` |
| `def` | `test_skill_scripts_are_self_contained_without_project_src_imports` |
| `def` | `test_portable_skill_compress_context_accepts_langchain_json_adapter` |

#### `test_skill_structured_controls.py`

Tests for structured segment controls in skill compression engine.

| Kind | Name |
|------|------|
| `def` | `test_structured_preserve_tag_keeps_segment` |
| `def` | `test_structured_rate_tag_overrides_global_ratio_for_tagged_block` |
| `def` | `test_structured_invalid_marker_raises_clear_error` |

#### `test_source_order_render_surfaces.py`

Source-order render locks for the code skeleton + multi-level skeleton surfaces.

| Kind | Name |
|------|------|
| `def` | `_chunk` |
| `class` | `TestCodeSkeletonRendersInSourceOrder` |
| `class` | `TestMultiLevelSkeletonRendersInInputOrder` |

#### `test_structural_summary.py`

Tests for structural_summary.py -- AST-based code outline generator.

| Kind | Name |
|------|------|
| `def` | `_auth_code` |
| `def` | `_auth_path` |
| `def` | `test_detect_python` |
| `def` | `test_detect_javascript` |
| `def` | `test_detect_typescript` |
| `def` | `test_detect_unknown` |
| `def` | `test_detect_no_extension` |
| `def` | `test_summary_contains_class_name` |
| `def` | `test_summary_contains_function_names` |
| `def` | `test_summary_no_function_bodies` |
| `def` | `test_summary_has_imports` |
| `def` | `test_summary_preserves_type_hints` |
| `def` | `test_summary_savings_above_80pct` |
| `def` | `test_summary_token_count_smaller` |
| `def` | `test_summary_result_fields` |
| `def` | `test_summary_line_count_correct` |
| `def` | `test_summary_dataclass_fields_shown` |
| `def` | `test_summary_empty_file` |
| `def` | `test_summary_syntax_error_fallback` |
| `def` | `test_summary_generic_fallback_non_python` |

#### `test_structured_bundles.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `_sample_evidence` |
| `def` | `_sample_artifact` |
| `def` | `setup_function` |
| `def` | `test_registry_stores_and_lists_scoped_bundles` |
| `def` | `test_registry_rejects_scope_mismatch_on_get` |
| `def` | `test_registry_replays_bundle_in_requested_format` |

#### `test_structured_content.py`

#190 structured-content detection + record splitting (model-free).

| Kind | Name |
|------|------|
| `class` | `TestDetectStructuredContent` |
| `class` | `TestSplitJsonRecords` |
| `class` | `TestGroupRecordsBySize` |
| `class` | `TestSplitJsonlRecords` |
| `class` | `TestSplitCsvRecords` |

#### `test_structured_logging.py`

Tests for structured logging with async context propagation.

| Kind | Name |
|------|------|
| `def` | `reset_logger` |
| `def` | `get_logger_with_capture` |
| `class` | `TestBasicLogging` |
| `class` | `TestJSONFormatter` |
| `class` | `TestHumanFormatter` |
| `class` | `TestContextPropagation` |
| `class` | `TestOpenTelemetryIntegration` |
| `class` | `TestConfiguration` |
| `class` | `TestPerformance` |
| `class` | `TestContextVariables` |
| `class` | `TestErrorHandling` |
| `class` | `TestContextMerging` |
| `class` | `TestLogLevelFiltering` |
| `class` | `TestRequestIDGeneration` |
| `class` | `TestEdgeCases` |

#### `test_summary_admonition_strip.py`

Dogfood 2026-07-11: a live compress of httpx's mkdocs docs surfaced

| Kind | Name |
|------|------|
| `def` | `_bare` |
| `class` | `TestSummaryAdmonitionStrip` |

#### `test_tee_recovery.py`

Tests for the tee/recovery system.

| Kind | Name |
|------|------|
| `class` | `TestTeeEntry` |
| `class` | `TestTeeStoreModes` |
| `class` | `TestTeeStoreOperations` |
| `class` | `TestTeeStoreListDelete` |
| `class` | `TestTeeStoreEviction` |
| `class` | `TestTeeStorePersistence` |
| `class` | `TestTeeStoreStats` |
| `class` | `TestCreateTeeStore` |
| `class` | `TestGenerateId` |

#### `test_temporal_graph.py`

Tests for temporal fact lifecycle tracking.

| Kind | Name |
|------|------|
| `def` | `test_record_document_state_and_invalidation` |
| `def` | `test_reingest_invalidates_removed_facts` |
| `def` | `test_search_timeline_filters_by_query` |

#### `test_temporal_handlers.py`

Tests for temporal MCP handlers and retrieval filtering.

| Kind | Name |
|------|------|
| `async def` | `test_temporal_handlers_round_trip` |
| `async def` | `test_temporal_handlers_include_invalidated_when_requested` |

#### `test_temporal_hypothesis.py`

Property tests for temporal fact lifecycle semantics.

| Kind | Name |
|------|------|
| `def` | `test_invalidated_facts_are_visible_before_but_not_after` |

#### `test_tenancy_hypothesis.py`

Property-based tests for scope identity normalization and isolation.

| Kind | Name |
|------|------|
| `def` | `test_unscoped_round_trip_preserves_raw_file_id` |
| `def` | `test_scoped_round_trip_preserves_visible_identity` |
| `def` | `test_scope_matching_rejects_other_workspaces` |
| `def` | `test_partial_scope_filters_only_requested_dimensions` |

#### `test_tenancy_scoping.py`

Phase 1 tenancy tests for scoped document identity and isolation.

| Kind | Name |
|------|------|
| `def` | `test_scoped_identity_round_trip` |
| `def` | `test_persistence_scopes_same_file_id` |
| `async def` | `test_handler_scope_isolates_same_file_id` |

#### `test_tensor_grep_integration.py`

Tests for tensor_grep_integration.py

| Kind | Name |
|------|------|
| `def` | `_make_completed_process` |
| `def` | `_make_popen` |
| `class` | `TestIsAvailable` |
| `class` | `TestGetRepoMap` |
| `class` | `TestCodeSearch` |
| `class` | `TestASTSearch` |
| `class` | `TestGetContextRender` |
| `class` | `TestScanRuleset` |

#### `test_token_estimation.py`

Tests for src/token_estimation.py (TokenEstimator class).

| Kind | Name |
|------|------|
| `def` | `estimator` |
| `def` | `test_fast_estimate_uses_len_div_4` |
| `def` | `test_fast_estimate_on_short_text` |
| `def` | `test_fast_estimate_on_empty_string` |
| `def` | `test_json_estimate_uses_len_div_2` |
| `def` | `test_json_estimate_on_empty_string` |
| `def` | `test_byte_count_is_exact` |
| `def` | `test_byte_count_on_empty_string` |
| `def` | `test_unicode_content_byte_difference` |
| `def` | `test_ascii_bytes_equal_char_count` |
| `def` | `test_accurate_count_is_positive_for_nonempty` |
| `def` | `test_accurate_count_zero_for_empty` |
| `def` | `test_accurate_count_matches_tiktoken` |
| `def` | `test_fallback_when_tiktoken_unavailable` |
| `def` | `test_estimate_all_returns_dict` |
| `def` | `test_estimate_all_has_required_keys` |
| `def` | `test_empty_string_returns_zeros` |
| `def` | `test_estimate_all_values_are_non_negative` |
| `def` | `test_estimate_all_json_estimate_gte_fast_estimate` |
| `def` | `test_code_content_estimation` |
| `def` | `test_code_content_byte_count_matches_utf8` |
| `def` | `test_bytes_field_matches_count_bytes` |
| `def` | `test_estimated_field_matches_estimate_fast` |
| `def` | `test_json_estimated_field_matches_estimate_json` |

#### `test_token_refiner.py`

Tests for src/token_refiner.py.

| Kind | Name |
|------|------|
| `def` | `make_refiner` |
| `class` | `TestScoreToken` |
| `class` | `TestScoreTokens` |
| `class` | `TestRefineRemoval` |
| `class` | `TestRefinePreservation` |
| `class` | `TestRefineShortSentences` |
| `class` | `TestRefineEdgeCases` |
| `class` | `TestRefineCompressionRatio` |
| `class` | `TestRefinedResult` |
| `class` | `TestCustomConfiguration` |

#### `test_token_savings.py`

Token Savings Benchmark Tests

| Kind | Name |
|------|------|
| `class` | `TestTokenSavings` |
| `class` | `TestSCARTokenSavings` |
| `class` | `TestEndToEndSavings` |
| `def` | `print_summary_report` |

#### `test_tool_profile_service.py`

Contract tests for app-layer tool profile bootstrap service.

| Kind | Name |
|------|------|
| `def` | `test_profile_bootstrap_returns_active_profile_and_names` |
| `def` | `test_profile_bootstrap_logs_fallback_warning` |
| `def` | `test_profile_bootstrap_request_contract_declared` |
| `def` | `test_profile_bootstrap_validate_request_map_rejects_extra_key` |

#### `test_tool_profiles.py`

Contract tests for MCP tool profile behavior.

| Kind | Name |
|------|------|
| `def` | `test_core_stable_profile_contract` |
| `def` | `test_full_profile_contains_core_stable_tools` |

#### `test_toon_bundle_roundtrip.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_handoff_bundle_toon_roundtrip` |

#### `test_toon_serializer_coverage.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `def` | `test_serialize_search_results_handles_types_and_empty` |
| `def` | `test_serialize_domain_helpers_and_stats` |
| `def` | `test_format_response_variants_and_savings` |

#### `test_training_utils_coverage.py`

(no module docstring — see symbols below)

| Kind | Name |
|------|------|
| `class` | `_TinyCompressor` |
| `class` | `_TinyAlignment` |
| `def` | `_tiny_config` |
| `def` | `test_dataset_contracts_and_synthetic_data_shapes` |
| `def` | `test_trainer_compressor_train_eval_checkpoint` |
| `def` | `test_trainer_alignment_train_eval` |

#### `test_transcript_extractor.py`

Tests for transcript extraction pipeline (src/transcript_extractor.py).

| Kind | Name |
|------|------|
| `def` | `reset_memory` |
| `class` | `TestSplitSentences` |
| `class` | `TestHasSignal` |
| `class` | `TestStripRolePrefix` |
| `class` | `TestExtractInsights` |
| `class` | `TestIngestTranscript` |

#### `test_url_fetcher.py`

Tests for SSRF-hardened URL fetcher (src/url_fetcher.py).

| Kind | Name |
|------|------|
| `def` | `_make_transport` |
| `async def` | `test_fetch_url_happy_path_returns_decoded_text` |
| `async def` | `test_fetch_url_returns_json_content` |
| `async def` | `test_mitigation_1_http_scheme_rejected` |
| `async def` | `test_mitigation_1_ftp_scheme_rejected` |
| `async def` | `test_mitigation_1_no_hostname_rejected` |
| `async def` | `test_mitigation_3_301_redirect_rejected` |
| `async def` | `test_mitigation_3_302_redirect_rejected` |
| `async def` | `test_mitigation_3_307_redirect_rejected` |
| `async def` | `test_mitigation_4_content_length_header_too_large` |
| `async def` | `test_mitigation_4_body_too_large_no_content_length` |
| `async def` | `test_mitigation_4_exactly_10_mb_is_allowed` |
| `async def` | `test_mitigation_4_size_cap_aborts_stream_without_buffering_whole_body` |
| `async def` | `test_mitigation_6_binary_content_type_rejected` |
| `async def` | `test_mitigation_6_image_content_type_rejected` |
| `async def` | `test_mitigation_6_text_html_allowed` |
| `async def` | `test_mitigation_6_application_xml_allowed` |
| `async def` | `test_mitigation_6_application_yaml_allowed` |
| `async def` | `test_mitigation_6_application_x_yaml_allowed` |
| `async def` | `test_non_2xx_status_raises_error` |
| `async def` | `test_server_error_raises_error` |
| `async def` | `test_urlfetcherror_has_code_attribute` |
| `def` | `test_urlfetcherror_construction` |

#### `test_url_fetcher_ip_pinning.py`

#282 — DNS-rebind TOCTOU close: pin the validated IP to the connection.

| Kind | Name |
|------|------|
| `def` | `_fake_getaddrinfo_for` |
| `class` | `TestPinRequest` |
| `class` | `TestFetchUrlPinsConnection` |

#### `test_url_fetcher_nonblocking.py`

#219: DNS resolution in ``fetch_url`` must run OFF the event loop.

| Kind | Name |
|------|------|
| `async def` | `test_dns_resolution_does_not_block_event_loop` |

#### `test_url_fetcher_ssrf_fence_6b.py`

Phase 2 Chunk 6B SSRF regression fence for token-saver url_fetcher.py.

| Kind | Name |
|------|------|
| `def` | `_make_transport` |
| `class` | `TestClassESizeCap` |
| `class` | `TestClassFSchemeAndRedirectFence` |

#### `test_url_fetcher_ssrf_hardening.py`

Tests for the 4 new SSRF hardening mitigations added in Phase 2 Chunk 6A.

| Kind | Name |
|------|------|
| `def` | `_make_transport` |
| `def` | `_fake_getaddrinfo_for` |
| `class` | `TestIPv4MappedIPv6Normalisation` |
| `class` | `TestDNSRebindingDetection` |
| `class` | `TestBlockedHostnamePreCheck` |
| `class` | `TestInternalTLDBlock` |

#### `test_ux_improvements.py`

UX Improvements Test Suite (v0.4.1)

| Kind | Name |
|------|------|
| `class` | `TestFidelityAdvisor` |
| `class` | `TestSmartError` |
| `class` | `TestCompressionAdvisor` |
| `class` | `TestUXIntegration` |

#### `test_validation_hooks_file_id.py`

Tests for ``_validate_ingest`` — file_id character class.

| Kind | Name |
|------|------|
| `class` | `TestIngestFileIdCharacterClass` |

#### `test_visualization.py`

Tests for Graph Visualization (v0.6.0)

| Kind | Name |
|------|------|
| `def` | `compressor` |
| `def` | `sample_document` |
| `async def` | `ingested_document` |
| `def` | `visualizer` |
| `class` | `TestASCIIRendering` |
| `class` | `TestJSONExport` |
| `class` | `TestGraphMLExport` |
| `class` | `TestHTMLVisualization` |
| `class` | `TestCompressionDecisionExplanation` |
| `class` | `TestMCPToolIntegration` |

#### `test_web_connector.py`

Tests for web connector normalization.

| Kind | Name |
|------|------|
| `def` | `test_collects_pages_into_documents` |

#### `test_wiring_integration.py`

Integration tests for Phase 5 wiring — verifies that modules are actually

| Kind | Name |
|------|------|
| `def` | `compressor` |
| `def` | `sample_text` |
| `def` | `handler_context` |
| `class` | `TestCompressorInitialization` |
| `class` | `TestIngestWiring` |
| `class` | `TestSkeletonWiring` |
| `class` | `TestSearchWiring` |
| `class` | `TestPhase5ToolsWork` |
| `class` | `TestSemanticChunking` |
| `class` | `TestIntraDocDedup` |
| `class` | `TestQueryAdaptive` |
| `class` | `TestWorkflowGuidance` |

#### `test_worldclass_batch1.py`

World-Class Compression Sprint — Batch 1 regression locks (2026-07-01).

| Kind | Name |
|------|------|
| `def` | `_make_context` |
| `def` | `_make_skeleton` |
| `def` | `_make_estimate` |
| `async def` | `test_ingest_surfaces_reasoning_confidence_and_estimated_compressed` |
| `async def` | `test_ingest_small_doc_returns_honesty_note` |
| `def` | `test_hidden_boilerplate_hoisted_to_header_once` |

#### `test_worldclass_batch2.py`

World-Class Compression Sprint — Task 4: summary + entity quality (2026-07-02).

| Kind | Name |
|------|------|
| `def` | `_bare_compressor` |
| `class` | `TestEntityStopwordFilter` |
| `class` | `TestEntityMarkdownLinkStrip` |
| `class` | `TestSummaryMarkdownStrip` |

### Other files

- `f11_gac_fixture_harness_receipt.json`

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
