# Folder guide: `src/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../CLAUDE.md).

## Contents

### Python modules

#### `__init__.py`

Semantic Modulator - Adaptive Semantic Fidelity MCP Server

_No top-level classes or functions (may re-export only)._

#### `__main__.py`

Allow running as `python -m src`.

_No top-level classes or functions (may re-export only)._

#### `ace_framework.py`

ACE Framework for Token Saver 5000

| Kind | Name |
|------|------|
| `class` | `BulletType` |
| `class` | `ACEBullet` |
| `class` | `ACEContext` |
| `class` | `ACEGenerator` |
| `class` | `ACEReflector` |
| `class` | `ACECurator` |
| `class` | `ACEFramework` |

#### `adaptive_rate_allocator.py`

Adaptive Rate Allocator

| Kind | Name |
|------|------|
| `class` | `AdaptiveRateAllocator` |
| `class` | `ContextWindowAdapter` |
| `class` | `MultiLevelSemanticEncoder` |

#### `afm.py`

Adaptive Focus Memory (AFM) for Language Models

| Kind | Name |
|------|------|
| `class` | `FidelityLevel` |
| `class` | `ImportanceLevel` |
| `class` | `Message` |
| `class` | `AFMConfig` |
| `class` | `PackingStats` |
| `class` | `TokenCounter` |
| `class` | `Compressor` |
| `class` | `HeuristicCompressor` |
| `class` | `LLMCompressor` |
| `class` | `Embedder` |
| `class` | `SentenceTransformerEmbedder` |
| `class` | `HashingEmbedder` |
| `class` | `ImportanceClassifier` |
| `class` | `FocusManager` |

#### `attention_pruning.py`

Attention-guided pruning for context compression.

| Kind | Name |
|------|------|
| `def` | `score_nodes_by_relevance` |
| `def` | `prune_by_relevance` |

#### `audited_graph.py`

Audited Semantic Graph with Provenance Tracking

| Kind | Name |
|------|------|
| `class` | `NodeProvenance` |
| `class` | `QualityHistory` |
| `class` | `AuditedSemanticNode` |
| `class` | `CompositionEdge` |
| `class` | `AuditedSemanticGraph` |

#### `batch_manager.py`

Batch Compression Manager for Token Saver 5000

| Kind | Name |
|------|------|
| `class` | `BatchDocument` |
| `class` | `BatchResult` |
| `class` | `BatchProgress` |
| `class` | `BatchProgressTracker` |
| `class` | `BatchCompressionManager` |
| `async def` | `batch_ingest_from_dict` |
| `async def` | `batch_ingest_from_files` |

#### `benchmark_guard.py`

Benchmark regression guard utilities.

| Kind | Name |
|------|------|
| `class` | `GuardViolation` |
| `def` | `load_json` |
| `def` | `evaluate_report_against_thresholds` |

#### `benchmark_harness.py`

Benchmark harness for stable token-savings regression tracking.

| Kind | Name |
|------|------|
| `class` | `BenchmarkCase` |
| `class` | `BenchmarkResult` |
| `class` | `BenchmarkSummary` |
| `class` | `BenchmarkExecutionDetail` |
| `def` | `_project_root` |
| `def` | `default_corpus_path` |
| `def` | `load_benchmark_cases` |
| `def` | `_savings_pct` |
| `def` | `_f1` |
| `def` | `_quality_overlap_metrics` |
| `def` | `run_benchmark_cases_detailed` |
| `def` | `run_benchmark_cases` |
| `def` | `filter_cases` |
| `def` | `summary_to_dict` |
| `def` | `write_summary` |
| `def` | `run_method_comparison` |

#### `blind_spot_detector.py`

Blind Spot Detector - Self-Correcting Context Loop

| Kind | Name |
|------|------|
| `class` | `BlindSpot` |
| `class` | `BlindSpotReport` |
| `class` | `BlindSpotDetector` |
| `class` | `HaloEffectDetector` |

#### `bm25_utils.py`

BM25 utility functions for semantic search.

| Kind | Name |
|------|------|
| `def` | `bm25_tokenize` |
| `def` | `bm25_idf` |
| `def` | `bm25_term_freq_with_stemming` |
| `def` | `_query_has_quoted_phrase` |
| `def` | `query_has_lexical_shape` |
| `def` | `bm25_top1_is_discriminative` |
| `def` | `bm25_scores` |

#### `budget_monitor.py`

Token budget monitoring with configurable limits and alerts.

| Kind | Name |
|------|------|
| `class` | `BudgetLimit` |
| `class` | `BudgetCheckResult` |
| `class` | `TokenBudgetMonitor` |
| `def` | `create_budget_monitor` |

#### `bundle_distiller.py`

Build token-efficient, auditable handoff bundle artifacts.

| Kind | Name |
|------|------|
| `def` | `_utc_now` |
| `def` | `_search_results` |
| `def` | `_context_block` |
| `def` | `_replay_text` |
| `def` | `distill_handoff_bundle` |

#### `cache_compatibility.py`

Provider + harness prompt-cache compatibility guidance.

| Kind | Name |
|------|------|
| `def` | `_normalize_harness` |
| `def` | `assess_cache_compatibility` |

#### `cache_diagnostics.py`

Root-cause diagnostics for prompt cache misses.

| Kind | Name |
|------|------|
| `def` | `_hash_text` |
| `def` | `_normalize_line_endings` |
| `def` | `_find_numeric_field` |
| `def` | `_diff_preview` |
| `def` | `extract_section_order` |
| `def` | `_extract_sections` |
| `def` | `_normalize_blank_lines` |
| `def` | `_normalize_json_content` |
| `def` | `_normalize_text_content` |
| `def` | `detect_semantic_equivalence_drift` |
| `def` | `detect_section_interleaving` |
| `def` | `diagnose_cache_miss` |

#### `cache_strategy_advisor.py`

Provider-aware cache strategy advisor.

| Kind | Name |
|------|------|
| `class` | `CacheStrategy` |
| `def` | `advise_cache_strategy` |

#### `cli_output_optimizer.py`

CLIOutputOptimizer — RTK-inspired CLI output filtering for Token Saver 5000.

| Kind | Name |
|------|------|
| `class` | `FilterResult` |
| `class` | `CLIOutputOptimizer` |

#### `client_config.py`

Client and model configuration for context-window-aware compression.

| Kind | Name |
|------|------|
| `def` | `_detect_provider` |
| `def` | `get_recommended_ratio` |
| `class` | `ClientConfig` |
| `class` | `SessionConfigStore` |

#### `code_compression_adapter.py`

Code Compression Adapter - Unified Interface for Text and Code Compression

| Kind | Name |
|------|------|
| `class` | `AdapterNode` |
| `class` | `CodeCompressionAdapter` |

#### `code_compressor.py`

Code-Specific Semantic Compressor

| Kind | Name |
|------|------|
| `class` | `CodeLanguage` |
| `class` | `CodeChunk` |
| `class` | `CodeSemanticCompressor` |

#### `compression_advisor.py`

Compression Size Preview and Estimation

| Kind | Name |
|------|------|
| `class` | `CompressionEstimate` |
| `class` | `CompressionAdvisor` |
| `def` | `estimate_compression_size` |

#### `compression_pipeline.py`

Composable multi-pass pipeline for read-skeleton compression flows.

| Kind | Name |
|------|------|
| `def` | `_resolve_auto_mode` |
| `def` | `_extract_h1_query` |
| `def` | `_generate_skeleton` |
| `def` | `_stage_payload` |
| `def` | `run_read_skeleton_pipeline` |

#### `compression_presets.py`

Compression profile presets for common use cases.

| Kind | Name |
|------|------|
| `class` | `CompressionPreset` |
| `def` | `get_preset` |
| `def` | `list_presets` |

#### `compression_profiles.py`

Named compression profile presets for Semantic Modulator.

| Kind | Name |
|------|------|
| `class` | `CompressionProfile` |
| `def` | `_build_profile` |
| `def` | `get_profile` |
| `def` | `get_default_profile` |
| `def` | `list_profiles` |
| `class` | `ProfileManager` |
| `def` | `apply_urgency` |
| `def` | `apply_profile` |
| `def` | `auto_select_profile` |

#### `compression_replay.py`

Compression replay log for tracking and optimizing compression strategies.

| Kind | Name |
|------|------|
| `class` | `CompressionReplayLog` |

#### `compression_rewards.py`

Decomposed Compression Rewards System

| Kind | Name |
|------|------|
| `class` | `RewardComponent` |
| `class` | `SchemaValidationResult` |
| `class` | `SemanticPreservationResult` |
| `class` | `FidelityAdherenceResult` |
| `class` | `CompositionIntegrityResult` |
| `class` | `MemoryDisciplineResult` |
| `class` | `CompressionReward` |
| `class` | `CompressionRewardCalculator` |
| `class` | `ProgressiveRewardShaper` |

#### `compression_verifier.py`

Compression Verifier with Contract System

| Kind | Name |
|------|------|
| `class` | `ContractType` |
| `class` | `ContractViolation` |
| `class` | `VerificationResult` |
| `class` | `CompressionContract` |
| `class` | `CompressionVerifier` |
| `class` | `BatchVerifier` |

#### `connector_registry.py`

Registry for managed connector feeds and available connector types.

| Kind | Name |
|------|------|
| `def` | `_utc_now` |
| `class` | `ConnectorFeedRecord` |
| `class` | `ConnectorRegistry` |

#### `constants.py`

Configuration Constants for Semantic Modulator

| Kind | Name |
|------|------|
| `def` | `get_semantic_chunk_boundary_threshold` |

#### `context_advisor.py`

Context window advisor.

| Kind | Name |
|------|------|
| `def` | `advise_context` |

#### `context_blocks.py`

Context block synthesis from temporal facts and recent access history.

| Kind | Name |
|------|------|
| `def` | `build_context_block` |

#### `context_decay.py`

Context decay and eviction for stale document management.

| Kind | Name |
|------|------|
| `class` | `AccessTracker` |
| `def` | `compute_decay_score` |

#### `dataset_registry.py`

Named dataset registry for experiment runs.

| Kind | Name |
|------|------|
| `def` | `_utc_now` |
| `def` | `_case_from_mapping` |
| `class` | `DatasetRecord` |
| `class` | `DatasetRegistry` |

#### `dead_code_detector.py`

Dead code detection via import graph analysis.

| Kind | Name |
|------|------|
| `class` | `DeadCodeReport` |
| `def` | `_extract_imports` |
| `def` | `_module_name_to_possible_files` |
| `def` | `_is_entry_point` |
| `def` | `_is_never_dead` |
| `def` | `detect_dead_files` |

#### `embedding_cache.py`

LRU Embedding Cache for Token Saver 5000 (v0.6.0)

| Kind | Name |
|------|------|
| `def` | `_persist_cache_at_exit` |
| `class` | `LRUEmbeddingCache` |
| `def` | `get_embedding_cache` |

#### `embedding_quantizer.py`

TurboQuant-inspired embedding quantization for semantic compression.

| Kind | Name |
|------|------|
| `class` | `QuantizedEmbedding` |
| `class` | `EmbeddingQuantizer` |

#### `embeddings.py`

Centralized Embedding Management for Semantic Modulator (v0.6.0)

| Kind | Name |
|------|------|
| `class` | `_EmbeddingManagerAdapter` |
| `class` | `EmbeddingTier` |
| `class` | `EmbeddingManager` |
| `def` | `get_embedding_manager` |

#### `embeddings_onnx.py`

ONNX Embedding Manager for Token Saver 5000 (v0.6.0)

| Kind | Name |
|------|------|
| `def` | `_uses_cls_pooling` |
| `class` | `ONNXEmbeddingManager` |
| `def` | `get_onnx_embedding_manager` |

#### `embeddings_tfidf.py`

TF-IDF Embedding Fallback for Token Saver 5000 (v0.6.0)

| Kind | Name |
|------|------|
| `class` | `TFIDFEmbeddingManager` |
| `def` | `get_tfidf_embedding_manager` |

#### `error_helpers.py`

Smart Error Message System with Fuzzy Matching

| Kind | Name |
|------|------|
| `class` | `SmartError` |
| `def` | `suggest_file_id` |
| `def` | `suggest_node_id` |
| `def` | `suggest_enum` |

#### `error_types.py`

Custom Exception Types for Reliability Infrastructure

| Kind | Name |
|------|------|
| `class` | `ReliabilityError` |
| `class` | `OperationTimeoutError` |
| `class` | `CircuitBreakerOpenError` |
| `class` | `RetryExhaustedError` |
| `class` | `RateLimitExceededError` |
| `class` | `GracefulDegradationError` |

#### `evidence_bundle.py`

Evidence Bundle System for Audited Compression Operations

| Kind | Name |
|------|------|
| `class` | `ContractStatus` |
| `class` | `ContractCheck` |
| `class` | `ContractResult` |
| `class` | `QualityMetrics` |
| `class` | `EvidenceBundle` |
| `class` | `EvidenceStore` |
| `def` | `get_evidence_store` |
| `def` | `reset_evidence_store` |

#### `experience_synthesis.py`

Experience Synthesis for Adversarial Testing

| Kind | Name |
|------|------|
| `class` | `TestCategory` |
| `class` | `SyntheticDocument` |
| `class` | `StressTestResult` |
| `class` | `BoundaryTestSuite` |
| `class` | `ExperienceSynthesizer` |

#### `experiment_tracker.py`

Tracked experiment runs over benchmark and evaluation datasets.

| Kind | Name |
|------|------|
| `def` | `_utc_now` |
| `class` | `ExperimentRunRecord` |
| `class` | `ExperimentTracker` |

#### `extractive_baseline.py`

Query-aware extractive compression baseline.

| Kind | Name |
|------|------|
| `class` | `ExtractiveCompressor` |

#### `fidelity_advisor.py`

Fidelity Level Recommendation System

| Kind | Name |
|------|------|
| `class` | `UseCase` |
| `class` | `FidelityRecommendation` |
| `class` | `FidelityAdvisor` |
| `def` | `recommend_fidelity` |

#### `fidelity_scoring.py`

Compression fidelity scoring.

| Kind | Name |
|------|------|
| `def` | `compute_fidelity_score` |

#### `file_sync_manager.py`

File Sync Manager for Semantic Modulator

| Kind | Name |
|------|------|
| `class` | `FileMetadata` |
| `class` | `FileSyncManager` |

#### `filter_rules.py`

Custom filter rules engine with TOML DSL.

| Kind | Name |
|------|------|
| `class` | `FilterRule` |
| `class` | `InlineTest` |
| `class` | `FilterRuleSet` |
| `class` | `FilterRuleEngine` |

#### `generative_rewrite.py`

Generative rewrite prompt template generator.

| Kind | Name |
|------|------|
| `def` | `_estimate_tokens` |
| `def` | `generate_rewrite_prompt` |

#### `graceful_degradation.py`

Graceful Degradation Strategies for Production Resilience

| Kind | Name |
|------|------|
| `class` | `GracefulDegradation` |

#### `graph_visualizer.py`

Graph Visualization Module for Token Saver 5000

| Kind | Name |
|------|------|
| `class` | `VisualizationConfig` |
| `class` | `GraphVisualizer` |

#### `health.py`

Health check and diagnostics for production monitoring.

| Kind | Name |
|------|------|
| `class` | `HealthStatus` |
| `class` | `ComponentHealth` |
| `class` | `HealthChecker` |
| `def` | `get_health` |

#### `history_compaction.py`

Stable-prefix-preserving conversation history compaction.

| Kind | Name |
|------|------|
| `def` | `_count_tokens` |
| `def` | `_render_messages` |
| `def` | `compact_conversation_history` |

#### `http_server.py`

HTTP Server for Health Checks and Metrics Endpoints

| Kind | Name |
|------|------|
| `async def` | `handle_liveness` |
| `async def` | `handle_readiness` |
| `async def` | `handle_diagnostics` |
| `async def` | `handle_metrics` |
| `async def` | `handle_root` |
| `async def` | `create_app` |
| `async def` | `start_http_server` |
| `def` | `is_http_enabled` |
| `def` | `get_http_config` |

#### `identifier_preservation.py`

Execution-critical identifier preservation for compressed tool output.

| Kind | Name |
|------|------|
| `def` | `extract_critical_identifiers` |
| `def` | `apply_identifier_guard` |

#### `identity_scope.py`

Helpers for tenant-aware document identity and filtering.

| Kind | Name |
|------|------|
| `def` | `_encode` |
| `def` | `_decode` |
| `def` | `_scope_values` |
| `def` | `compose_scoped_file_id` |
| `def` | `parse_scoped_file_id` |
| `def` | `display_file_id` |
| `def` | `has_scope` |
| `def` | `scope_matches` |

#### `intra_doc_dedup.py`

Intra-document redundancy detection and collapse.

| Kind | Name |
|------|------|
| `def` | `find_intra_duplicates` |
| `def` | `collapse_redundant_nodes` |

#### `keyword_anchoring.py`

Keyword anchoring for compression.

| Kind | Name |
|------|------|
| `def` | `apply_keyword_anchoring` |

#### `knowledge_compiler.py`

Knowledge compiler — periodic consolidation of flat memories into

| Kind | Name |
|------|------|
| `def` | `_tokenize` |
| `def` | `_similarity` |
| `def` | `_utc_now_iso` |
| `class` | `ConceptArticle` |
| `class` | `CompilationResult` |
| `class` | `KnowledgeCompiler` |

#### `knowledge_lint.py`

Knowledge lint — quality checks on stored memories and compiled articles.

| Kind | Name |
|------|------|
| `def` | `_similarity` |
| `def` | `_parse_iso` |
| `class` | `LintFinding` |
| `class` | `LintReport` |
| `class` | `KnowledgeLinter` |

#### `mcp_install.py`

Install or render Token Saver 5000 MCP configuration.

| Kind | Name |
|------|------|
| `def` | `repo_root` |
| `def` | `detect_claude_config_path` |
| `def` | `detect_project_mcp_config_path` |
| `def` | `detect_server_command` |
| `def` | `resolve_project_root_for_cli` |
| `def` | `recommend_setup_target` |
| `def` | `build_status_report` |
| `def` | `build_server_config` |
| `def` | `render_mcp_config` |
| `def` | `merge_mcp_server` |
| `def` | `remove_mcp_server` |
| `def` | `load_existing_config` |
| `def` | `inspect_mcp_installation` |
| `def` | `write_merged_mcp_config` |
| `def` | `remove_mcp_server_from_file` |
| `def` | `install_mcp` |
| `def` | `uninstall_mcp` |
| `def` | `install_project_mcp` |
| `def` | `uninstall_project_mcp` |
| `def` | `install_portable_project_mcp` |
| `def` | `detect_agent_config_path` |
| `def` | `_build_agents_md_section` |
| `def` | `_build_gemini_config` |
| `def` | `install_agent_config` |
| `def` | `uninstall_agent_config` |
| `def` | `inspect_all_agents` |
| `def` | `build_arg_parser` |
| `def` | `_format_all_agents_report` |
| `def` | `main` |

#### `media_metadata.py`

Helpers for multimodal media metadata inspection and normalization.

| Kind | Name |
|------|------|
| `def` | `detect_media_kind` |
| `def` | `inspect_media_path` |
| `def` | `summarize_payload_bytes` |

#### `memory_api.py`

Scoped explicit memory service for add/search/list/delete/profile flows.

| Kind | Name |
|------|------|
| `def` | `_utc_now` |
| `def` | `_scope_request` |
| `def` | `_scope_matches` |
| `def` | `_tokenize` |
| `class` | `MemoryEntry` |
| `class` | `MemoryAPI` |

#### `memory_classifier.py`

Memory classification system for auto-categorizing insights.

| Kind | Name |
|------|------|
| `class` | `ClassificationResult` |
| `def` | `classify_insight` |
| `def` | `classify_insights` |

#### `meta_tokens.py`

Lossless meta-token compression inspired by arXiv 2506.00307.

| Kind | Name |
|------|------|
| `class` | `MetaTokenEntry` |
| `class` | `MetaTokenResult` |
| `class` | `MetaTokenCompressor` |

#### `metrics.py`

Prometheus metrics collection with cardinality control.

| Kind | Name |
|------|------|
| `class` | `TokenSavingsTelemetry` |
| `def` | `compute_cost_savings` |
| `class` | `MetricsCollector` |
| `class` | `NoOpMetricsCollector` |
| `def` | `get_metrics` |

#### `model_optimizer.py`

Model-aware optimization recommendations for cost, fidelity, and prompt shaping.

| Kind | Name |
|------|------|
| `def` | `_find_numeric_field` |
| `def` | `_require_numeric_field` |
| `def` | `_find_provider_cache_read_tokens` |
| `def` | `summarize_provider_cache_usage` |
| `def` | `advise_cache_threshold` |
| `def` | `build_prompt_cache_key` |
| `def` | `optimize_for_model` |

#### `multi_level_skeleton.py`

Multi-level skeleton output.

| Kind | Name |
|------|------|
| `def` | `generate_multi_level_skeleton` |

#### `multimodal_compressor.py`

Multi-Modal Semantic Compressor

| Kind | Name |
|------|------|
| `class` | `ModalityType` |
| `class` | `MultiModalNode` |
| `class` | `MultiModalCompressor` |

#### `multimodal_ingestion.py`

Stable multimodal ingestion and search service.

| Kind | Name |
|------|------|
| `class` | `MultimodalProjectRecord` |
| `class` | `MultimodalIngestionService` |
| `def` | `build_multimodal_request` |
| `def` | `estimate_multimodal_payload_bytes` |

#### `node_identity.py`

Node identity helpers shared across server and handlers.

| Kind | Name |
|------|------|
| `def` | `extract_file_id_from_node` |
| `def` | `collect_file_ids` |

#### `observability.py`

OpenTelemetry distributed tracing with async context propagation.

| Kind | Name |
|------|------|
| `class` | `NoOpSpan` |
| `class` | `ObservabilityManager` |
| `def` | `get_observability` |
| `def` | `configure_observability` |

#### `path_validator.py`

Path Validator for File Sync Operations

| Kind | Name |
|------|------|
| `class` | `PathValidator` |

#### `persistence.py`

Persistent Storage Layer for Semantic Modulator

| Kind | Name |
|------|------|
| `class` | `PersistenceManager` |

#### `personalization.py`

User preference extraction and profile synthesis for explicit memory APIs.

| Kind | Name |
|------|------|
| `def` | `_extract_preferences` |
| `def` | `_extract_topics` |
| `def` | `build_user_profile` |
| `def` | `summarize_user_memories` |

#### `prompt_cache_audit.py`

Prompt cacheability auditing helpers for stable-prefix prompt design.

| Kind | Name |
|------|------|
| `def` | `_normalize_sections` |
| `def` | `_render_sections` |
| `def` | `audit_prompt_cacheability` |

#### `prompt_cache_middleware.py`

Bridge rendered prompt expectations to provider cache telemetry validation.

| Kind | Name |
|------|------|
| `def` | `_utc_now` |
| `def` | `_hash_text` |
| `def` | `_first_difference_index` |
| `def` | `_extract_cached_tokens` |
| `class` | `PromptCacheExpectation` |
| `class` | `PromptCacheMiddleware` |

#### `prompt_cache_stability_guard.py`

Preventive stable-prefix guard for cache-friendly prompt assembly.

| Kind | Name |
|------|------|
| `def` | `evaluate_prompt_stability` |

#### `prompt_registry.py`

Versioned prompt template registry with deployment labels.

| Kind | Name |
|------|------|
| `def` | `_utc_now` |
| `def` | `_normalize_variables` |
| `def` | `_stable_prefix_text` |
| `def` | `_stable_prefix_hash` |
| `def` | `_initial_stable_prefix_analysis` |
| `class` | `PromptVersion` |
| `class` | `PromptTemplateRecord` |
| `class` | `PromptRegistry` |

#### `provider_profiles.py`

Provider/model profile registry for cost and cache-aware optimization.

| Kind | Name |
|------|------|
| `class` | `ProviderProfile` |
| `def` | `get_provider_profile` |
| `def` | `list_provider_profiles` |

#### `provider_telemetry_normalizer.py`

Canonical cache telemetry normalization for observability surfaces.

| Kind | Name |
|------|------|
| `def` | `normalize_cache_telemetry` |

#### `proxy_cli.py`

Entry point for token-saver-proxy CLI command.

| Kind | Name |
|------|------|
| `def` | `main` |

#### `quality_gate.py`

Independent, sealed-fixture utility oracle for compression-quality gating (MF1).

| Kind | Name |
|------|------|
| `class` | `GradeResult` |
| `def` | `grade_answerability` |
| `def` | `grade_byte_identity` |
| `def` | `grade_source_order` |
| `def` | `grade_modulate_region_roundtrip` |
| `class` | `UtilityFixtureLike` |
| `class` | `CorpusFixtureReport` |
| `class` | `CorpusReport` |
| `def` | `evaluate_compressor` |
| `def` | `identity_compressor` |
| `def` | `empty_compressor` |
| `def` | `first_paragraph_compressor` |

#### `quality_predictor.py`

PoC quality predictor for semantic compression.

| Kind | Name |
|------|------|
| `class` | `QualityPredictor` |

#### `query_adaptive.py`

Query-adaptive compression ratios.

| Kind | Name |
|------|------|
| `def` | `compute_section_ratios` |

#### `rate_limiter.py`

Rate Limiting Infrastructure for Resource Protection

| Kind | Name |
|------|------|
| `class` | `RateLimiter` |
| `def` | `get_rate_limiter` |
| `def` | `configure_rate_limiter` |

#### `reliability.py`

Reliability Infrastructure for Production MCP Server

| Kind | Name |
|------|------|
| `class` | `TimeoutManager` |
| `class` | `CircuitBreaker` |
| `class` | `RetryPolicy` |

#### `reranker_gate.py`

Recall-gated cross-encoder rerank stage (#187, WORLD-CLASS #1).

| Kind | Name |
|------|------|
| `class` | `RerankScorer` |
| `class` | `RerankConfig` |
| `def` | `rerank_candidates` |

#### `resource_manager.py`

Resource Management for Semantic Modulator

| Kind | Name |
|------|------|
| `class` | `ResourceLimits` |
| `class` | `ResourceManager` |

#### `response_formatter.py`

Response formatter for MCP tool results.

| Kind | Name |
|------|------|
| `def` | `cache_stable_format` |
| `def` | `_serialized_size` |
| `def` | `_make_continuation_token` |
| `def` | `_is_more_compact_fidelity` |
| `class` | `ResponseFormatter` |

#### `savings_dashboard.py`

Cross-session savings dashboard for gotcontext.ai.

| Kind | Name |
|------|------|
| `class` | `AggregatedReport` |
| `class` | `SavingsDashboard` |
| `def` | `format_summary` |
| `def` | `format_daily` |
| `def` | `format_by_tool` |
| `def` | `format_cost` |
| `def` | `report_to_dict` |
| `def` | `report_to_csv` |
| `def` | `build_arg_parser` |
| `def` | `main` |

#### `savings_discover.py`

Discover missed token savings opportunities.

| Kind | Name |
|------|------|
| `class` | `SavingsOpportunity` |
| `class` | `DiscoveryReport` |
| `class` | `ContextAnalyzer` |
| `def` | `format_report` |

#### `savings_tracker.py`

Real-time token savings tracker for gotcontext.ai.

| Kind | Name |
|------|------|
| `class` | `SavingsEvent` |
| `class` | `SavingsReport` |
| `class` | `SavingsTracker` |

#### `scar_compressor.py`

SCAR-Inspired Learnable Semantic Compression

| Kind | Name |
|------|------|
| `class` | `CompressionStats` |
| `class` | `LearnableSemanticCompressor` |
| `class` | `SemanticAlignmentModule` |
| `class` | `SCAREnhancedCompressor` |

#### `search_compress_pipeline.py`

Search-then-compress pipeline: tensor-grep search -> semantic compression.

| Kind | Name |
|------|------|
| `class` | `SearchCompressResult` |
| `def` | `_glob_search` |
| `def` | `_count_tokens` |
| `def` | `search_then_compress` |

#### `segment_compression_cache.py`

Reusable cache for segment-level compression artifacts.

| Kind | Name |
|------|------|
| `def` | `_hash_value` |
| `class` | `SegmentCompressionCache` |

#### `semantic_chunking.py`

Semantic chunking for context ingestion.

| Kind | Name |
|------|------|
| `def` | `_word_token_count` |
| `def` | `detect_semantic_boundaries` |
| `def` | `chunk_by_semantics` |

#### `semantic_compressor.py`

Fidelity-Preserving Semantic Compressor

| Kind | Name |
|------|------|
| `def` | `_node_belongs_to_file` |
| `def` | `_gate_should_fuse_g` |
| `class` | `FidelityLevel` |
| `class` | `SemanticNode` |
| `class` | `SkeletonResponse` |
| `class` | `EvidenceResult` |
| `class` | `DiffReingestionResult` |
| `def` | `compute_adaptive_ratio` |
| `class` | `SemanticCompressor` |

#### `semantic_ssim.py`

Semantic SSIM (Structural Similarity Index for Semantic Content)

| Kind | Name |
|------|------|
| `class` | `SemanticSSIM` |
| `def` | `interpret_ssim_score` |

#### `server.py`

Semantic Modulator MCP Server

| Kind | Name |
|------|------|
| `class` | `SemanticModulatorServer` |
| `async def` | `async_main` |
| `def` | `main` |

#### `session_journal.py`

Session Journal (v0.13.0)

| Kind | Name |
|------|------|
| `class` | `JournalEvent` |
| `class` | `SessionSummary` |
| `class` | `SessionJournal` |

#### `setup_cli.py`

Guided setup CLI for Token Saver MCP onboarding.

| Kind | Name |
|------|------|
| `def` | `build_arg_parser` |
| `def` | `_selected_target` |
| `def` | `_apply_target` |
| `def` | `_remove_target` |
| `def` | `main` |

#### `structural_summary.py`

Structural summary generator inspired by repowise.

| Kind | Name |
|------|------|
| `class` | `StructuralSummary` |
| `def` | `generate_structural_summary` |
| `def` | `_detect_language` |
| `def` | `_summarize_python` |
| `def` | `_summarize_class` |
| `def` | `_function_signature` |
| `def` | `_annotation_str` |
| `def` | `_name_str` |
| `def` | `_summarize_generic` |

#### `structured_bundles.py`

Scoped registry for durable handoff bundles.

| Kind | Name |
|------|------|
| `def` | `_utc_now` |
| `class` | `HandoffBundleRecord` |
| `class` | `HandoffBundleRegistry` |

#### `structured_logging.py`

Structured logging with async context propagation and OpenTelemetry integration.

| Kind | Name |
|------|------|
| `def` | `_redact_context` |
| `class` | `JSONFormatter` |
| `class` | `HumanFormatter` |
| `class` | `StructuredLogger` |
| `def` | `configure_structlog` |
| `def` | `get_logger` |

#### `team_export.py`

Team dashboard data export.

| Kind | Name |
|------|------|
| `class` | `TeamMemberStats` |
| `class` | `TeamReport` |
| `class` | `TeamExporter` |

#### `tee_recovery.py`

Tee/Recovery system for preserving original content before compression.

| Kind | Name |
|------|------|
| `class` | `TeeEntry` |
| `class` | `TeeStore` |
| `def` | `get_default_tee_dir` |
| `def` | `create_tee_store` |

#### `temporal_graph.py`

Temporal fact graph for lifecycle-aware retrieval and invalidation.

| Kind | Name |
|------|------|
| `def` | `coerce_timestamp` |
| `def` | `format_timestamp` |
| `class` | `TemporalFactVersion` |
| `class` | `TemporalEvent` |
| `class` | `TemporalGraph` |

#### `tensor_grep_integration.py`

Optional tensor-grep integration for AST-aware code compression and search.

| Kind | Name |
|------|------|
| `class` | `RepoMapResult` |
| `class` | `CodeSearchResult` |
| `class` | `ASTSearchResult` |
| `class` | `ContextRenderResult` |
| `class` | `ScanFinding` |
| `class` | `ScanResult` |
| `def` | `is_available` |
| `def` | `get_repo_map` |
| `def` | `code_search` |
| `def` | `ast_search` |
| `def` | `get_context_render` |
| `def` | `scan_ruleset` |

#### `token_estimation.py`

Token estimation utilities for Semantic Modulator.

| Kind | Name |
|------|------|
| `def` | `_load_tiktoken_encoder` |
| `class` | `TokenEstimator` |

#### `token_refiner.py`

Token-level skeleton refinement inspired by LLMLingua.

| Kind | Name |
|------|------|
| `class` | `RefinedResult` |
| `class` | `MIGConfig` |
| `class` | `MIGScorer` |
| `class` | `TokenRefiner` |

#### `token_threshold.py`

Token threshold monitoring for proactive compression suggestions.

| Kind | Name |
|------|------|
| `class` | `ContextBudgetResult` |
| `def` | `check_context_budget` |

#### `toon_serializer.py`

TOON Serializer for Token Saver 5000

| Kind | Name |
|------|------|
| `class` | `OutputFormat` |
| `class` | `TOONSerializer` |
| `def` | `format_response` |
| `def` | `estimate_token_savings` |

#### `training_utils.py`

Training Utilities for SCAR Modules

| Kind | Name |
|------|------|
| `class` | `TrainingConfig` |
| `class` | `SemanticCompressionDataset` |
| `class` | `AlignmentDataset` |
| `class` | `SCARTrainer` |
| `def` | `create_synthetic_training_data` |

#### `transcript_extractor.py`

Transcript-to-memory extraction pipeline.

| Kind | Name |
|------|------|
| `class` | `ExtractionResult` |
| `def` | `_split_sentences` |
| `def` | `_has_signal` |
| `def` | `_strip_role_prefix` |
| `def` | `extract_insights` |
| `def` | `ingest_transcript` |

#### `types.py`

Type definitions for Token Saver 5000 MCP Server.

| Kind | Name |
|------|------|
| `class` | `HandlerContext` |
| `class` | `ToolArguments` |

#### `url_fetcher.py`

SSRF-hardened URL fetcher for ingest_context file_url parameter.

| Kind | Name |
|------|------|
| `class` | `URLFetchError` |
| `def` | `_normalize_ip_address` |
| `def` | `_is_private_ip` |
| `def` | `_is_blocked_hostname` |
| `def` | `_resolve_host` |
| `def` | `_check_resolved_ips` |
| `def` | `_resolve_and_check_host` |
| `def` | `_check_dns_rebinding` |
| `async def` | `fetch_url` |

#### `validation_hooks.py`

Input validation hooks for MCP tool calls.

| Kind | Name |
|------|------|
| `def` | `_register` |
| `def` | `validate_tool_input` |
| `def` | `_validate_search` |
| `def` | `_validate_ingest` |
| `def` | `_validate_modulate` |
| `def` | `_validate_delete` |
| `def` | `_validate_batch_ingest` |

#### `version_manager.py`

Version Manager for Semantic Modulator

| Kind | Name |
|------|------|
| `class` | `DocumentVersion` |
| `class` | `VersionManager` |

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
