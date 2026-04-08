# Folder guide: `scripts/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../CLAUDE.md).

## Contents

### Python modules

#### `benchmark.py`

Performance Benchmarking Script for Token Saver 5000

| Kind | Name |
|------|------|
| `class` | `BenchmarkResult` |
| `class` | `PerformanceBenchmark` |
| `def` | `benchmark_document_ingestion` |
| `def` | `benchmark_search_performance` |
| `def` | `benchmark_fidelity_modulation` |
| `def` | `benchmark_compression_ratios` |
| `def` | `benchmark_scar_features` |
| `def` | `main` |

#### `benchmark_cujs.py`

Critical User Journey Baselines for gotcontext.ai

| Kind | Name |
|------|------|
| `class` | `StepResult` |
| `class` | `CUJResult` |
| `class` | `CUJBaseline` |
| `def` | `_now_ms` |
| `def` | `_elapsed_ms` |
| `def` | `_savings_pct` |
| `def` | `_count_tokens` |
| `def` | `run_cuj_1_solo_dev_codebase` |
| `def` | `run_cuj_2_long_document` |
| `def` | `run_cuj_3_cli_output_filtering` |
| `def` | `run_cuj_4_query_focused_search` |
| `def` | `run_cuj_5_session_recovery` |
| `def` | `run_cuj_6_savings_report` |
| `def` | `run_cuj_7_schema_compression` |
| `def` | `run_cuj_8_code_compression` |
| `def` | `run_cuj_9_dialogue_memory` |
| `def` | `run_cuj_10_budget_governance` |
| `def` | `run_cuj_11_tee_recovery` |
| `def` | `run_cuj_12_team_dashboard` |
| `def` | `run_cuj_13_knowledge_compilation` |
| `def` | `run_all_cujs` |
| `def` | `_print_table` |
| `def` | `main` |

#### `benchmark_token_savings.py`

Token Saver 5000 Benchmark: Compare token usage with and without compression.

| Kind | Name |
|------|------|
| `def` | `_print_search_compress_table` |
| `def` | `main` |

#### `check_claude_folder_guides_sync.py`

Regenerate per-folder claude.md guides and fail if git working tree differs.

| Kind | Name |
|------|------|
| `def` | `iter_guide_paths` |
| `def` | `main` |

#### `check_setup.py`

Setup Verification Script for Token Saver 5000

| Kind | Name |
|------|------|
| `def` | `check_python_version` |
| `def` | `check_dependencies` |
| `def` | `check_imports` |
| `def` | `check_embedding_model` |
| `def` | `quick_smoke_test` |
| `def` | `main` |

#### `generate_claude_folder_guides.py`

Regenerate per-folder claude.md guides and print a manifest for CLAUDE.md.

| Kind | Name |
|------|------|
| `def` | `iter_doc_dirs` |
| `def` | `skip_path` |
| `def` | `module_summary` |
| `def` | `extract_symbols` |
| `def` | `analyze_py` |
| `def` | `list_non_py` |
| `def` | `rel_posix` |
| `def` | `build_folder_markdown` |
| `def` | `main` |

#### `migrate_pickle.py`

Migrate Legacy Pickle Files to Safe JSON + NumPy Format

| Kind | Name |
|------|------|
| `def` | `find_legacy_pickle_files` |
| `def` | `load_pickle_safely` |
| `def` | `save_as_safe_format` |
| `def` | `_save_graph_data` |
| `def` | `_save_chunks_data` |
| `def` | `_save_afm_data` |
| `def` | `determine_data_type` |
| `def` | `migrate_file` |
| `def` | `main` |

#### `profile_memory.py`

Memory Profiling Script for Token Saver 5000

| Kind | Name |
|------|------|
| `def` | `format_memory` |
| `def` | `print_snapshot_diff` |
| `def` | `test_singleton_memory_leak` |
| `def` | `delete_document_from_compressor` |
| `def` | `test_document_ingestion_leak` |
| `def` | `test_ace_context_persistence` |
| `def` | `test_file_sync_metadata` |
| `def` | `test_version_history_accumulation` |
| `def` | `run_full_profile` |
| `def` | `main` |

#### `quickstart.py`

Token Saver 5000 - Quickstart Script

| Kind | Name |
|------|------|
| `def` | `print_header` |
| `def` | `print_step` |
| `def` | `check_python_version` |
| `def` | `install_dependencies` |
| `def` | `download_embedding_model` |
| `def` | `run_tests` |
| `def` | `run_demo` |
| `def` | `print_next_steps` |
| `def` | `main` |

#### `test_simulation.py`

Channel Pressure Test - Empirical Validation

| Kind | Name |
|------|------|
| `class` | `SemanticGraph` |
| `def` | `generate_synthetic_document` |
| `def` | `calculate_semantic_ssim` |
| `def` | `simulate_compression` |
| `def` | `run_channel_stress_test` |

#### `token_saver_proxy.py`

Token Saver MCP Proxy: transparent compression for any MCP server.

| Kind | Name |
|------|------|
| `def` | `build_arg_parser` |
| `def` | `_print_dry_run` |
| `def` | `main` |

### Other files

- `install_mcp.sh`
- `README.md`

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
