# Changelog

All notable changes to Token Saver 5000.

## [Unreleased]

### Added (v0.12.0 - Enterprise Launch & gotcontext.ai)

#### Phase 1: Launch Blockers
- **Multi-agent MCP setup** (`src/mcp_install.py`): one-command install for 8 AI agents
  (Claude Desktop, Claude Code, Cursor, Windsurf, Cline, VS Code Copilot, Codex, Gemini CLI).
  `token-saver-install-mcp --agent cursor` or `--doctor-all` for status. 25 tests.
- **Savings dashboard CLI** (`src/savings_dashboard.py`): cross-session savings analysis.
  Entry point: `token-saver-stats` / `gotcontext-stats`. Modes: `--daily`, `--weekly`,
  `--by-tool`, `--cost`, `--json`, `--csv`. 18 tests.
- **Proxy savings tracking** (`src/proxy/proxy_server.py`): per-call metrics recording,
  savings percentage, token estimates, `--verbose` per-call logging, enhanced exit summary.
  8 tests.

#### Phase 2: Competitive Parity (RTK-Inspired)
- **CLI output optimizer enhancements** (`src/cli_output_optimizer.py`): expanded from 10
  to **11 command-specific strategies** (added `docker_compact`). Enhanced git_diff (per-file
  previews), test_failure_focus (structured pytest parser), lint_group (JSON grouping),
  json_structure (recursive schema). 51 tests total.
- **Tee/recovery system** (`src/tee_recovery.py`): preserves original content before
  compression for later retrieval. LRU-evicting store with 3 modes (failures/always/never),
  JSON persistence. 3 new MCP tools: `get_original_output`, `list_tee_entries`,
  `tee_store_stats`. 42 tests.
- **Missed savings discovery** (`src/savings_discover.py`): scans directories/files to
  find compression opportunities. Per-extension compression ratio estimates, ranked
  opportunity report. New MCP tool: `discover_savings`. 27 tests.
- **Session continuity validation**: 8 tests proving skeleton survival across JSON
  round-trips, truncation, and re-ingestion.

#### Phase 3: Differentiation
- **Custom filter rules DSL** (`src/filter_rules.py`): user-defined output filtering via
  `.gotcontext.toml` TOML config. 8-stage pipeline (strip_ansi → strip_lines → keep_lines
  → truncate → head → tail → max → on_empty). Project + user-global precedence, inline
  test verification. 26 tests.
- **ROI calculator** MCP tool (`calculate_roi`): monthly cost comparison with/without
  compression, Pro plan ROI, net savings, payback analysis. 20+ model pricing database.
  6 tests.
- **Token budget monitoring** (`src/budget_monitor.py`): configurable per-session, daily,
  and monthly token budget limits. Alert thresholds (ok/info/warning/critical), usage
  projections. MCP tool: `check_budget`. Env vars: `TOKEN_BUDGET_SESSION`,
  `TOKEN_BUDGET_DAILY`, `TOKEN_BUDGET_MONTHLY`. 14 tests.
- **Team dashboard export** (`src/team_export.py`): aggregated team savings in JSON, CSV,
  and Prometheus exposition formats. MCP tool: `export_team_data`. 12 tests.

#### Phase 4: Polish
- **GTM benchmark reproduction suite** (`tests/test_gtm_benchmarks.py`): 26 tests
  validating all go-to-market claims (compression ratios, CLI strategies, schema
  compression, ROI calculations, tool count, feature existence).
- **Documentation overhaul**: README.md, CLAUDE.md, copilot-instructions.md updated
  with all new features, tool counts, and entry points.
- **gotcontext entry points**: `gotcontext-mcp`, `gotcontext-setup`, `gotcontext-stats`,
  `gotcontext-proxy` alongside legacy `token-saver-*` aliases.
- **Python 3.14 support**: fixed `requires-python` upper bound from `<3.14` to `<3.15`.

- **Total MCP tools**: 121 (was 112).
- **Full test suite**: 3,409+ tests across 90+ test files.
- **Benchmark results**: medium corpus 4.97x, large corpus 7.80x, avg 83.5% savings.
- **CUJ benchmark suite**: expanded from 6 to **12 journeys** covering schema compression,
  code-aware compression, AFM dialogue memory, budget governance, tee/recovery, and
  team dashboard export. All 12 pass. Aggregate: 4.49M input → 591K output (86.8% savings).

### Fixed
- **ROI calculator** (`token_optimization_handlers.py`): safe dict access for model pricing
  rates — prevents `KeyError` on unknown models.
- **Team export handler** (`token_optimization_handlers.py`): type guard on member items —
  non-dict entries in `members` list are now skipped instead of crashing.
- **Filter rules regex** (`filter_rules.py`): catch `re.error` on user-supplied regex
  patterns in strip/keep pipeline stages — invalid patterns skip the stage gracefully.
- **Prometheus label injection** (`team_export.py`): escape backslashes, quotes, and
  newlines in `user_id` labels to prevent metric format corruption.
- **Filter rules dead code** (`cli_output_optimizer.py`): `FilterRuleEngine` is now wired
  as a post-filter stage in `CLIOutputOptimizer` — user-defined `.gotcontext.toml` rules
  are applied after the built-in strategy.
- **Version mismatch** (`src/__init__.py`): `__version__` now derived from package metadata
  via `importlib.metadata.version()` — pyproject.toml is the single source of truth.
  README version/Python range updated to match (0.11.0, 3.10-3.14).
- **Logging side effect** (`afm.py`): removed `logging.basicConfig()` call at import time
  that could override the application's structured logging configuration.
- **Pytest local ergonomics** (`pyproject.toml`): removed `--cov*` flags from default
  `addopts` — focused test runs no longer fail due to global coverage threshold. Use
  `--cov=src --cov-fail-under=70` explicitly for CI coverage.
- **Training utils** (`training_utils.py`): fixed bare `from scar_compressor import` to
  package-relative `from src.scar_compressor import`, and added `weights_only=True` to
  `torch.load()` for safer checkpoint loading.
- **Tool error envelope** (`router_binding.py`): tool failures now return structured JSON
  `{"error": "...", "message": "...", "tool": "..."}` instead of plain `"Error: ..."` text.
- **Contract validation DRY** (app layer): extracted shared `contract_key_mismatch_message`
  and `validate_contract_keys` into `contract_validation.py`, replacing 8 identical copies
  across the app-layer services.
- **Portable MCP config** (`mcp_install.py`): added doc comment explaining why portable
  mode uses generic `"python"` command and when to prefer desktop install instead.

### Added (v0.11.0 - Cross-Platform Token Optimization)
- **Token estimation module** (`src/token_estimation.py`): dual-mode estimation with tiktoken
  (accurate), `len//4` fast, `len//2` JSON-density, Gemini-compatible (0.25/ASCII, 1.3/non-ASCII),
  and raw byte count. Exposed as `estimate_tokens` MCP tool.
- **Response formatter** (`src/response_formatter.py`): size-aware MCP tool response formatting.
  Enforces soft limit (40K chars) and hard limit (49K chars) to stay within Claude Code's 50K
  per-tool cap. Three truncation strategies: `paginate` (default, continuation token),
  `proportional` (20% head + 80% tail, matches Gemini CLI), `head` (first N chars).
  Optional `_header` field (< 200 chars) survives any truncation strategy.
- **Client configuration** (`src/client_config.py`): model-aware compression tuning.
  Maps model IDs to context window sizes and compression trigger ratios. Auto-tunes skeleton
  ratio based on window size and client compression aggressiveness. Supports Claude (200K/1M,
  triggers at 93%), Gemini (1M, triggers at 50%), GPT (128K, no auto-compress), and explicit
  overrides. Session-scoped via `SessionConfigStore`.
- **Compression profiles** (`src/compression_profiles.py`): named presets
  (minimal/summary/balanced/detailed/full) bundling skeleton_ratio, fidelity, and chunk_size.
  Session-scoped via `ProfileManager`. Urgency support: `compact` (cap ratio 0.15) and
  `emergency` (force ratio 0.05, ABSTRACT fidelity) for tight context budgets.
- **4 new MCP tools**: `estimate_tokens`, `configure_for_client`, `set_compression_profile`,
  `get_compression_profile` -- all routed via `token_optimization_handlers.py`.
- **Model database**: `KNOWN_MODEL_CONTEXT_WINDOWS` expanded with 5 Gemini models
  (2.5-pro/flash, 3.1-pro/flash/flash-lite at 1,048,576 tokens).
  New `KNOWN_MODEL_COMPRESSION_TRIGGERS` dict for per-model compression aggressiveness.
- **Schema stability**: tool schemas now sorted alphabetically for prompt cache stability.
  Dynamic content (version strings, temporal language) removed from 2 tool descriptions.
- **156 new tests** across 6 test files: `test_response_formatter.py` (19),
  `test_token_estimation.py` (23), `test_client_config.py` (20), `test_compression_profiles.py`
  (22), `test_schema_stability.py` (11), `test_gemini_enhancements.py` (56),
  plus tool count updates in `test_mcp_routing.py` and `test_experimental_handlers.py`.
- **Total MCP tools**: 109 (was 108).
- **Token savings tracker** (`src/savings_tracker.py`): real-time ROI tracking for every
  compression operation. Computes tokens saved, dollar savings (model-aware), compression
  ratios, monthly projections, ROI vs Pro plan ($29/mo), and breakeven analysis.
  Per-tool breakdown shows which operations save the most. Persists to SQLite via
  SessionJournal. New MCP tools: `get_savings_report`, `get_savings_inline`. 26 tests.
- **Total MCP tools**: 111 (was 109).
- **OpenCode CLI support** (`docs/opencode-token-optimization-enhancements.md`): full analysis
  of OpenCode's 12+ provider, 50+ model architecture. Added 16 models to KNOWN_MODEL_CONTEXT_WINDOWS
  (GPT-4.1 family, Groq Llama-4, Grok-3, DeepSeek R1, etc.) with pricing and compression triggers.
  OpenCode benchmark provider added to harness. 52 new tests.
- **Cache strategy advisor** (`src/cache_strategy_advisor.py`): provider-aware caching recommendations.
  Returns optimal strategy per model: Anthropic (explicit, 90% discount), OpenAI (automatic, 50%),
  Gemini 2.5+ (implicit, 90%), Groq/XAI/Local (none/limited). New MCP tool: `advise_cache_strategy`.
- **Provider detection extended**: `_detect_provider()` now handles groq, xai, local, ollama prefixes.
- **Total MCP tools**: 112 (was 111).
- **Full test suite**: 3,045+ tests across 90+ test files.
- **CLI output optimizer** (`src/cli_output_optimizer.py`): RTK-inspired command-aware
  filtering for CLI tool output. Auto-detects 10 command types (git diff, pytest, npm install,
  lint, JSON, logs, progress bars, etc.) and applies optimal filtering strategy. Strips ANSI
  codes, extracts stats, groups errors, focuses on failures, deduplicates logs.
  Integrated into proxy pipeline as Stage 0 (before TokenRefiner). Optional RTK binary
  fallback when installed. New MCP tool: `filter_cli_output`. 51 tests.
- **Transparent MCP proxy** (`src/proxy/`, `scripts/token_saver_proxy.py`): drop-in proxy
  that wraps ANY MCP server and compresses tool responses automatically. Pipeline: TokenRefiner
  -> MetaTokenCompressor -> ResponseFormatter. Separate entry point: `token-saver-proxy`.
  Optionally replaces N upstream tools with 3 meta-tools (`--schema-compression`).
  49 proxy tests.
- **Session continuity** (`src/session_journal.py`): SQLite-backed event journal that survives
  conversation compaction. Records ingestions, configs, profiles, tool calls. New MCP tool:
  `recover_session` returns compact summary of all prior work. 16 tests.
- **Tensor-grep integration** (`src/tensor_grep_integration.py`): optional AST-aware code
  compression via `tg map` (repo structure) and `tg search` (trigram index). New MCP tools:
  `compress_codebase`, `search_code`. Graceful fallback when tg not installed. 14 tests.
- **Meta-token compression** (`src/meta_tokens.py`): lossless LZ77-inspired subsequence
  replacement (arXiv 2506.00307). Finds repeated token n-grams, replaces with §N symbols +
  prepended dictionary. Fully reversible via `decompress()`. New MCP tool: `compress_meta_tokens`.
  26 tests.
- **COMI MIG scoring** (`src/token_refiner.py`): Marginal Information Gain token scoring
  (arXiv 2602.01719, ICLR 2026). Query-aware scoring that balances relevance vs redundancy.
  New `MIGScorer` class, `scoring_mode="mig"` option on `TokenRefiner`. 17 tests.
- **PoC quality predictor** (`src/quality_predictor.py`): performance-oriented compression
  (arXiv 2603.19733). Specify quality floor instead of ratio. Auto-selects most aggressive
  profile meeting the floor. New MCP tool: `recommend_compression`. 27 tests.
- **Codex CLI support**: Added `gpt-5.1-codex` and `codex-mini` to model database
  (200K context, 0.80 compression trigger). Codex pricing in `pricing.py`.
  Codex JSONL output parser in `providers.py`. 30 new Codex tests.
- **Cross-platform benchmark harness** (`src/cli_benchmark/`, `scripts/benchmark_token_savings.py`):
  Measures real API token usage with vs without Token Saver compression across Claude Code,
  Gemini CLI, and Codex. Two modes: skill-based (pre-compression) and MCP-based (live
  integration). Ships with 3 corpus files (small/medium/large) and 62 benchmark tests.
  Supports `--dry-run`, `--providers`, `--sizes`, `--output` JSON results + ASCII table.
- **Proven benchmark results** (large corpus, 2,206 lines, 16K tokens, with token refinement):
  - Document compression: 13.0x (16,461 -> 1,269 tokens), up from 5.9x before refinement
  - Claude Code: 26.4% input savings, 3.5-50% cost savings (varies with cache)
  - Codex: 38.2% input savings, 36.8% cost savings
  - Gemini CLI: 55.7% input savings, 53.8% cost savings
  - All savings measured using total content tokens (cache-independent) for stability
- **Gemini benchmark fix:** switched from `input` (billed, cache-dependent) to `prompt`
  (total content, cache-independent) for stable savings comparison across runs
- **Findings documents**: `docs/claude-code-token-optimization-enhancements.md`,
  `docs/gemini-cli-token-optimization-enhancements.md`,
  `docs/codex-cli-token-optimization-enhancements.md`
- **Cache-stable response ordering** (`src/response_formatter.py`): provider-aware key
  ordering that maximizes prompt cache hits. Stable fields (status, file_id) first for
  Claude/Gemini prefix caching; mirrored at tail for Codex middle-truncation resilience.
- **Provider format hints** (`src/client_config.py`): auto-detects provider from model ID
  (anthropic/google/openai) and sets optimal output format, truncation strategy, and
  cache ordering per provider. Claude gets TOON+paginate, Gemini gets JSON+proportional,
  Codex gets TOON+head.
- **Token-level skeleton refinement** (`src/token_refiner.py`): LLMLingua-inspired
  post-processing that removes low-importance tokens (articles, fillers, hedges) from
  compressed skeletons while preserving semantic anchors (numbers, code identifiers,
  URLs, acronyms). Achieves 20-40% additional reduction on top of semantic compression.
  Pure Python, no external dependencies. 45 tests.
- **TurboQuant-inspired embedding quantization** (`src/embedding_quantizer.py`): reduces
  384-dim float32 embeddings to 96-dim int8 (13x memory reduction) using random orthogonal
  rotation (PolarQuant) + int8 quantization + 1-bit residual error correction (QJL).
  >0.99 cosine fidelity in reduced space. Pure numpy. 22 tests.

### Changed
- Added a local extractive compression baseline plus reusable segment-level compression cache for lower-latency token trimming experiments.
- Added stable-prefix-preserving history compaction utilities for conversation workflows.
- Extended `optimize_for_model` and provider profiles with cache-threshold guidance and deterministic `prompt_cache_key` helper output.
- Extended benchmark harness support with semantic-vs-extractive method comparison metrics.
- Added provider+harness cache compatibility assessment via `assess_cache_compatibility`, including Gemini CLI stats visibility checks and Codex/OpenAI routing-stickiness guidance.
- Extended model-aware optimization and provider profiles with OpenAI/Codex `prompt_cache_key` routing guidance and Codex-family aliases.
- Extended cache telemetry summarization to understand Gemini CLI-style camelCase stats exports (`inputTokens`, `outputTokens`, `cachedTokens`).
- Added portable skill external input adapters (`raw_json`, `langchain_json`, `llamaindex_json`, `auto`) across profile/compress/evidence/workflow scripts with JSON input support and adapter diagnostics in output payloads.
- Added explainability payloads to portable skill segment outputs with score decomposition (`relevance`, `position`, `final`) and `selection_reason` (`forced_by_tag`, `local_rate_policy`, `top_rank`, `not_selected`).
- Added structured segment controls to skill compression via `<llmlingua ...>` tags, including `compress=False` preserve blocks and per-block `rate` overrides that can supersede global skeleton ratios for tagged sections.
- Refactored skill compression engine to use a composable pipeline execution model (`split` -> `score` -> `select`) while preserving output contracts.
- Extended benchmark harness quality reporting with query-aware overlap metrics (`precision_at_k`, `recall_at_k`, `f1_at_k`) and aggregate quality averages/counts in benchmark summaries.
- Hardened ingest persistence path to safely await coroutine-returning persistence hooks (`save_document`, `save_file_sync_metadata`) in `src/handlers/compression_handlers.py`.
- Extended skill TOON-vs-JSON benchmark output contract to include auto-format selection diagnostics and guard expectations for uniform vs mixed payloads.
- Hardened remaining app-layer service boundaries with explicit request-envelope contract validation in `lifecycle_service`, `progress_service`, `persistence_orchestration_service`, `tool_profile_service`, and `router_binding`.
- Added canonical contract key mismatch messaging and fail-fast keyset drift checks across those app services while preserving runtime behavior.
- Normalized node identity parsing in `src/node_identity.py` to treat only `"_n<digits>"` as text-node suffixes.
- Updated `validate_node_ids` in `src/handlers/compression_handlers.py` to use shared node-ID parsing and file-id matching logic.
- `src/handlers/mcp_core.py` now supports profile-aware tool listing and routing gates.
- `src/server.py` now reads `MCP_TOOL_PROFILE` and applies profile-aware `list_tools`/routing behavior.
- `src/server.py` now logs startup MCP profile diagnostics (`mcp_tool_profile_active`) with enabled tool count.
- `check_environment` now reports runtime tool profile diagnostics (`profile`, `enabled_tool_count`, `enabled_tools`).
- `tool_help` documentation for `check_environment` now includes tool profile diagnostics guidance.
- `tool_help(check_environment)` now publishes explicit `output_fields`, with tests enforcing alignment to runtime profile diagnostics keys.
- Added canonical `check_environment` output-field utility in `resource_handlers`, and wired `tool_help` to reuse it to reduce docs/runtime drift.
- Added canonical `search_semantic` output-field utility in `compression_handlers`, and wired `tool_help(search_semantic)` to reuse it with contract tests.
- Added canonical `read_skeleton` output-field utility in `compression_handlers`, and wired `tool_help(read_skeleton)` to reuse it with contract tests.
- Added canonical `ingest_context` output-field utility in `compression_handlers`, and wired `tool_help(ingest_context)` to reuse it with estimate-field contract tests.
- Added canonical `recommend_fidelity` output-field utility in `compression_handlers`, and wired `tool_help(recommend_fidelity)` to reuse it with contract tests.
- Fixed `tool_help(modulate_region)` parameter naming drift to use `fidelity_level` consistently with handler and MCP schema.
- Cleared repo-wide lint/format debt across audited graph, verifier, rewards, evidence, synthesis, and related tests so `ruff` and `black --check` pass on `src/tests/scripts`.
- Fixed `AuditedSemanticGraph.get_node_with_history` to include the node creation bundle via provenance-linked `creation_bundle_id`.
- Fixed compression postcondition contracts to allow empty `node_map` in `RAW` mode.
- Added Phase 1 enterprise namespace scaffolding under `src/semantic_modulator/` (`app/bootstrap`, `api/mcp/registry`, `api/mcp/router`) as compatibility facades over existing server/router modules.
- `src/server.py` now consumes MCP registry/router through `src.semantic_modulator.api.mcp` facades as the canonical import path (behavior unchanged).
- Added app-layer MCP tooling gateway (`src/semantic_modulator/app/tooling.py`) and moved profile resolution/listing/routing orchestration behind it to further thin `src/server.py`.
- Added app-layer context service (`src/semantic_modulator/app/context_service.py`) and moved handler-context assembly + validation logic behind it while preserving server method compatibility.
- Added app-layer lifecycle service (`src/semantic_modulator/app/lifecycle_service.py`) and moved startup/shutdown sequencing behind it while preserving server lifecycle behavior.
- Added app-layer progress rendering service (`src/semantic_modulator/app/progress_service.py`) and moved progress-bar formatting behind it while preserving server helper compatibility.
- Added app-layer persistence orchestration service (`src/semantic_modulator/app/persistence_orchestration_service.py`) and moved persisted document + file-sync load/save flows behind it while preserving server wrapper compatibility.
- Added app-layer tool profile bootstrap service (`src/semantic_modulator/app/tool_profile_service.py`) and moved profile fallback + startup diagnostics logging behind it while preserving server constructor behavior.
- Added app-layer server factory service (`src/semantic_modulator/app/server_factory_service.py`) and moved server composition wiring behind it while preserving constructor-observable behavior.
- Expanded app bootstrap (`src/semantic_modulator/app/bootstrap.py`) to own `async_main`/`main` runtime entrypoints, with `src/server.py` delegating through thin wrappers.
- Added app-layer server service adapter (`src/semantic_modulator/app/server_service_adapter.py`) and routed server helper wrappers (`_load_*`, `_build_context`, `_validate_*`, `_create_progress_bar`) through it to further thin protocol wiring.
- Extracted `ACEContextManager` to `src/semantic_modulator/app/ace_context_manager.py` and kept `src/server.py` re-export compatibility to reduce server module surface area.
- Added app-layer router binding helper (`src/semantic_modulator/app/router_binding.py`) and routed `_setup_handlers` through it, further thinning server protocol wiring.
- Added app-layer runtime execution service (`src/semantic_modulator/app/runtime_service.py`) and routed `SemanticModulatorServer.run()` through it as a thin delegation wrapper.
- Synchronized `skills/token-saver-context-compression/` with the new self-contained `.claude` skill package, including TOON/JSON/auto output routing and local benchmark guard scripts.
- Centralized server-factory contract key sources (`BUILD_DEFAULT_REQUEST_KEYS`, `FACTORY_VALIDATION_RESULT_KEYS`, `DEFAULT_BUILD_INPUTS_KEYS`, `BUILD_REQUEST_KEYS`, `BUILD_KWARGS_KEYS`) and reused them across validator paths.
- Unified factory contract key-mismatch messaging via `contract_key_mismatch_message(...)`/`validate_contract_keys(...)`, and hardened default-build envelope validation with nested checks and subclass-dispatch-safe validator routing.
- Updated remaining factory contract validators to class-dispatch (`validate_default_class_map`, `validate_build_kwargs_map`, `validate_build_default_request_map`, `validate_factory_validation_result_map`) so derived factories can override shared validation policy consistently.
- Hardened `ServerContextService` with canonical context-key contract validation (`CONTEXT_MAP_KEYS`, `validate_context_map(...)`) and unified key-mismatch messaging; `build_context(...)` now validates output envelope before returning.
- Hardened `RuntimeService` with contract-validated run envelope (`RunRequest`, `validate_run_request_map(...)`) and class-dispatch-safe validator routing in `run(...)`.
- Hardened `MCPToolingGateway` with contract-validated profile state envelope (`ProfileState`, `validate_profile_state_map(...)`) and centralized `set_profile_state(...)` orchestration used by profile resolution and listing paths.

### Added
- Added portable skill adapter contract tests in `tests/test_skill_external_adapters.py` and CLI integration coverage for `compress_context.py --input-adapter langchain_json` in `tests/test_skill_scripts.py`.
- Added portable-skill explainability contract coverage in `tests/test_skill_scripts.py` for both `compress_context.py` and `run_skill_workflow.py`.
- Added structured-control tests in `tests/test_skill_structured_controls.py` covering preserve behavior, rate override behavior, and invalid-marker validation errors.
- Added skill pipeline primitives in `skills/token-saver-context-compression/scripts/_pipeline.py` with named-stage sync/async execution and stage-local failure context.
- Added pipeline contract tests in `tests/test_skill_pipeline.py` for stage ordering, mixed sync/async execution, and error attribution.
- Added competitor analysis report `docs/research/COMPETITOR_CODEBASE_COMPARISON_2026-02-15.md` and TDD execution roadmap `docs/guides/TDD_OUTPERFORM_PLAN_2026-02-15.md` for outperform strategy planning.
- Added benchmark harness contracts for quality metric fields in `tests/test_benchmark_harness.py`.
- Added skill portability contract tests to ensure self-contained skill scripts avoid project-root imports and path mutation bootstraps.
- Added benchmark guard contract tests for TOON/JSON auto-format selection behavior and benchmark output schema.
- Added ingest handler async-persistence regression test to enforce awaiting coroutine-based file-sync metadata save hooks.
- Added TDD execution specification `docs/guides/TDD_EXECUTION_SPEC_2026-02-15.md` with research-backed checkpoints, gates, and slice-level acceptance criteria.
- Added app contract tests for lifecycle/progress/persistence/tool-profile/router-binding request-envelope schemas and validation failures.
- Added targeted node identity tests in `tests/test_node_identity.py`.
- Added a validation regression test for non-index `_n` segments in `tests/test_compression_handlers.py`.
- Added MCP tool profile support (`full`, `core_stable`) for gradual surface simplification.
- Added profile contract tests in `tests/test_tool_profiles.py` and server profile env tests in `tests/test_server_unit.py`.
- Added dedicated CI workflow `.github/workflows/mcp-profile-guard.yml` for MCP profile regression checks.
- Added enterprise layout contract tests in `tests/test_enterprise_layout.py` to lock namespace/bootstrap/router wrapper behavior.
- Added enterprise import-path contract coverage to ensure `SemanticModulatorServer` initializes tool setup via the new facade module.
- Added gateway contract tests in `tests/test_mcp_tooling_gateway.py` for profile fallback/state and routing delegation.
- Added context service contract tests in `tests/test_context_service.py` and kept server validation/context tests green via delegation wrappers.
- Added lifecycle service contract tests in `tests/test_lifecycle_service.py` and kept server lifecycle/load/save tests green via delegation wrappers.
- Added progress service contract tests in `tests/test_progress_service.py` and kept server progress helper tests green via delegation wrapper.
- Added persistence orchestration service contract tests in `tests/test_persistence_orchestration_service.py` and kept server load/save lifecycle tests green via delegation wrappers.
- Added tool profile service contract tests in `tests/test_tool_profile_service.py` and kept server profile initialization tests green via delegation wiring.
- Added server factory service contract tests in `tests/test_server_factory_service.py` and kept server initialization/lifecycle/profile regressions green after composition extraction.
- Added bootstrap contract tests in `tests/test_bootstrap.py` for server creation and runtime-entrypoint delegation semantics.
- Added server service adapter contract tests in `tests/test_server_service_adapter.py` and kept helper-wrapper regression tests green via adapter delegation.
- Added ACE context manager module contract tests in `tests/test_ace_context_manager_module.py` to enforce module export and server re-export compatibility.
- Added router binding contract tests in `tests/test_router_binding.py` to enforce handler registration and result/error wrapping semantics.
- Added runtime service and wrapper compatibility tests in `tests/test_runtime_service.py` and `tests/test_server_runtime_wrapper.py` to enforce stdio execution + delegation semantics.

- Updated factory composition wiring so ServerFactoryService.build(...) now constructs and returns runtime_service plus service_adapter, removing duplicate constructor wiring from src/server.py.
- Added constructor wiring contract test tests/test_server_wiring_contract.py to lock factory ownership of runtime + adapter dependencies.

- Added ServerFactoryService.build_default(...) to centralize production class wiring while preserving the existing explicit uild(...) factory contract.
- src/server.py now uses uild_default(...) with class_overrides from module-level aliases to preserve unit-test patch compatibility while reducing constructor wiring verbosity.
- Added a default-factory delegation contract test in 	ests/test_server_factory_service.py.

- Added app helper module src/semantic_modulator/app/server_aliases.py to centralize the server class-alias override contract.
- src/server.py now builds factory override maps via uild_server_class_overrides(globals()), preserving patch-compatible module aliases while reducing constructor wiring logic in the server module.
- Added alias contract tests in 	ests/test_server_aliases.py for required-key mapping and helpful missing-key failures.

- Hardened ServerFactoryService.build_default(...) override handling with fail-fast validation for unknown class_overrides keys to prevent silent wiring typos.
- Added factory contract coverage in 	ests/test_server_factory_service.py for unknown override-key rejection.

- Refactored class override handling in ServerFactoryService through esolve_class_overrides(...) to centralize merge + unknown-key validation logic and reduce drift in build-default wiring.
- Added merge-contract test coverage in 	ests/test_server_factory_service.py to ensure known overrides replace defaults while untouched defaults remain stable.

- Added ServerFactoryService.default_class_map() as a dedicated source of production wiring defaults, and updated uild_default() to consume it through esolve_class_overrides(...) for cleaner composition semantics.
- Added factory contracts in 	ests/test_server_factory_service.py to lock default-map coverage and enforce that uild_default() uses resolved override output for downstream build wiring.

- Added uild_kwargs_from_resolved_classes(...) in ServerFactoryService to centralize alias-to-build-argument translation and reduce mapping drift risk in uild_default().
- Added factory contracts in 	ests/test_server_factory_service.py for build-kwargs mapping correctness and uild_default() delegation through the new helper path.

- Centralized factory override-key policy in src/semantic_modulator/app/server_aliases.py via APP_FACTORY_ONLY_KEYS, ALLOWED_FACTORY_OVERRIDE_KEYS, and alidate_override_keys(...).
- Updated ServerFactoryService.resolve_class_overrides(...) to delegate override-key validation to shared alias-policy helper, reducing duplicated validation logic.
- Added alias-policy contract tests in 	ests/test_server_aliases.py for allowed-key coverage and unknown-key rejection behavior.

- Refactored `ServerFactoryService.build(...)` to consume centralized helper configs for code adapter, AFM, resource limits, ACE defaults, and context-window monitor shape.
- Added factory contract tests in `tests/test_server_factory_service.py` to lock helper default values and constructor-kwargs wiring behavior.
- Converted `ServerFactoryService.build(...)` to class-dispatch (`@classmethod`) so helper config overrides are extensible for derived factories without altering default behavior.
- Added subclass-dispatch contract coverage in `tests/test_server_factory_service.py` to verify `build()` uses helper overrides via `cls` dispatch.

- Added centralized factory logging payload helpers in `ServerFactoryService` (`file_sync_log_kwargs`, `path_validator_log_kwargs`, `ace_framework_log_kwargs`) and routed `build()` logging through class-dispatchable helper calls.
- Added logging-helper contracts in `tests/test_server_factory_service.py`, including subclass override dispatch coverage for logger payloads.

- Added `ServerFactoryService.build_service_layer(...)` to centralize service-layer construction (context/lifecycle/progress/persistence/tool-profile/runtime + adapter wiring) and reduce constructor orchestration complexity.
- Updated `build()` to delegate service assembly through the new helper and added factory contracts in `tests/test_server_factory_service.py` for helper wiring and delegation behavior.

- Added `ServerFactoryService.build_core_runtime_layer(...)` to centralize foundational runtime assembly (focus/persistence/resource/file-sync/version/path-validator/ACE) and associated startup logging.
- Updated `build()` to delegate foundational runtime construction through the new helper and added delegation+wiring contracts in `tests/test_server_factory_service.py` to lock behavior.

- Declared typed factory artifact contracts in `server_factory_service` (`CoreRuntimeArtifacts`, `ServiceLayerArtifacts`, `BuildArtifacts`) and updated factory helper/build return annotations to use them.
- Added artifact-contract coverage in `tests/test_server_factory_service.py` to lock typed key schemas and reduce dict-shape drift risk across helper boundaries.

- Added typed class-alias contract `FactoryClassMap` in `server_factory_service` and tightened class-map-related method annotations to reduce override wiring drift risk.
- Added `validate_default_class_map(...)` fail-fast guard and wired `build_default()` to validate class-map key parity with `ALLOWED_OVERRIDE_KEYS` before override resolution.
- Added factory contract tests in `tests/test_server_factory_service.py` for class-map type/schema alignment and early drift rejection semantics.

- Added typed build-kwargs contract `BuildKwargsMap` and updated `build_kwargs_from_resolved_classes(...)` to return the typed mapping for stronger alias-to-constructor parity semantics.
- Added `validate_build_kwargs_map(...)` and wired `build_default()` to fail fast when build-kwargs keys drift from the canonical constructor kwargs contract.
- Added factory contract tests in `tests/test_server_factory_service.py` for build-kwargs schema declaration/alignment and early drift rejection before server build invocation.

## [0.10.0] - 2026-02-26

**Experimental Module Exposure + Production Hardening**

### Added
- TOON serialization MCP tools: `toon_encode`, `toon_decode` (~40% smaller than JSON)
- SCAR compression MCP tools: `scar_compress`, `scar_get_stats` (experimental: untrained weights)
- Multimodal ingest MCP tool: `multimodal_ingest` (text + code + images via Pillow)
- `experimental_handlers.py`: 300+ lines, 30+ tests, all responses include `"experimental": true` flag
- `help_handlers.py`: `tool_help` MCP tool for structured per-tool documentation
- `visualization_handlers.py`: graph visualization MCP tools
- MCP tool profile system: `MCP_TOOL_PROFILE` env var selects `full` (49), `core`, or `minimal` tool set
- HTTP server (`src/http_server.py`) for Kubernetes health probes (`HTTP_ENABLED=true`)
- Docker multi-stage build targeting <500MB image with non-root security
- Kubernetes production manifests in `deployment/kubernetes/` (HPA, ServiceMonitor, PrometheusRule)
- GitHub Actions CI/CD workflows (test matrix, lint, build, deploy)
- `.env.example` documenting all 13 environment variables
- `.claude/context/state.json` cross-session metadata schema (v1.0.0)
- ChromaDB moved to optional extra (`pip install ".[chromadb]"`)

### Changed
- `pyproject.toml`: `requires-python` tightened to `>=3.10,<3.14`; added `structlog`, `orjson`, `msgpack` to core deps
- Handler count: 7 → 10 modules in `src/handlers/`
- Total MCP tools: 39 → 49
- Test count: 1,063 → 1,171 (78 test modules)
- Source module count: 50 → 71 (incl. `semantic_modulator` subpackage)

### Fixed
- `.claude/settings.json` rewritten to valid Claude Code hooks format (command-based)
- `.claude/.mcp.json` phantom packages replaced with `@modelcontextprotocol/server-filesystem`
- Hook scripts created as proper Python executables in `.claude/hooks/`

## [0.9.0] - 2025-11-29

**Code Compression Adapter with Semantic Fidelity Encoding**

### Added

**CodeCompressionAdapter** (src/code_compression_adapter.py, 702 lines):
- Unified API for text and code compression
- AST-based code skeleton generation with semantic node mapping
- Dual embedding model support:
  * Text: `all-MiniLM-L6-v2` (384-dim)
  * Code: `microsoft/codebert-base` (768-dim)
- Domain-aware search routing based on file type
- Delimiter-aware prefix checks for data isolation

**New MCP Tools** (2 tools, 37 total):
- `ingest_directory`: Bulk code file ingestion with glob patterns
  * Path validation (CWE-22 prevention)
  * Configurable patterns, exclusions, max_files
  * Concurrent processing via BatchCompressionManager
- `get_help`: Interactive documentation system
  * Context-aware help for all 37 MCP tools
  * Usage examples and tips

**Help Handlers** (src/handlers/help_handlers.py, 415 lines):
- Comprehensive tool documentation
- Category-based help organization
- Usage examples for each tool

### Fixed

**Critical Bug Fixes (6 P0-P2 issues):**
- **P0-1**: SkeletonResponse constructor signature mismatch (code ingestion failed)
- **P0-2**: FidelityLevel.BALANCED → DETAILED for code nodes (invalid fidelity)
- **P1-1**: node_map population for code skeletons (empty node maps)
- **P1-2**: Search uses appropriate embedding model per domain (mismatched dimensions)
- **P2-1**: Exclude pattern matching using PurePath.match() (fnmatch failed on **)
- **P2-2**: Search/deletion filtering with delimiter-aware checks (data leakage)

**Code Quality:**
- Fix asyncio deprecation warnings (`get_event_loop` → `get_running_loop`)
- Remove unused `event_loop_policy` fixture from conftest.py
- Update `handle_ingest` docstring to specify JSON return type
- Add delimiter-aware prefix checks in `get_stats()` for consistency

### Changed
- Test count: 1064 → 1077 tests (14 new CodeCompressionAdapter tests)
- Coverage: 73% maintained
- Python 3.16+ compatible (asyncio deprecation fixes)

### Tests Added
- **test_code_compression_adapter.py** (489 lines, 14 tests):
  * Text and code ingestion tests
  * Search with correct embedding model tests
  * Skeleton generation with node_map tests
  * Deletion isolation tests (TestDeletionDoesNotOverreach)
  * Statistics aggregation tests

---

## [0.7.0] - 2025-11-27 🚧 IN PROGRESS

**Enterprise Production Readiness - Week 1-2: Reliability Infrastructure**

Goal: Achieve 95/100 production readiness through systematic hardening across reliability, testing, observability, and DevOps.

### Added (Week 1-2 Complete)

**Reliability Infrastructure (Zero Server Hangs, Cascading Failure Prevention)**
- **TimeoutManager** (src/reliability.py, 108 lines):
  * Configurable timeout enforcement for all async operations
  * Per-operation timeout configuration (embedding: 30s, compression: 120s, persistence: 10s)
  * Prevents server hangs from indefinite operations
  * OperationTimeoutError with operation context
  * configure_timeout() for runtime adjustment
- **CircuitBreaker** (src/reliability.py, 103 lines):
  * Prevents cascading failures with CLOSED/OPEN/HALF_OPEN states
  * Configurable failure threshold and timeout
  * Automatic state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
  * Circuit breaker statistics (failure_count, success_count, last_failure_time)
  * Manual reset capability
- **RetryPolicy** (src/reliability.py, 104 lines):
  * Exponential backoff for transient errors
  * Configurable max_retries, base_delay, max_delay, backoff_factor
  * Retryable exception configuration (OSError, TimeoutError, ConnectionError)
  * Automatic retry with increasing delays
  * RetryExhaustedError after max attempts
- **RateLimiter** (src/rate_limiter.py, 223 lines):
  * Token bucket rate limiting to prevent resource exhaustion
  * Configurable rate (tokens/second) and capacity (burst limit)
  * Blocking and non-blocking modes
  * Automatic token refill based on elapsed time
  * Rate limiter statistics (rejection rate, total wait time)
  * Global rate limiters for common operations (ingest, batch_ingest, compression)
- **GracefulDegradation** (src/graceful_degradation.py, 220 lines):
  * Embedding fallback: PyTorch → ONNX → TF-IDF
  * Persistence fallback: Disk → In-memory only (with warning)
  * File sync fallback: Full validation → Cached metadata
  * Version history fallback: Full diffs → Metadata only
  * Maintains partial functionality when components fail
- **Custom Exception Types** (src/error_types.py, 90 lines):
  * OperationTimeoutError (timeout exceeded)
  * CircuitBreakerOpenError (too many failures)
  * RetryExhaustedError (all retries exhausted)
  * RateLimitExceededError (rate limit hit)
  * GracefulDegradationError (fallback active)
  * Hierarchical exception structure with ReliabilityError base
- **Test Coverage:** 29 comprehensive reliability tests
  * 4 TimeoutManager tests (timeout enforcement, configuration)
  * 6 CircuitBreaker tests (state transitions, OPEN/HALF_OPEN behavior)
  * 5 RetryPolicy tests (exponential backoff, exhaustion)
  * 7 RateLimiter tests (token bucket, refill, blocking/non-blocking)
  * 3 Global rate limiter configuration tests
  * 3 Graceful degradation fallback tests
  * 2 Integration tests (timeout+retry, circuit breaker+retry)

### Changed (Week 1-2)
- Test count: 735 → 764 tests (29 new reliability tests)
- Code formatted with black (zero warnings)
- Code linted with ruff (zero warnings)

### Added (Week 3-4 Complete)

**Comprehensive Testing Suite (100 New Tests - 90%+ Production Test Confidence)**
- **Shared Test Infrastructure** (tests/conftest.py, 380 lines):
  * Centralized fixtures for all test files (compressor, handler_context, managers)
  * Sample data fixtures (short/medium/large text, code, documents, dialogue)
  * Temporary file helpers (temp_dir, temp_file, temp_code_file)
  * Performance testing fixtures (100 documents, 10k token large docs)
  * Chaos engineering fixtures (mock_disk_full, mock_network_partition, mock_model_crash)
  * Assertion helpers (assert_valid_skeleton, assert_valid_embedding)
- **Integration Workflow Tests** (tests/test_integration_workflows.py, 1479 lines, 50 tests):
  * 10 basic workflows (ingest → compress → expand, multi-fidelity, batch, concurrent)
  * 10 file sync workflows (staleness detection, auto-refresh, checksum validation)
  * 10 version history (create diffs, view history, automatic pruning, rollback)
  * 5 ACE workflows (generate → reflect → curate, LRU management)
  * 5 AFM workflows (add → retrieve → forget, recency weighting, critical memory)
  * 5 batch processing integration (progress tracking, error isolation, retry)
  * 5 cross-feature integration (compression + ACE + AFM + file sync + versions)
- **Performance Benchmark Tests** (tests/test_performance.py, 634 lines, 15 tests):
  * 3 throughput tests (single-doc, batch vs sequential, compression speed)
  * 3 latency tests (p50, p95, p99 latency measurement)
  * 3 memory usage tests (baseline, large docs, batch processing with leak detection)
  * 3 cache effectiveness (hit rate, memory overhead, eviction performance)
  * 3 burst capacity (10, 50, 100 concurrent documents with rate limiting)
- **Chaos Engineering Tests** (tests/test_chaos_engineering.py, 1056 lines, 20 tests):
  * 5 disk failures (ENOSPC, permission denied, corrupted files, slow I/O, recovery)
  * 5 model crashes (CUDA OOM, timeout, corrupted weights, retry on transient, all tiers fail)
  * 5 network issues (partition, timeout, connection refused, intermittent, circuit breaker recovery)
  * 5 data corruption (NaN/Inf embeddings, malformed JSON, invalid diffs, corrupted cache)
- **End-to-End Scenario Tests** (tests/test_e2e_scenarios.py, 1238 lines, 15 tests):
  * 5 research paper workflows (compress & navigate, multi-fidelity comparison, version tracking, ACE enhancement, batch processing)
  * 5 codebase documentation workflows (multi-file compression, code/docs separation, file sync & refresh, version evolution, multimodal)
  * 5 dialogue management workflows (AFM compression, recency vs importance, budget exhaustion, multi-session persistence, full pipeline integration)

### Changed (Week 3-4)
- Test count: 764 → 864 tests (100 new comprehensive tests)
- Test infrastructure: Centralized conftest.py with reusable fixtures
- Code formatted with black (zero warnings)
- Code linted with ruff (zero warnings)
- Production test confidence: 70% → 90%+ (complete workflow coverage)

### Added (Week 5-6 Complete)

**Observability & Monitoring Infrastructure (<100ms Overhead, Production-Grade Visibility)**
- **Structured Logging** (src/structured_logging.py, 540 lines):
  * JSON and human-readable formatters with ISO 8601 timestamps
  * Async-aware context propagation via contextvars
  * Operation tracking context manager with automatic request ID generation
  * OpenTelemetry trace correlation (trace_id, span_id in logs)
  * Log sampling (1% DEBUG in production, configurable)
  * Performance: <10ms overhead per log
- **Prometheus Metrics** (src/metrics.py, 330 lines):
  * 7 production metrics (compression_ratio, processing_latency, documents_processed, cache_hit_ratio, active_documents, errors, batch_size)
  * Cardinality control with validated label values (prevents explosion)
  * Histogram buckets optimized for compression workloads
  * Graceful degradation (NoOp when prometheus_client unavailable)
  * Prometheus text format export for scraping
- **OpenTelemetry Tracing** (src/observability.py, 717 lines):
  * Distributed tracing with OTLP export (console fallback)
  * Async-safe context propagation via contextvars
  * Span creation context manager with attributes
  * Trace sampling (10% production, 100% development)
  * Integration with structured logging (trace correlation)
  * Exception recording with span status tracking
  * Performance: <50ms overhead per operation
- **Health Checks & Diagnostics** (src/health.py, 500 lines):
  * Three-tier health checks (liveness, readiness, diagnostics)
  * Component health monitoring (embedding manager, persistence, cache, disk space)
  * Performance metrics (p50/p95/p99 latency percentiles)
  * Resource usage tracking (memory, disk, cache)
  * 10-second result caching (avoids expensive checks)
  * Health status: healthy/degraded/unhealthy
- **Test Coverage:** 168 comprehensive observability tests
  * 43 structured logging tests (91% coverage: JSON/human formatters, async context, OTEL integration)
  * 29 Prometheus metrics tests (86% coverage: all 7 metrics, cardinality control, graceful degradation)
  * 53 OpenTelemetry tracing tests (85% coverage: span creation, async propagation, exception handling, OTLP export)
  * 43 health check tests (91% coverage: liveness/readiness/diagnostics, component health, caching)

### Changed (Week 5-6)
- Test count: 864 → 1,032 tests (168 new observability tests)
- Dependencies added: prometheus-client>=0.19.0 (already had opentelemetry)
- Code formatted with black (zero warnings)
- Code linted with ruff (zero warnings)
- Observability modules: 88% average coverage (91% logging, 86% metrics, 85% tracing, 91% health)

### Added (Week 7-8 Complete)

**DevOps & Operational Excellence (Production Deployment Infrastructure)**
- **HTTP Server for Health & Metrics** (src/http_server.py, 415 lines):
  * Async aiohttp web server for production monitoring endpoints
  * GET /health/liveness - Kubernetes liveness probe (always returns healthy if running)
  * GET /health/readiness - Kubernetes readiness probe (checks component health, returns 503 if unhealthy)
  * GET /health/diagnostics - Detailed diagnostics with performance metrics and resource usage
  * GET /metrics - Prometheus metrics scraping endpoint (text format)
  * GET / - Root endpoint (redirects to readiness check)
  * Optional HTTP server (disabled by default for backward compatibility)
  * Environment variable configuration: HTTP_ENABLED, HTTP_HOST, HTTP_PORT
  * Integration with existing src/health.py and src/metrics.py modules
  * Graceful shutdown handling via asyncio cancellation
  * Zero-overhead when disabled (stdio-only mode unchanged)
- **Docker Multi-Stage Build** (Dockerfile, 134 lines):
  * Builder stage: Install dependencies, download models (~600MB intermediate)
  * Runtime stage: Minimal image with only runtime artifacts (<500MB target, ~450MB expected)
  * Security hardening: Non-root user (uid 1000), read-only filesystem support, dropped capabilities
  * Health check: curl-based liveness probe (conditional on HTTP_ENABLED)
  * Optimizations: --no-cache-dir, virtual environment isolation, model caching
  * Volume support: /data for persistent semantic modulator data
  * Supports both stdio mode (default) and HTTP mode (Kubernetes)
- **Kubernetes Deployment Manifests** (deployment/kubernetes/, 14 files):
  * Core manifests (7 files): Namespace, ConfigMap, Secret template, Service, Deployment, HPA, ServiceMonitor, PrometheusRule
  * Deployment configuration: 2 replicas, resource limits (1-2GB memory, 0.5-2 CPU), three-tier health probes
  * HorizontalPodAutoscaler: 2-10 replicas, CPU target 70%, memory target 80%
  * ServiceMonitor: Prometheus Operator integration, /metrics scraping every 15s
  * PrometheusRule: 16 comprehensive alerting rules (availability, performance, application, infrastructure)
  * Helper files (7 files): kustomization.yaml, README.md (502 lines), validate.sh, quickstart.sh, MANIFEST_SUMMARY.md, .yamllint
  * Security: Pod anti-affinity, non-root user, read-only filesystem, dropped capabilities
  * High availability: Zero-downtime rolling updates (maxSurge: 1, maxUnavailable: 0)
- **GitHub Actions CI/CD Workflows** (.github/workflows/, 10 files):
  * test.yml (139 lines): Matrix testing (Python 3.10/3.11/3.12), pip caching, coverage enforcement (70%), Codecov integration
  * lint.yml (170 lines): Black, Ruff, Bandit security scanning, Radon complexity analysis
  * build.yml (216 lines): Docker multi-stage builds, GHCR push, Trivy CVE scanning, SBOM generation (SPDX)
  * deploy.yml (427 lines): Kubernetes deployment, staging auto-deploy, production approval-gated, health validation
  * Documentation (6 files): README.md, WORKFLOWS_SUMMARY.md, WORKFLOWS_SETUP_CHECKLIST.md, WORKFLOWS_QUICK_REFERENCE.md, WORKFLOWS_ARCHITECTURE.md, .github/README.md
  * Performance optimizations: Pip caching (40% speedup), Docker BuildKit layer caching (50% speedup), parallel testing (3x speedup)
  * Security: Trivy vulnerability scanning, SARIF output, SBOM generation, secret scanning
- **Test Coverage:** 32 comprehensive HTTP server tests
  * 11 endpoint tests: Liveness, readiness, diagnostics, metrics, root endpoint response validation
  * 3 integration tests: Real health.py and metrics.py module integration
  * 5 configuration tests: Environment variable configuration, route setup
  * 4 lifecycle tests: Server start/stop/error handling, concurrent requests
  * 4 edge case tests: Liveness never fails, HEAD requests, component field validation
  * 5 standalone tests: Config structure, async handlers, request parameters
  * 100% test pass rate (32/32 passing)

### Changed (Week 7-8)
- Test count: 1,032 → 1,064 tests (32 new HTTP server tests)
- Dependencies added: aiohttp>=3.9.0 (async HTTP server for health/metrics endpoints)
- Code formatted with black (zero warnings)
- Code linted with ruff (zero warnings)
- Deployment infrastructure: Production-ready Kubernetes + Docker + CI/CD
- Deployment model: Hybrid stdio (primary for MCP) + optional HTTP (secondary for monitoring)

---

## [0.6.0-beta] - 2025-11-26 ✅ COMPLETE

**Major release with 3 parallel implementation tracks:**
- 🚀 **Track 1:** Multi-Document Batch Processing (4× throughput improvement)
- 📊 **Track 2:** Interactive Graph Visualization (4 new MCP tools, multiple export formats)
- 💾 **Track 3:** Memory Optimization Engine (3-tier embedding system, 70% memory reduction)

### Added

**Track 1: Multi-Document Batch Processing**
- **BatchCompressionManager** (src/batch_manager.py, 439 lines):
  * Concurrent document ingestion with asyncio.gather()
  * Bounded parallelism with semaphore-based rate limiting (default: 4 concurrent)
  * Real-time progress tracking with callbacks
  * Error isolation (one failure doesn't block entire batch)
  * Automatic retry mechanism for transient failures
- **Batch Progress Tracking:**
  * BatchProgress dataclass with percentage, success rate calculation
  * BatchProgressTracker for live progress updates
  * Callback support for progress monitoring
- **Utility Functions:**
  * batch_ingest_from_dict() for dict-based batch ingestion
  * batch_ingest_from_files() for file path batch ingestion
- **MCP Tool:** batch_ingest_documents (handles 1-100 docs in single call)
- **Test Coverage:** 18 comprehensive batch processing tests
- **Performance:** 4× throughput improvement (measured: 4 docs in 8.2s vs 32.8s sequential)

**Track 2: Interactive Graph Visualization**
- **GraphVisualizer** (src/graph_visualizer.py, 472 lines):
  * render_ascii(): Terminal-friendly text visualization with importance scores
  * export_json(): Structured JSON export for programmatic access
  * export_graphml(): GraphML format for Gephi, Cytoscape, NetworkX analysis
  * visualize_html(): Interactive HTML with pyvis (draggable nodes, zoom, pan)
  * explain_compression_decision(): Detailed analysis of why nodes kept/dropped
- **VisualizationConfig:** Customizable max_nodes, min_importance, edge weights, layouts
- **4 New MCP Tools (35 total, was 31):**
  * export_graph_json: Export semantic graph as JSON
  * visualize_graph_html: Generate interactive HTML visualization
  * export_graph_graphml: Export as GraphML for analysis tools
  * explain_compression_decision: Explain compression decisions for specific nodes
- **Visualization Handlers** (src/handlers/visualization_handlers.py, 158 lines):
  * SmartError integration for consistent error handling
  * Validation for file_id, output_path, node_id parameters
- **Test Coverage:** 16 visualization tests (15 passing, 1 skipped for pyvis dependency)

**Track 3: Memory Optimization Engine**
- **ONNX Embedding Manager** (src/embeddings_onnx.py, 277 lines):
  * Quantized INT8 models for reduced memory footprint
  * 3-5× faster inference on CPU vs PyTorch
  * 60-70% memory reduction (~150MB vs ~400MB)
  * Automatic model download and caching
  * Memory usage tracking (RSS, VMS, percent)
- **TF-IDF Fallback** (src/embeddings_tfidf.py, 270 lines):
  * Lightweight sklearn-based vectorization (~10MB memory)
  * 100-1000× faster than neural models
  * 70-80% quality correlation with SBERT
  * Configurable vocabulary size, n-grams, document frequency
  * Auto-fit capability for first-time usage
- **LRU Embedding Cache** (src/embedding_cache.py, 413 lines):
  * Thread-safe OrderedDict-based cache with LRU eviction
  * Configurable capacity (default: 10k entries)
  * Optional disk persistence with msgpack serialization
  * TTL support for stale entry expiration
  * Batch operations (get_batch, put_batch) for efficiency
  * Cache statistics: hit rate, entries, memory usage
- **Multi-Tier Embedding System** (src/embeddings.py enhanced):
  * EmbeddingTier enum (STANDARD, ONNX, TFIDF)
  * encode() method with tier selection and automatic fallback
  * Tier switching with set_tier() and get_tier()
  * LRU cache integration (transparent, 60-80% hit rate)
  * Enhanced cache_stats(): tier info, LRU stats, memory breakdown
- **Dependencies Added:**
  * pyvis>=0.3.2 (HTML visualization)
  * onnxruntime>=1.16.0 (ONNX inference)
  * optimum[exporters]>=1.15.0 (ONNX model export)
  * msgpack>=1.0.7 (cache serialization)
  * transformers>=4.35.0 (ONNX tokenization)
- **Test Coverage:** 24 memory optimization tests
  * ONNX encoding and memory tracking (6 tests)
  * TF-IDF fit/transform and auto-fit (6 tests)
  * LRU cache operations and persistence (6 tests)
  * Tier switching and fallback logic (6 tests)

### Changed
- Threading bottleneck fixed: max_workers=1 → 4 in semantic_compressor.py (instant 2-4× speedup)
- MCP tool count: 31 → 35 tools
- Test count: 446 → 497 tests (51 new tests added)
- EmbeddingManager now supports tier selection with automatic fallback hierarchy
- requirements.txt updated with 5 new optional dependencies

### Performance
- **Batch Processing:** 4× throughput improvement for multi-document ingestion
- **Threading:** 2-4× speedup from increased worker pool size
- **Memory (ONNX tier):** 60-70% reduction (~150MB vs ~400MB)
- **Memory (TF-IDF tier):** 98% reduction (~10MB vs ~400MB)
- **Inference Speed (ONNX):** 3-5× faster than standard SentenceTransformer
- **Inference Speed (TF-IDF):** 100-1000× faster than neural models
- **Cache Hit Rate:** 60-80% for production workloads (eliminates redundant computation)

### Quality
- ✅ All 497 tests passing (was 446 in v0.5.0)
- ✅ Code formatted with `black` (zero warnings)
- ✅ Code linted with `ruff` (zero warnings)
- ✅ Backward compatibility maintained (all tiers optional, graceful degradation)
- ✅ Zero tech debt introduced
- ✅ 68% coverage on GraphVisualizer, 32% on visualization handlers

### Backward Compatibility
- All new features are **optional**:
  * Batch processing: Existing single-doc ingestion unchanged
  * Visualization: No impact on compression behavior
  * Memory tiers: Default tier is STANDARD (existing SentenceTransformer)
- Graceful degradation:
  * ONNX/TF-IDF tiers unavailable without dependencies → automatic fallback to STANDARD
  * LRU cache unavailable without msgpack → no caching, existing behavior
  * pyvis unavailable → HTML visualization skipped, other formats work

## [0.5.0-beta] - 2025-11-25 ✅ COMPLETE

### Added
- **Async Support (Phase 1):**
  - Async encoding wrapper (`_encode_async()`) using ThreadPoolExecutor for non-blocking embedding generation
  - `ingest_file_async()` method for async MCP server use
  - Backward-compatible `ingest_file()` synchronous wrapper for existing tests
  - 9 compression handlers converted to async (handle_ingest, handle_read_skeleton, etc.)
  - Async MCP router (`route_tool_call()`) with await support
  - 10 comprehensive async operation tests (test_async_operations.py)
  - Converted 14 MCP routing tests to async
  - Converted 27 compression handler tests to async

### Changed
- `SemanticCompressor.ingest_file()` split into async (`ingest_file_async()`) and sync (`ingest_file()`) variants
- All compression handlers now async-capable to prevent MCP timeout errors
- MCP router supports async handler invocation with `await`
- Test suite updated: 446/446 tests passing (100% success rate)

### Performance
- Non-blocking embedding generation prevents 5-50 second event loop blocks
- MCP server remains responsive during document ingestion
- Health checks respond <100ms even during active compression
- Zero MCP timeout errors during large document compression

### Quality
- ✅ All 446 tests passing
- ✅ Code formatted with `black` (zero warnings)
- ✅ Code linted with `ruff` (zero warnings)
- ✅ Backward compatibility maintained for existing synchronous code

## [0.4.4] - 2025-11-25

### Added
- MCP lifespan management with async context manager protocol (__aenter__/__aexit__)
- PageRank caching for performance optimization (O(1) lookup after first computation)
- Server lifecycle tests (6 tests in test_server_lifecycle.py)
- PageRank caching unit tests (3 tests in test_semantic_compressor_unit.py)

### Changed
- Moved state loading from __init__ to __aenter__ for proper resource initialization
- Moved state persistence from __del__ to __aexit__ for graceful shutdown
- Updated test count: 427 → 436 tests (all passing)

### Fixed
- 5 failing tests related to async context manager protocol
- Test expectations aligned with MCP best practices

## [0.4.3] - 2025-11-25

### Added
- TypedDict handler hints for IDE autocomplete and type safety
- Comprehensive test coverage improvements (427 tests, 59% overall coverage)
- Code compressor unit tests (47 tests, 99% coverage)
- Server unit tests (43 tests, 88% coverage)
- Semantic compressor unit tests (65 tests, 99% coverage)

### Changed
- Refactored AFM._pack_messages() into focused helper functions
- Standardized all handler signatures to use HandlerContext
- Updated sentence-transformers to >=3.1.0 (CVE fixes)
- Version synchronized across all files to 0.4.3

### Fixed
- Critical handler signature bug in compression_handlers.py
- Security vulnerabilities CVE-2024-11392/11393/11394
- Documentation metric inaccuracies

## [0.4.0] - 2025-11-24

### Added
- ACE Framework (Agentic Context Engineering) - 32% quality boost with 4× compression
- 7 new MCP tools for ACE operations (generate, reflect, curate)
- File sync with staleness detection (mtime + MD5 checksums)
- Full version history with diffs
- LRU eviction for version history, file sync metadata, ACE contexts
- Singleton embedding manager (~70% memory reduction)

### Changed
- Handler architecture refactoring (server.py: 1,911 → 299 lines, 84% reduction)
- Modular handler system with centralized routing
- Total MCP tools: 16 → 30

## [0.2.0] - 2025-11-22

### Added
- Persistent storage (ChromaDB + JSON fallback)
- Resource management (100MB/doc, 1GB total, 1000 docs max)
- AFM export/import for conversation state
- Automated MCP installation script

### Changed
- Total MCP tools: 13 → 16

## [0.1.0] - Initial Release

### Features
- Semantic compression (80-95% token reduction)
- 5 adaptive fidelity levels
- Graph-based structure preservation with PageRank
- Code compression with AST parsing
- Multi-modal support (text, code, images)
- Blind spot detection
- 13 MCP tools

### Research Foundations
- JSCCM (arXiv:2511.15699v1)
- FPQE (arXiv:2511.15695v1)
- SCAR (arXiv:2511.14063v1)
- AFM (arXiv:2511.12712v1)
- ACE (arXiv:2510.04618v1)






















