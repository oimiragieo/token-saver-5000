# Folder guide: `src/cli_benchmark/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../../CLAUDE.md).

## Contents

### Python modules

#### `__init__.py`

CLI benchmark harness for Token Saver 5000.

_No top-level classes or functions (may re-export only)._

#### `compressor.py`

Wraps Token Saver skill scripts for pre-compression.

| Kind | Name |
|------|------|
| `class` | `CompressResult` |
| `def` | `compress_file` |
| `def` | `compress_text` |

#### `corpus.py`

Corpus loader for benchmark harness.

| Kind | Name |
|------|------|
| `class` | `CorpusEntry` |
| `def` | `load_manifest` |
| `def` | `load_corpus` |
| `def` | `load_all_corpus` |
| `def` | `build_prompt` |

#### `pricing.py`

Cost computation for benchmark results.

| Kind | Name |
|------|------|
| `def` | `compute_cost` |
| `def` | `get_model_rates` |

#### `project_scaffold.py`

MCP-mode project folder generator.

| Kind | Name |
|------|------|
| `def` | `_mcp_settings_claude` |
| `def` | `_mcp_settings_gemini` |
| `def` | `create_vanilla` |
| `def` | `create_with_mcp` |
| `def` | `cleanup` |

#### `providers.py`

CLI provider wrappers for Claude Code and Gemini CLI.

| Kind | Name |
|------|------|
| `def` | `_find_cli` |
| `def` | `is_available` |
| `def` | `run_prompt` |
| `def` | `_build_command` |
| `def` | `_parse_json_output` |
| `def` | `_parse_claude_result` |
| `def` | `_parse_gemini_result` |
| `def` | `_parse_opencode_result` |
| `def` | `_parse_codex_result` |

#### `results.py`

Benchmark result dataclasses and output formatters.

| Kind | Name |
|------|------|
| `class` | `CLIResult` |
| `class` | `ComparisonResult` |
| `def` | `_aggregate_repeats` |
| `class` | `BenchmarkReport` |

#### `runner.py`

Main benchmark orchestrator.

| Kind | Name |
|------|------|
| `def` | `run_benchmark` |
| `def` | `run_search_compress_benchmark` |
| `def` | `_run_skill_comparison` |
| `def` | `_run_mcp_comparison` |

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
