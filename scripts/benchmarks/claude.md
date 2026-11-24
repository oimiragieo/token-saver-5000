# Folder guide: `scripts/benchmarks/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../../CLAUDE.md).

## Contents

### Python modules

#### `check_benchmark_guard.py`

Fail CI if benchmark reports regress below configured thresholds.

| Kind | Name |
|------|------|
| `def` | `build_parser` |
| `def` | `_evaluate_case_set` |
| `def` | `_build_markdown_summary` |
| `def` | `main` |

#### `dogfood_2026_sota.py`

Wave 5 Enterprise E2E Dogfooding Harness for Token Saver 5000 (2026 SOTA).

| Kind | Name |
|------|------|
| `def` | `_sha256_text` |
| `def` | `run_wave1_e2e` |
| `def` | `_make_80_tool_catalog` |
| `def` | `run_wave2_e2e` |
| `def` | `run_wave3_e2e` |
| `def` | `run_wave4_e2e` |
| `def` | `compute_paired_summary` |
| `def` | `_install_wheel_into_venv` |
| `def` | `run_wheel_check` |
| `def` | `run_dogfood_suite` |
| `def` | `build_parser` |
| `def` | `main` |

#### `run_benchmarks.py`

Run fixed-corpus token-savings benchmarks and emit JSON report.

| Kind | Name |
|------|------|
| `def` | `build_parser` |
| `def` | `main` |

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
