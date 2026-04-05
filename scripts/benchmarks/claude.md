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

#### `run_benchmarks.py`

Run fixed-corpus token-savings benchmarks and emit JSON report.

| Kind | Name |
|------|------|
| `def` | `build_parser` |
| `def` | `main` |

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
