# Folder guide: `tests/fixtures/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../../CLAUDE.md).

## Contents

### Python modules

#### `__init__.py`

(no module docstring — see symbols below)

_No top-level classes or functions (may re-export only)._

#### `f11_multi_node_fixtures.py`

Sealed multi-node fixture corpus for the F11 ranker decision (#250).

| Kind | Name |
|------|------|
| `def` | `content_words` |
| `def` | `term_count` |
| `class` | `F11Query` |
| `class` | `F11MultiNodeFixture` |

#### `quality_gate_fixtures.py`

Sealed ground-truth fixtures for the compression quality gate.

| Kind | Name |
|------|------|
| `class` | `QualityGateFixture` |

### Other files

- `benchmark_corpus.json`
- `skill_context_sample.txt`

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
