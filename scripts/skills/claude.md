# Folder guide: `scripts/skills/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../../CLAUDE.md).

## Contents

### Python modules

#### `compress_context.py`

Compress context with baseline/query-guided/evidence-aware modes.

| Kind | Name |
|------|------|
| `def` | `_read_text` |
| `def` | `build_parser` |
| `def` | `main` |

#### `profile_tokens.py`

Profile token counts for raw vs compressed context.

| Kind | Name |
|------|------|
| `def` | `_read_text` |
| `def` | `build_parser` |
| `def` | `main` |

#### `run_skill_workflow.py`

Run end-to-end token-saver skill workflow in one command.

| Kind | Name |
|------|------|
| `def` | `_read_text` |
| `def` | `build_parser` |
| `def` | `main` |

#### `validate_evidence.py`

Validate evidence sufficiency for a query against compressed context.

| Kind | Name |
|------|------|
| `def` | `_read_text` |
| `def` | `build_parser` |
| `def` | `main` |

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
