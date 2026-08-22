# Folder guide: `src/handlers/mcp_core/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../../../CLAUDE.md).

## Contents

### Python modules

#### `__init__.py`

MCP Core Routing Module

_No top-level classes or functions (may re-export only)._

#### `_constants.py`

Module-level constants for the MCP core routing package.

_No top-level classes or functions (may re-export only)._

#### `_profile.py`

Tool-profile filtering helpers. Moved verbatim from mcp_core.py.

| Kind | Name |
|------|------|
| `def` | `_normalize_tool_profile` |
| `def` | `_enabled_tool_names` |
| `def` | `_tools_for_profile` |

#### `dispatch.py`

route_tool_call: dispatch table + validation/logging wrapper.

| Kind | Name |
|------|------|
| `async def` | `route_tool_call` |

#### `schemas_afm_temporal.py`

Tool schemas: AFM Dialogue (afm) + Temporal (th). Split from mcp_core.py (N2 slice 2).

_No top-level classes or functions (may re-export only)._

#### `schemas_compression.py`

Tool schemas: Document Compression (ch). Split from mcp_core.py (N2 slice 2).

_No top-level classes or functions (may re-export only)._

#### `schemas_experimental.py`

Tool schemas: Experimental / NOT production-ready (exp). Split from mcp_core.py (N2 slice 2).

_No top-level classes or functions (may re-export only)._

#### `schemas_filesync_bundle.py`

Tool schemas: File Sync (fs) + Handoff Bundles (bh). Split from mcp_core.py (N2 slice 2).

_No top-level classes or functions (may re-export only)._

#### `schemas_memory.py`

Tool schemas: Memory handlers (mh). Split from mcp_core.py (N2 slice 2).

_No top-level classes or functions (may re-export only)._

#### `schemas_misc.py`

Tool schemas: Connector (coh) + Resource (rh) + Detection (dh) + Docs (doch) + Help (hh). Split from mcp_core.py (N2 slice 2).

_No top-level classes or functions (may re-export only)._

#### `schemas_model_experiment.py`

Tool schemas: Model handlers (moh) + Experiment handlers (eh). Split from mcp_core.py (N2 slice 2).

_No top-level classes or functions (may re-export only)._

#### `schemas_multimodal_viz.py`

Tool schemas: Multimodal (mmh) + Visualization (vh). Split from mcp_core.py (N2 slice 2).

_No top-level classes or functions (may re-export only)._

#### `schemas_prompts_ace.py`

Tool schemas: Prompt templates (ph) + ACE framework (ace). Split from mcp_core.py (N2 slice 2).

_No top-level classes or functions (may re-export only)._

#### `schemas_token_optimization.py`

Tool schemas: Token Optimization (toh). Split from mcp_core.py (N2 slice 2).

_No top-level classes or functions (may re-export only)._

#### `setup.py`

setup_mcp_tools: concatenate every schema-list module, sort, filter by profile.

| Kind | Name |
|------|------|
| `def` | `setup_mcp_tools` |

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
