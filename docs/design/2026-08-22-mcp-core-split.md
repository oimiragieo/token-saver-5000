# Design: split `src/handlers/mcp_core.py` (N2 slice 2)

Measured, not inferred. Commands quoted below; source file at commit HEAD of
`token-saver-5000` as of 2026-08-22, `wc -l` = 3670 lines.

## Step 0 — inventory (AST-derived, not regex)

Slice 1 taught: a regex inventory misses annotated assignments. Confirmed live
here — my first-pass regex `^[A-Z_][A-Z_0-9]* = ` missed
`CORE_STABLE_TOOL_NAMES: Set[str] = {...}` because of the type annotation.
Redone with `ast.parse` + walking `tree.body`:

```
$ python3 -c "import ast; ..." # walk top-level nodes
assign      logger                       L46-46
assign      SCOPE_PROPERTIES             L48-65     (dict, public — see blast radius)
annassign   CORE_STABLE_TOOL_NAMES       L67-75     (Set[str], PUBLIC — imported by registry.py)
assign      SUPPORTED_TOOL_PROFILES      L76-76     (frozenset/tuple-like, PUBLIC)
def         _normalize_tool_profile      L79-86     internal (leading underscore)
def         _enabled_tool_names          L89-93     internal
def         _tools_for_profile           L96-98     internal
def         setup_mcp_tools              L101-3426  PUBLIC — 3325 lines, almost the whole file
async def   route_tool_call              L3429-3670 PUBLIC — 241 lines
```

That is the **entire top-level symbol table**: 4 module-level names, 3 private
helpers, 2 public functions. No classes in this file at all — everything is
either a data literal or a routing function.

`setup_mcp_tools`'s 3325-line body is **one Python list literal**:

```
$ grep -c "Tool(" src/handlers/mcp_core.py         # 128
$ sed -n '123p' src/handlers/mcp_core.py            # "    all_tools = ["
```

`all_tools = [ ... 128 × Tool(...) ... ]`, then:

```python
all_tools.sort(key=lambda t: t.name)
return _tools_for_profile(all_tools, profile)
```

`route_tool_call`'s 241-line body is dominated by **one dict literal** — the
router table — plus ~40 lines of validation/logging around it:

```
$ sed -n '3468,3626p' src/handlers/mcp_core.py | grep -c '": '
```

Router entries: 128 (matches the 128 tool schemas exactly, 1:1, no orphans on
either side — verified with a script joining router keys against schema
`name=` values: `missing: []`).

**Module-level mutable state:** `CORE_STABLE_TOOL_NAMES` is a `Set[str]`
literal, never mutated anywhere in either repo (`grep -rn
"CORE_STABLE_TOOL_NAMES\.\(add\|update\|remove\|discard\)"` → 0 hits). `logger`
is a `get_logger(...)` instance used only inside `route_tool_call` itself
(`grep -n "logger\."` → 1 hit, L3666) and never imported by name elsewhere
(`grep -rn "mcp_core\.logger"` → 0 hits). **There is exactly one registry-shaped
object in this file (`CORE_STABLE_TOOL_NAMES`) and it has exactly one
definition today — the split MUST NOT create a second one.**

## Step 1 — blast radius (both repos)

```
$ grep -rn "mcp_core" src/ ../api/ --include=*.py
```

19 distinct call sites (excluding this file's own docstring/comments and
mentions inside other comments), all importing by **name**, never
`import mcp_core` + attribute access:

| Importer | Names pulled | Import style |
|---|---|---|
| `src/handlers/help_handlers.py:1579` | `setup_mcp_tools` | function-local |
| `src/semantic_modulator/api/mcp/registry.py:8` | `CORE_STABLE_TOOL_NAMES`, `SUPPORTED_TOOL_PROFILES`, `setup_mcp_tools` | module-level, then **re-exports all 3** via its own `__all__` (L11) |
| `src/semantic_modulator/api/mcp/router.py:6` | `route_tool_call` | module-level |
| `api/app/mcp_gateway.py:1240,1455` | `setup_mcp_tools`, `route_tool_call` | function-local (deferred import — the exact pattern the v1.22.4 outage came from; see `test_mcp_call_tool_dispatch_fence.py`) |
| `api/app/routers/v1/audit_cache.py:72` | `route_tool_call` | function-local |
| `api/app/routers/v1/check_budget.py:45,69` | `route_tool_call` (twice, once aliased `rtc`) | function-local |
| `api/app/routers/v1/detect_issues.py:74` | `route_tool_call` | function-local |
| `api/app/routers/v1/mcp_tools.py:166` | `setup_mcp_tools` | function-local |
| `api/tests/*.py` (7 files) | `setup_mcp_tools`, `route_tool_call`, or `patch("src.handlers.mcp_core.route_tool_call", ...)` | mixed — see below |

**This module is consumed by the parent API directly**, exactly as the brief
warned, via `api/app/mcp_gateway.py` (both the free-tier "core" session
manager path at L1240 and the full-profile path at L1455) and by three
`api/app/routers/v1/*.py` routers that call `route_tool_call` for
non-MCP-transport code paths (audit cache, budget check, issue detection).

**Two things in the parent test suite need explicit attention**, both new
risk classes beyond slice 1 (models.py had no monkeypatch targets like these):

1. `api/tests/test_mcp_reload.py:233,249,265` — `patch.dict("sys.modules",
   {"src.handlers.mcp_core": None or a Mock})`. This treats `mcp_core` as a
   **single module object** to swap out wholesale (simulating a hot-reload /
   import-failure). A package `mcp_core/__init__.py` is still one module
   object at that dotted path, so `sys.modules["src.handlers.mcp_core"]`
   continues to resolve correctly — but this test is the discriminating check
   that a naive split (e.g. leaving `route_tool_call` importable only from a
   submodule, not re-exported at the package root) would break. Confirm this
   test still passes post-split; it is cheap insurance for free.
2. `api/tests/test_audit_cache.py` (×3) and `test_mcp_gateway.py` (×2) —
   `mock.patch("src.handlers.mcp_core.route_tool_call", ...)`. This patches
   the **attribute on the package namespace**. It only works if
   `route_tool_call` is a real name bound in `mcp_core/__init__.py` (via
   `from .dispatch import route_tool_call`), because every caller does a
   fresh function-local `from src.handlers.mcp_core import route_tool_call`
   at call time — which re-reads the package attribute, so the patch takes
   effect. If the re-export shim instead did something indirect (e.g.
   `__getattr__`-based lazy re-export), `mock.patch` would still work via
   `setattr`/`delattr` on the module object, but a `from .dispatch import
   route_tool_call as route_tool_call` static rebind is simpler and provably
   correct — use that, not `__getattr__`.

**`api/app/services/compression.py:5`** only mentions `mcp_core` in a
docstring pointing at `mcp_gateway.py`'s import — no code dependency there.

## Step 2 — the split, from measured coupling

The schema list and the router dict are **already partitioned by the file's
own top-of-file imports** — every `Tool(name=...)` schema has a corresponding
`router["name"] = <module_alias>.<handler_func>`, and I joined them
programmatically (128/128 matched, 0 orphans):

```
$ python3 - <<'EOF'
# join router dict {name: module} against schema Tool() line spans
...
EOF
ch       tools=24  schema_lines=725
toh      tools=20  schema_lines=509
ph       tools=9   schema_lines=249
ace      tools=7   schema_lines=244
exp      tools=9   schema_lines=236
mh       tools=11  schema_lines=216
moh      tools=6   schema_lines=161
eh       tools=5   schema_lines=119
afm      tools=6   schema_lines=116
vh       tools=4   schema_lines=108
th       tools=4   schema_lines=105
mmh      tools=2   schema_lines=101
fs       tools=4   schema_lines=86
bh       tools=4   schema_lines=74
coh      tools=5   schema_lines=61
rh       tools=3   schema_lines=57
dh       tools=2   schema_lines=55
doch     tools=2   schema_lines=54
hh       tools=1   schema_lines=25
```

No cluster needs a second level (largest is `ch` at 725 lines, well under
1000). Proposed package:

```
src/handlers/mcp_core/
├── __init__.py           ~40 lines   — re-export shim (Step 2a)
├── _constants.py         ~30 lines   — SCOPE_PROPERTIES, SUPPORTED_TOOL_PROFILES, CORE_STABLE_TOOL_NAMES
├── _profile.py           ~25 lines   — _normalize_tool_profile, _enabled_tool_names, _tools_for_profile
├── setup.py              ~60 lines   — setup_mcp_tools(): imports + concatenates all schema lists, sorts, filters
├── dispatch.py           ~260 lines  — route_tool_call(): the router dict + validation/logging wrapper (unchanged logic)
├── schemas_compression.py            ~730 lines  — ch  (24 tools: ingest/skeleton/modulate/search/stats/... )
├── schemas_token_optimization.py     ~515 lines  — toh (20 tools: discover_savings/calculate_roi/check_budget/export_team_data/...)
├── schemas_prompts_ace.py            ~500 lines  — ph (9) + ace (7): prompt handlers + ACE framework
├── schemas_experimental.py           ~240 lines  — exp (9): "NOT production-ready" / ASG-SI — kept isolated, matches its own warning comment
├── schemas_memory.py                 ~220 lines  — mh (11)
├── schemas_model_experiment.py       ~285 lines  — moh (6) + eh (5)
├── schemas_afm_temporal.py           ~225 lines  — afm (6) + th (4)
├── schemas_multimodal_viz.py         ~215 lines  — mmh (2) + vh (4)
├── schemas_filesync_bundle.py        ~165 lines  — fs (4) + bh (4)
└── schemas_misc.py                   ~260 lines  — coh (5) + rh (3) + dh (2) + doch (2) + hh (1): connector/resource/detection/docs/help
```

15 files, largest 730 lines, smallest 25-line functions folded into
`_profile.py`. Grouping mirrors the file's own existing handler-module
aliases (`ch`, `afm`, `bh`, ... imported at the top of today's file) — no new
taxonomy invented, so a reviewer can verify each schema landed in the right
place by checking it against the *existing* router-comment groups (`#
Document Compression (9 tools)`, `# AFM Dialogue (6 tools)`, etc. at
L3472-3540 today).

### Step 2a — the re-export shim (MANDATORY, zero edits in either repo)

`__init__.py` must re-export **every symbol Step 0 found**, not just the two
public functions — this is the surprise-2 lesson, and `CORE_STABLE_TOOL_NAMES`
already proved it live in Step 1 (`registry.py` imports it by name):

```python
# src/handlers/mcp_core/__init__.py
from ._constants import (
    SCOPE_PROPERTIES,
    SUPPORTED_TOOL_PROFILES,
    CORE_STABLE_TOOL_NAMES,
)
from ._profile import (
    _normalize_tool_profile,
    _enabled_tool_names,
    _tools_for_profile,
)
from .setup import setup_mcp_tools
from .dispatch import route_tool_call

__all__ = [
    "SCOPE_PROPERTIES",
    "SUPPORTED_TOOL_PROFILES",
    "CORE_STABLE_TOOL_NAMES",
    "setup_mcp_tools",
    "route_tool_call",
]
```

The three `_`-prefixed helpers are re-exported too even though nothing
outside this file currently imports them (`grep -rn
"_normalize_tool_profile\|_enabled_tool_names\|_tools_for_profile"
src/ ../api/` shows zero external hits) — cheap insurance, symmetric with
`CORE_STABLE_TOOL_NAMES` being "surely internal" until it wasn't. Do **not**
put them in `__all__` (preserves the current "private by convention" signal
for anything doing `from mcp_core import *`), but do bind them as real names
on the package so `mcp_core._normalize_tool_profile` still resolves if
anything reaches for it directly.

### Module-level mutable state: single owner

`CORE_STABLE_TOOL_NAMES` is defined **once**, in `_constants.py`. `setup.py`
and `_profile.py` both import it from there (`from ._constants import
CORE_STABLE_TOOL_NAMES`), never redefine it. `dispatch.py` does not need it at
all (verified: `route_tool_call` never references it — profile filtering
happens via `_enabled_tool_names`, called from within `route_tool_call` but
importing the *function*, not the constant, so there is no second copy to
accidentally create).

### Import direction (no cycles)

```
_constants.py   (leaf — only stdlib/typing + mcp.types.Tool)
     ^
_profile.py     (imports _constants)
     ^
schemas_*.py    (each imports Tool, ToolAnnotations, SCOPE_PROPERTIES from
                 _constants, and its own handler module e.g. `from .. import
                 compression_handlers as ch` — same relative-import pattern
                 the file uses today, unchanged)
     ^
setup.py        (imports _constants.SUPPORTED_TOOL_PROFILES is NOT needed —
                 setup.py imports _profile._tools_for_profile and every
                 schemas_*.py's tool-list constant, concatenates, sorts)
     ^
dispatch.py     (imports every schemas_*.py's *handler module* — actually
                 dispatch.py needs the same 19 handler-module imports the
                 original file has at its top, e.g. `from .. import
                 compression_handlers as ch` — NOT imports from setup.py or
                 the schemas_*.py files, since the router dict only needs the
                 handler functions, not the Tool schemas)
     ^
__init__.py     (imports setup.setup_mcp_tools, dispatch.route_tool_call,
                 _constants.*, _profile.* — the only file that imports
                 "downward" from both setup.py and dispatch.py)
```

Dependency direction is one-way: `_constants` → `_profile` → `{schemas_*,
dispatch}` → `setup` → `__init__`. `dispatch.py` and the `schemas_*.py` files
are siblings (both import the 19 handler-module aliases directly from
`src/handlers/`, exactly as today) and never import each other — that
symmetry is what keeps this cycle-free without needing dispatch.py to import
19 schema-list constants it has no use for.

## Step 3 — the oracle: byte-identical tool catalogue

Slice 1's oracle (`alembic check` diff) was **already red before the split**
(247 pre-existing differences on a fresh DB) — the split team baselined it
first and only compared delta-vs-delta. Same discipline here, because I have
not run this baseline myself as part of this design task (design-only,
no production code touched):

**Before writing any code**, capture on the pre-split tree:

```bash
cd token-saver-5000
python3 -c "
import json, sys
sys.path.insert(0, '.')
from src.handlers.mcp_core import setup_mcp_tools
tools = setup_mcp_tools(profile='full')
catalogue = [
    {'name': t.name, 'description': t.description, 'inputSchema': t.inputSchema,
     'annotations': t.annotations.model_dump() if t.annotations else None}
    for t in tools
]
json.dump(catalogue, open('/tmp/pre-split-catalogue-full.json', 'w'), indent=2, sort_keys=True)
" 
python3 -c "... profile='core_stable' ..." > /tmp/pre-split-catalogue-core.json  # both profiles — the filter fn is also being touched
```

**If this pre-split capture is not byte-stable across two consecutive runs**
(it should be — `all_tools.sort(key=lambda t: t.name)` already exists
specifically for determinism per the file's own comment at L3423), STOP and
treat that as a pre-existing bug to baseline around, exactly as slice 1 did
with `alembic check`. Do not assume today's oracle is green — prove it twice
before trusting the diff.

**After the split**, regenerate both JSON files from the new package import
path (same `from src.handlers.mcp_core import setup_mcp_tools` — the shim
means the call site is unchanged) and `diff -u` byte-for-byte. Any difference
is a bug: a dropped tool, a reordered schema, a changed default, or (the
dangerous one) two tools silently sharing a name because a router dict got
merged wrong.

**Second oracle, also mandatory:** run
`api/tests/test_mcp_call_tool_dispatch_fence.py` against the split tree. It
does not import `mcp_core` directly (it audits `mcp_gateway.py`'s own
function-body imports + the `_FREE_TOOLS`/`_PRO_TOOLS` allowlist vs
`_dispatch_tool`), but its Test 1 (AST-walk every deferred `from X import Y`
in `mcp_gateway.py`, then `hasattr(module, name)`) will directly exercise
`hasattr(mcp_core_package, "setup_mcp_tools")` and `hasattr(mcp_core_package,
"route_tool_call")` the moment `mcp_gateway.py` is imported — this is the
exact regression class (v1.22.4) the split risks reintroducing if the shim
is incomplete. Green here is a second, independent confirmation beyond the
catalogue diff.

**Third, run the existing test files that already touch this module** rather
than inventing new ones: `test_mcp_gateway.py`, `test_mcp_reload.py`,
`test_mcp_tools_call_dispatch.py`, `test_readonly_hint_annotations.py`,
`test_saas_filesystem_filter.py`, `test_audit_cache.py`,
`verify_all_tools.py` (the last is a manual script, not pytest-collected —
run it explicitly, it exercises every tool against a real server context and
is the closest thing to an end-to-end smoke test this module has).

## Step 4 — ranked task list

Each task 2-5 minutes, one commit boundary per task, dependency-ordered.

1. **Baseline the oracle** (no code change): run the pre-split catalogue
   capture script above for both profiles, twice each, confirm byte-stable.
   Commit the two JSON files under `docs/design/baselines/` (gitignored from
   prod, kept only as a local diff aid — do not ship them in the package).
2. **Create `src/handlers/mcp_core/` directory + `_constants.py`**: move
   `SCOPE_PROPERTIES`, `SUPPORTED_TOOL_PROFILES`, `CORE_STABLE_TOOL_NAMES`
   verbatim (byte-identical dict/set literals) from the old file.
3. **Create `_profile.py`**: move `_normalize_tool_profile`,
   `_enabled_tool_names`, `_tools_for_profile` verbatim; import
   `CORE_STABLE_TOOL_NAMES`, `SUPPORTED_TOOL_PROFILES` from `_constants`.
4. **Create the 10 `schemas_*.py` files**, one commit each (or batched 2-3 per
   commit if review bandwidth allows — but keep each file's diff reviewable
   against the line-span table in Step 2): move the exact `Tool(...)` literals
   for that cluster verbatim, each file exporting a module-level list constant
   (e.g. `COMPRESSION_TOOLS: List[Tool] = [...]`) plus the handler-module
   import lines it needs (`from .. import compression_handlers as ch`, etc.,
   copied unchanged from the original top-of-file imports).
5. **Create `setup.py`**: `from ._profile import _tools_for_profile`; import
   every `*_TOOLS` constant from the 10 schema files; concatenate in the
   original order (order does not matter for behavior since the function
   sorts, but keep the original file's grouping order for reviewability);
   `all_tools.sort(key=lambda t: t.name)`; `return _tools_for_profile(...)`.
6. **Create `dispatch.py`**: move `route_tool_call` verbatim, including its
   19 handler-module imports (`from .. import compression_handlers as ch`,
   etc. — copy the exact import block from the top of the original file) and
   the full router dict body, unchanged.
7. **Create `__init__.py`** per Step 2a's exact shim.
8. **Delete `src/handlers/mcp_core.py`** (the old flat file) in the same
   commit as step 7, or the immediately following commit — do not leave both
   the old file and the new package on disk simultaneously (Python will
   silently prefer the package over the same-named `.py` file in most import
   orders, which would mask a broken shim rather than fail loudly).
9. **Run the oracle**: regenerate post-split catalogue JSON for both
   profiles, `diff -u` against the Step 1 baseline — must be empty.
10. **Run `test_mcp_call_tool_dispatch_fence.py`** and the 7 existing test
    files listed in Step 3, plus `verify_all_tools.py` manually.
11. **Regenerate folder guides**: `python
    scripts/generate_claude_folder_guides.py` then `python
    scripts/check_claude_folder_guides_sync.py` — CI reddened on slice 1 for
    skipping this; `src/handlers/` gains a new subpackage and needs its own
    guide.
12. **Update `src/handlers/__init__.py`**: it currently lists `"mcp_core"` as
    a flat-file entry in its docstring/`__all__`-like listing (L8, L20) —
    confirm the string still resolves correctly as a package name (it will,
    Python doesn't distinguish) but update the docstring's "mcp_core.py: Core
    MCP infrastructure" line to say "mcp_core/: ..." so the doc doesn't lie
    about the file being flat.

## Risks beyond what the audit assumed

- **This is the file the parent API's production request path touches on
  every single MCP tool call and on 3 non-MCP REST routes** (`audit_cache`,
  `check_budget`, `detect_issues`). Slice 1 (`models.py`) is loaded once at
  process start; this module's `route_tool_call` is on the hot path for every
  tool invocation. A broken shim here doesn't fail at import time in a way
  CI's collection step would necessarily catch first — it could fail per-call
  at runtime if, say, `dispatch.py` and a `schemas_*.py` file diverge on a
  tool name (router has an entry, no schema — or vice versa) in a way that
  only 1 of 128 tools exercises. The byte-identical catalogue diff (Step 3)
  is the only check that covers all 128 uniformly; do not skip it for "a
  couple of tools looked fine in manual testing."
- **`sys.modules` patching test** (`test_mcp_reload.py`) is the one place a
  package-vs-flat-module distinction could theoretically bite, even though
  reasoning above says it shouldn't — run it explicitly rather than trusting
  the general test sweep to catch it, since it's testing exactly the
  "swap the whole module out" scenario a split changes the shape of.
- **`registry.py` re-exports `CORE_STABLE_TOOL_NAMES` via its own `__all__`**
  — a second layer of re-export sitting on top of this module's shim. Not
  broken by this split (it imports by name, unaffected by flat-vs-package),
  but worth grep-confirming post-split since it's an easy thing to forget
  exists when focused on the shim one level down.
- **The `exp` cluster (9 tools, "Experimental ... NOT production-ready")**
  and the ASG-SI comment (`# ASG-SI (4 tools) - Experimental
  self-improvement framework`) inside it — keep this cluster in its own file
  (`schemas_experimental.py`) rather than folding it into a neighbor, so the
  file boundary preserves the existing "these are marked experimental"
  signal for the next reader, rather than diluting it into a mixed file.
- **Folder-guide CI gate**: flagged explicitly per the brief — this bit
  slice 1 today and will bite again here since a brand-new subpackage is
  being introduced, not just files moved within an existing one.

## Summary (5 bullets)

- **Measured symbol count**: 4 module-level names (`logger`,
  `SCOPE_PROPERTIES`, `CORE_STABLE_TOOL_NAMES`, `SUPPORTED_TOOL_PROFILES`) + 3
  private helper functions + 2 public functions (`setup_mcp_tools` at
  3325 lines, `route_tool_call` at 241 lines) — an AST walk, not a regex,
  after regex missed the annotated `CORE_STABLE_TOOL_NAMES: Set[str] = {...}`.
- **Importer count**: 19 call sites across both repos (`token-saver-5000` +
  parent `api/`), including 2 production hot-path imports in
  `api/app/mcp_gateway.py` and 3 non-MCP REST routers that call
  `route_tool_call` directly, plus `sys.modules`-level patching in
  `test_mcp_reload.py` and attribute-patching (`mock.patch`) in
  `test_audit_cache.py` / `test_mcp_gateway.py` that require
  `route_tool_call` to be a real re-exported name on the package, not a lazy
  `__getattr__` shim.
- **Proposed submodules**: 15 files (`__init__.py`, `_constants.py`,
  `_profile.py`, `setup.py`, `dispatch.py`, 10 `schemas_*.py` files grouped
  by the file's own existing handler-module aliases), largest at ~730 lines
  (`schemas_compression.py`, 24 tools) — all comfortably under the 1000-line
  ceiling with no need for a second split level.
- **Module-level mutable state found**: exactly one, `CORE_STABLE_TOOL_NAMES`
  (a `Set[str]`, never mutated at runtime in either repo), owned solely by
  `_constants.py` post-split; `logger` is a second module-level object but is
  purely internal (never imported by name elsewhere) so carries no
  duplication risk.
- **Single biggest risk**: this module sits on the **production hot path**
  for every MCP tool call (unlike slice 1's process-start-only `models.py`),
  and is consumed by 3 non-MCP REST routers in the parent repo in addition to
  the MCP gateway itself — a subtly broken re-export or a router/schema
  mismatch on even 1 of 128 tools would not necessarily fail at import time,
  making the byte-identical full-catalogue diff (Step 3) load-bearing rather
  than optional, and the two monkeypatch-shaped parent tests
  (`test_mcp_reload.py`'s `sys.modules` swap, `test_audit_cache.py`'s
  attribute patch) the two tests most likely to catch a shim that "looks"
  correct but isn't.
