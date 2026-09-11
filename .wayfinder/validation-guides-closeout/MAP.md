# Wayfinder MAP — Validation + Guides Closeout (Slice 4)

> **Done and right** means: backlog open items closed with execution evidence, CI gates green, folder guides synced to split modules, and VAL contracts classified pass/waived with receipts.

## Success criteria

| Goal | Definition of done |
|------|------------------|
| Folder guides | `generate_claude_folder_guides.py` + `check_claude_folder_guides_sync.py` pass on committed tree |
| VAL matrix | `artifacts/validation-run-2026-08-28.md` lists every VAL-* ID with pass/fail/waived + command evidence |
| Quality gates | `audit_env_example.py`, MCP 128 tools, async/concurrent ingest tests pass |
| No regressions | `test_compression_handlers`, `test_f11_gated_fusion`, `test_all_tools_have_handlers` pass |

## Out of scope (waived for this repo)

- VAL-UI-* — no Next.js frontend in this repository
- VAL-DEGRADE-* — requires live Redis/Supabase/Postgres stack (document as deferred)
- VAL-DOCKER-001/003/004 — Docker smoke (optional local run; CI docker job if present)

## File touch map

| Area | Files |
|------|-------|
| Guides | `src/claude.md`, `src/handlers/claude.md`, `tests/claude.md`, … |
| Handler fix | `compression_handlers_ingest.py` (private helper imports) |
| Receipt | `artifacts/validation-run-2026-08-28.md` |
| Tracking | `backlog.md`, `workflow-ledger.md`, `.orchestrator/state.json` |

## Risks

- Star-import split modules miss `_private` helpers → explicit imports from `compression_handlers_common`
- Guide sync fails until `claude.md` files are committed after regen
