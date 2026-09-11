# MAP — Growth VAL park relocate (engine → gotcontext-main)

## Destination

token-saver-5000 `backlog.md` no longer enumerates 22 SaaS Growth VAL-* IDs as if they were engine work. Ownership lives in gotcontext-main with a single pointer from the engine. **No SaaS features are implemented in either repo in this slice.**

## Open questions

(none — CEO update 2026-08-29 recommended default B: move list to platform)

## Out of scope

- Implementing VAL-TEAM / WEBHOOK / ENTERPRISE / CROSS
- Changing `artifacts/validation-contract-growth.md` contract text (stays as contract source in engine)
- Touching in-flight branches `fix/b1b2-*`
- New product features

## Answers

### Engine backlog is a pointer, not a VAL table

**Answer:** Replace the 22-row VAL table in `token-saver-5000/backlog.md` with one short section pointing at the platform park file + defer brief.

**Check:** `rg 'VAL-TEAM-001' backlog.md` → 0; park file ID set == TEAM∪WEBHOOK∪ENTERPRISE∪CROSS (measured 23); VAL-DOCKER stays engine Done.
**Judged by:** run it

### Platform owns the enumerated park

**Answer:** Add `gotcontext-main/docs/growth-val-park.md` (or BACKLOG section) listing all 22 IDs with blocker **CEO-gate / product-home**, source contract path, and re-trigger from defer brief.

**Check:** file exists; contains VAL-TEAM-001 through VAL-CROSS-006.
**Judged by:** run it

### Defer brief + orchestrator updated

**Answer:** Defer brief notes relocation complete; `.orchestrator/state.json` last_slice=`growth-val-park-relocate`, open_backlog reflects pointer-only.

**Check:** files updated; no claim that VAL features are built.
**Judged by:** read it
