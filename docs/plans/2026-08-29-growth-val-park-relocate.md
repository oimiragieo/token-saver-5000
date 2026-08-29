# Plan — Growth VAL park relocate

**Wayfinder:** `.wayfinder/growth-val-park-relocate/MAP.md`  
**Depth:** solo docs  
**Fable (round 1):** CHANGES_REQUIRED — hardcoded "22" stale; add ID-set equality DoD + VAL-DOCKER disposition  
**Fable (round 2 target):** APPROVED after fixes below

## Premise checks

- [x] Engine backlog had Growth VAL table (relocated in L2)
- [x] `gotcontext-main` previously had zero VAL-TEAM matches
- [x] Contracts file `artifacts/validation-contract-growth.md` **kept**
- [x] `VAL-DOCKER-*` stays engine Done (not parked)
- [x] Do not touch `fix/b1b2-*`

## Leaves

| Leaf | Repo | Action |
|------|------|--------|
| L1 | gotcontext-main | `docs/growth-val-park.md` + BACKLOG pointer |
| L2 | token-saver-5000 | Pointer-only Open section |
| L3 | token-saver-5000 | Defer brief + orchestrator + ledger |

## DoD (ID-set equality — no hardcoded count)

```
Growth IDs = VAL-TEAM-* ∪ VAL-WEBHOOK-* ∪ VAL-ENTERPRISE-* ∪ VAL-CROSS-*
park file must contain every Growth ID formerly listed in engine backlog
engine backlog.md must contain zero VAL-TEAM/WEBHOOK/ENTERPRISE/CROSS rows
VAL-DOCKER-* remains only in engine Done / validation receipts (not in park as open SaaS work)
```

Measured: TEAM6 + WEBHOOK6 + ENTERPRISE5 + CROSS6 = **23** Growth IDs.

## Pre-CI dependency graph

Docs-only (no `src/` edits) → run `python scripts/check_claude_folder_guides_sync.py` only. No Full Validation wait required for this slice's risk class; still push and note CI run.

## Definition of done

Wayfinder Checks pass · park ⊇ Growth ID set · engine backlog has no Growth VAL rows · no SaaS code · contracts file present
