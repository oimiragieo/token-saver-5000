# Plan: Validation + Guides Closeout

**Wayfinder:** `.wayfinder/validation-guides-closeout/MAP.md`  
**Date:** 2026-08-28

## Steps

1. Fix `_SMALL_INPUT_TOKEN_THRESHOLD` import in `compression_handlers_ingest.py`
2. Run pytest: async operations, compression handlers, f11, all_tools_have_handlers
3. Run `python scripts/audit_env_example.py`
4. Regenerate folder guides; verify sync check
5. Write `artifacts/validation-run-2026-08-28.md` with full VAL matrix
6. Update `backlog.md`, `workflow-ledger.md`, `.orchestrator/state.json`

## Verification

```bash
python scripts/audit_env_example.py
python scripts/check_claude_folder_guides_sync.py
pytest tests/test_async_operations.py tests/test_compression_handlers.py \
  tests/test_f11_gated_fusion.py tests/test_all_tools_have_handlers.py -q --no-cov
```
