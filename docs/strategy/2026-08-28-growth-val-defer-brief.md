# Defer brief — Growth VAL-* in token-saver-5000

**Date:** 2026-08-28  
**Skill:** `research-council-defer` (no research council run — demand signal already absent; BLOCK not DEFER-by-enthusiasm)

## Context

`artifacts/validation-contract-growth.md` lists Teams / Webhooks / Enterprise / Cross-area VAL-* for gotcontext.ai SaaS. This repository is the MCP compression server (`semantic-modulator`). Grep of `src/` shows no `/api/teams`, webhook, or license phone-home surfaces.

## Decision

**BLOCK** (not soft defer): do not implement Growth VAL-* inside `token-saver-5000`.

## Named prereqs

1. SaaS surfaces exist in the product home (expected: `gotcontext-main` or successor), **or**
2. CEO explicitly relocates a named VAL-* (e.g. MCP-only rewrite of VAL-CROSS-002) into this repo’s contract set.

## Re-trigger

Behavior ≥ payment or shipped SaaS API in the product repo that maps 1:1 to a VAL-* ID — then open a wayfinder in *that* repo, not here.

## Integration shape (if triggered)

Keep MCP Docker ONNX proofs under VAL-DOCKER-*; rewrite any cross-item that only needs self-hosted image smoke into an Areas 1–4 ID so Growth stays SaaS-scoped.
