# GitHub Actions Workflows - Architecture & Flow Diagrams

This document visualizes the current CI/CD workflow architecture, trigger events, and execution flow.

`ci.yml` is the canonical broad validation workflow. `test.yml` and `lint.yml` are retained only as manual-only deprecated compatibility shims.

---

## Overall CI/CD Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Git Repository Events                            │
└───────────────────────┬───────────────────────────────┬──────────────────────┘
                        │                               │
             ┌──────────▼──────────┐         ┌──────────▼──────────┐
             │ Push / Pull Request │         │   Create Tag v*     │
             └──────────┬──────────┘         └──────────┬──────────┘
                        │                               │
        ┌───────────────┼────────────────┐              │
        │               │                │              │
        ▼               ▼                ▼              ▼
 ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
 │   ci.yml     │ │ focused      │ │ legacy test/ │ │  build.yml   │
 │ canonical CI │ │ guard flows  │ │ lint shims   │ │ docker build │
 └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
        │                │                │                │
        │                │                │                ├───────────────┐
        │                │                │                │               │
        ▼                ▼                ▼                ▼               ▼
 ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
 │ required PR  │ │ path-scoped  │ │ manual notice│ │ push image   │ │ deploy.yml   │
 │ validation   │ │ specialists  │ │ only         │ │ to registry  │ │ staging/prod │
 └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

---

## Trigger Event Matrix

```
┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Event            │ ci.yml       │ focused      │ legacy shims │ build/deploy │
│                  │              │ guards       │              │              │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Push main        │ RUN          │ path-based   │ SKIP         │ build.yml    │
│ Push develop     │ RUN          │ path-based   │ SKIP         │ build.yml    │
│ Push claude/**   │ RUN          │ path-based   │ SKIP         │ SKIP         │
│ Push feature/*   │ RUN          │ path-based   │ SKIP         │ SKIP         │
│ Pull Request     │ RUN          │ path-based   │ SKIP         │ SKIP         │
│ Tag v*           │ SKIP         │ SKIP         │ SKIP         │ build+deploy │
│ Workflow Dispatch│ RUN          │ depends      │ RUN          │ optional     │
└──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

Focused guards currently include:

- `skill-ci.yml`
- `benchmark-guard.yml`
- `mcp-profile-guard.yml`

These run only when matching files change, so they complement `ci.yml` instead of duplicating it.

---

## Canonical Validation Flow

```
Feature Branch / Pull Request
        │
        ▼
   ci.yml starts
        │
        ├─ quality-gate
        │   - Black
        │   - Ruff
        │   - workflow/package contract tests
        │
        ├─ compatibility
        │   - Python 3.10
        │   - Python 3.11
        │   - Python 3.12
        │   - import + smoke coverage
        │
        ├─ package-validation
        │   - python -m build
        │   - twine check
        │   - install built package
        │   - MCP installer smoke
        │
        └─ full-validation
            - full pytest command
            - no-cov
            - perf test excluded
        │
        ▼
   Required repository validation result
        │
   PASS ───────────────► merge / continue
   FAIL ───────────────► fix locally and push again
```

---

## `ci.yml` Job Topology

```
ci.yml
  │
  ├─ quality-gate
  │    Purpose:
  │    - fail fast on repo-wide formatting/lint/contract drift
  │
  ├─ compatibility
  │    Matrix:
  │    - python-version = [3.10, 3.11, 3.12]
  │    Purpose:
  │    - validate install/import behavior across supported runtimes
  │
  ├─ package-validation
  │    Purpose:
  │    - verify sdist/wheel metadata
  │    - verify installed MCP tooling is invokable
  │
  └─ full-validation
       Purpose:
       - run the canonical repo-wide pytest command
       - catch integration drift not covered by smoke/contract checks
```

Design intent:

- `quality-gate` fails quickly before the expensive jobs finish.
- `compatibility` verifies supported Python runtimes explicitly.
- `package-validation` protects the productized install surface, not just source-tree execution.
- `full-validation` remains the broadest correctness gate.

---

## Focused Guard Workflow Role

```
Changed files
    │
    ├─ skills / help handlers ─────────────► skill-ci.yml
    │
    ├─ benchmark harness / compressor ─────► benchmark-guard.yml
    │
    └─ MCP profile / packaging surfaces ───► mcp-profile-guard.yml
```

These workflows exist for fast, high-signal specialist checks on narrow file sets.

They are not replacements for `ci.yml`. They are additional safety rails when sensitive areas change.

---

## Legacy Workflow Compatibility Layer

```
Manual dispatch only
        │
        ├─ test.yml
        │    └─ posts deprecation notice
        │
        └─ lint.yml
             └─ posts deprecation notice
```

Behavior:

- no push trigger
- no pull request trigger
- no required status role
- purpose is migration compatibility only

This prevents duplicate broad CI runs while preserving the old workflow names long enough for maintainers to update habits and branch protection settings.

---

## Branch Protection Model

```
Protected branch
      │
      ├─ Require PR review
      ├─ Require ci.yml to pass
      └─ Optionally require focused guards
           when the team wants path-sensitive hardening
```

Recommended rule of thumb:

- Always require `ci.yml`.
- Require focused guards only if your branch protection strategy wants explicit specialist gates.
- Do not require `test.yml` or `lint.yml`.

---

## Cache and Performance Strategy

```
Cold run
  ├─ install dependencies from network
  ├─ build package artifacts
  └─ execute full validation

Warm run
  ├─ restore pip cache
  ├─ reuse previously downloaded dependencies
  └─ shorten setup time before validation work begins
```

Performance notes:

- `ci.yml` uses pip caching keyed from dependency metadata.
- `build.yml` uses Docker layer caching.
- focused guards reduce unnecessary specialist checks on unrelated changes.
- manual-only legacy shims eliminate duplicate broad validation work.

---

## Build and Release Flow

```
Merge / push to main or develop
        │
        └─ build.yml
            ├─ build Docker image
            ├─ push image when allowed
            ├─ run Trivy scan
            └─ generate SBOM

Create semantic version tag
        │
        ├─ build.yml
        │    └─ produce tagged image
        │
        └─ deploy.yml
             ├─ deploy-staging automatically
             └─ deploy-production after approval
```

---

## Deployment Flow

```
Tag vX.Y.Z pushed
      │
      ▼
 build.yml completes
      │
      ▼
 deploy-staging
      ├─ authenticate to cluster
      ├─ build manifests
      ├─ patch image tag
      ├─ apply manifests
      ├─ wait for rollout
      ├─ run health checks
      └─ inspect logs / status
      │
      ▼
 staging live
      │
      ▼
 deploy-production waits for approval
      │
      ├─ approve and deploy
      ▼
 production live
```

---

## Error Recovery Flow

```
ci.yml failure
    │
    ├─ GitHub blocks merge
    ├─ developer inspects failing job
    ├─ developer reproduces locally
    ├─ developer fixes code/config
    └─ push re-runs canonical validation

build.yml failure
    │
    ├─ image not published
    ├─ tag/release pipeline stalls
    └─ developer fixes Docker/build issue

deploy.yml failure
    │
    ├─ rollout or health checks fail
    ├─ environment remains blocked
    └─ maintainer fixes manifests/runtime issue and redeploys
```

---

## Concurrency and Cancellation

```
New push on same branch
      │
      ├─ old ci.yml run cancelled
      └─ new ci.yml run starts on latest commit

Focused guards
      │
      └─ same principle: stale in-progress runs can be cancelled

Deployments
      │
      └─ serialized per environment where safety matters
```

Intent:

- prefer the newest validation result for code-review workflows
- avoid wasting runner time on stale branches
- avoid risky deployment cancellation behavior

---

## Security Coverage Map

```
Source / package validation
    ├─ black / ruff / contract checks
    ├─ package metadata validation
    └─ installed entrypoint smoke tests

Image security
    ├─ Trivy scan
    └─ SBOM generation

Runtime security
    ├─ environment protection / approvals
    ├─ deployment auth checks
    └─ Kubernetes runtime hardening
```

---

## Practical Workflow Guidance

Use this mental model:

1. `ci.yml` answers: "Is the product valid overall?"
2. focused guards answer: "Did this sensitive area get its extra checks?"
3. `build.yml` answers: "Can we produce and inspect the release artifact?"
4. `deploy.yml` answers: "Can we safely roll the artifact out?"
5. `test.yml` / `lint.yml` answer only: "This workflow name is deprecated; use `ci.yml`."

---

## Glossary

| Term | Meaning |
|------|---------|
| **Canonical CI** | The one broad validation workflow the repo relies on by default |
| **Focused guard** | A narrow workflow triggered by changes in a sensitive area |
| **Compatibility shim** | A retained legacy workflow that exists only for migration compatibility |
| **Matrix** | A job running multiple times with different runtime inputs |
| **Artifact** | A generated output such as a package, report, or SBOM |
| **Environment** | A deployment target such as staging or production |
| **Concurrency** | A policy that controls overlapping workflow runs |
