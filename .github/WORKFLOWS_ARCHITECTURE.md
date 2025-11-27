# GitHub Actions Workflows - Architecture & Flow Diagrams

This document visualizes the CI/CD workflow architecture, trigger events, and execution flow.

---

## Overall CI/CD Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Git Repository Events                         │
└───────────────────┬──────────────────┬──────────────────────────┘
                    │                  │
         ┌──────────▼────────┐   ┌─────▼────────────┐
         │ Push to main/      │   │  Create Tag v*   │
         │ develop/claude/**  │   │  (Semantic Ver)  │
         └──────────┬────────┘   └─────┬────────────┘
                    │                   │
        ┌───────────┴───────────┐       │
        │                       │       │
        ▼                       ▼       ▼
    ┌───────────┐          ┌────────────────┐
    │ test.yml  │          │  build.yml     │
    │ lint.yml  │          │ (Docker build) │
    │ (CI)      │          └────────┬───────┘
    └───────────┘                   │
                        ┌───────────┴──────────┐
                        │                      │
                        ▼                      ▼
                    ┌─────────────┐    ┌──────────────────┐
                    │  Push Image │    │ deploy-staging   │
                    │ to Registry │    │ (Auto-deploy)    │
                    └─────────────┘    └────────┬─────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │ Deploy-production   │
                                    │ (Requires approval) │
                                    └─────────────────────┘
```

---

## Trigger Event Matrix

```
┌──────────────────┬───────────────────┬──────────┬──────────┐
│ Event            │ test.yml          │ lint.yml │ build    │
├──────────────────┼───────────────────┼──────────┼──────────┤
│ Push main        │ RUN               │ RUN      │ RUN      │
│ Push develop     │ RUN               │ RUN      │ RUN      │
│ Push claude/**   │ RUN               │ RUN      │ SKIP     │
│ Push feature/*   │ RUN (no push)     │ RUN      │ SKIP     │
│ Pull Request     │ RUN (required)    │ RUN      │ SKIP     │
│ Tag v*           │ SKIP              │ SKIP     │ RUN      │
│ Workflow Disp.   │ Optional          │ Optional │ OPTIONAL │
└──────────────────┴───────────────────┴──────────┴──────────┘
```

---

## Workflow Dependency Graph

```
Feature Branch
     │
     ├─────────────────────────────────────┐
     │                                     │
     ▼                                     ▼
test.yml (8-12 min)                  lint.yml (2-4 min)
     │                                     │
     ├─────── REQUIRED PASS ───────────────┤
     │                                     │
     └──────────┬──────────────────────────┘
                │
                ▼
         Pull Request Checks
                │
         ┌──────┴─────────┐
         │                │
      PASS            FAIL → Fix locally
         │                │  & push again
         │                │
         ▼                │
  Review & Approve        │
         │                │
         ▼                │
     Merge to main ◄──────┘
         │
         ├─────────────────────────────────┐
         │                                 │
         ▼                                 ▼
    build.yml (1-8 min)             All status checks pass
    - Docker build                   (push to main)
    - Push to registry
    - Trivy scan
    - SBOM generation
         │
         ├────────────────┬────────────────┐
         │                │                │
         ▼                ▼                ▼
    Test (PR)    staging (tag)    production (tag)
    - Local OK   - Auto-deploy     - Approval needed
    - Skip push  - Health checks    - Auto-deploy when
                                      approved
```

---

## Python Test Matrix Parallelization

```
test.yml Job Execution (Runs in Parallel)
┌────────────────────────────────────────────────────────┐
│  Matrix: python-version = [3.10, 3.11, 3.12]          │
└────────────────────────────────────────────────────────┘

     Parallel Worker 1            Parallel Worker 2            Parallel Worker 3
     ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
     │ Python 3.10      │         │ Python 3.11      │         │ Python 3.12      │
     ├──────────────────┤         ├──────────────────┤         ├──────────────────┤
     │ 1. Checkout      │         │ 1. Checkout      │         │ 1. Checkout      │
     │ 2. Setup Python  │         │ 2. Setup Python  │         │ 2. Setup Python  │
     │ 3. Install deps  │         │ 3. Install deps  │         │ 3. Install deps  │
     │ 4. Lint          │         │ 4. Lint          │         │ 4. Lint          │
     │ 5. Tests         │         │ 5. Tests         │         │ 5. Tests         │
     │ 6. Coverage      │         │ 6. Coverage      │         │ 6. Coverage      │
     │ 7. Upload        │         │ 7. Upload        │         │ 7. Upload        │
     │                  │         │                  │         │                  │
     │ Duration: 3-4min │         │ Duration: 3-4min │         │ Duration: 3-4min │
     │ Total: ~3-4 min  │         │ (All in parallel)│         │ (Not 12 min)     │
     └──────────────────┘         └──────────────────┘         └──────────────────┘
            │                              │                             │
            └──────────────┬───────────────┴─────────────┬───────────────┘
                          │
                    All workers complete
                          │
                          ▼
              test-matrix-complete (1 sec)
                          │
                          ▼
                 Report overall status
```

---

## Caching Strategy

```
Workflow Run 1 (Cold Cache)
┌─────────────────────────────────────────┐
│ Install dependencies                    │
│ - Download all pip packages (5-10 min)  │
│ - No cache hit (first run)              │
└─────────────────────────────────────────┘
           │
           ▼
   Store cache (hashFiles requirements.txt)
           │
           ▼
   Cache saved for future runs

Workflow Run 2 (Warm Cache)
┌─────────────────────────────────────────┐
│ Install dependencies                    │
│ - Load from GHA cache (30 sec)          │
│ - No pip downloads needed               │
└─────────────────────────────────────────┘
           │
           ▼
   Saves 2-5 minutes per run!

Cache Invalidation
┌─────────────────────────────────────────┐
│ requirements.txt changes                │
│ - Hash changes                          │
│ - Cache invalidated                     │
│ - New download on next run              │
└─────────────────────────────────────────┘
```

---

## Docker Image Build Caching

```
Dockerfile Multi-Stage Build
┌──────────────────────────┐
│ Stage 1: Builder         │
├──────────────────────────┤
│ FROM python:3.12-slim    │
│ Install build tools      │
│ Copy requirements.txt    │
│ pip install all deps     │
│ Download ML models      │ ◄─── Layer Cache
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Stage 2: Runtime         │
├──────────────────────────┤
│ FROM python:3.12-slim    │
│ COPY /opt/venv           │◄─── Use builder cache
│ COPY src/                │
│ COPY models              │
│ Set environment vars     │
└────────────┬─────────────┘
             │
             ▼
        Final Image
        ~450MB size

Build Times:
First build:     8 minutes
Cached build:    1-2 minutes (2-4x faster)
CI rebuild:      2-4 minutes (GHA cache)
```

---

## Kubernetes Deployment Flow

```
Deploy Staging (Automatic on Tags)
────────────────────────────────────

Tag v0.6.1 pushed
       │
       ▼
  build.yml completes
       │
       ▼
deploy-staging triggered automatically
       │
       ├─ Auth to K8s cluster
       │
       ├─ Build manifests with kustomize
       │
       ├─ Patch image tag in YAML
       │
       ├─ Apply manifests (kubectl apply)
       │
       ├─ Wait for rollout (rolling update)
       │     - New pod starts
       │     - Liveness probe (30s initial, 10s period)
       │     - Readiness probe (10s initial, 10s period)
       │     - Service traffic switched
       │
       ├─ Verify pod status
       │
       ├─ Run health checks (HTTP endpoints)
       │
       ├─ Check logs for errors
       │
       └─ Report deployment status
           │
           ▼
       Staging LIVE


Deploy Production (Approval Required)
────────────────────────────────────

deploy-staging completes
       │
       ▼
deploy-production awaits approval
       │
       ├─ GitHub shows "Review deployments"
       │
       ├─ Reviewers (1-2) notified
       │
       └─ Approval action taken
           │
           ├─ "Approve and deploy"
           │
           ▼
       deploy-production triggered
           │
           ├─ Same K8s steps as staging
           │  - Auth
           │  - Build manifests
           │  - Apply with rolling update
           │  - Health checks
           │  - Verify
           │
           └─ Production LIVE
               (May have auto-rollback on failure)
```

---

## Health Check Sequence

```
Pod Startup and Readiness
─────────────────────────

0s     Container starts
       Startup Probe begins
       (httpGet /health/liveness)
       initialDelaySeconds: 0
       periodSeconds: 10
       timeoutSeconds: 5
       failureThreshold: 12

0-120s Startup probe checks (up to 12 * 10 = 120s)
       │
       ├─ Probe 1: Request timeout → retry
       ├─ Probe 2: 500 error → retry
       ├─ Probe 3: Connection refused → retry
       │  ...
       └─ Probe N: 200 OK ✓
          │
          ├─ Container marked as "started"
          │
          ├─ Liveness probe begins
          │  (httpGet /health/liveness)
          │  initialDelaySeconds: 30
          │  periodSeconds: 10
          │
          └─ Readiness probe begins
             (httpGet /health/readiness)
             initialDelaySeconds: 10
             periodSeconds: 10

30-∞s  Liveness probe (every 10s)
       - Ensures container is running
       - Restarts on 3 consecutive failures (30s)

10-∞s  Readiness probe (every 10s)
       - Ensures container ready for traffic
       - Removes from service on 3 failures
       - Re-adds when succeeds

Result: Pod ready for traffic ✓
```

---

## Error Recovery Flow

```
Test Failure
     │
     ▼
Workflow blocked
     │
     ├─ GitHub shows failed check
     │
     ├─ Developer notified
     │
     ├─ Cannot merge to main
     │
     └─ Developer action:
        │
        ├─ Pull latest code
        │
        ├─ Run tests locally
        │  (pytest tests/ --cov=src)
        │
        ├─ Fix failing tests
        │
        ├─ Verify locally
        │  (All tests pass, coverage OK)
        │
        ├─ Commit and push
        │
        └─ GitHub re-runs test.yml
           │
           └─ If passes → workflow unblocked


Build Failure
     │
     ▼
Image not pushed
     │
     ├─ Staging deployment blocked
     │
     └─ Developer action:
        │
        ├─ Check build.yml logs
        │
        ├─ Identify error:
        │  - Dockerfile syntax error
        │  - Missing dependency
        │  - Model download failed
        │
        ├─ Test locally:
        │  docker build -t test:latest .
        │
        ├─ Fix issue
        │
        ├─ Commit and push
        │
        └─ build.yml re-runs
           │
           └─ If succeeds → Image pushed


Deployment Failure
     │
     ▼
Pod doesn't start
     │
     ├─ Rollout status shows failure
     │
     └─ Automatic retry? OR Manual action:
        │
        ├─ Check pod logs
        │  kubectl logs -n token-saver-5000 pod-name
        │
        ├─ Identify issue:
        │  - Image not found
        │  - Port conflict
        │  - Missing ConfigMap
        │
        ├─ Fix K8s manifests
        │
        ├─ Commit and push
        │
        ├─ Create new tag or manual dispatch
        │
        └─ deploy.yml re-runs
           │
           └─ If succeeds → Deployment complete
```

---

## Concurrency and Cancellation

```
Feature Branch: feature/auth
Push 1: Add auth module
   │
   ├─ test.yml starts (concurrency: test-feature/auth)
   │
Push 2: Fix auth bug (3 minutes later)
   │
   ├─ Previous test.yml CANCELLED (in-progress)
   │
   └─ New test.yml starts with latest code
      (Only one test.yml per branch)


Main Branch: main
Push 1: Release v0.6.1 tag
   │
   ├─ build.yml starts (concurrency: build-refs/tags/v0.6.1)
   │
   ├─ deploy-staging starts (concurrency: deploy-staging)
   │
Push 2: Hotfix pushed
   │
   ├─ New build.yml starts
   │
   ├─ Previous deploy-staging NOT cancelled
   │  (Different concurrency group)
   │
   ├─ New deploy-staging waits for previous to complete
   │  (Only one deploy-staging at a time)
   │
   └─ Then starts with new code


Production Deployment: deploy-production
   │
   ├─ No cancellation (concurrency: cancel-in-progress: false)
   │  Deployments are too risky to cancel
   │
   └─ Queue waits for current deployment to complete
```

---

## Performance Metrics

```
Typical CI/CD Run (Cold Cache vs Warm Cache)

                                Time (min)
Task                     First Run    Cached Run    Speedup
─────────────────────────────────────────────────────────────
test.yml setup + tests      5-8          2-3         2-3x
lint.yml                    2-3          1-2         2x
build.yml                   5-8          1-2         5x
deploy-staging              3-5          3-5         1x
─────────────────────────────────────────────────────────────
Total CI                    7-12         3-5         2.5x
Total with deploy          10-17         6-10        1.7x

Caching Impact:
- Pip cache: Saves ~3-5 minutes (largest impact)
- Docker layer cache: Saves ~3-5 minutes
- Combined: 6-10 minute reduction per run

For continuous deployment:
- First deployment: 25-30 minutes (cold)
- Typical deployment: 17-20 minutes (warm cache)
```

---

## Security Flow

```
Code Security (lint.yml)
     │
     ├─ Bandit scan
     │  - Hardcoded passwords
     │  - SQL injection risks
     │  - Insecure hashing
     │
     └─ Results in artifacts (non-blocking)

Image Security (build.yml)
     │
     ├─ Trivy vulnerability scan
     │  - CVE detection in dependencies
     │  - OS package vulnerabilities
     │
     ├─ SBOM generation
     │  - Software bill of materials
     │  - Dependency tracking
     │
     └─ Results in GitHub Security tab

Deployment Security (deploy.yml)
     │
     ├─ RBAC authentication
     │  - Service account verification
     │
     ├─ Pod security
     │  - Non-root user (uid: 1000)
     │  - Read-only filesystem
     │  - Drop all capabilities
     │
     └─ Network policies enforced
        - Pod-to-pod communication limited

Overall: Defense in depth at code, image, and runtime levels
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **Concurrency** | Prevent multiple jobs from running simultaneously |
| **Matrix** | Run job multiple times with different inputs |
| **Cache** | Store build artifacts to speed up future runs |
| **Artifact** | File output from workflow (logs, reports, etc.) |
| **Environment** | Deployment target (staging, production) |
| **Secret** | Encrypted variable for sensitive data |
| **Step** | Individual action in a workflow job |
| **Job** | Set of steps that run on a runner |
| **Workflow** | Automation triggered by events |
| **Runner** | Virtual machine that executes workflows |
| **Action** | Reusable workflow unit (uses/docker/script) |
| **YAML** | Configuration language for workflows |

---

## Architecture Principles

1. **Automation First:** Minimize manual steps
2. **Fast Feedback:** Tests run in parallel (3-5 min vs 10-12 min)
3. **Defense in Depth:** Multiple validation layers
4. **Clear Boundaries:** Each workflow has single responsibility
5. **Observable:** Detailed logs and reports
6. **Repeatable:** Same result every time (no manual tweaks)
7. **Safe:** Production requires approval
8. **Efficient:** Caching and parallel execution
