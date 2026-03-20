# GitHub Actions CI/CD Workflows

This directory contains 8 GitHub Actions workflows for Token Saver 5000: canonical CI, focused guards, release automation, and legacy compatibility shims.

## Workflows Overview

### 0. ci.yml - Canonical Product CI
**Purpose:** Runs the canonical repository-wide validation pipeline used to check the full product surface.

**Trigger Events:**
- Push to: `main`, `develop`, `claude/**` branches
- Pull requests to: `main`, `develop`
- Manual: `workflow_dispatch`

**Key Features:**
- **Fast Quality Gate:** Runs Black and Ruff using the same repo-wide commands validated locally
- **Workflow/Packaging Contracts:** Runs `tests/test_ci_workflows.py` and `tests/test_mcp_packaging.py` early
- **Python Compatibility Matrix:** Verifies install/import + smoke tests on Python 3.10, 3.11, and 3.12
- **Package Validation:** Builds sdist/wheel, runs `twine check`, reinstalls the wheel, and smoke-tests the installed `token-saver-install-mcp --print-config` entrypoint via Python's scripts directory for runner-stable PATH handling
- **Full Validation:** Runs `python -m pytest tests/ -q --no-cov --ignore=tests/test_performance.py`
- **Deterministic by Design:** Avoids `scripts/check_setup.py` and Docker model downloads in CI because they depend on optional packages or network availability

**Jobs:**
- `quality-gate`
- `compatibility`
- `package-validation`
- `full-validation`

**Performance:**
- Typical run time: 10-20 minutes depending on cache warmth
- Uses pip cache keyed from `requirements.txt` and `pyproject.toml`

---

### 1. test.yml - Deprecated Compatibility Shim
**Purpose:** Preserves the legacy workflow name while redirecting maintainers to `ci.yml`.

**Trigger Events:**
- Manual: `workflow_dispatch`

**Key Features:**
- Manual-only compatibility workflow
- Writes a deprecation notice to the GitHub step summary
- Directs maintainers to `ci.yml` for real repository validation

**Jobs:**
- `legacy-test-notice`

**Artifacts Generated:**
- None

**Performance:**
- Typical run time: <1 minute

---

### 2. lint.yml - Deprecated Compatibility Shim
**Purpose:** Preserves the legacy workflow name while redirecting maintainers to `ci.yml`.

**Trigger Events:**
- Manual: `workflow_dispatch`

**Key Features:**
- Manual-only compatibility workflow
- Writes a deprecation notice to the GitHub step summary
- Directs maintainers to `ci.yml` for real repository validation

**Jobs:**
- `legacy-lint-notice`

**Artifacts Generated:**
- None

**Performance:**
- Typical run time: <1 minute

---

### 3. build.yml - Docker Image Build & Push
**Purpose:** Builds Docker images with multi-stage caching and pushes to container registry

**Trigger Events:**
- Push to: `main`, `develop` branches
- Tags: Semantic version tags (`v*`)
- Manual: workflow_dispatch for testing

**Key Features:**
- **Multi-Stage Build:** Separates builder and runtime stages for smaller images
- **BuildKit Caching:**
  - GitHub Actions cache layer (type=gha)
  - Registry cache layer (type=registry)
  - Saves 2-5 minutes on cache hit
- **Image Tagging Strategy:**
  - `latest`: On default branch
  - Branch name: `develop`, `main`
  - Semantic version: `v0.6.1`, `v0.6.1-beta`
  - Short SHA: `main-a1b2c3d`
  - Date stamp: `2025-11-27`
- **Container Registry:** GitHub Container Registry (GHCR)
- **Image Scanning:**
  - Trivy vulnerability scanning (CVE detection)
  - SARIF format for GitHub Security tab
  - Non-blocking (informational only)
- **SBOM Generation:**
  - Software Bill of Materials in SPDX JSON format
  - Useful for compliance and vulnerability tracking
- **Smoke Testing:** Basic image validation on PRs (local build only)
- **Size Reporting:** Tracks Docker image size (target: <500MB)

**Jobs:**
- `build`: Single job handling all build, push, scan, and report tasks

**Outputs:**
- Docker image pushed to: `ghcr.io/username/token-saver-5000:TAG`
- Artifacts:
  - `sbom`: Software Bill of Materials (SPDX)
  - Trivy scan report (GitHub Security tab)

**Pull Request Behavior:**
- Builds image locally (no push to registry)
- Runs smoke test to verify image builds
- Skips Trivy scan (not needed for PR testing)

**Performance:**
- First build: 5-8 minutes (downloads all dependencies)
- Cached build: 1-2 minutes (GHA cache layer)

---

### 4. deploy.yml - Kubernetes Deployment
**Purpose:** Deploys to Kubernetes clusters (staging and production) with zero-downtime updates

**Trigger Events:**
- Tags: Semantic version tags (`v*`)
- Manual: workflow_dispatch with custom inputs
  - Environment selection: `staging` or `production`
  - Namespace: Configurable (default: `token-saver-5000`)
  - Image tag: Custom image tag to deploy

**Key Features:**
- **Dual Environment Support:**
  - Staging: Automatic deployment on tags
  - Production: Manual trigger (requires approval)
- **Kustomize Integration:**
  - Environment-specific manifests
  - Image tag patching
  - ConfigMap/Secret injection
  - Validation (syntax checking)
- **Zero-Downtime Deployment:**
  - Rolling update strategy (maxSurge: 1, maxUnavailable: 0)
  - Health checks before marking ready
  - Pod anti-affinity for HA
- **Comprehensive Validation:**
  - Kubernetes cluster access verification
  - Deployment status monitoring
  - Pod readiness checks
  - Health endpoint validation
  - Log analysis for errors
- **Smoke Tests:** Basic functionality tests after deployment
- **Security:**
  - Non-root user deployment (uid: 1000)
  - Read-only root filesystem
  - Security context enforcement
- **High Availability:**
  - Pod anti-affinity across nodes
  - Liveness/readiness/startup probes
  - Resource limits (memory: 1-2GB, CPU: 0.5-2 cores)
- **Concurrency Control:**
  - Only one deployment per environment at a time
  - Prevents concurrent deployments (too risky)
- **Observability:**
  - Prometheus metrics scraping
  - OpenTelemetry trace collection
  - Detailed deployment reports
- **Rollback Instructions:**
  - Auto-generated on failure
  - Manual rollback procedure documented

**Jobs:**
- `deploy-staging`: Deploys to staging cluster
  - Automatic on semantic version tags
  - Manual via workflow_dispatch
- `deploy-production`: Deploys to production cluster
  - Requires approval (environment protection)
  - Only after staging deployment succeeds
  - Only on semantic version tags or manual dispatch

**Environment URLs:**
- Staging: `https://staging.token-saver-5000.example.com`
- Production: `https://api.token-saver-5000.example.com`

**Secrets Required:**
- `KUBECONFIG_STAGING`: Base64-encoded kubeconfig for staging cluster
- `KUBECONFIG_PRODUCTION`: Base64-encoded kubeconfig for production cluster

**Health Checks Performed:**
1. Kubernetes cluster connectivity
2. Deployment authorization
3. Pod startup (startup probe: 120s timeout)
4. Pod readiness (readiness probe: 30s initial, 10s period)
5. Liveness (liveness probe: 30s initial, 10s period)
6. Service endpoint availability
7. Metrics endpoint validation
8. Error log analysis
9. Smoke tests (basic functionality)

**Performance:**
- Staging deployment: 3-5 minutes (including health checks)
- Production deployment: 5-10 minutes (longer validation)

---

### 5. skill-ci.yml - Skill Script Fast Checks
**Purpose:** Fast feedback loop for Claude skill scripts and help metadata.

**Trigger Events:**
- Push/PR changes scoped to:
  - `scripts/skills/**`
  - `skills/**`
  - `tests/test_skill_scripts.py`
  - `tests/test_help_handlers.py`
  - `src/handlers/help_handlers.py`
  - workflow file itself

**Key Features:**
- Python 3.12 setup with pip cache
- Black check scoped to skill-related files
- Ruff check scoped to skill-related files
- Fast test run:
  - `tests/test_skill_scripts.py`
  - `tests/test_help_handlers.py`

**Performance:**
- Typical run time: 2-4 minutes
- Runs only when skill-related files change

---

### 6. benchmark-guard.yml - Benchmark Regression Guard
**Purpose:** Prevent token-savings regressions in benchmark-sensitive changes.

**Trigger Events:**
- Push/PR changes scoped to:
  - `src/semantic_compressor.py`
  - `src/benchmark_harness.py`
  - `scripts/benchmarks/**`
  - `tests/test_benchmark_harness.py`
  - `tests/fixtures/benchmark_corpus.json`
  - workflow file itself

**Key Features:**
- Python 3.12 setup with pip cache
- Scope-limited Black + Ruff checks for benchmark files
- Benchmark harness test run
- Regression gate command:
  - `python scripts/benchmarks/run_benchmarks.py --compare baseline,query_guided,evidence_aware`
- Uploads benchmark JSON artifacts for inspection

**Performance:**
- Typical run time: 3-6 minutes
- Runs only on benchmark-related file changes

---

## Configuration & Setup

### Secrets Configuration

Add these to GitHub repository settings (Settings > Secrets and variables > Actions):

```
KUBECONFIG_STAGING - Base64-encoded kubeconfig for staging
KUBECONFIG_PRODUCTION - Base64-encoded kubeconfig for production
```

To generate base64 kubeconfig:
```bash
cat ~/.kube/config | base64 -w 0 | pbcopy  # macOS
cat ~/.kube/config | base64 -w 0 | xclip  # Linux
```

### Environment Configuration

For production deployments, add GitHub environments (Settings > Environments):

**Staging Environment:**
- Deployment branches: `main`
- Required reviewers: 0 (automatic)

**Production Environment:**
- Deployment branches: `main`
- Required reviewers: 1-2 (add your team members)
- Prevent self-review: Checked

---

## Workflow Execution Timeline

### Typical Development Flow

```
1. Developer pushes code to feature branch
   └─ ci.yml: Runs the canonical validation pipeline
   └─ focused guards: Run when matching paths change

2. Developer creates pull request
   └─ ci.yml runs automatically
   └─ Blocking checks should be based on ci.yml and any chosen focused guards

3. After merge to main
   └─ ci.yml: Final validation
   └─ build.yml: Builds Docker image (2-5 min)

4. Create semantic version tag (v0.6.2)
   └─ build.yml: Builds and pushes image
   └─ deploy.yml[staging]: Auto-deploys to staging (3-5 min)
   └─ deploy.yml[production]: Awaits approval
   └─ (Reviewer approves deployment)
   └─ deploy.yml[production]: Deploys to production (5-10 min)
```

### Concurrent Workflow Runs

Workflows use concurrency groups to prevent duplicate runs:

- `ci-${{ github.workflow }}-${{ github.ref }}`: One canonical CI run per branch/workflow
- `build-${{ github.ref }}`: One build run per branch
- `deploy-staging`, `deploy-production`: One deployment at a time

---

## Troubleshooting

### Test Failures

1. Check the test output in GitHub Actions logs
2. Coverage threshold: Ensure coverage >= 70%
3. Black formatting: Run `black src/ tests/` locally
4. Ruff linting: Run `ruff check src/ tests/`

### Lint Failures

1. Format with Black: `black src/ tests/`
2. Fix Ruff issues: `ruff check --fix src/ tests/`
3. Check imports: `isort src/ tests/`

### Build Failures

1. Check Docker build logs
2. Verify Dockerfile syntax
3. Ensure base image is available
4. Check pip dependencies in requirements.txt

### Deployment Failures

1. Verify kubeconfig secrets are configured
2. Check cluster connectivity: `kubectl cluster-info`
3. Verify RBAC permissions
4. Check pod logs: `kubectl logs -n token-saver-5000 -l app=token-saver-5000`
5. Check deployment status: `kubectl describe deployment token-saver-5000 -n token-saver-5000`

### Image Not Found in Deploy

1. Ensure build.yml completed successfully
2. Verify image tag matches deployment
3. Check registry credentials in kubeconfig
4. Verify imagePullPolicy in deployment.yaml

---

## Performance Optimization

### Pip Caching

The workflows use GitHub Actions pip caching:
- Dependency cache: Cached by `hashFiles('**/requirements.txt')`
- Saves 2-5 minutes per run on cache hit
- Cache is per-branch and per-Python-version

### Docker Build Caching

Build.yml uses multi-layer Docker caching:
- GitHub Actions cache (type=gha): Saves intermediate layers
- BuildKit cache: Leverages Docker daemon cache
- Multi-stage Dockerfile: Separates builder and runtime

### Parallel Job Execution

- Compatibility matrix: 3 Python versions run in parallel
- Quality gate and package validation stay separate for faster failure visibility
- Build job: Single serial job (caching makes it fast)
- Deploy jobs: Staging runs first, production waits for approval

---

## Best Practices

1. **Commit Messages:** Use conventional commits (feat:, fix:, docs:)
2. **Pull Requests:** Reference issues and include testing notes
3. **Testing:** Add tests for new features before merging
4. **Coverage:** Maintain >70% code coverage through the canonical CI path and local validation
5. **Tagging:** Use semantic versioning for releases (v1.0.0, v0.6.1)
6. **Secrets:** Never hardcode secrets; use GitHub Secrets
7. **Docker:** Use multi-stage builds for smaller images
8. **Kubernetes:** Always include resource limits and probes

---

## File Locations

```
.github/
└── workflows/
    ├── ci.yml        (canonical validation workflow)
    ├── test.yml      (deprecated manual compatibility shim)
    ├── lint.yml      (deprecated manual compatibility shim)
    ├── build.yml     (216 lines) - Docker image building
    ├── deploy.yml    (427 lines) - K8s deployment
    └── README.md     (This file) - Workflow documentation
```

---

## Integration Points

### With CI/CD Tools

- **Codecov:** Coverage reports uploaded for trend tracking
- **GitHub Security:** Trivy scan results in Security tab
- **GitHub Environments:** Production deployment approval

### With Repository

- **Branches:** main, develop, claude/*, feature/*, fix/*
- **Tags:** Semantic versioning (v*)
- **Secrets:** KUBECONFIG_STAGING, KUBECONFIG_PRODUCTION
- **Environments:** staging, production

---

## Future Enhancements

1. **Semantic Release:** Auto-generate release notes and tags
2. **SBOM Export:** GitHub releases integration
3. **Performance Benchmarks:** Compare before/after metrics
4. **Slack Notifications:** Deployment status to Slack
5. **ArgoCD Integration:** GitOps-style deployments
6. **Multi-Region Deployments:** Deploy to multiple clusters
7. **Canary Deployments:** Gradual rollout with traffic splitting
8. **Automated Rollback:** Auto-rollback on health check failures

---

## Support & Questions

For issues or questions about these workflows:
1. Check the GitHub Actions logs for detailed error messages
2. Review this README for troubleshooting tips
3. Check workflow syntax: `yamllint .github/workflows/*.yml`
4. Consult GitHub Actions documentation: https://docs.github.com/actions
