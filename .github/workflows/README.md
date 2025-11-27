# GitHub Actions CI/CD Workflows

This directory contains 4 production-grade GitHub Actions workflows for Token Saver 5000 with comprehensive automation, caching, and validation.

## Workflows Overview

### 1. test.yml - Continuous Integration Testing
**Purpose:** Validates code quality and test coverage on every push and pull request

**Trigger Events:**
- Push to: `main`, `develop`, `claude/**` branches
- Pull requests to: `main`, `develop`

**Key Features:**
- **Matrix Testing:** Tests across Python 3.10, 3.11, 3.12 in parallel
- **Pip Caching:** Saves 2-5 minutes per run using `hashFiles('**/requirements.txt')`
- **Coverage Enforcement:** Fails if coverage drops below 70% threshold
- **Multi-Report Support:**
  - Terminal output (human-readable)
  - XML report (Codecov integration)
  - HTML report (artifacts)
- **Setup Verification:** Runs `scripts/check_setup.py` to validate environment
- **Code Formatting:** Checks Black formatting compliance
- **Linting:** Validates with Ruff (fast Python linter)
- **Type Checking:** Optional mypy type safety checks

**Jobs:**
- `test` (matrix): Runs tests for each Python version
- `test-matrix-complete`: Validates all matrix jobs passed

**Artifacts Generated:**
- `coverage-report-py3.10`, `coverage-report-py3.11`, `coverage-report-py3.12` (HTML coverage reports)
- Coverage reports uploaded to Codecov for historical tracking

**Performance:**
- Typical run time: 8-12 minutes (3 Python versions in parallel)
- Pip cache hit: Saves ~2-3 minutes per run on cache hit

---

### 2. lint.yml - Code Quality & Security
**Purpose:** Enforces code style, security standards, and maintainability on every push

**Trigger Events:**
- Push to: `main`, `develop`, `claude/**` branches
- Pull requests to: `main`, `develop`

**Key Features:**
- **Black Formatting:** Strict PEP 8 compliance checking with diff output
- **Ruff Linting:** 10-100x faster Python linting with 500+ rules
- **Format Verification:** Additional formatting checks using Ruff formatter
- **Type Safety:** Optional mypy type checking (informational)
- **Security Scanning:**
  - Bandit: Scans for common Python security vulnerabilities
  - Identifies hardcoded passwords, SQL injection risks, etc.
- **Documentation Check:** pydocstyle validates docstring coverage (Google style)
- **Import Sorting:** isort ensures consistent import organization
- **Complexity Analysis:**
  - Radon McCabe complexity score
  - Maintainability index calculation
- **Concurrency Control:** Cancels previous runs if new commits pushed

**Jobs:**
- `lint`: All linting and security checks in single job

**Artifacts Generated:**
- `bandit-security-report`: JSON report of security findings
- `complexity-reports`: Radon complexity and maintainability indices

**Fail Conditions (Blocking):**
- Black formatting check fails
- Ruff lint check fails
- Ruff format check fails

**Informational Only (Non-blocking):**
- Type checking warnings
- Security scan results
- Docstring coverage gaps
- Import sorting issues
- Code complexity metrics

**Performance:**
- Typical run time: 2-4 minutes
- Single Python version (3.12 latest)

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
   └─ test.yml: Runs tests (5-8 min)
   └─ lint.yml: Checks code quality (2-3 min)

2. Developer creates pull request
   └─ All workflows run automatically
   └─ Blocking checks (test, lint) must pass

3. After merge to main
   └─ test.yml: Final validation
   └─ lint.yml: Final code quality check
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

- `test-${{ github.ref }}`: One test run per branch
- `lint-${{ github.ref }}`: One lint run per branch
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

- Test matrix: 3 Python versions run in parallel (3x speedup)
- Lint job: Single serial job (fast enough)
- Build job: Single serial job (caching makes it fast)
- Deploy jobs: Staging runs first, production waits for approval

---

## Best Practices

1. **Commit Messages:** Use conventional commits (feat:, fix:, docs:)
2. **Pull Requests:** Reference issues and include testing notes
3. **Testing:** Add tests for new features before merging
4. **Coverage:** Maintain >70% code coverage (enforced by test.yml)
5. **Tagging:** Use semantic versioning for releases (v1.0.0, v0.6.1)
6. **Secrets:** Never hardcode secrets; use GitHub Secrets
7. **Docker:** Use multi-stage builds for smaller images
8. **Kubernetes:** Always include resource limits and probes

---

## File Locations

```
.github/
└── workflows/
    ├── test.yml      (139 lines) - CI test automation
    ├── lint.yml      (170 lines) - Code quality enforcement
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
