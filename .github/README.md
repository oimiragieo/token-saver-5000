# GitHub Configuration for Token Saver 5000

> This is **not** the main product README.
>
> For what Token Saver 5000 is, how to run it locally, MCP vs non-MCP usage, and tool walkthroughs, read:
> - [../README.md](../README.md)

This directory contains GitHub-specific configuration including the complete CI/CD automation suite.

## Quick Navigation

### CI/CD Workflows
All workflows are located in `workflows/` directory:

| Workflow | Purpose | Trigger | Duration |
|----------|---------|---------|----------|
| **ci.yml** | Canonical full-product validation | Push, PR, Manual | 10-20 min |
| **test.yml** | Deprecated compatibility shim | Manual only | <1 min |
| **lint.yml** | Deprecated compatibility shim | Manual only | <1 min |
| **build.yml** | Docker image build & push | Push (main), Tags | 1-8 min |
| **deploy.yml** | Kubernetes deployments | Tags, Manual | 3-10 min |

### Documentation

Start with one of these based on your role:

**I want to...**
- [ ] **Understand the complete system** → Read [WORKFLOWS_SUMMARY.md](WORKFLOWS_SUMMARY.md)
- [ ] **Set up CI/CD workflows** → Follow [WORKFLOWS_SETUP_CHECKLIST.md](WORKFLOWS_SETUP_CHECKLIST.md)
- [ ] **Quickly reference how workflows work** → Use [WORKFLOWS_QUICK_REFERENCE.md](WORKFLOWS_QUICK_REFERENCE.md)
- [ ] **Understand the architecture** → Review [WORKFLOWS_ARCHITECTURE.md](WORKFLOWS_ARCHITECTURE.md)
- [ ] **Know details about each workflow** → Read [workflows/README.md](workflows/README.md)

## File Structure

```
.github/
├── workflows/
│   ├── ci.yml                      # Canonical product CI validation
│   ├── test.yml                    # Deprecated manual compatibility shim
│   ├── lint.yml                    # Deprecated manual compatibility shim
│   ├── build.yml                   # Docker image build & push
│   ├── deploy.yml                  # Kubernetes deployment automation
│   └── README.md                   # Detailed workflow documentation
├── README.md                       # This file (navigation guide)
├── WORKFLOWS_SUMMARY.md            # Executive summary & metrics
├── WORKFLOWS_SETUP_CHECKLIST.md    # Step-by-step implementation
├── WORKFLOWS_QUICK_REFERENCE.md    # Developer quick lookup
└── WORKFLOWS_ARCHITECTURE.md       # Technical design & diagrams
```

## At a Glance

### Workflows Overview

**ci.yml** - Canonical Product CI
- Runs on: Every push, PR, or manual dispatch
- Checks: Black, Ruff, workflow/package contracts, Python 3.10-3.12 compatibility smoke, package build, MCP installer smoke, full pytest suite
- Duration: 10-20 min depending on cache and runner state
- Notes: This is the preferred required-status workflow for broad repository validation

**test.yml / lint.yml** - Deprecated Compatibility Shims
- Run on: Manual dispatch only
- Purpose: Preserve old workflow names during migration while directing maintainers to `ci.yml`
- Notes: These workflows no longer participate in automatic CI or branch protection

**build.yml** - Docker Image Build
- Runs on: Push to main, semantic version tags, manual trigger
- Actions: Build multi-stage Docker image, push to GHCR
- Scanning: Trivy vulnerability scan, SBOM generation
- Duration: 1-2 min (cached), 5-8 min (cold)
- Caching: GitHub Actions + Docker BuildKit layers

**deploy.yml** - Kubernetes Deployment
- Runs on: Semantic version tags, manual trigger
- Targets: Staging (auto), Production (approval-gated)
- Validation: Health checks, pod readiness, error logs
- Strategy: Rolling update (zero-downtime)
- Duration: 3-5 min (staging), 5-10 min (production)

### Key Features

- **Automated Testing:** Multi-version Python matrix testing with parallel execution
- **Coverage Enforcement:** Fails if coverage drops below 70%
- **Code Quality:** Black formatting + Ruff linting + type checking
- **Security:** Bandit code scanning + Trivy image scanning + SBOM
- **Performance:** Pip caching saves 2-5 min, Docker caching 3-5 min
- **Deployments:** Zero-downtime rolling updates to Kubernetes
- **Production Safety:** Manual approval required before production deploy
- **Observability:** Codecov integration, GitHub Security tab, detailed artifacts

## Getting Started

### For Developers
1. Read: [WORKFLOWS_QUICK_REFERENCE.md](WORKFLOWS_QUICK_REFERENCE.md) (5 min read)
2. Run: `pytest tests/ -v --cov=src` (before pushing)
3. Push: Code automatically tested by GitHub Actions

### For DevOps/Platform Teams
1. Read: [WORKFLOWS_SETUP_CHECKLIST.md](WORKFLOWS_SETUP_CHECKLIST.md)
2. Configure: Secrets (kubeconfig), Environments, Branch protection
3. Test: Each workflow individually
4. Deploy: Enable in production

### For Architects/Leads
1. Read: [WORKFLOWS_SUMMARY.md](WORKFLOWS_SUMMARY.md) (executive summary)
2. Review: [WORKFLOWS_ARCHITECTURE.md](WORKFLOWS_ARCHITECTURE.md) (design)
3. Plan: Integration with existing systems

## Workflow Triggers

### On Every Push to main/develop
Core validation plus build workflows run:
```
Push → ci.yml (10-20 min) + focused guards (path-based) + build.yml (2-5 min)
```

### On Pull Request
Canonical validation and fast quality workflows run (no image push):
```
PR → ci.yml (10-20 min) + focused guards (path-based)
```

### On Semantic Version Tag (v0.6.2)
Build, staging, and optionally production:
```
Tag → build.yml (2-5 min) → deploy-staging (3-5 min) → deploy-production (manual)
```

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Workflows | 5 core + 3 focused guards |
| Total YAML Lines | 952 |
| Documentation | 5 guides + 5000+ lines |
| Test Coverage Threshold | 70% (enforced) |
| Docker Image Size Target | <500MB |
| Python Versions Tested | 3.10, 3.11, 3.12 |
| K8s Environments | 2 (staging, production) |
| Build Time (cold) | 18-29 min |
| Build Time (warm) | 10-17 min |
| Performance Gain | 2-5x speedup with caching |

## Common Commands

### Before Pushing Code
```bash
# Run tests locally (same as CI)
pytest tests/ -v --cov=src --cov-report=term

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/
```

### Creating a Release
```bash
# Create semantic version tag
git tag -a v0.6.2 -m "Release v0.6.2"

# Push tag (triggers build + staging deploy)
git push origin v0.6.2

# Approve production deployment
# (Go to GitHub Actions > deploy-production > Review deployments)
```

### Troubleshooting
```bash
# Check workflow status
gh run list --workflow=ci.yml

# View detailed logs
gh run view <run-id> --log

# Rebuild image locally
docker build -t token-saver-5000:latest .

# Check Kubernetes deployment
kubectl get deployment -n token-saver-5000
kubectl logs -n token-saver-5000 -l app=token-saver-5000
```

## Setup Required

Before workflows can run:

1. **Configure Secrets** (Settings > Secrets and variables > Actions)
   - `KUBECONFIG_STAGING` - Base64 kubeconfig for staging
   - `KUBECONFIG_PRODUCTION` - Base64 kubeconfig for production

2. **Create Environments** (Settings > Environments)
   - `staging` - Auto-deploy (no approval)
   - `production` - Requires approval

3. **Branch Protection** (Settings > Branches)
   - Require ci.yml to pass
   - Require PR approval

4. **Enable Actions** (Settings > Actions)
   - Allow GitHub-owned actions
   - Allow Actions from Marketplace

See [WORKFLOWS_SETUP_CHECKLIST.md](WORKFLOWS_SETUP_CHECKLIST.md) for detailed setup.

## Documentation Index

1. **[WORKFLOWS_SUMMARY.md](WORKFLOWS_SUMMARY.md)** (Executive Summary)
   - Complete overview and metrics
   - Architecture and technical specs
   - File structure and usage scenarios
   - Success metrics and sign-off

2. **[WORKFLOWS_SETUP_CHECKLIST.md](WORKFLOWS_SETUP_CHECKLIST.md)** (Implementation)
   - Step-by-step setup instructions
   - Secret configuration
   - Testing each workflow
   - Troubleshooting guide

3. **[WORKFLOWS_QUICK_REFERENCE.md](WORKFLOWS_QUICK_REFERENCE.md)** (Developer Guide)
   - Quick lookup tables
   - Common commands
   - Failure scenarios and fixes
   - Performance benchmarks

4. **[WORKFLOWS_ARCHITECTURE.md](WORKFLOWS_ARCHITECTURE.md)** (Technical Design)
   - ASCII diagrams and flow charts
   - Caching strategy details
   - Parallelization patterns
   - Security architecture

5. **[workflows/README.md](workflows/README.md)** (Detailed Reference)
   - Complete workflow documentation
   - Feature descriptions
   - Configuration examples
   - Best practices

## Support

**For setup issues:** Follow [WORKFLOWS_SETUP_CHECKLIST.md](WORKFLOWS_SETUP_CHECKLIST.md)

**For usage questions:** Check [WORKFLOWS_QUICK_REFERENCE.md](WORKFLOWS_QUICK_REFERENCE.md)

**For design questions:** Review [WORKFLOWS_ARCHITECTURE.md](WORKFLOWS_ARCHITECTURE.md)

**For troubleshooting:** See troubleshooting sections in documentation

**For GitHub Actions help:** https://docs.github.com/actions

## Status

- **Created:** November 27, 2025
- **Version:** 1.0
- **Validation:** All YAML syntax verified
- **Documentation:** Complete with 5 workflow guides
- **Testing:** Ready for workflow validation
- **Status:** Ready for implementation

## Next Steps

1. Read appropriate documentation based on your role
2. Follow setup checklist to configure GitHub
3. Test `ci.yml` with a push or PR; optionally trigger legacy shims manually
4. Train team on CI/CD process
5. Start using for development and releases

---

For complete implementation instructions, see [WORKFLOWS_SETUP_CHECKLIST.md](WORKFLOWS_SETUP_CHECKLIST.md).
