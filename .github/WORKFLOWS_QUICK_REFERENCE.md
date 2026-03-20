# GitHub Actions Workflows - Quick Reference

## At a Glance

| Workflow | Trigger | Duration | Key Checks |
|----------|---------|----------|-----------|
| **ci.yml** | Push, PR, Manual | 10-20 min | Quality gate, package validation, compatibility, full pytest |
| **test.yml** | Manual | <1 min | Deprecated compatibility notice |
| **lint.yml** | Manual | <1 min | Deprecated compatibility notice |
| **build.yml** | Push, Tag, Manual | 1-8 min | Docker build, Image scan (Trivy), SBOM generation |
| **deploy.yml** | Tag, Manual | 3-10 min | Staging auto, Production approval, Health checks |

---

## When Each Workflow Runs

### On Every Push to `main` / `develop`
1. ci.yml (10-20 min) - Canonical product validation
2. Focused guard workflows - Path-scoped specialist checks
3. build.yml (1-8 min) - Docker build & push
4. GitHub should treat `ci.yml` as the primary required check

### On Pull Request
1. ci.yml - Required canonical validation
2. Focused guard workflows - Optional/required based on changed paths
3. build.yml skipped (PR don't push to registry)

### On Semantic Version Tag (`v0.6.1`)
1. build.yml - Builds image with version tag
2. deploy.yml[staging] - Auto-deploys to staging
3. deploy.yml[production] - Awaits approval > deploys

---

## Local Commands (Before Pushing)

```bash
# Run all tests locally (same as CI)
pytest tests/ -v --cov=src --cov-report=term

# Format code with Black
black src/ tests/ examples/

# Check formatting without changes
black --check src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type check with mypy
mypy src/ --ignore-missing-imports

# Run setup verification
python scripts/check_setup.py

# Build Docker image locally
docker build -t token-saver-5000:latest .
```

---

## Failing Workflow? What to Do

### Test Failures
```bash
# Run tests locally first
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=term

# Fix coverage: add missing tests or improve code
```

### Lint Failures
```bash
# Fix all auto-fixable issues
black src/ tests/
ruff check --fix src/ tests/
isort src/ tests/

# Check what's left
ruff check src/ tests/
```

### Build Failures
```bash
# Test locally
docker build -t token-saver-5000:test .

# Check Dockerfile syntax
docker build --progress=plain -t token-saver-5000:test .

# Check requirements.txt
pip install -r requirements.txt
```

### Deployment Failures
```bash
# Check kubeconfig is configured
echo $KUBECONFIG

# Verify cluster access
kubectl cluster-info

# Check pod status
kubectl get pods -n token-saver-5000

# Check logs
kubectl logs -n token-saver-5000 -l app=token-saver-5000 --tail=50
```

---

## How to Create a Release

```bash
# 1. Update version in code (if needed)
# 2. Create semantic version tag
git tag -a v0.6.2 -m "Release v0.6.2: Feature X"

# 3. Push tag (triggers build.yml + deploy.yml)
git push origin v0.6.2

# 4. Watch workflows
# - build.yml: Builds and pushes image to ghcr.io
# - deploy.yml[staging]: Auto-deploys to staging
# - deploy.yml[production]: Waits for approval

# 5. Approve production deployment in GitHub Actions UI
```

---

## GitHub Actions Artifacts & Reports

### Coverage Reports
- **Where:** GitHub Actions > ci.yml
- **Access:** Review logs and package-validation outputs from the canonical CI run

### Security Reports
- **Where:** GitHub Security > Code scanning
- **From:** Trivy scan in build.yml
- **View:** Settings > Security tab

### SBOM (Software Bill of Materials)
- **Where:** GitHub Actions > build.yml > Artifacts
- **File:** sbom.spdx.json
- **Use:** Compliance, vulnerability tracking

### Complexity Reports
- **Where:** Legacy `lint.yml` is now manual-only and does not produce canonical reports
- **Use instead:** Review `ci.yml` output plus any focused guard workflow artifacts

---

## Blocked by Failing Check?

### Test Check Failing
**Root cause:** Tests fail or coverage < 70%
**Fix:** Run `pytest tests/ --cov=src --cov-report=term` locally
**Prevention:** Add tests for new code

### Lint Check Failing
**Root cause:** Black or Ruff violations
**Fix:** Run `black src/ tests/` and `ruff check --fix src/`
**Prevention:** Install pre-commit hook locally

### Build Check Failing
**Root cause:** Docker build error
**Fix:** Run `docker build .` locally to debug
**Prevention:** Test Docker builds locally before pushing

### Deploy Check Failing
**Root cause:** Kubernetes deployment error
**Fix:** Check `kubectl logs` and `kubectl describe deployment`
**Prevention:** Test manifests with `kubectl apply --dry-run`

---

## Deployment Flow for Team

### Developer
1. Feature branch → push code
2. GitHub Actions runs ci.yml automatically
3. Fix any issues until all checks pass
4. Create pull request
5. Get code review
6. Merge to main

### CI/CD Pipeline
1. Push to main → ci.yml, focused guards, and build.yml run
2. Image built and pushed to ghcr.io
3. Tag created (v0.6.2)
4. Tag push → build.yml, deploy-staging, deploy-production

### Release Manager / Approver
1. PR merged to main
2. Create semantic version tag: `git tag -a v0.6.2 -m "Release 0.6.2"`
3. Push tag: `git push origin v0.6.2`
4. build.yml runs automatically
5. deploy-staging runs automatically
6. deploy-production waits for approval
7. Go to GitHub Actions > deploy.yml > deploy-production
8. Click "Review deployments"
9. Select "production" environment
10. Click "Approve and deploy"
11. Watch deployment complete

---

## Performance Benchmarks

| Workflow | Time (First) | Time (Cached) | Bottleneck |
|----------|--------------|---------------|-----------|
| ci.yml | 20 min | 10 min | Full validation breadth |
| test.yml | <1 min | <1 min | Deprecated notice only |
| lint.yml | <1 min | <1 min | Deprecated notice only |
| build.yml | 8 min | 2 min | Layer download/build |
| deploy-staging | 5 min | 5 min | Pod startup |

**Total CI/CD time:** 25 min (first run) → 17 min (cached)

---

## Common Misconfigurations

### Kubeconfig Not Found
**Error:** `unable to load config file`
**Fix:** Add KUBECONFIG_STAGING and KUBECONFIG_PRODUCTION secrets

### Image Push Failing
**Error:** `denied: permission denied`
**Fix:** Verify GHCR push permissions in GitHub token

### Pod Not Starting
**Error:** `ImagePullBackOff`
**Fix:** Image may not exist; check build.yml succeeded

### Health Check Failing
**Error:** `connection refused`
**Fix:** Ensure HTTP_ENABLED=true in deployment ConfigMap

---

## Quick Debugging Checklist

- [ ] Run `pytest tests/` locally - tests pass?
- [ ] Run `black --check src/` - formatting OK?
- [ ] Run `ruff check src/` - no lint errors?
- [ ] Run `docker build .` - Docker builds OK?
- [ ] Check kubeconfig: `kubectl cluster-info`
- [ ] Check pod logs: `kubectl logs -n token-saver-5000 ...`
- [ ] Check deployment: `kubectl describe deployment ...`

---

## Useful Links

- **GitHub Actions Docs:** https://docs.github.com/actions
- **Workflow Syntax:** https://docs.github.com/actions/using-workflows/workflow-syntax-for-github-actions
- **Artifacts:** https://docs.github.com/actions/managing-workflow-runs-and-artifacts
- **Environments:** https://docs.github.com/actions/deployment/targeting-different-environments
- **Secrets:** https://docs.github.com/actions/security-guides/encrypted-secrets

---

## Getting Help

1. Check GitHub Actions logs: Actions tab > Workflow name > Latest run
2. Check workflow output for step that failed
3. Scroll down for detailed error messages
4. Compare with local command execution
5. Ask team for help with specific error message

---

## Files Modified

- `.github/workflows/ci.yml` - Canonical repository validation
- `.github/workflows/test.yml` - Deprecated manual compatibility shim
- `.github/workflows/lint.yml` - Deprecated manual compatibility shim
- `.github/workflows/build.yml` - New - Docker build & push
- `.github/workflows/deploy.yml` - New - Kubernetes deployment
- `.github/workflows/README.md` - New - Detailed documentation
- `.github/WORKFLOWS_QUICK_REFERENCE.md` - New - This file
