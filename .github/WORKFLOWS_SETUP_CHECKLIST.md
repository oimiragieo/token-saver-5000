# GitHub Actions Workflows - Setup Checklist

Complete this checklist to enable all 4 CI/CD workflows for Token Saver 5000.

## Pre-Setup (Organization/Repository Setup)

- [ ] Repository has GitHub Actions enabled (Settings > Actions > Allow all actions and reusable workflows)
- [ ] Repository visibility is public or Actions available for private repos
- [ ] Admin has permissions to create environments and manage secrets

---

## 1. Configure GitHub Secrets

**Location:** Settings > Secrets and variables > Actions

### For Kubernetes Deployments (Required for deploy.yml)

#### KUBECONFIG_STAGING
1. Get kubeconfig from staging cluster:
   ```bash
   # From staging cluster admin
   kubectl config view --raw
   ```

2. Encode to base64:
   ```bash
   # macOS/Linux
   cat ~/.kube/config | base64 | pbcopy

   # Or pipe to file and copy
   cat ~/.kube/config | base64 > kubeconfig.b64
   cat kubeconfig.b64  # Copy output

   # Windows PowerShell
   [Convert]::ToBase64String([System.IO.File]::ReadAllBytes("$env:USERPROFILE\.kube\config")) | Set-Clipboard
   ```

3. Create secret in GitHub:
   - Go to Settings > Secrets and variables > Actions
   - Click "New repository secret"
   - Name: `KUBECONFIG_STAGING`
   - Value: Paste base64-encoded kubeconfig
   - Click "Add secret"

#### KUBECONFIG_PRODUCTION
Repeat the process for production cluster:
   - Name: `KUBECONFIG_PRODUCTION`
   - Value: Base64-encoded production kubeconfig

### Optional: Docker Registry Secrets

If using Docker Hub instead of GitHub Container Registry:
- `DOCKER_HUB_USERNAME` - Docker Hub username
- `DOCKER_HUB_PASSWORD` - Docker Hub personal access token

(Currently using GHCR which uses automatic GITHUB_TOKEN)

**Status:**
- [ ] KUBECONFIG_STAGING configured
- [ ] KUBECONFIG_PRODUCTION configured
- [ ] Secrets tested (optional)

---

## 2. Configure GitHub Environments

**Location:** Settings > Environments

### Create Staging Environment

1. Click "New environment"
2. Name: `staging`
3. Required reviewers: Leave unchecked (auto-deploy)
4. Deployment branches: `main` only
5. Save

**Status:**
- [ ] Staging environment created
- [ ] Auto-deployment enabled (no approval required)

### Create Production Environment

1. Click "New environment"
2. Name: `production`
3. Required reviewers: Check and add 1-2 team members
4. Prevent self-review: Check
5. Deployment branches: `main` only
6. Save

**Status:**
- [ ] Production environment created
- [ ] Approval required (1-2 reviewers)
- [ ] Prevent self-review enabled

---

## 3. Configure Branch Protection Rules

**Location:** Settings > Branches > Branch protection rules

### For `main` Branch

1. Go to Settings > Branches
2. Click "Add rule"
3. Branch name pattern: `main`
4. Enable protection options:
   - [ ] Require a pull request before merging
   - [ ] Dismiss stale PR approvals when new commits are pushed
   - [ ] Require approval of the most recent reviewers before merging
   - [ ] Require status checks to pass before merging:
     - [ ] Require branches to be up to date before merging
     - [ ] Status checks required: Select `ci.yml`, any focused guards you want to enforce, and `build.yml` if needed
   - [ ] Restrict who can push to matching branches: (optional)
5. Save

**Status:**
- [ ] Branch protection enabled for `main`
- [ ] Status checks required: ci.yml, any chosen focused guards, build.yml
- [ ] PR review required before merge

### For `develop` Branch (Optional)

Similar to `main`, but with fewer restrictions:
- Require ci.yml to pass
- Require 1 reviewer (not required)

**Status:**
- [ ] Branch protection enabled for `develop` (optional)

---

## 4. Configure Codecov Integration (Optional)

**Purpose:** Track coverage trends over time

### Enable Codecov

1. Go to https://codecov.io
2. Sign in with GitHub
3. Click "Add new repository"
4. Select `token-saver-5000`
5. Codecov provides integration automatically

### Verify Integration

1. Go to repository > Actions
2. Re-run ci.yml
3. Check that "Upload coverage to Codecov" step succeeds
4. Visit Codecov dashboard to see coverage trends

**Status:**
- [ ] Codecov account created
- [ ] Repository added to Codecov
- [ ] Coverage reports uploading successfully

---

## 5. Configure SBOM Export (Optional)

**Purpose:** Track software dependencies for compliance

### Enable SBOM

The build.yml workflow already generates SBOM using `anchore/sbom-action`.

### Verify

1. Go to repository > Actions > build.yml
2. Check "Artifacts" section for `sbom` (SPDX format)
3. Download and review sbom.spdx.json

**Status:**
- [ ] SBOM generation working (check build.yml artifacts)

---

## 6. Test Workflow Execution

### Test ci.yml

1. Create feature branch: `git checkout -b test/workflows`
2. Make small change to src/
3. Commit and push: `git push origin test/workflows`
4. Go to Actions tab
5. Click "ci.yml"
6. Wait for completion
7. Verify: quality gate, compatibility, package validation, and full validation all pass

**Status:**
- [ ] ci.yml runs on push
- [ ] Compatibility matrix runs for all supported Python versions
- [ ] Package validation passes
- [ ] Full validation passes

### Test legacy compatibility workflows

1. Go to Actions tab
2. Trigger `test.yml` manually
3. Trigger `lint.yml` manually
4. Verify each workflow posts a deprecation notice pointing to `ci.yml`

**Status:**
- [ ] test.yml is manual-only
- [ ] lint.yml is manual-only
- [ ] Legacy workflows clearly point maintainers to ci.yml

### Test build.yml

1. No special setup needed
2. build.yml auto-runs on push to main
3. Check Actions > build.yml
4. Verify: Docker image built
5. Verify: Image pushed to ghcr.io

**Status:**
- [ ] build.yml runs on push to main
- [ ] Docker image builds successfully
- [ ] Image pushed to ghcr.io/username/token-saver-5000
- [ ] Trivy scan completes (informational)
- [ ] SBOM generated

### Test deploy.yml (Staging)

1. Create semantic version tag: `git tag -a v0.0.1-test -m "Test deployment"`
2. Push tag: `git push origin v0.0.1-test`
3. Go to Actions > deploy.yml
4. Wait for staging deployment (5-10 min)
5. Verify:
   - [ ] deploy-staging job runs automatically
   - [ ] Kubectl applies manifests
   - [ ] Pod starts successfully
   - [ ] Health checks pass
   - [ ] Logs show no errors

**Status:**
- [ ] deploy-staging auto-runs on tags
- [ ] Kubernetes cluster accessible
- [ ] Pod deploys and starts
- [ ] Health checks pass

### Test deploy.yml (Production - Optional)

Only if you want to test production approval flow:

1. Create another test tag: `git tag -a v0.0.2-test -m "Test prod approval"`
2. Push tag: `git push origin v0.0.2-test`
3. Go to Actions > deploy.yml
4. Watch deploy-staging complete
5. Click on deploy-production job
6. Click "Review deployments"
7. Select "production"
8. Click "Approve and deploy"
9. Wait for production deployment (5-10 min)
10. Verify same checks as staging

**Status:**
- [ ] deploy-production awaits approval
- [ ] Approval flow works
- [ ] Production deployment succeeds

---

## 7. Configure Docker Registry (If Not Using GHCR)

### For Docker Hub

1. Create Docker Hub account at https://hub.docker.com
2. Create personal access token:
   - Account > Settings > Security > New Access Token
3. Add secrets to GitHub:
   - `DOCKER_HUB_USERNAME` = Docker Hub username
   - `DOCKER_HUB_PASSWORD` = Personal access token
4. Update build.yml:
   - Change `registry: ghcr.io` to `registry: docker.io`
   - Update login step to use Docker Hub credentials

### For AWS ECR

1. Create IAM user for ECR push
2. Add to GitHub secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_DEFAULT_REGION`
3. Update build.yml with ECR login action

**Status:**
- [ ] Container registry configured
- [ ] Registry credentials added to secrets
- [ ] build.yml updated with registry details

---

## 8. Configure Pre-commit Hook (Developers)

**Purpose:** Catch issues locally before pushing to GitHub

### Install Pre-commit

```bash
pip install pre-commit
```

### Create .pre-commit-config.yaml

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.10.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: [types-all]
```

### Enable Pre-commit

```bash
pre-commit install
```

**Status:**
- [ ] pre-commit configured (developers only)
- [ ] Hooks run on every commit
- [ ] Issues caught before push to GitHub

---

## 9. Configure Notifications (Optional)

### Slack Integration

1. Create Slack app: https://api.slack.com/apps
2. Get webhook URL
3. Add to GitHub secret: `SLACK_WEBHOOK_URL`
4. Update workflows to send notifications

### Email Notifications

GitHub provides automatic email on workflow failure:
- Settings > Notifications > Actions
- Select when to notify

**Status:**
- [ ] Slack integration configured (optional)
- [ ] Email notifications enabled

---

## 10. Final Validation Checklist

Before declaring setup complete:

### Workflows Present
- [ ] `.github/workflows/ci.yml` exists
- [ ] `.github/workflows/test.yml` exists as a deprecated manual shim
- [ ] `.github/workflows/lint.yml` exists as a deprecated manual shim
- [ ] `.github/workflows/build.yml` exists
- [ ] `.github/workflows/deploy.yml` exists
- [ ] `.github/workflows/README.md` exists

### Secrets Configured
- [ ] KUBECONFIG_STAGING configured
- [ ] KUBECONFIG_PRODUCTION configured
- [ ] (Optional) DOCKER_HUB credentials if needed

### Environments Configured
- [ ] staging environment created
- [ ] production environment created
- [ ] Production requires approval

### Branch Protection
- [ ] main branch protection enabled
- [ ] ci.yml required
- [ ] legacy workflows not required
- [ ] build.yml required (or informational)

### Workflows Tested
- [ ] ci.yml runs successfully
- [ ] legacy workflows show deprecation notices successfully when triggered manually
- [ ] build.yml runs successfully
- [ ] deploy-staging runs successfully
- [ ] deploy-production approval/deploy works

### Documentation
- [ ] README.md reviewed
- [ ] WORKFLOWS_QUICK_REFERENCE.md reviewed
- [ ] Team trained on workflow usage
- [ ] Deployment approval process documented

---

## Common Issues & Fixes

### Workflows Not Appearing in Actions Tab

**Cause:** GitHub doesn't always show workflows immediately
**Fix:**
1. Wait 1-2 minutes
2. Refresh page
3. Push a new commit to trigger workflows
4. Check workflow files are in `.github/workflows/`

### Tests Failing on Coverage

**Cause:** Code coverage below 70%
**Fix:**
```bash
pytest tests/ --cov=src --cov-report=term
# Add tests to increase coverage
```

### Docker Build Failing

**Cause:** Image size > 500MB or dependency not found
**Fix:**
```bash
docker build -t test:latest .
docker images test:latest  # Check size
```

### Deployment Failing

**Cause:** Kubeconfig not configured or invalid
**Fix:**
1. Verify secret is base64-encoded correctly
2. Test kubeconfig locally: `kubectl --kubeconfig=<secret> cluster-info`
3. Check permissions: Does service account have required RBAC?

### Approval Not Working

**Cause:** Production environment not set to require approval
**Fix:**
1. Go to Settings > Environments > production
2. Check "Required reviewers"
3. Add 1-2 reviewers
4. Save

---

## Next Steps After Setup

1. **Train Team:**
   - Share WORKFLOWS_QUICK_REFERENCE.md with team
   - Demonstrate approval flow for production
   - Explain when each workflow runs

2. **Monitor Performance:**
   - Track workflow run times
   - Watch for cache hits (2-4x speedup)
   - Monitor coverage trends on Codecov

3. **Iterate:**
   - Gather feedback on workflow experience
   - Optimize for team's workflow
   - Add additional checks as needed

4. **Document SOP:**
   - Create release playbook
   - Document rollback procedures
   - Add team-specific guidelines

---

## Setup Complete!

Once all items above are checked:
1. Core workflows plus focused guards are active and tested, with `ci.yml` serving as the canonical validation pipeline
2. CI/CD pipeline is fully operational
3. Team can push code with confidence
4. Automatic deployments to staging/production working
5. You're ready for production releases!

**Time to complete setup:** 30-60 minutes (one-time)
**Time per release:** 5-10 minutes (mostly automated)
