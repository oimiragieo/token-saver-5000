#!/bin/bash
#
# validate.sh - Validate Kubernetes manifests for Token Saver 5000
#
# Usage: ./validate.sh [options]
#
# Options:
#   --dry-run          Perform dry-run validation (default)
#   --apply            Actually apply manifests (use with caution!)
#   --namespace NAME   Override namespace (default: token-saver-5000)
#   --skip-monitoring  Skip ServiceMonitor and PrometheusRule (if Prometheus Operator not installed)
#   --help             Show this help message

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DRY_RUN=true
NAMESPACE="token-saver-5000"
SKIP_MONITORING=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --apply)
      DRY_RUN=false
      shift
      ;;
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --skip-monitoring)
      SKIP_MONITORING=true
      shift
      ;;
    --help)
      head -n 12 "$0" | tail -n 10
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Helper functions
print_section() {
  echo -e "\n${GREEN}=== $1 ===${NC}"
}

print_info() {
  echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
  if ! command -v "$1" &> /dev/null; then
    print_error "$1 is not installed. Please install it first."
    return 1
  fi
  return 0
}

# Validation steps
print_section "Validating Prerequisites"

# Check kubectl
if ! check_command kubectl; then
  exit 1
fi
print_success "kubectl found: $(kubectl version --client --short 2>/dev/null || echo 'installed')"

# Check cluster connectivity
if ! kubectl cluster-info &> /dev/null; then
  print_error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
  exit 1
fi
print_success "Connected to cluster: $(kubectl config current-context)"

# Check metrics-server (for HPA)
print_section "Checking Optional Components"

if kubectl get deployment metrics-server -n kube-system &> /dev/null; then
  print_success "Metrics Server is installed (required for HPA)"
else
  print_error "Metrics Server not found (HPA will not work)"
  print_info "Install: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml"
fi

# Check Prometheus Operator (for ServiceMonitor/PrometheusRule)
if ! $SKIP_MONITORING; then
  if kubectl get crd servicemonitors.monitoring.coreos.com &> /dev/null; then
    print_success "Prometheus Operator is installed (ServiceMonitor/PrometheusRule will work)"
  else
    print_error "Prometheus Operator not found (ServiceMonitor/PrometheusRule will be ignored)"
    print_info "Install: helm install prometheus prometheus-community/kube-prometheus-stack"
    print_info "Or skip monitoring with: --skip-monitoring"
  fi
fi

# Validate manifests
print_section "Validating Manifests"

MANIFESTS=(
  "namespace.yaml"
  "configmap.yaml"
  "secret.yaml"
  "deployment.yaml"
  "service.yaml"
  "hpa.yaml"
)

# Add monitoring manifests if not skipped
if ! $SKIP_MONITORING; then
  MANIFESTS+=("servicemonitor.yaml" "prometheusrule.yaml")
fi

for manifest in "${MANIFESTS[@]}"; do
  manifest_path="$SCRIPT_DIR/$manifest"

  if [[ ! -f "$manifest_path" ]]; then
    print_error "Manifest not found: $manifest"
    continue
  fi

  # Validate YAML syntax
  if kubectl apply --dry-run=client -f "$manifest_path" &> /dev/null; then
    print_success "Valid: $manifest"
  else
    print_error "Invalid: $manifest"
    kubectl apply --dry-run=client -f "$manifest_path"
    exit 1
  fi
done

# Check for placeholder secrets
print_section "Checking Configuration"

if grep -q "PLACEHOLDER" "$SCRIPT_DIR/secret.yaml"; then
  print_error "secret.yaml contains PLACEHOLDER values. Update with real secrets before deploying!"
  print_info "Use: kubectl create secret generic token-saver-5000-secrets --from-literal=api-key=<value>"
else
  print_success "secret.yaml does not contain PLACEHOLDER values"
fi

# Validate resource limits
print_section "Validating Resource Configuration"

cpu_request=$(grep -A 2 "requests:" "$SCRIPT_DIR/deployment.yaml" | grep "cpu:" | awk '{print $2}' | tr -d '"')
memory_request=$(grep -A 2 "requests:" "$SCRIPT_DIR/deployment.yaml" | grep "memory:" | awk '{print $2}' | tr -d '"')
cpu_limit=$(grep -A 2 "limits:" "$SCRIPT_DIR/deployment.yaml" | grep "cpu:" | awk '{print $2}' | tr -d '"')
memory_limit=$(grep -A 2 "limits:" "$SCRIPT_DIR/deployment.yaml" | grep "memory:" | awk '{print $2}' | tr -d '"')

print_info "Resource requests: CPU=$cpu_request, Memory=$memory_request"
print_info "Resource limits: CPU=$cpu_limit, Memory=$memory_limit"

# Dry-run or apply
print_section "Deployment Action"

if $DRY_RUN; then
  print_info "Performing dry-run validation (--dry-run=server)..."

  for manifest in "${MANIFESTS[@]}"; do
    manifest_path="$SCRIPT_DIR/$manifest"

    if kubectl apply --dry-run=server -f "$manifest_path" &> /dev/null; then
      print_success "Server dry-run valid: $manifest"
    else
      print_error "Server dry-run failed: $manifest"
      kubectl apply --dry-run=server -f "$manifest_path"
      exit 1
    fi
  done

  print_success "All manifests are valid!"
  print_info "To apply manifests, run: ./validate.sh --apply"

else
  print_info "Applying manifests to namespace: $NAMESPACE"

  # Create namespace first
  kubectl apply -f "$SCRIPT_DIR/namespace.yaml"

  # Apply configuration
  kubectl apply -f "$SCRIPT_DIR/configmap.yaml"
  kubectl apply -f "$SCRIPT_DIR/secret.yaml"

  # Apply core resources
  kubectl apply -f "$SCRIPT_DIR/deployment.yaml"
  kubectl apply -f "$SCRIPT_DIR/service.yaml"

  # Apply autoscaling
  kubectl apply -f "$SCRIPT_DIR/hpa.yaml"

  # Apply monitoring (if not skipped)
  if ! $SKIP_MONITORING; then
    kubectl apply -f "$SCRIPT_DIR/servicemonitor.yaml"
    kubectl apply -f "$SCRIPT_DIR/prometheusrule.yaml"
  fi

  print_success "All manifests applied successfully!"

  # Wait for deployment
  print_info "Waiting for deployment to be ready..."
  kubectl wait --for=condition=available --timeout=300s \
    deployment/token-saver-5000 -n "$NAMESPACE"

  # Show status
  print_section "Deployment Status"
  kubectl get pods,svc,hpa -n "$NAMESPACE"

  print_info "Check logs: kubectl logs -n $NAMESPACE -l app=token-saver-5000 --tail=50"
  print_info "Test health: kubectl port-forward -n $NAMESPACE svc/token-saver-5000 8080:8080"
fi

print_section "Validation Complete"
