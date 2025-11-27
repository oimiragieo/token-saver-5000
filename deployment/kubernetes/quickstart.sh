#!/bin/bash
#
# quickstart.sh - One-command deployment for Token Saver 5000 on Kubernetes
#
# Usage: ./quickstart.sh [options]
#
# Options:
#   --namespace NAME   Namespace to deploy to (default: token-saver-5000)
#   --skip-monitoring  Skip Prometheus Operator resources (if not installed)
#   --delete           Delete deployment instead of creating
#   --help             Show this help message
#
# Examples:
#   ./quickstart.sh                          # Deploy to default namespace
#   ./quickstart.sh --namespace production   # Deploy to 'production' namespace
#   ./quickstart.sh --skip-monitoring        # Deploy without monitoring
#   ./quickstart.sh --delete                 # Delete deployment

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
NAMESPACE="token-saver-5000"
SKIP_MONITORING=false
DELETE_MODE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --skip-monitoring)
      SKIP_MONITORING=true
      shift
      ;;
    --delete)
      DELETE_MODE=true
      shift
      ;;
    --help)
      head -n 15 "$0" | tail -n 13
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Helper functions
print_banner() {
  echo -e "${BLUE}"
  echo "╔═══════════════════════════════════════════════════════════╗"
  echo "║                                                           ║"
  echo "║           Token Saver 5000 - Kubernetes Deploy           ║"
  echo "║                                                           ║"
  echo "╚═══════════════════════════════════════════════════════════╝"
  echo -e "${NC}"
}

print_section() {
  echo -e "\n${GREEN}▶ $1${NC}"
}

print_info() {
  echo -e "  ${YELLOW}ℹ${NC} $1"
}

print_success() {
  echo -e "  ${GREEN}✓${NC} $1"
}

print_error() {
  echo -e "  ${RED}✗${NC} $1"
}

check_prereqs() {
  local missing=0

  if ! command -v kubectl &> /dev/null; then
    print_error "kubectl not found"
    missing=1
  fi

  if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    missing=1
  fi

  if [[ $missing -eq 1 ]]; then
    return 1
  fi

  return 0
}

# Main deployment function
deploy() {
  print_banner

  # Check prerequisites
  print_section "Checking Prerequisites"
  if ! check_prereqs; then
    print_error "Prerequisites check failed"
    exit 1
  fi
  print_success "kubectl found and cluster accessible"

  # Check optional components
  if ! kubectl get deployment metrics-server -n kube-system &> /dev/null; then
    print_info "Metrics Server not found (HPA may not work)"
  else
    print_success "Metrics Server detected"
  fi

  if ! $SKIP_MONITORING && ! kubectl get crd servicemonitors.monitoring.coreos.com &> /dev/null; then
    print_info "Prometheus Operator not found (using --skip-monitoring)"
    SKIP_MONITORING=true
  elif ! $SKIP_MONITORING; then
    print_success "Prometheus Operator detected"
  fi

  # Create namespace
  print_section "Creating Namespace: $NAMESPACE"
  if kubectl get namespace "$NAMESPACE" &> /dev/null; then
    print_info "Namespace already exists"
  else
    kubectl create namespace "$NAMESPACE"
    print_success "Namespace created"
  fi

  # Apply ConfigMap
  print_section "Applying ConfigMap"
  kubectl apply -f "$SCRIPT_DIR/configmap.yaml" -n "$NAMESPACE"
  print_success "ConfigMap applied"

  # Check Secret
  print_section "Checking Secret"
  if grep -q "PLACEHOLDER" "$SCRIPT_DIR/secret.yaml"; then
    print_error "secret.yaml contains PLACEHOLDER values!"
    print_info "For quick testing, creating empty secret..."
    kubectl create secret generic token-saver-5000-secrets \
      --from-literal=api-key="" \
      --namespace="$NAMESPACE" \
      --dry-run=client -o yaml | kubectl apply -f -
    print_info "Note: Update secrets before production use!"
  else
    kubectl apply -f "$SCRIPT_DIR/secret.yaml" -n "$NAMESPACE"
    print_success "Secret applied"
  fi

  # Apply Deployment
  print_section "Deploying Application"
  kubectl apply -f "$SCRIPT_DIR/deployment.yaml" -n "$NAMESPACE"
  print_success "Deployment created"

  # Apply Service
  print_section "Creating Service"
  kubectl apply -f "$SCRIPT_DIR/service.yaml" -n "$NAMESPACE"
  print_success "Service created"

  # Apply HPA
  print_section "Configuring Autoscaling"
  kubectl apply -f "$SCRIPT_DIR/hpa.yaml" -n "$NAMESPACE"
  print_success "HPA configured"

  # Apply monitoring (if not skipped)
  if ! $SKIP_MONITORING; then
    print_section "Setting up Monitoring"
    kubectl apply -f "$SCRIPT_DIR/servicemonitor.yaml" -n "$NAMESPACE"
    kubectl apply -f "$SCRIPT_DIR/prometheusrule.yaml" -n "$NAMESPACE"
    print_success "Monitoring configured"
  fi

  # Wait for deployment
  print_section "Waiting for Deployment"
  print_info "This may take 1-2 minutes (downloading embedding models)..."

  if kubectl wait --for=condition=available --timeout=300s \
    deployment/token-saver-5000 -n "$NAMESPACE" &> /dev/null; then
    print_success "Deployment ready!"
  else
    print_error "Deployment timeout - check pod logs"
    print_info "kubectl logs -n $NAMESPACE -l app=token-saver-5000 --tail=50"
  fi

  # Show status
  print_section "Deployment Status"
  kubectl get pods,svc,hpa -n "$NAMESPACE"

  # Show next steps
  print_section "Next Steps"
  print_info "1. Test health endpoint:"
  echo "     kubectl port-forward -n $NAMESPACE svc/token-saver-5000 8080:8080"
  echo "     curl http://localhost:8080/health/liveness"
  print_info "2. View logs:"
  echo "     kubectl logs -n $NAMESPACE -l app=token-saver-5000 --tail=50 -f"
  print_info "3. View metrics:"
  echo "     curl http://localhost:8080/metrics"
  print_info "4. Scale manually:"
  echo "     kubectl scale deployment token-saver-5000 --replicas=5 -n $NAMESPACE"

  echo -e "\n${GREEN}✓ Deployment complete!${NC}\n"
}

# Delete deployment
delete() {
  print_banner

  print_section "Deleting Deployment from: $NAMESPACE"

  # Confirm deletion
  read -p "Are you sure you want to delete Token Saver 5000 from '$NAMESPACE'? [y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Deletion cancelled"
    exit 0
  fi

  # Delete monitoring
  if ! $SKIP_MONITORING; then
    print_info "Deleting monitoring resources..."
    kubectl delete -f "$SCRIPT_DIR/servicemonitor.yaml" -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete -f "$SCRIPT_DIR/prometheusrule.yaml" -n "$NAMESPACE" --ignore-not-found=true
  fi

  # Delete HPA
  print_info "Deleting autoscaler..."
  kubectl delete -f "$SCRIPT_DIR/hpa.yaml" -n "$NAMESPACE" --ignore-not-found=true

  # Delete Service
  print_info "Deleting service..."
  kubectl delete -f "$SCRIPT_DIR/service.yaml" -n "$NAMESPACE" --ignore-not-found=true

  # Delete Deployment
  print_info "Deleting deployment..."
  kubectl delete -f "$SCRIPT_DIR/deployment.yaml" -n "$NAMESPACE" --ignore-not-found=true

  # Delete ConfigMap & Secret
  print_info "Deleting configuration..."
  kubectl delete -f "$SCRIPT_DIR/configmap.yaml" -n "$NAMESPACE" --ignore-not-found=true
  kubectl delete -f "$SCRIPT_DIR/secret.yaml" -n "$NAMESPACE" --ignore-not-found=true

  # Delete namespace (optional)
  read -p "Delete namespace '$NAMESPACE' as well? [y/N] " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    print_success "Namespace deleted"
  else
    print_info "Namespace preserved"
  fi

  echo -e "\n${GREEN}✓ Deletion complete!${NC}\n"
}

# Main execution
if $DELETE_MODE; then
  delete
else
  deploy
fi
