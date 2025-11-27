# Kubernetes Deployment for Token Saver 5000

This directory contains production-ready Kubernetes manifests for deploying Token Saver 5000 MCP server with high availability, monitoring, and autoscaling.

## Overview

Token Saver 5000 is deployed as a stateless service with:
- **High Availability**: 2-10 replicas with HPA (Horizontal Pod Autoscaler)
- **Health Monitoring**: Liveness and readiness probes via HTTP endpoints
- **Metrics**: Prometheus metrics scraping via ServiceMonitor
- **Autoscaling**: CPU/memory-based autoscaling with HPA
- **Security**: Non-root containers, read-only filesystems, security contexts
- **Alerting**: PrometheusRule with comprehensive alerts

## Prerequisites

### Required
- Kubernetes cluster 1.20+ (tested on 1.25+)
- `kubectl` CLI configured with cluster access
- 2+ worker nodes (for high availability pod anti-affinity)

### Optional (for full feature set)
- **Prometheus Operator** (for ServiceMonitor and PrometheusRule)
  - Install via [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
- **Metrics Server** (for HPA CPU/memory metrics)
  - Install: `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml`
- **External Secrets Operator** (for secret management)
  - Install: [External Secrets Operator](https://external-secrets.io/latest/)
- **Persistent Volume** (for production data persistence)

## Quick Start

### 1. Create Namespace

```bash
kubectl create namespace token-saver-5000
```

### 2. Apply Manifests

Apply in this order (dependencies matter):

```bash
# Configuration (no dependencies)
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml  # Update with real secrets first!

# Core resources (depend on ConfigMap/Secret)
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Autoscaling (depends on Deployment)
kubectl apply -f hpa.yaml

# Monitoring (depends on Service, requires Prometheus Operator)
kubectl apply -f servicemonitor.yaml
kubectl apply -f prometheusrule.yaml
```

Or apply all at once:

```bash
kubectl apply -f deployment/kubernetes/
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods -n token-saver-5000

# Check deployment status
kubectl get deployment token-saver-5000 -n token-saver-5000

# Check HPA status
kubectl get hpa token-saver-5000-hpa -n token-saver-5000

# Check service endpoints
kubectl get endpoints token-saver-5000 -n token-saver-5000

# Check logs
kubectl logs -n token-saver-5000 -l app=token-saver-5000 --tail=100
```

### 4. Test Health Endpoints

```bash
# Port-forward to access health endpoints
kubectl port-forward -n token-saver-5000 svc/token-saver-5000 8080:8080

# In another terminal, test endpoints
curl http://localhost:8080/health/liveness
curl http://localhost:8080/health/readiness
curl http://localhost:8080/metrics
```

## Manifest Files

### Core Resources

#### `deployment.yaml`
- **Deployment** with 2 replicas (baseline, controlled by HPA)
- **Resource limits**: 1GB-2GB memory, 0.5-2 CPU cores
- **Health probes**: Liveness, readiness, startup probes via HTTP
- **Security context**: Non-root user, read-only filesystem
- **Pod anti-affinity**: Spread pods across nodes for HA
- **Volume mounts**: `/data` for persistence, `/tmp/.cache` for models

#### `service.yaml`
- **ClusterIP Service** for internal access
- **Port 8080**: Health checks (liveness, readiness)
- **Port 9090**: Prometheus metrics
- **Headless Service** for direct pod access (debugging)

#### `configmap.yaml`
- **Environment variables**: HTTP config, logging, storage backend
- **Application config files**: JSON configuration (optional)
- **Tunable parameters**: Cache size, batch concurrency, embedding tier

#### `secret.yaml`
- **Template** for secrets (DO NOT commit real secrets!)
- **Placeholder values**: API keys, auth tokens, DB passwords
- **External Secrets examples**: AWS Secrets Manager, Sealed Secrets

### Autoscaling

#### `hpa.yaml`
- **HorizontalPodAutoscaler**: Scale 2-10 replicas
- **CPU target**: 70% utilization
- **Memory target**: 80% utilization
- **Scaling behavior**: Aggressive scale-up, conservative scale-down
- **VPA example**: Vertical Pod Autoscaler (commented out)

### Monitoring

#### `servicemonitor.yaml`
- **ServiceMonitor** for Prometheus Operator
- **Scrape endpoint**: `/metrics` on port 9090
- **Scrape interval**: 15 seconds
- **Relabeling**: Add namespace, pod, service labels
- **PodMonitor example**: Direct pod scraping (commented out)

#### `prometheusrule.yaml`
- **PrometheusRule** with 16 alerting rules across 4 groups:
  1. **Availability**: Pod not ready, no pods running, crash looping
  2. **Performance**: High CPU, high memory, OOM risk, high latency
  3. **Application**: High error rate, low cache hit rate, compression degradation
  4. **Infrastructure**: Insufficient replicas, HPA maxed out, PV near full
- **Severity levels**: Critical, Warning, Info
- **Runbook URLs**: Links to troubleshooting guides

## Configuration

### Environment Variables (ConfigMap)

Key configuration options in `configmap.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_ENABLED` | `true` | Enable HTTP endpoints (health, metrics) |
| `HTTP_HOST` | `0.0.0.0` | Bind to all interfaces (required for K8s) |
| `HTTP_PORT` | `8080` | HTTP server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | `json` | Log format (json, text) |
| `STORAGE_BACKEND` | `json` | Storage backend (json, chromadb) |
| `DATA_DIR` | `/data` | Persistent storage directory |
| `METRICS_ENABLED` | `true` | Enable Prometheus metrics |

See `configmap.yaml` for full list of configuration options.

### Resource Limits

Default resource allocation in `deployment.yaml`:

| Resource | Request | Limit |
|----------|---------|-------|
| Memory | 1GB | 2GB |
| CPU | 500m | 2000m |

**Recommendations:**
- **Small workloads**: 1GB memory, 0.5 CPU (requests = limits)
- **Medium workloads**: 1-2GB memory, 0.5-1 CPU (default)
- **Large workloads**: 2-4GB memory, 1-2 CPU (increase limits)

### Autoscaling Targets

HPA thresholds in `hpa.yaml`:

| Metric | Target | Action |
|--------|--------|--------|
| CPU | 70% | Scale up if > 70%, scale down if < 70% |
| Memory | 80% | Scale up if > 80%, scale down if < 80% |

**Scaling behavior:**
- **Scale up**: Immediate (0s delay), aggressive (double replicas)
- **Scale down**: 5-minute delay, conservative (50% reduction)
- **Replica bounds**: Min 2 (HA), Max 10 (cost control)

## Production Checklist

Before deploying to production:

### 1. Secrets Management
- [ ] Replace placeholder secrets in `secret.yaml` with real values
- [ ] Use External Secrets Operator or Sealed Secrets (recommended)
- [ ] Never commit real secrets to version control

### 2. Persistent Storage
- [ ] Replace `emptyDir` with `persistentVolumeClaim` in `deployment.yaml`
- [ ] Create PersistentVolumeClaim for `/data` volume
- [ ] Configure backup strategy for persistent data

### 3. Image Management
- [ ] Build and push Docker image to private registry
- [ ] Update `image:` in `deployment.yaml` with versioned tag (not `latest`)
- [ ] Configure `imagePullSecrets` if using private registry

### 4. Monitoring Setup
- [ ] Install Prometheus Operator (kube-prometheus-stack)
- [ ] Verify ServiceMonitor is discovered by Prometheus
- [ ] Configure AlertManager for alert notifications
- [ ] Test alert routing (email, Slack, PagerDuty)

### 5. Resource Tuning
- [ ] Load test to determine optimal resource requests/limits
- [ ] Tune HPA thresholds based on actual traffic patterns
- [ ] Monitor resource usage over 1-2 weeks before finalizing

### 6. Security Hardening
- [ ] Review and apply NetworkPolicies (restrict pod-to-pod traffic)
- [ ] Enable Pod Security Standards (restricted profile)
- [ ] Configure RBAC (ServiceAccount with minimal permissions)
- [ ] Scan container images for vulnerabilities

### 7. High Availability
- [ ] Deploy to 3+ availability zones (pod anti-affinity)
- [ ] Configure PodDisruptionBudget (ensure min 1 pod during disruptions)
- [ ] Test failover scenarios (node drain, pod eviction)

## Advanced Configuration

### Persistent Storage

Replace `emptyDir` with PersistentVolumeClaim:

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: token-saver-5000-data
  namespace: token-saver-5000
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard  # Or your storage class
```

Update `deployment.yaml`:

```yaml
volumes:
- name: data
  persistentVolumeClaim:
    claimName: token-saver-5000-data
```

### NetworkPolicy

Restrict network access:

```yaml
# networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: token-saver-5000-netpol
  namespace: token-saver-5000
spec:
  podSelector:
    matchLabels:
      app: token-saver-5000
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # Allow Prometheus scraping
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8080
  egress:
  # Allow DNS
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
  # Allow external API calls (model downloads, etc.)
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443
```

### PodDisruptionBudget

Ensure high availability during disruptions:

```yaml
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: token-saver-5000-pdb
  namespace: token-saver-5000
spec:
  minAvailable: 1  # Always keep at least 1 pod running
  selector:
    matchLabels:
      app: token-saver-5000
```

### Ingress

Expose service externally (optional):

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: token-saver-5000-ingress
  namespace: token-saver-5000
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - token-saver.example.com
    secretName: token-saver-tls
  rules:
  - host: token-saver.example.com
    http:
      paths:
      - path: /health
        pathType: Prefix
        backend:
          service:
            name: token-saver-5000
            port:
              number: 8080
      - path: /metrics
        pathType: Prefix
        backend:
          service:
            name: token-saver-5000
            port:
              number: 9090
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod -n token-saver-5000 -l app=token-saver-5000

# Check logs
kubectl logs -n token-saver-5000 -l app=token-saver-5000 --tail=100

# Common issues:
# - Image pull errors: Check image name and imagePullSecrets
# - Resource limits: Check node resources with kubectl describe node
# - ConfigMap/Secret missing: Verify kubectl get cm,secret -n token-saver-5000
```

### HPA Not Scaling

```bash
# Check HPA status
kubectl describe hpa token-saver-5000-hpa -n token-saver-5000

# Check metrics server
kubectl top pods -n token-saver-5000

# Common issues:
# - Metrics server not installed: Install metrics-server
# - Resource requests not set: Verify deployment.yaml has resources.requests
# - HPA conditions: Check kubectl get hpa -n token-saver-5000 -o yaml
```

### Prometheus Not Scraping

```bash
# Check ServiceMonitor
kubectl get servicemonitor -n token-saver-5000

# Check Prometheus targets (Prometheus UI)
# Navigate to Status > Targets, search for "token-saver-5000"

# Common issues:
# - ServiceMonitor label mismatch: Verify prometheus: kube-prometheus label
# - Service port name mismatch: Ensure port names match between Service and ServiceMonitor
# - Prometheus RBAC: Verify Prometheus ServiceAccount has RBAC to discover ServiceMonitors
```

### Alerts Not Firing

```bash
# Check PrometheusRule
kubectl get prometheusrule -n token-saver-5000

# Check Prometheus rules (Prometheus UI)
# Navigate to Status > Rules, search for "token-saver-5000"

# Check AlertManager (AlertManager UI)
# Verify alerts are routing to correct receivers

# Common issues:
# - PrometheusRule label mismatch: Verify prometheus: kube-prometheus label
# - Alert expression errors: Check Prometheus logs for evaluation errors
# - AlertManager routing: Verify alertmanager.yaml configuration
```

## Monitoring Dashboards

### Grafana Dashboard

Import pre-built Grafana dashboard for Token Saver 5000:

1. Navigate to Grafana UI
2. Click **+ > Import**
3. Use dashboard ID: **15759** (Kubernetes Pod Monitoring)
4. Customize panels with Token Saver 5000 metrics:
   - `compression_ratio`
   - `compression_latency_seconds`
   - `cache_hit_ratio`
   - `errors_total`

### Key Metrics to Monitor

| Metric | Description | Good Value |
|--------|-------------|------------|
| `compression_ratio` | Token reduction ratio | > 0.7 (70%) |
| `compression_latency_seconds` | P95 compression latency | < 0.5s (500ms) |
| `cache_hit_ratio` | Embedding cache hit rate | > 0.6 (60%) |
| `errors_total` | Total error count | < 10/min |
| `kube_pod_status_ready` | Pod readiness status | 1 (ready) |
| `container_memory_working_set_bytes` | Memory usage | < 1.8GB (90% of limit) |
| `container_cpu_usage_seconds_total` | CPU usage rate | < 1.8 cores (90% of limit) |

## Cleanup

Remove all resources:

```bash
# Delete all manifests
kubectl delete -f deployment/kubernetes/

# Delete namespace (WARNING: deletes all data)
kubectl delete namespace token-saver-5000
```

Delete specific resources:

```bash
# Delete monitoring resources only
kubectl delete -f servicemonitor.yaml
kubectl delete -f prometheusrule.yaml

# Delete autoscaling only
kubectl delete -f hpa.yaml

# Delete deployment only (keep ConfigMap/Secret)
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml
```

## Support

For issues and questions:
- **GitHub Issues**: [token-saver-5000/issues](https://github.com/your-org/token-saver-5000/issues)
- **Documentation**: [DEPLOYMENT.md](../DEPLOYMENT.md)
- **Runbooks**: [Wiki Runbooks](https://github.com/your-org/token-saver-5000/wiki)

## License

See [LICENSE](../../LICENSE) file.
