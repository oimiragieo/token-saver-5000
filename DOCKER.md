# Docker Deployment

Run Token Saver 5000 as a Docker container for easy deployment and isolation.

> **Production Kubernetes Deployment:** For production environments, see [deployment/kubernetes/README.md](deployment/kubernetes/README.md) for comprehensive Kubernetes manifests with health checks, autoscaling, and monitoring.

## Quick Start

```bash
# Build the multi-stage Docker image
docker build -t token-saver-5000:latest .

# Run in stdio mode (default, for local Claude Desktop/Code integration)
docker run -d \
  --name token-saver-mcp \
  -v token-saver-data:/data \
  token-saver-5000:latest

# Run with HTTP server enabled (for Kubernetes health checks and metrics)
docker run -d \
  --name token-saver-mcp \
  -e HTTP_ENABLED=true \
  -p 8080:8080 \
  -v token-saver-data:/data \
  token-saver-5000:latest
```

## Claude Desktop Integration

**Add to your Claude Desktop config:**

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "token-saver": {
      "command": "docker",
      "args": ["exec", "-i", "token-saver-mcp", "python", "-m", "src.server"]
    }
  }
}
```

Restart Claude Desktop. The 30 MCP tools will be available.

## Docker Image Features

The multi-stage Dockerfile provides:

- **Security:** Non-root user (uid 1000), read-only filesystem support, dropped capabilities
- **Size Optimization:** <500MB target image (~450MB expected) via builder + runtime stages
- **Hybrid Deployment:** Supports both stdio mode (MCP) and HTTP mode (Kubernetes)
- **Health Checks:** Built-in health check for liveness probe (when HTTP_ENABLED=true)
- **Model Caching:** Pre-downloads sentence-transformers model during build

### Environment Variables

- `HTTP_ENABLED` (default: false) - Enable HTTP server for health checks and metrics
- `HTTP_HOST` (default: 0.0.0.0) - HTTP server bind address
- `HTTP_PORT` (default: 8080) - HTTP server port
- `DATA_DIR` (default: /data) - Persistent data directory

### Build Options

```bash
# Standard build
docker build -t token-saver-5000:latest .

# Build with custom tag
docker build -t token-saver-5000:v0.7.0 .

# Multi-platform build (requires buildx)
docker buildx build --platform linux/amd64,linux/arm64 -t token-saver-5000:latest .
```

## Persistent Storage

Data persists in the `token-saver-data` volume:
- Compressed documents
- Version history
- File sync metadata
- AFM dialogue state

To reset: `docker volume rm token-saver-data`

## Troubleshooting

**Container won't start:**
```bash
docker logs token-saver-mcp
```

**Model download fails:**
- First run downloads ~80MB embedding model
- Check internet connectivity
- Allow 2-3 minutes for initial setup

**Memory issues:**
- Default: 2GB limit in docker-compose.yml
- Increase for large document sets (>1000 docs)

## Production Deployment

> **Recommended:** For production Kubernetes deployments, see [deployment/kubernetes/README.md](deployment/kubernetes/README.md) for:
> - Horizontal Pod Autoscaling (2-10 replicas based on CPU/memory)
> - Prometheus metrics and alerting (16 production-ready alerts)
> - Zero-downtime rolling updates
> - Health probes (liveness, readiness, startup)
> - High availability (pod anti-affinity)

### Docker Standalone Production

For standalone Docker production deployments:

```bash
# Run with HTTP server for monitoring
docker run -d \
  --name token-saver-mcp \
  --restart unless-stopped \
  -e HTTP_ENABLED=true \
  -p 8080:8080 \
  -v /opt/token-saver/data:/data \
  --memory="2g" \
  --cpus="1.0" \
  --read-only \
  --security-opt=no-new-privileges \
  token-saver-5000:latest
```

**Health Check Endpoints:**
- `GET /health/liveness` - Always returns 200 if container is running
- `GET /health/readiness` - Returns 200 if all components healthy, 503 if unhealthy
- `GET /health/diagnostics` - Detailed performance metrics and resource usage
- `GET /metrics` - Prometheus metrics in text format

**Monitoring:**
```bash
# Check health
curl http://localhost:8080/health/readiness

# View metrics
curl http://localhost:8080/metrics
```
