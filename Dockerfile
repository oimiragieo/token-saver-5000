# Multi-stage Dockerfile for Token Saver 5000 MCP Server
# Builder stage: Install dependencies and download models
# Runtime stage: Minimal image with only runtime dependencies
# Target image size: <500MB

# ==============================================================================
# Builder Stage: Install dependencies and download models
# ==============================================================================
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in a virtual environment
# Using venv ensures clean separation from system packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with pip cache.
# Force CPU torch: default PyPI wheels pull CUDA (~4GB+) and blow past the
# <600MB image-size contract. Runtime never needs GPU in this MCP image.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^torch([>=<]|$)' requirements.txt > /tmp/requirements-no-torch.txt && \
    pip install --no-cache-dir -r /tmp/requirements-no-torch.txt && \
    pip install --no-cache-dir "optimum[onnxruntime]>=1.15.0"

# Pre-cache DEFAULT_TEXT_MODEL (must match src.constants.DEFAULT_TEXT_MODEL).
# Also export ONNX so VAL-DOCKER-001 can take the torch-free ORT path as mcp.
COPY src/ /app/src/
COPY pyproject.toml /app/
ENV HF_HOME=/root/.cache/huggingface \
    PYTHONPATH=/app
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" && \
    python -c "from src.embeddings_onnx import ONNXEmbeddingManager; print(ONNXEmbeddingManager().encode(['warmup']).shape)"

# ==============================================================================
# Runtime Stage: Minimal image with only runtime dependencies
# ==============================================================================
FROM python:3.12-slim AS runtime

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
# Running as non-root is a security best practice
RUN useradd -m -u 1000 -s /bin/bash mcp && \
    mkdir -p /app /data && \
    chown -R mcp:mcp /app /data

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=mcp:mcp src/ /app/src/
COPY --chown=mcp:mcp pyproject.toml /app/

# Copy HF + ONNX model caches from builder (includes ~/.cache/token-saver-5000)
COPY --from=builder /root/.cache /home/mcp/.cache
RUN chown -R mcp:mcp /home/mcp/.cache

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/mcp/.cache/huggingface \
    # HTTP server configuration (optional, disabled by default)
    HTTP_ENABLED=false \
    HTTP_HOST=0.0.0.0 \
    HTTP_PORT=8080 \
    # Data storage directory
    DATA_DIR=/data \
    # Python path
    PYTHONPATH=/app

# Switch to non-root user
USER mcp

# Health check (requires HTTP_ENABLED=true)
# Kubernetes will use /health/liveness and /health/readiness endpoints
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD if [ "$HTTP_ENABLED" = "true" ]; then curl -f http://localhost:${HTTP_PORT}/health/liveness || exit 1; else exit 0; fi

# Expose HTTP port (for health checks and metrics)
# Only used if HTTP_ENABLED=true
EXPOSE 8080

# Volume for persistent data (semantic modulator data, version history)
VOLUME ["/data"]

# Default command: Run MCP server in stdio mode
# Override with HTTP-enabled startup for Kubernetes deployment
CMD ["python", "-m", "src.server"]

# ==============================================================================
# Build Instructions
# ==============================================================================
# Build image:
#   docker build -t token-saver-5000:latest .
#
# Run with stdio mode (default):
#   docker run -i token-saver-5000:latest
#
# Run with HTTP server enabled (for Kubernetes):
#   docker run -e HTTP_ENABLED=true -p 8080:8080 token-saver-5000:latest
#
# Run with volume for persistent data:
#   docker run -v $(pwd)/data:/data -i token-saver-5000:latest
#
# Development mode with source code mounted:
#   docker run -v $(pwd)/src:/app/src -e HTTP_ENABLED=true -p 8080:8080 token-saver-5000:latest
#
# ==============================================================================
# Image Size Optimization
# ==============================================================================
# Expected image size: ~500-800MB with CPU torch + bge-small + ONNX export.
# Previous CUDA torch wheels inflated the image to ~9GB.
# Further cuts: drop torch entirely and serve ONNX-only (~200-400MB).
# ==============================================================================
