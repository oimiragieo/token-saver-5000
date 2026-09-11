# Multi-stage Dockerfile for Token Saver 5000 MCP Server
# Builder: full deps + torch (CPU) to export + quantize ONNX
# Runtime: ONNX-only venv (no torch / sentence-transformers / transformers)
# Target image size: <600MB (VAL-DOCKER-003)

# ==============================================================================
# Builder Stage: export + quantize ONNX (needs torch + optimum)
# ==============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv-build
ENV PATH="/opt/venv-build/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^torch([>=<]|$)' requirements.txt > /tmp/requirements-no-torch.txt && \
    pip install --no-cache-dir -r /tmp/requirements-no-torch.txt && \
    pip install --no-cache-dir "optimum[onnxruntime]>=1.15.0"

COPY src/ /app/src/
COPY pyproject.toml /app/
ENV HF_HOME=/root/.cache/huggingface \
    PYTHONPATH=/app

# Export DEFAULT_TEXT_MODEL, then dynamic-quantize weights for a smaller runtime cache.
RUN python - <<'PY'
from pathlib import Path
from src.embeddings_onnx import ONNXEmbeddingManager
from onnxruntime.quantization import QuantType, quantize_dynamic

mgr = ONNXEmbeddingManager()
print("warmup", mgr.encode(["warmup"]).shape)
model_dir = Path.home() / ".cache" / "token-saver-5000" / "BAAI_bge-small-en-v1.5"
onnx_path = model_dir / "model.onnx"
quant_path = model_dir / "model.quant.onnx"
quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QInt8)
onnx_path.unlink()
quant_path.rename(onnx_path)
print("quantized_bytes", onnx_path.stat().st_size)
# Prove torch-free path still loads the quantized file.
mgr2 = ONNXEmbeddingManager.__new__(ONNXEmbeddingManager)
mgr2.__init__()
print("requant_ok", mgr2.encode(["warmup"]).shape)
PY

# Stage only the exported model dir (not the HF hub blob mirror).
RUN mkdir -p /opt/onnx-export && \
    cp -a /root/.cache/token-saver-5000/BAAI_bge-small-en-v1.5 /opt/onnx-export/

# ==============================================================================
# Runtime Stage: slim ONNX-only image
# ==============================================================================
FROM python:3.12-slim AS runtime

RUN useradd -m -u 1000 -s /bin/bash mcp \
    && mkdir -p /app /data /home/mcp/.cache/huggingface /home/mcp/.cache/token-saver-5000 \
    && chown -R mcp:mcp /app /data /home/mcp

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-docker-runtime.txt /tmp/requirements-docker-runtime.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-docker-runtime.txt && \
    pip uninstall -y pip setuptools wheel huggingface-hub hf_xet 2>/dev/null || true && \
    find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name 'tests' -path '*/site-packages/*' -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /tmp/requirements-docker-runtime.txt /root/.cache/pip /tmp/*

COPY --chown=mcp:mcp src/ /app/src/
COPY --chown=mcp:mcp pyproject.toml /app/
COPY --chown=mcp:mcp --from=builder /opt/onnx-export/BAAI_bge-small-en-v1.5 \
    /home/mcp/.cache/token-saver-5000/BAAI_bge-small-en-v1.5

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/mcp/.cache/huggingface \
    EMBEDDING_TIER=onnx \
    HTTP_ENABLED=false \
    HTTP_HOST=0.0.0.0 \
    HTTP_PORT=8080 \
    DATA_DIR=/data \
    PYTHONPATH=/app

USER mcp

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD if [ "$HTTP_ENABLED" = "true" ]; then python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('HTTP_PORT','8080')+'/health/liveness')" || exit 1; else exit 0; fi

EXPOSE 8080
VOLUME ["/data"]

CMD ["python", "-m", "src.server"]

# Build: docker build -t gotcontext:test .
# Size:  docker images gotcontext:test --format "{{.Size}}"   # must be <600MB
