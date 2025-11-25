# Token Saver 5000 MCP Server - Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Pre-download embedding model (avoids first-run delay)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create directories for persistent data
RUN mkdir -p /data/chromadb /data/json_backup /data/afm_exports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STORAGE_BACKEND=json
ENV DATA_DIR=/data

# Expose volume for persistence
VOLUME ["/data"]

# Health check (optional - checks if server starts)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src import __version__; print(__version__)" || exit 1

# Run the MCP server via stdio
ENTRYPOINT ["python", "-m", "src.server"]
