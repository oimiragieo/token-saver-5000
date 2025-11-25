# Docker Deployment

Run Token Saver 5000 as a Docker container for easy deployment and isolation.

## Quick Start

```bash
# Start the container
docker-compose up -d

# Test it works
echo '{"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1}' | \
  docker exec -i token-saver-mcp python -m src.server
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

## Manual Build

```bash
# Build image
docker build -t token-saver-5000:latest .

# Run container
docker run -d \
  --name token-saver-mcp \
  -v token-saver-data:/data \
  token-saver-5000:latest
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

For production, configure resource limits in `docker-compose.yml`:

```yaml
services:
  token-saver-mcp:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 1G
```

Mount persistent storage:

```yaml
    volumes:
      - ./data:/data  # Explicit local directory
```

Enable health checks:

```yaml
    healthcheck:
      test: ["CMD", "python", "-c", "import src.server"]
      interval: 30s
      timeout: 10s
      retries: 3
```
