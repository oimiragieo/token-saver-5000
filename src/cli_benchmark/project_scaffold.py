"""MCP-mode project folder generator."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLAUDE_MD_TOKENSAVER = """\
# Token Saver Benchmark Project

You have the Token Saver 5000 MCP server available.

## MANDATORY Workflow

Before answering any questions about the document:

1. Read corpus.txt
2. Call `ingest_context` with the file content and file_id="corpus"
3. Call `read_skeleton` with file_id="corpus" to get compressed overview
4. Use the compressed skeleton to answer the questions

Do NOT read the raw file and answer directly. You MUST compress first.
"""

GEMINI_MD_TOKENSAVER = """\
# Token Saver Benchmark Project

You have the Token Saver 5000 MCP server available.

## MANDATORY Workflow

Before answering any questions about the document:

1. Read corpus.txt
2. Call `ingest_context` with the file content and file_id="corpus"
3. Call `read_skeleton` with file_id="corpus" to get compressed overview
4. Use the compressed skeleton to answer the questions

Do NOT read the raw file and answer directly. You MUST compress first.
"""


def _mcp_settings_claude() -> dict:
    """Generate .claude/settings.json with Token Saver MCP."""
    return {
        "mcpServers": {
            "token-saver": {
                "command": "token-saver-mcp",
                "args": [],
                "cwd": str(PROJECT_ROOT),
                "env": {"PYTHONPATH": str(PROJECT_ROOT)},
            }
        }
    }


def _mcp_settings_gemini() -> dict:
    """Generate .gemini/settings.json with Token Saver MCP."""
    return {
        "mcpServers": {
            "token-saver": {
                "command": "token-saver-mcp",
                "args": [],
                "cwd": str(PROJECT_ROOT),
                "env": {"PYTHONPATH": str(PROJECT_ROOT)},
            }
        }
    }


def create_vanilla(corpus_path: Path, provider: str) -> Path:
    """Create a vanilla project folder with just the corpus file."""
    tmp = Path(tempfile.mkdtemp(prefix=f"bench_vanilla_{provider}_"))
    shutil.copy2(corpus_path, tmp / "corpus.txt")
    return tmp


def create_with_mcp(corpus_path: Path, provider: str) -> Path:
    """Create a project folder with Token Saver MCP configured."""
    tmp = Path(tempfile.mkdtemp(prefix=f"bench_mcp_{provider}_"))
    shutil.copy2(corpus_path, tmp / "corpus.txt")

    if provider == "claude":
        config_dir = tmp / ".claude"
        config_dir.mkdir()
        with open(config_dir / "settings.json", "w") as f:
            json.dump(_mcp_settings_claude(), f, indent=2)
        (tmp / "CLAUDE.md").write_text(CLAUDE_MD_TOKENSAVER, encoding="utf-8")
    elif provider == "gemini":
        config_dir = tmp / ".gemini"
        config_dir.mkdir()
        with open(config_dir / "settings.json", "w") as f:
            json.dump(_mcp_settings_gemini(), f, indent=2)
        (tmp / "GEMINI.md").write_text(GEMINI_MD_TOKENSAVER, encoding="utf-8")

    return tmp


def cleanup(path: Path) -> None:
    """Remove a scaffold directory."""
    if path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
