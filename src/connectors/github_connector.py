"""GitHub export connector normalization."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseConnector, ConnectorDocument, sanitize_segment

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GitHubConnector(BaseConnector):
    connector_type = "github"
    description = "Normalize exported GitHub repo files into ingestible documents."

    def validate_config(self, config: dict[str, Any]) -> None:
        repo = config.get("repo")
        files = config.get("files")
        if not isinstance(repo, str) or not _REPO_RE.match(repo.strip()):
            raise ValueError("github connector requires repo in 'owner/repo' format")
        if not isinstance(files, list) or not files:
            raise ValueError("github connector requires a non-empty 'files' list")
        for file_entry in files:
            if not isinstance(file_entry, dict):
                raise ValueError("github connector files must be objects")
            if not isinstance(file_entry.get("path"), str) or not file_entry["path"].strip():
                raise ValueError("github connector files require non-empty 'path'")

    def collect_documents(self, config: dict[str, Any]) -> list[ConnectorDocument]:
        self.validate_config(config)
        repo = config["repo"].strip()
        documents: list[ConnectorDocument] = []
        for file_entry in config["files"]:
            path = file_entry["path"].strip()
            ref = str(file_entry.get("ref") or config.get("ref") or "HEAD")
            content = file_entry.get("content") or f"Captured GitHub file {repo}:{path}@{ref}"
            documents.append(
                ConnectorDocument(
                    source_id=f"github://{repo}/{path}@{ref}",
                    file_id=f"github-{sanitize_segment(repo)}-{sanitize_segment(path)}",
                    text=str(content),
                    metadata={
                        "connector_type": self.connector_type,
                        "repo": repo,
                        "path": path,
                        "ref": ref,
                        "source": "github",
                    },
                )
            )
        return documents
