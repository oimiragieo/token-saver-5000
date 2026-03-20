"""Slack export connector normalization."""

from __future__ import annotations

from typing import Any

from .base import BaseConnector, ConnectorDocument, sanitize_segment


class SlackExportConnector(BaseConnector):
    connector_type = "slack_export"
    description = "Normalize Slack export channels/messages into channel transcript documents."

    def validate_config(self, config: dict[str, Any]) -> None:
        channels = config.get("channels")
        if not isinstance(channels, list) or not channels:
            raise ValueError("slack_export connector requires a non-empty 'channels' list")
        for channel in channels:
            if not isinstance(channel, dict):
                raise ValueError("slack_export channels must be objects")
            if not isinstance(channel.get("name"), str) or not channel["name"].strip():
                raise ValueError("slack_export channels require non-empty 'name'")

    def collect_documents(self, config: dict[str, Any]) -> list[ConnectorDocument]:
        self.validate_config(config)
        workspace = str(config.get("workspace") or "workspace")
        documents: list[ConnectorDocument] = []
        for channel in config["channels"]:
            channel_name = channel["name"].strip()
            messages = channel.get("messages") or []
            transcript_lines = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                user = str(message.get("user") or "unknown")
                text = str(message.get("text") or "")
                ts = str(message.get("ts") or "")
                transcript_lines.append(f"[{ts}] {user}: {text}".strip())
            transcript = "\n".join(transcript_lines) or f"Slack export for #{channel_name}"
            documents.append(
                ConnectorDocument(
                    source_id=f"slack://{workspace}/{channel_name}",
                    file_id=f"slack-{sanitize_segment(workspace)}-{sanitize_segment(channel_name)}",
                    text=transcript,
                    metadata={
                        "connector_type": self.connector_type,
                        "workspace": workspace,
                        "channel": channel_name,
                        "message_count": len(messages),
                        "source": "slack_export",
                    },
                )
            )
        return documents
