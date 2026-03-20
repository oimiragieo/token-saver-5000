"""Helpers for multimodal media metadata inspection and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".md", ".txt", ".html", ".docx"}


def detect_media_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if suffix in SUPPORTED_AUDIO_EXTENSIONS:
        return "audio"
    if suffix in SUPPORTED_DOCUMENT_EXTENSIONS:
        return "document"
    return "unknown"


def inspect_media_path(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"Media path does not exist: {path}")

    kind = detect_media_kind(path)
    return {
        "path": str(file_path),
        "name": file_path.name,
        "suffix": file_path.suffix.lower(),
        "media_kind": kind,
        "size_bytes": file_path.stat().st_size,
    }


def summarize_payload_bytes(items: list[dict[str, Any]]) -> int:
    total = 0
    for item in items:
        content = item.get("content")
        if isinstance(content, bytes):
            total += len(content)
        elif isinstance(content, str):
            total += len(content.encode("utf-8"))
    return total
