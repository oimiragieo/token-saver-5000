"""Stable multimodal ingestion and search service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from .media_metadata import inspect_media_path, summarize_payload_bytes
from .multimodal_compressor import MultiModalCompressor


@dataclass
class MultimodalProjectRecord:
    doc_id: str
    content_types: list[str]
    asset_count: int
    manifest: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content_types": list(self.content_types),
            "asset_count": self.asset_count,
            "manifest": [dict(item) for item in self.manifest],
            "stats": dict(self.stats),
        }


class MultimodalIngestionService:
    _instance: "MultimodalIngestionService | None" = None

    def __init__(self):
        self._lock = RLock()
        self._compressor = MultiModalCompressor(
            use_clip_for_images=False, use_codebert_for_code=False
        )
        self._projects: dict[str, MultimodalProjectRecord] = {}

    @classmethod
    def get_service(cls) -> "MultimodalIngestionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        cls._instance = None

    def ingest(
        self, doc_id: str, content_items: list[dict[str, Any]], manifest: list[dict[str, Any]]
    ) -> dict[str, Any]:
        with self._lock:
            stats = self._compressor.ingest_mixed_content(content_items, doc_id)
            record = MultimodalProjectRecord(
                doc_id=doc_id,
                content_types=sorted({item["type"] for item in content_items}),
                asset_count=len(content_items),
                manifest=manifest,
                stats=stats,
            )
            self._projects[doc_id] = record
            return record.to_dict()

    def search(
        self,
        doc_id: str,
        query: str | bytes,
        *,
        query_type: str = "text",
        top_k: int = 5,
        filter_modality: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if doc_id not in self._projects:
                raise ValueError(f"Multimodal project '{doc_id}' not found")

            results = self._compressor.search_cross_modal(
                query,
                query_type=query_type,
                project_id=doc_id,
                top_k=top_k,
                filter_modality=filter_modality,
            )
            enriched = []
            for node_id, score, modality in results:
                node_payload = self._compressor.get_node_content(node_id)
                enriched.append(
                    {
                        "node_id": node_id,
                        "score": round(score, 4),
                        "modality": modality,
                        "metadata": node_payload.get("metadata", {}),
                        "content_preview": (
                            node_payload.get("content", "")[:120]
                            if node_payload.get("content_type") == "text"
                            else node_payload.get("content_type")
                        ),
                    }
                )
            return {
                "doc_id": doc_id,
                "total_results": len(enriched),
                "results": enriched,
            }

    def get_project(self, doc_id: str) -> dict[str, Any]:
        with self._lock:
            if doc_id not in self._projects:
                raise ValueError(f"Multimodal project '{doc_id}' not found")
            return self._projects[doc_id].to_dict()


def build_multimodal_request(
    args: dict[str, Any],
    *,
    path_validator: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    content_items: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    text_content = args.get("text_content")
    if text_content:
        content_items.append(
            {"type": "text", "content": str(text_content), "metadata": {"source": "text"}}
        )
        manifest.append({"type": "text", "size_bytes": len(str(text_content).encode("utf-8"))})

    code_content = args.get("code_content")
    if code_content:
        metadata = {"source": "code", "language": args.get("code_language", "python")}
        content_items.append({"type": "code", "content": str(code_content), "metadata": metadata})
        manifest.append({"type": "code", "language": metadata["language"]})

    image_captions = args.get("image_captions") or {}
    image_ocr_text = args.get("image_ocr_text") or {}
    for raw_path in args.get("image_paths", []) or []:
        if path_validator is None:
            raise ValueError("PathValidator not configured - cannot safely handle image paths")
        validated_path = path_validator.validate(raw_path)
        media_info = inspect_media_path(validated_path)
        if media_info["media_kind"] != "image":
            raise ValueError(f"Unsupported image asset type for '{raw_path}'")
        image_bytes = Path(validated_path).read_bytes()
        metadata = {"source": "image", "file": Path(validated_path).name}
        content_items.append({"type": "image", "content": image_bytes, "metadata": metadata})
        manifest.append(media_info)
        if raw_path in image_captions:
            content_items.append(
                {
                    "type": "text",
                    "content": str(image_captions[raw_path]),
                    "metadata": {"source": "image_caption", "image_path": validated_path},
                }
            )
        if raw_path in image_ocr_text:
            content_items.append(
                {
                    "type": "text",
                    "content": str(image_ocr_text[raw_path]),
                    "metadata": {"source": "image_ocr", "image_path": validated_path},
                }
            )

    for audio_item in args.get("audio_items", []) or []:
        transcript = str(audio_item.get("transcript") or "").strip()
        if not transcript:
            raise ValueError("Audio items require a non-empty transcript")
        metadata = {
            "source": "audio",
            "title": audio_item.get("title"),
            "path": audio_item.get("path"),
        }
        content_items.append({"type": "text", "content": transcript, "metadata": metadata})
        manifest.append(
            {"type": "audio", "title": audio_item.get("title"), "path": audio_item.get("path")}
        )

    for document_item in args.get("document_items", []) or []:
        body = str(document_item.get("text") or "").strip()
        if body:
            content_items.append(
                {
                    "type": "text",
                    "content": body,
                    "metadata": {"source": "document", "title": document_item.get("title")},
                }
            )
            manifest.append({"type": "document", "title": document_item.get("title")})
        nested_paths = document_item.get("image_paths") or []
        nested_args = {
            "image_paths": nested_paths,
            "image_captions": document_item.get("image_captions") or {},
            "image_ocr_text": document_item.get("image_ocr_text") or {},
        }
        nested_content, nested_manifest = build_multimodal_request(
            nested_args,
            path_validator=path_validator,
        )
        content_items.extend(nested_content)
        manifest.extend(nested_manifest)

    if args.get("video_items"):
        raise ValueError(
            "Video ingestion is not production-ready yet; provide extracted transcripts/images instead"
        )

    if not content_items:
        raise ValueError(
            "At least one supported multimodal payload is required: text_content, code_content, image_paths, audio_items, or document_items"
        )

    return content_items, manifest


def estimate_multimodal_payload_bytes(content_items: list[dict[str, Any]]) -> int:
    return summarize_payload_bytes(content_items)
