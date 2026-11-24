"""Wiring tests for Wave 1 handlers (CL 4)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.handlers.compression_handlers_extended import handle_advise_context
from src.handlers.compression_handlers_ingest import handle_modulate_region
from src.handlers.token_optimization_handlers import (
    _get_tracker,
    _savings_trackers,
    handle_get_original_output,
)


@pytest.mark.asyncio
async def test_modulate_region_records_expansion_debit():
    _savings_trackers.pop("w1-mod-session", None)
    compressor = MagicMock()
    compressor.modulate_region.return_value = "expanded text " * 40
    sync_manager = MagicMock()
    sync_manager.file_metadata = {}
    context = {
        "compressor": compressor,
        "sync_manager": sync_manager,
        "retrieval_history": {},
    }

    with (
        patch("src.handlers.compression_handlers_ingest.validate_node_ids"),
        patch(
            "src.handlers.compression_handlers_ingest.extract_file_id_from_node",
            return_value="doc1",
        ),
        patch("src.savings_tracker.SavingsTracker._init_journal"),
    ):
        out = await handle_modulate_region(
            context,
            {"node_ids": ["doc1_n0"], "fidelity_level": "RAW", "session_id": "w1-mod-session"},
        )

    assert "expanded text" in out
    report = _get_tracker("w1-mod-session").get_report()
    assert report.total_operations == 1
    assert report.total_tokens_saved < 0
    _savings_trackers.pop("w1-mod-session", None)


@pytest.mark.asyncio
async def test_get_original_output_records_expansion_debit():
    _savings_trackers.pop("w1-tee-session", None)
    fake_entry = SimpleNamespace(
        entry_id="tee1",
        original_text="original output " * 20,
        source="cli",
        command_hint="pytest",
        compression_pct=80.0,
        timestamp=123.0,
    )
    fake_store = MagicMock()
    fake_store.get.return_value = fake_entry

    with (
        patch("src.handlers.token_optimization_handlers._get_tee_store", return_value=fake_store),
        patch("src.savings_tracker.SavingsTracker._init_journal"),
    ):
        raw = await handle_get_original_output(
            {}, {"entry_id": "tee1", "session_id": "w1-tee-session"}
        )

    payload = json.loads(raw)
    assert payload["status"] == "success"
    report = _get_tracker("w1-tee-session").get_report()
    assert report.total_operations == 1
    assert report.total_tokens_saved < 0
    _savings_trackers.pop("w1-tee-session", None)


@pytest.mark.asyncio
async def test_handle_advise_context_with_explain_true():
    compressor = MagicMock()
    compressor.graphs = {}
    compressor.chunks = {}
    out = await handle_advise_context({"compressor": compressor}, {"explain": True})
    payload = json.loads(out)
    assert "explain_analyze_trace" in payload
    assert payload["explain_analyze_trace"]["decision"] == "abstain"


def test_get_tracker_does_not_overwrite_existing_model_when_omitted():
    _savings_trackers.pop("w1-model-keep", None)
    try:
        t1 = _get_tracker("w1-model-keep", "gpt-5.1-codex")
        assert t1.model == "gpt-5.1-codex"
        t2 = _get_tracker("w1-model-keep")
        assert t2.model == "gpt-5.1-codex"
    finally:
        _savings_trackers.pop("w1-model-keep", None)


def test_prompt_registry_wires_cache_align_sections():
    from src.prompt_registry import PromptRegistry

    registry = PromptRegistry(seed_defaults=False)
    registry.create_template(
        name="t1",
        description="desc",
        system_prompt="Be concise.",
        user_prompt_template="Summarize {topic}",
        variables=["topic"],
    )
    rendered = registry.render_prompt("t1", variables={"topic": "diffs"})
    assert "cache_alignment" in rendered
    assert rendered["cache_alignment"]["has_volatile_prefix_leak"] is False


def test_file_sync_manager_wires_session_diff_tracker():
    from src.file_sync_manager import FileSyncManager

    fsm = FileSyncManager()
    first = fsm.track_or_diff_reread("s1", "a.py", "line 1\nline 2\n")
    assert first["mode"] == "full"
    second = fsm.track_or_diff_reread("s1", "a.py", "line 1\nline 2\n")
    assert second["mode"] == "unchanged"


def test_advise_context_schema_declares_explain_properties():
    from src.handlers.mcp_core.schemas_compression_batch import COMPRESSION_BATCH_TOOLS

    tool = next(t for t in COMPRESSION_BATCH_TOOLS if t.name == "advise_context")
    props = tool.inputSchema["properties"]
    assert "explain" in props
    assert "model" in props
    assert "expected_hit_rate" in props


@pytest.mark.asyncio
async def test_check_file_sync_wires_track_or_diff_reread(tmp_path):
    from src.file_sync_manager import FileSyncManager
    from src.handlers.file_sync_handlers import handle_check_file_sync
    from src.identity_scope import compose_scoped_file_id

    target = tmp_path / "sample.py"
    lines = [f"line {i}: sample code content for diff reread" for i in range(1, 31)]
    initial = "\n".join(lines) + "\n"
    target.write_text(initial, encoding="utf-8")

    fsm = FileSyncManager()
    scoped_id = compose_scoped_file_id("doc_reread", session_id="s_reread")
    fsm.register_file(scoped_id, str(target), initial)
    ctx = {"sync_manager": fsm}

    out1 = json.loads(
        await handle_check_file_sync(
            ctx, {"file_id": "doc_reread", "session_id": "s_reread", "diff_on_reread": True}
        )
    )
    assert out1["reread"]["mode"] == "full"

    out2 = json.loads(
        await handle_check_file_sync(
            ctx, {"file_id": "doc_reread", "session_id": "s_reread", "diff_on_reread": True}
        )
    )
    assert out2["reread"]["mode"] == "unchanged"

    updated_lines = list(lines)
    updated_lines[10] = "line 11: sample code content updated"
    target.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    out3 = json.loads(
        await handle_check_file_sync(
            ctx, {"file_id": "doc_reread", "session_id": "s_reread", "diff_on_reread": True}
        )
    )
    assert out3["reread"]["mode"] == "diff"


@pytest.mark.asyncio
async def test_check_file_sync_preserves_crlf_on_reread(tmp_path):
    from src.file_sync_manager import FileSyncManager
    from src.handlers.file_sync_handlers import handle_check_file_sync
    from src.identity_scope import compose_scoped_file_id

    target = tmp_path / "crlf_sample.py"
    lines = [f"line {i}: crlf content for reread check" for i in range(1, 31)]
    initial = "\r\n".join(lines) + "\r\n"
    target.write_bytes(initial.encode("utf-8"))

    fsm = FileSyncManager()
    scoped_id = compose_scoped_file_id("doc_crlf", session_id="s_crlf")
    fsm.register_file(scoped_id, str(target), initial)
    ctx = {"sync_manager": fsm}

    out1 = json.loads(
        await handle_check_file_sync(
            ctx, {"file_id": "doc_crlf", "session_id": "s_crlf", "diff_on_reread": True}
        )
    )
    assert out1["reread"]["mode"] == "full"
    assert "\r\n" in out1["reread"]["content"]

    out2 = json.loads(
        await handle_check_file_sync(
            ctx, {"file_id": "doc_crlf", "session_id": "s_crlf", "diff_on_reread": True}
        )
    )
    assert out2["reread"]["mode"] == "unchanged"

    updated_lines = list(lines)
    updated_lines[10] = "line 11: crlf content updated"
    target.write_bytes(("\r\n".join(updated_lines) + "\r\n").encode("utf-8"))
    out3 = json.loads(
        await handle_check_file_sync(
            ctx, {"file_id": "doc_crlf", "session_id": "s_crlf", "diff_on_reread": True}
        )
    )
    assert out3["reread"]["mode"] == "diff"
