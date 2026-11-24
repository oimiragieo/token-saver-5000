"""Tests for Wave 3 DTOC manage_context, TeeStore redaction & scope isolation."""

from __future__ import annotations

import hashlib
import json

import pytest

from src.afm import FocusManager
from src.context_visibility import ContextVisibilityManager, ManageContextRequest
from src.handlers.mcp_core.dispatch import route_tool_call
from src.handlers.mcp_core.setup import setup_mcp_tools
from src.savings_tracker import SavingsTracker
from src.tee_recovery import TEE_MODE_ALWAYS, TeeStore


class TestTeeStoreSecurityAndTokens:
    def test_estimated_tokens_and_visibility_populated(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        eid = store.store(
            original_text="hello world " * 20,
            compressed_text="hello",
            compression_pct=90.0,
            source="cli_optimizer",
            workspace_id="ws_alpha",
            session_id="sess_1",
        )
        assert eid is not None
        entry = store.get(eid, workspace_id="ws_alpha", session_id="sess_1")
        assert entry is not None
        assert entry.estimated_tokens > 0
        assert entry.visible is True

    def test_secret_redaction_before_disk_persistence(self, tmp_path):
        canary_anthropic = "sk-ant-api03-" + "canary_secret_value_123456"
        canary_github = "ghp_" + "canary_token_9876543210"
        raw_text = f"AUTH={canary_anthropic}\nGH={canary_github}\nstatus=ok"

        store = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=str(tmp_path))
        eid = store.store(raw_text, "status=ok", 90.0, "cli_optimizer")
        assert eid is not None

        persisted_json = (tmp_path / f"{eid}.json").read_text(encoding="utf-8")
        assert canary_anthropic not in persisted_json
        assert canary_github not in persisted_json
        assert "[REDACTED_SECRET]" in persisted_json

    def test_scope_isolation_on_get_and_set_visibility(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        eid = store.store(
            "sensitive log",
            "log",
            90.0,
            "cli_optimizer",
            workspace_id="ws_1",
            session_id="s_1",
        )
        assert eid is not None

        with pytest.raises(PermissionError):
            store.get_original(eid, workspace_id="ws_2", session_id="s_1")

        with pytest.raises(PermissionError):
            store.set_visibility(eid, False, workspace_id="ws_1", session_id="s_2")


class TestContextVisibilityManager:
    def test_disable_and_enable_with_sha256_and_debit(self, tmp_path):
        trackers: dict[str, SavingsTracker] = {}

        def _factory(sid: str) -> SavingsTracker:
            if sid not in trackers:
                trackers[sid] = SavingsTracker(session_id=sid, storage_dir=tmp_path)
            return trackers[sid]

        mgr = ContextVisibilityManager(savings_tracker_factory=_factory)
        original = "def authenticate(user_id):\n    return verify_jwt(user_id)\n" * 10
        expected_sha = hashlib.sha256(original.encode("utf-8")).hexdigest()

        handle = mgr.register(
            tool_key="auth_trace",
            original_text=original,
            summary="JWT auth trace",
            workspace_id="ws_a",
            session_id="sess_a",
        )
        assert handle.sha256 == expected_sha
        assert handle.visible is True

        dis = mgr.disable(["auth_trace"], workspace_id="ws_a", session_id="sess_a")
        assert dis["disabled"] == ["auth_trace"]
        assert dis["handles"]["auth_trace"].startswith("[CTX_HIDDEN key=auth_trace ")
        assert (
            mgr.render("auth_trace", workspace_id="ws_a", session_id="sess_a")
            == dis["handles"]["auth_trace"]
        )

        en = mgr.enable(["auth_trace"], workspace_id="ws_a", session_id="sess_a")
        assert en["enabled"] == ["auth_trace"]
        assert en["restored"]["auth_trace"] == original
        assert en["tokens_debited"] > 0

        report = trackers["sess_a"].get_report()
        assert report.total_tokens_saved == -en["tokens_debited"]

    def test_cross_scope_access_raises_permission_error(self):
        mgr = ContextVisibilityManager()
        mgr.register("k1", "payload", workspace_id="ws_1", session_id="s_1")

        with pytest.raises(PermissionError):
            mgr.disable(["k1"], workspace_id="ws_2", session_id="s_1")

        with pytest.raises(PermissionError):
            mgr.enable(["k1"], workspace_id="ws_1", session_id="s_2")

    def test_focus_manager_renders_hidden_handle(self):
        mgr = ContextVisibilityManager()
        mgr.register(
            "build_log",
            "Compiling module A...\n" * 25,
            summary="Build output",
            workspace_id="ws_fm",
            session_id="s_fm",
        )
        mgr.disable(["build_log"], workspace_id="ws_fm", session_id="s_fm")

        fm = FocusManager(context_visibility=mgr)
        fm.add_message(
            "user",
            "Compiling module A...\n" * 25,
            tool_key="build_log",
            workspace_id="ws_fm",
            session_id="s_fm",
        )
        packed, _ = fm.build_context("Compiling module A", budget_tokens=500)
        assert any("[CTX_HIDDEN key=build_log" in content for _, content in packed)


class TestManageContextMCPTool:
    def test_manage_context_registered_in_setup_mcp_tools(self):
        names = {tool.name for tool in setup_mcp_tools("full")}
        assert "manage_context" in names

    @pytest.mark.asyncio
    async def test_route_tool_call_manage_context_roundtrip(self, tmp_path):
        mgr = ContextVisibilityManager(
            savings_tracker_factory=lambda sid: SavingsTracker(session_id=sid, storage_dir=tmp_path)
        )
        mgr.register(
            "tool_1", "alpha beta gamma " * 15, summary="sample", workspace_id="w1", session_id="s1"
        )

        raw_disable = await route_tool_call(
            "manage_context",
            {"disable": ["tool_1"], "workspace_id": "w1", "session_id": "s1"},
            {"context_visibility": mgr},
        )
        dis_payload = json.loads(raw_disable)
        assert dis_payload["status"] == "success"
        assert dis_payload["disabled"] == ["tool_1"]
        assert "[CTX_HIDDEN key=tool_1" in dis_payload["handles"]["tool_1"]

        raw_denied = await route_tool_call(
            "manage_context",
            {"enable": ["tool_1"], "workspace_id": "w2", "session_id": "s1"},
            {"context_visibility": mgr},
        )
        denied_payload = json.loads(raw_denied)
        assert denied_payload["status"] == "error"

        raw_enable = await route_tool_call(
            "manage_context",
            {"enable": ["tool_1"], "workspace_id": "w1", "session_id": "s1"},
            {"context_visibility": mgr},
        )
        en_payload = json.loads(raw_enable)
        assert en_payload["status"] == "success"
        assert en_payload["enabled"] == ["tool_1"]
        assert en_payload["tokens_debited"] > 0

        # ManageContextRequest direct dataclass check
        req = ManageContextRequest(
            enable=["tool_1"], disable=[], workspace_id="w1", session_id="s1"
        )
        assert req.enable == ["tool_1"]


class TestW3AuditRegressions:
    def test_tee_id_uses_full_content_and_scope(self):
        prefix = "A" * 512
        id1 = TeeStore.generate_id(prefix + "tail_one", "cli", "ws1", "s1")
        id2 = TeeStore.generate_id(prefix + "tail_two", "cli", "ws1", "s1")
        id3 = TeeStore.generate_id(prefix + "tail_one", "cli", "ws2", "s1")
        assert id1 != id2
        assert id1 != id3

    def test_tee_unscoped_blocked_for_scoped_entries_and_allowed_for_legacy(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        legacy_id = store.store("legacy data", "legacy", 90.0, "cli")
        global_id = store.store(
            "global data", "global", 90.0, "cli", workspace_id="__global__", session_id="__global__"
        )
        scoped_id = store.store(
            "secret scoped", "scoped", 90.0, "cli", workspace_id="ws_priv", session_id="s_priv"
        )
        assert legacy_id and global_id and scoped_id

        assert store.get_original(legacy_id) == "legacy data"
        assert store.get_original(global_id) == "global data"
        with pytest.raises(PermissionError):
            store.get(scoped_id)
        with pytest.raises(PermissionError):
            store.get_original(scoped_id)
        with pytest.raises(PermissionError):
            store.delete(scoped_id)

        unscoped_ids = {e["entry_id"] for e in store.list_entries()}
        assert scoped_id not in unscoped_ids
        assert legacy_id in unscoped_ids
        assert store.get_stats()["entry_count"] == 2
        assert store.get_stats(workspace_id="ws_priv", session_id="s_priv")["entry_count"] == 1
        assert store.clear(workspace_id="ws_priv", session_id="s_priv") == 1
        assert store.get(legacy_id) is not None

    def test_tee_recursive_metadata_redaction_and_flag(self, tmp_path):
        canary_key = "sk-ant-api03-" + "recursive_meta_secret_999"
        canary_gh = "ghp_" + "nested_list_secret_888"
        store = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=str(tmp_path))
        eid = store.store(
            original_text="clean body",
            compressed_text="clean",
            compression_pct=90.0,
            source="cli",
            command_hint=f"curl -H 'Authorization: Bearer {canary_key}'",
            metadata={"nested": {"tokens": [canary_gh], "ok": 1}},
        )
        assert eid is not None
        raw_disk = (tmp_path / f"{eid}.json").read_text(encoding="utf-8")
        assert canary_key not in raw_disk
        assert canary_gh not in raw_disk
        assert "[REDACTED_SECRET]" in raw_disk

        reloaded = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=str(tmp_path))
        loaded_entry = reloaded.get(eid)
        assert loaded_entry is not None
        assert loaded_entry.redacted_on_disk is True

        clean_id = store.store("plain text", "plain", 90.0, "cli")
        assert clean_id is not None
        reloaded_clean = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=str(tmp_path))
        assert reloaded_clean.get(clean_id).redacted_on_disk is False

    @pytest.mark.asyncio
    async def test_handle_get_original_output_enforces_scope(self):
        from src.handlers import token_optimization_handlers as toh

        store = toh._get_tee_store("s_scope")
        eid = store.store(
            "scoped original",
            "comp",
            90.0,
            "cli",
            workspace_id="ws_owner",
            session_id="s_scope",
        )
        assert eid is not None

        denied = json.loads(
            await toh.handle_get_original_output(
                {}, {"entry_id": eid, "workspace_id": "ws_other", "session_id": "s_scope"}
            )
        )
        assert denied["status"] == "error"

        allowed = json.loads(
            await toh.handle_get_original_output(
                {}, {"entry_id": eid, "workspace_id": "ws_owner", "session_id": "s_scope"}
            )
        )
        assert allowed["status"] == "success"
        assert allowed["original_text"] == "scoped original"

    def test_multi_scope_cvm_isolation_and_atomic_rollback(self, tmp_path):
        trackers: dict[str, SavingsTracker] = {}

        def _factory(sid: str) -> SavingsTracker:
            if sid not in trackers:
                trackers[sid] = SavingsTracker(session_id=sid, storage_dir=tmp_path)
            return trackers[sid]

        mgr = ContextVisibilityManager(savings_tracker_factory=_factory)
        mgr.register("shared_key", "alpha payload " * 10, workspace_id="ws1", session_id="s1")
        mgr.register("shared_key", "beta payload " * 10, workspace_id="ws2", session_id="s1")
        mgr.register("k2", "gamma payload " * 10, workspace_id="ws1", session_id="s1")

        # Disabling same key twice or with duplicates counts newly hidden tokens once
        res = mgr.disable(["shared_key", "shared_key"], workspace_id="ws1", session_id="s1")
        assert res["disabled"] == ["shared_key"]
        first_hidden = res["tokens_hidden"]
        assert first_hidden > 0
        assert (
            mgr.disable(["shared_key"], workspace_id="ws1", session_id="s1")["tokens_hidden"] == 0
        )
        # Other scope untouched
        assert mgr.get_handle("shared_key", workspace_id="ws2", session_id="s1").visible is True

        # Atomic rollback on batch failure in enable (missing second key)
        with pytest.raises(KeyError):
            mgr.enable(["shared_key", "nonexistent_key"], workspace_id="ws1", session_id="s1")
        assert mgr.get_handle("shared_key", workspace_id="ws1", session_id="s1").visible is False
        assert "s1" not in trackers

        # Atomic rollback on manage() when enable fails after disable
        with pytest.raises(KeyError):
            mgr.manage(
                ManageContextRequest(
                    disable=["k2"],
                    enable=["missing_key"],
                    workspace_id="ws1",
                    session_id="s1",
                )
            )
        assert mgr.get_handle("k2", workspace_id="ws1", session_id="s1").visible is True

    @pytest.mark.asyncio
    async def test_manage_context_validation_and_schema(self):
        from src.handlers.mcp_core.schemas_token_optimization import TOKEN_OPTIMIZATION_TOOLS

        schema = next(t.inputSchema for t in TOKEN_OPTIMIZATION_TOOLS if t.name == "manage_context")
        assert "user_id" not in schema["properties"]
        assert "agent_id" not in schema["properties"]

        mgr = ContextVisibilityManager()
        mgr.register("k1", "hello world", workspace_id="w1", session_id="s1")

        bad_cases = [
            {"enable": "k1"},
            {"disable": [123]},
            {"enable": ["   "]},
            {"enable": ["k1"], "disable": ["k1"]},
            {"enable": ["k" * 257]},
            {"disable": [f"k{i}" for i in range(101)]},
            {"enable": ["k1"], "workspace_id": ""},
            {"enable": ["k1"], "session_id": 42},
        ]
        for args in bad_cases:
            resp = json.loads(
                await route_tool_call("manage_context", args, {"context_visibility": mgr})
            )
            assert resp["status"] == "error", f"Expected error for {args}"

    @pytest.mark.asyncio
    async def test_production_wiring_filter_cli_and_server_factory(self):
        from src.handlers.token_optimization_handlers import (
            get_context_visibility_manager,
            handle_filter_cli_output,
        )
        from src.semantic_modulator.app.server_factory_service import ServerFactoryService

        raw = "line 1\nline 2\n" * 10
        res = json.loads(
            await handle_filter_cli_output(
                {}, {"text": raw, "workspace_id": "ws_prod", "session_id": "s_prod"}
            )
        )
        assert res["status"] == "success"
        assert "tool_key" in res
        cvm = get_context_visibility_manager()
        assert cvm.render(res["tool_key"], workspace_id="ws_prod", session_id="s_prod") == raw

        artifacts = ServerFactoryService.build_default(
            preload_code_model=False,
            cwd=".",
            home_dir=".",
            max_ace_contexts=5,
            logger=type("L", (), {"info": lambda *a, **k: None, "warning": lambda *a, **k: None})(),
        )
        assert artifacts["focus_manager"].context_visibility is cvm

    def test_tee_entry_id_traversal_rejected(self, tmp_path):
        store = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=str(tmp_path))
        for bad_id in ("../escape", "..\\escape", "sub/dir", "", "a" * 129, "bad.id"):
            with pytest.raises(ValueError):
                store.get(bad_id)
            with pytest.raises(ValueError):
                store.delete(bad_id)
            with pytest.raises(ValueError):
                store._entry_path(bad_id)

    def test_tee_field_aware_and_source_redaction(self, tmp_path):
        canary_src = "sk-ant-api03-" + "source_secret_98765"
        store = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=str(tmp_path))
        eid = store.store(
            original_text="clean body",
            compressed_text="clean",
            compression_pct=90.0,
            source=f"cli {canary_src}",
            metadata={"api_key": "ordinary-value-123", "nested": {"Auth-Token": "plain-tok"}},
        )
        assert eid is not None
        raw_disk = (tmp_path / f"{eid}.json").read_text(encoding="utf-8")
        persisted = json.loads(raw_disk)
        assert "ordinary-value-123" not in raw_disk
        assert "plain-tok" not in raw_disk
        assert canary_src not in raw_disk
        assert persisted["metadata"]["api_key"] == "[REDACTED_SECRET]"
        assert persisted["metadata"]["nested"]["Auth-Token"] == "[REDACTED_SECRET]"
        assert "[REDACTED_SECRET]" in persisted["source"]
        assert persisted["redacted_on_disk"] is True

    def test_mixed_manage_rollback_on_tee_store_failure(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        mgr = ContextVisibilityManager(tee_store=store)
        mgr.register("k1", "payload one " * 5, visible=True)
        mgr.register("k2", "payload two " * 5, visible=False)

        k1_tee_id = mgr._tee_ids[("default", "default", "k1")]
        k2_tee_id = mgr._tee_ids[("default", "default", "k2")]
        assert store.delete(k2_tee_id) is True

        with pytest.raises(KeyError):
            mgr.manage(ManageContextRequest(disable=["k1"], enable=["k2"]))

        assert mgr.get_handle("k1").visible is True
        assert store.get(k1_tee_id).visible is True
        assert mgr.get_handle("k2").visible is False

    @pytest.mark.asyncio
    async def test_manage_context_rejects_unknown_keys(self):
        mgr = ContextVisibilityManager()
        mgr.register("k1", "hello world")
        resp = json.loads(
            await route_tool_call(
                "manage_context",
                {"enable": ["k1"], "unexpected_flag": True},
                {"context_visibility": mgr},
            )
        )
        assert resp["status"] == "error"
        assert "Unknown argument" in resp["error"]

    @pytest.mark.asyncio
    async def test_afm_add_message_forwards_dtoc_scope_and_schema(self):
        from src.handlers.mcp_core.schemas_afm_temporal import AFM_TOOLS

        schema = next(t.inputSchema for t in AFM_TOOLS if t.name == "afm_add_message")
        assert "tool_key" in schema["properties"]
        assert "workspace_id" in schema["properties"]
        assert "session_id" in schema["properties"]

        mgr = ContextVisibilityManager()
        mgr.register(
            "build_log",
            "Compiling module B...\n" * 20,
            summary="Build B",
            workspace_id="ws_afm",
            session_id="s_afm",
        )
        mgr.disable(["build_log"], workspace_id="ws_afm", session_id="s_afm")
        fm = FocusManager(context_visibility=mgr)

        await route_tool_call(
            "afm_add_message",
            {
                "role": "user",
                "content": "Compiling module B...\n" * 20,
                "tool_key": "build_log",
                "workspace_id": "ws_afm",
                "session_id": "s_afm",
            },
            {"focus_manager": fm},
        )
        assert fm.messages[-1].tool_key == "build_log"
        assert fm.messages[-1].workspace_id == "ws_afm"
        assert fm.messages[-1].session_id == "s_afm"
        packed, _ = fm.build_context("Compiling module B", budget_tokens=500)
        assert any("[CTX_HIDDEN key=build_log" in content for _, content in packed)


def test_mixed_manage_rollback_on_tee_persist_failure(tmp_path, monkeypatch):
    tee = TeeStore(mode="always", persist_dir=str(tmp_path))
    mgr = ContextVisibilityManager(tee_store=tee)
    h1 = mgr.register("k1", "first output " * 5, workspace_id="w1", session_id="s1")
    h2 = mgr.register("k2", "second output " * 5, workspace_id="w1", session_id="s1")
    mgr.disable(["k2"], workspace_id="w1", session_id="s1")

    tid1 = mgr._tee_ids[("w1", "s1", "k1")]
    tid2 = mgr._tee_ids[("w1", "s1", "k2")]
    assert tee.get(tid1, workspace_id="w1", session_id="s1").visible is True
    assert tee.get(tid2, workspace_id="w1", session_id="s1").visible is False

    orig_persist = tee._persist_entry

    def fail_on_second(entry):
        if entry.entry_id == tid2:
            raise OSError("disk full")
        return orig_persist(entry)

    monkeypatch.setattr(tee, "_persist_entry", fail_on_second)
    with pytest.raises(OSError, match="disk full"):
        mgr.manage(
            ManageContextRequest(disable=["k1"], enable=["k2"], workspace_id="w1", session_id="s1")
        )

    assert h1.visible is True
    assert h2.visible is False
    assert tee.get(tid1, workspace_id="w1", session_id="s1").visible is True
    assert tee.get(tid2, workspace_id="w1", session_id="s1").visible is False
