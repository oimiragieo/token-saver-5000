"""Baseline characterization tests for Wave 3 dependencies (CL 0)."""

from __future__ import annotations

import pytest

from src.ace_framework import ACEFramework, BulletType
from src.afm import AFMConfig, FocusManager
from src.compression_replay import CompressionReplayLog
from src.tee_recovery import TEE_MODE_ALWAYS, TEE_MODE_FAILURES, TeeStore


class TestTeeStoreBaseline:
    def test_store_and_recover_roundtrip(self, tmp_path):
        store = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=str(tmp_path))
        entry_id = store.store(
            original_text="line1\nline2\nline3\n",
            compressed_text="line1\n",
            compression_pct=66.7,
            source="cli_optimizer",
            command_hint="pytest",
        )
        assert entry_id is not None
        assert store.get_original(entry_id) == "line1\nline2\nline3\n"

        # Reload from disk into a fresh store instance
        reloaded = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=str(tmp_path))
        assert reloaded.get_original(entry_id) == "line1\nline2\nline3\n"

    def test_failures_mode_skips_below_threshold(self):
        store = TeeStore(mode=TEE_MODE_FAILURES, compression_threshold=80.0)
        assert store.store("raw", "comp", 50.0, "cli_optimizer") is None


class TestFocusManagerBaseline:
    def test_add_message_and_build_context_respects_budget(self):
        fm = FocusManager(AFMConfig(tau_high=0.45, tau_mid=0.25))
        fm.add_message("user", "We need to fix the authentication timeout in login.py.")
        fm.add_message("assistant", "Checking login.py session expiry logic now.")
        fm.add_message("user", "Remember: never disable CSRF validation in production.")

        packed, stats = fm.build_context("authentication timeout", budget_tokens=120)
        assert len(packed) >= 1
        assert stats.total_tokens <= 120
        assert fm.get_stats()["total_messages"] == 3

    def test_empty_message_raises(self):
        fm = FocusManager()
        with pytest.raises(ValueError):
            fm.add_message("user", "   ")


class TestCompressionReplayAndACEBaseline:
    def test_replay_log_insights_and_recommendation(self):
        log = CompressionReplayLog()
        log.record(
            "doc1", "code", input_tokens=1000, output_tokens=200, ratio=0.2, fidelity_score=0.92
        )
        log.record(
            "doc2", "code", input_tokens=1000, output_tokens=100, ratio=0.1, fidelity_score=0.60
        )
        insights = log.get_insights()
        assert insights["code"]["count"] == 2
        assert insights["code"]["best_ratio"] == 0.2
        assert log.recommend_ratio("code", min_fidelity=0.8) == 0.2

    def test_ace_framework_cycle_updates_context(self):
        ace = ACEFramework()
        ctx = ace.create_initial_context([("Preserve stack traces", BulletType.CONSTRAINT)])
        updated_ctx, trajectory = ace.execute_ace_cycle(
            task="Compress pytest failure log",
            context=ctx,
            outcome="Kept AssertionError and line number",
            success=True,
        )
        assert updated_ctx.version >= 1
        assert len(trajectory) >= 1


class TestAFMMessagePositionalAndPersistence:
    def test_message_positional_constructor_compatibility(self):
        from src.afm import FidelityLevel, ImportanceLevel, Message

        msg = Message(
            "user",
            "hello world",
            0,
            "m0",
            None,
            ImportanceLevel.TRIVIAL,
            0.0,
            FidelityLevel.PLACEHOLDER,
            None,
            None,
            1234.5,
        )
        assert msg.timestamp == 1234.5
        assert msg.tool_key is None
        assert msg.workspace_id == "default"
        assert msg.session_id == "default"

    def test_persistence_roundtrip_preserves_dtoc_metadata(self, tmp_path):
        from src.afm import Message
        from src.persistence import PersistenceManager

        pm = PersistenceManager(storage_dir=str(tmp_path))
        msg = Message(
            role="user",
            content="persisted output",
            turn_index=0,
            tool_key="build_log",
            workspace_id="ws_p",
            session_id="s_p",
        )
        assert pm.save_afm_history("s_p", [msg], turn_counter=1) is True
        loaded = pm.load_afm_history("s_p")
        assert loaded is not None
        restored_msg = loaded["messages"][0]
        assert restored_msg.tool_key == "build_log"
        assert restored_msg.workspace_id == "ws_p"
        assert restored_msg.session_id == "s_p"
