"""Regression lock: file_id / session_id must not escape their storage dir.

Verified bug (audit 2026-09-12): the file_id validation hook only rejected
".." and allowed "/" in its charset, but never checked for an absolute path
or a drive letter/UNC prefix. `Path.__truediv__` silently REPLACES the whole
path when the right-hand operand is absolute, so a file_id of "/etc/passwd"
(or "C:/Windows/..." on Windows) escaped `documents_dir` entirely once
joined in `persistence.py`. `session_journal.py`'s `session_id` had NO
validation at all before building a SQLite db path from it.

These tests exercise the REAL validator (`src.safe_identifier`), the REAL
MCP validation hook, and the REAL `PersistenceManager` / `SessionJournal`
classes against a real filesystem (tmp_path) -- not mocks -- so a regression
that deletes the absolute-path check would fail these tests, not just a
mocked assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.safe_identifier import validate_safe_storage_id
from src.session_journal import SessionJournal
from src.validation_hooks import validate_tool_input


class TestValidateSafeStorageId:
    """Unit tests for the shared allowlist validator."""

    def test_legit_relative_file_id_with_subdirs_accepted(self):
        assert (
            validate_safe_storage_id("docs/audits/foo/bar.md", "file_id", allow_subdirs=True)
            is None
        )

    def test_posix_absolute_path_rejected(self):
        error = validate_safe_storage_id("/etc/passwd", "file_id", allow_subdirs=True)
        assert error and "absolute" in error

    def test_leading_slash_without_full_absolute_semantics_rejected(self):
        error = validate_safe_storage_id("//etc/passwd", "file_id", allow_subdirs=True)
        assert error

    def test_windows_drive_letter_rejected(self):
        error = validate_safe_storage_id(
            "C:/Windows/System32/config", "file_id", allow_subdirs=True
        )
        assert error

    def test_windows_drive_letter_backslash_rejected(self):
        error = validate_safe_storage_id(
            r"C:\Windows\System32\config", "file_id", allow_subdirs=True
        )
        assert error

    def test_unc_path_rejected(self):
        error = validate_safe_storage_id(r"\\server\share\file", "file_id", allow_subdirs=True)
        assert error

    def test_dotdot_still_rejected(self):
        error = validate_safe_storage_id("../secret", "file_id", allow_subdirs=True)
        assert error

    def test_no_subdirs_mode_rejects_any_slash(self):
        error = validate_safe_storage_id("a/b", "session_id", allow_subdirs=False)
        assert error

    def test_no_subdirs_mode_accepts_simple_id(self):
        assert validate_safe_storage_id("session-A_1.2", "session_id", allow_subdirs=False) is None

    def test_no_subdirs_mode_rejects_absolute_path(self):
        error = validate_safe_storage_id("/etc/passwd", "session_id", allow_subdirs=False)
        assert error

    def test_empty_value_is_not_an_error(self):
        assert validate_safe_storage_id("", "file_id", allow_subdirs=True) is None


class TestIngestFileIdEscapeRejected:
    """The MCP entry-point hook must reject escape shapes for ingest_context."""

    def test_absolute_posix_path_rejected(self):
        errors = validate_tool_input("ingest_context", {"text": "hi", "file_id": "/etc/passwd"})
        assert errors

    def test_windows_drive_letter_rejected(self):
        errors = validate_tool_input(
            "ingest_context",
            {"text": "hi", "file_id": "C:/Windows/System32/config"},
        )
        assert errors

    def test_unc_path_rejected(self):
        errors = validate_tool_input(
            "ingest_context",
            {"text": "hi", "file_id": r"\\attacker\share\payload"},
        )
        assert errors

    def test_legit_relative_file_id_still_accepted(self):
        """Regression guard: the fix must not reject real usage."""
        assert (
            validate_tool_input(
                "ingest_context", {"text": "hi", "file_id": "docs/audits/foo/bar.md"}
            )
            == []
        )


class TestDeleteDocumentFileIdEscapeRejected:
    def test_absolute_path_rejected(self):
        errors = validate_tool_input("delete_document", {"file_id": "/etc/passwd"})
        assert errors

    def test_legit_file_id_accepted(self):
        assert validate_tool_input("delete_document", {"file_id": "src/auth.py"}) == []


class TestBatchIngestDocumentsFileIdEscapeRejected:
    def test_one_malicious_document_file_id_rejected(self):
        errors = validate_tool_input(
            "batch_ingest_documents",
            {
                "documents": [
                    {"file_id": "safe-doc", "text": "hi"},
                    {"file_id": "/etc/passwd", "text": "hi"},
                ]
            },
        )
        assert errors

    def test_all_legit_file_ids_accepted(self):
        errors = validate_tool_input(
            "batch_ingest_documents",
            {
                "documents": [
                    {"file_id": "safe-doc-1", "text": "hi"},
                    {"file_id": "safe/doc-2.md", "text": "hi"},
                ]
            },
        )
        assert errors == []


class TestPersistenceManagerRejectsEscapingFileId:
    """Real filesystem, real PersistenceManager -- no mocking the join."""

    def _manager(self, tmp_path: Path):
        from src.persistence import PersistenceManager

        return PersistenceManager(storage_dir=str(tmp_path / ".semantic_modulator_data"))

    def test_save_document_with_absolute_file_id_does_not_escape_documents_dir(self, tmp_path):
        pm = self._manager(tmp_path)
        outside_target = tmp_path / "outside.json"
        assert not outside_target.exists()

        # An attacker-controlled absolute file_id that, pre-fix, would have
        # been joined directly onto documents_dir and (via Path.__truediv__
        # replacing the whole path) written wherever the absolute path
        # pointed. Point it at a path outside the storage tree entirely.
        malicious_file_id = str(tmp_path / "outside")
        result = pm.save_document(
            malicious_file_id,
            chunks={},
            graph_data={},
            metadata={},
        )

        assert result is False
        assert not outside_target.exists()
        # Nothing escaped into documents_dir either, under any name.
        assert list(pm.documents_dir.iterdir()) == []

    def test_load_document_with_absolute_file_id_returns_none(self, tmp_path):
        pm = self._manager(tmp_path)
        # Plant a real secret file OUTSIDE documents_dir that a traversal
        # would have been able to read back via _load_document_json.
        secret = tmp_path / "secret.json"
        secret.write_text('{"leaked": true}', encoding="utf-8")

        loaded = pm.load_document(str(tmp_path / "secret"))
        assert loaded is None

    def test_delete_document_with_absolute_file_id_does_not_touch_real_file(self, tmp_path):
        pm = self._manager(tmp_path)
        victim = tmp_path / "victim_graph.json"
        victim.write_text("{}", encoding="utf-8")

        malicious_file_id = str(tmp_path / "victim")
        result = pm.delete_document(malicious_file_id)

        assert result is False
        assert victim.exists()

    def test_legit_relative_file_id_still_saves_and_loads(self, tmp_path):
        pm = self._manager(tmp_path)
        ok = pm.save_document(
            "docs/audits/report",
            chunks={},
            graph_data={"nodes": []},
            metadata={"title": "t"},
        )
        assert ok is True
        loaded = pm.load_document("docs/audits/report")
        assert loaded is not None
        assert loaded["metadata"]["title"] == "t"


class TestSessionJournalRejectsUnsafeSessionId:
    """Real SQLite construction, no mocking -- the chokepoint is __init__."""

    def test_absolute_path_session_id_raises(self, tmp_path):
        malicious_session_id = str(tmp_path / "escape")
        with pytest.raises(ValueError):
            SessionJournal(session_id=malicious_session_id, storage_dir=tmp_path / "sessions")

    def test_traversal_session_id_raises(self, tmp_path):
        with pytest.raises(ValueError):
            SessionJournal(session_id="../../evil", storage_dir=tmp_path / "sessions")

    def test_slash_session_id_raises(self, tmp_path):
        with pytest.raises(ValueError):
            SessionJournal(session_id="a/b", storage_dir=tmp_path / "sessions")

    def test_legit_session_id_still_works(self, tmp_path):
        journal = SessionJournal(session_id="session-A_1", storage_dir=tmp_path / "sessions")
        try:
            journal.write_event(
                "ingest", {"file_id": "x", "original_tokens": 10, "compressed_tokens": 2}
            )
            assert journal.event_count() == 1
        finally:
            journal.close()

    def test_no_db_file_created_outside_storage_dir_for_traversal_attempt(self, tmp_path):
        storage_dir = tmp_path / "sessions"
        with pytest.raises(ValueError):
            SessionJournal(session_id="../escape", storage_dir=storage_dir)
        # Nothing should have been created one level up from storage_dir.
        assert not (tmp_path / "escape.db").exists()
