"""Regression lock: code-ingest path must never write to stdout (2026-07-06).

Architecture plan `docs/superpowers/plans/2026-07-06-ultimate-compression-
architecture.md` section 1.4 / Move 5 flagged ``CodeSemanticCompressor.
ingest_code_file`` (``src/code_compressor.py``) as using bare ``print(...)``
calls on every code ingestion. The MCP server's default transport is stdio —
JSON-RPC frames are read from / written to stdout — so any stray ``print()``
on a hot code path corrupts the protocol stream for every client using the
default (non-HTTP) transport. This is a LIVE bug: ``CodeCompressionAdapter.
ingest_file_async`` calls ``ingest_code_file`` directly on every code-file
ingest (``ingest_directory`` / any ``ingest_context`` call whose file_path
resolves to a code extension), not behind an opt-in flag.

Fixed by replacing the prints with ``logger.info`` (matches the rest of the
module, which already uses ``logging`` for the surrounding lines).
"""

from unittest.mock import patch

import numpy as np

from src.code_compressor import CodeSemanticCompressor

SAMPLE_PYTHON_CODE = '''
import os


def add(a, b):
    """Add two numbers."""
    return a + b


class Greeter:
    """Greets people."""

    def greet(self, name):
        return f"Hello, {name}!"
'''


@patch("src.code_compressor.EmbeddingManager")
class TestIngestCodeFileNeverPrintsToStdout:
    """LOAD-BEARING: a real ingest must produce ZERO stdout bytes.

    ``capsys`` captures the real OS-level stdout stream, so this fails if ANY
    ``print(...)`` fires anywhere inside ``ingest_code_file`` — including ones
    added in the future.
    """

    def _make_compressor(self, mock_embedding_manager_class):
        mock_model = type(
            "MockModel",
            (),
            {
                "encode": lambda self, texts, show_progress_bar=False: np.random.rand(
                    len(texts), 384
                )
            },
        )()
        mock_manager = type(
            "MockManager", (), {"get_code_embedder": lambda self, *a, **kw: mock_model}
        )()
        mock_embedding_manager_class.return_value = mock_manager
        return CodeSemanticCompressor()

    def test_python_ingest_writes_nothing_to_stdout(self, mock_embedding_manager_class, capsys):
        compressor = self._make_compressor(mock_embedding_manager_class)

        compressor.ingest_code_file(
            code=SAMPLE_PYTHON_CODE, file_id="stdio_safety_py", filepath="thing.py"
        )

        captured = capsys.readouterr()
        assert captured.out == "", (
            "ingest_code_file() wrote to stdout — this corrupts the MCP stdio "
            f"JSON-RPC transport. Captured stdout: {captured.out!r}"
        )

    def test_unknown_language_ingest_writes_nothing_to_stdout(
        self, mock_embedding_manager_class, capsys
    ):
        """The line-based fallback path (unknown/no filepath) is a separate
        branch through the same function — cover it too."""
        compressor = self._make_compressor(mock_embedding_manager_class)

        compressor.ingest_code_file(
            code="some plain text\nmore text\nyet more\n",
            file_id="stdio_safety_unknown",
            filepath=None,
        )

        captured = capsys.readouterr()
        assert captured.out == ""


def test_no_bare_print_calls_left_in_ingest_code_file():
    """Static guard: no bare ``print(`` token inside ``ingest_code_file``'s body.

    Cheap source-level backstop alongside the capsys behavioral test above.
    """
    import inspect

    from src.code_compressor import CodeSemanticCompressor

    source = inspect.getsource(CodeSemanticCompressor.ingest_code_file)
    assert "print(" not in source, (
        "ingest_code_file() contains a bare print( call — this corrupts the "
        "MCP stdio JSON-RPC transport. Use logger.info/debug instead."
    )
