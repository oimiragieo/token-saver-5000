"""Structured-content detection + record splitting for the JSON/table
compression path (#190).

Pure functions, no model load. These are the foundation of the structured-data
compression fix: a raw JSON array otherwise collapses to ONE mega-node under the
markdown/sentence chunker (dogfood 2026-07-11: a 100-record array -> 1 node,
skeleton hides ~99% = DATA LOSS, unrecoverable by agents). Splitting on RECORD
boundaries lets records become individually rankable / queryable nodes instead
of one hidden blob. The chunker (next wiring step) groups the returned records
by its existing size target.
"""

from __future__ import annotations

import csv as _csv
import io
import json
from typing import Optional

# A 1-element array/collection is not worth the structured path (nothing to
# split); require at least this many records.
_MIN_RECORDS = 2


def detect_structured_content(text: str) -> Optional[str]:
    """Classify ``text`` as structured data.

    Returns ``"json_array"`` | ``"json_object"`` | ``"jsonl"`` | ``"csv"`` |
    ``None``. Conservative on purpose: prose and markdown return ``None`` so the
    normal text path is unaffected.
    """
    if not text or not text.strip():
        return None
    stripped = text.strip()

    # JSON array or object (parse the whole thing first — a pretty-printed array
    # spanning many lines must resolve here, not fall through to JSONL).
    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except (ValueError, TypeError, RecursionError):
            parsed = None
        if isinstance(parsed, list) and len(parsed) >= _MIN_RECORDS:
            return "json_array"
        if isinstance(parsed, dict):
            return "json_object"

    lines = [ln for ln in stripped.splitlines() if ln.strip()]

    # JSONL: >=2 non-empty lines, each a standalone JSON value.
    if len(lines) >= _MIN_RECORDS and all(ln.strip()[:1] in "[{" for ln in lines):
        ok = 0
        for ln in lines:
            try:
                json.loads(ln)
                ok += 1
            except (ValueError, TypeError, RecursionError):
                break
        if ok == len(lines):
            return "jsonl"

    # CSV: >=2 rows, first row has a delimiter, consistent column count >=2.
    if len(lines) >= _MIN_RECORDS and "," in lines[0]:
        try:
            rows = list(_csv.reader(io.StringIO(stripped)))
        except (_csv.Error, ValueError):
            rows = []
        rows = [r for r in rows if r]
        if len(rows) >= _MIN_RECORDS:
            ncol = len(rows[0])
            if ncol >= 2 and all(len(r) == ncol for r in rows):
                return "csv"

    return None


def split_json_records(text: str) -> Optional[list[str]]:
    """Split a JSON array into its top-level elements, PRESERVING each element's
    ORIGINAL source substring.

    Walks element boundaries with ``raw_decode`` and slices the source rather than
    deserialize/reserialize — so duplicate keys, exact numeric literals (e.g.
    ``1e400``), and formatting survive intact (fidelity is the whole point of the
    structured path). Returns the record strings, or ``None`` when ``text`` is not
    a JSON array of >=2 elements.
    """
    stripped = (text or "").strip()
    if not stripped or stripped[0] != "[":
        return None
    decoder = json.JSONDecoder()
    records: list[str] = []
    i = 1  # past the opening '['
    n = len(stripped)
    try:
        while i < n:
            while i < n and stripped[i] in " \t\r\n,":
                i += 1
            if i >= n or stripped[i] == "]":
                break
            _element, end = decoder.raw_decode(stripped, i)
            records.append(stripped[i:end].strip())
            i = end
    except (ValueError, TypeError, RecursionError):
        return None
    if len(records) < _MIN_RECORDS:
        return None
    return records


def group_records_by_size(records, max_tokens, count_tokens):
    """Greedily pack record strings into newline-joined chunks up to ``max_tokens``.

    Keeps records ATOMIC — an oversized single record becomes its own chunk
    rather than being split mid-record. This is what lets a big JSON array become
    N size-bounded, rankable nodes instead of one hidden mega-node.

    KNOWN LIMITATION (#190, codex P1): a single record larger than the embedding
    window (~512 tok) stays atomic, so its tail is present + readable in the
    skeleton but not embedding-searchable (the encoder truncates). Data is
    preserved; only dense retrieval-by-tail on a huge single record degrades.
    Hierarchical sub-record chunking is deferred (splitting a record breaks it).

    Args:
        records: list of record strings (e.g. from ``split_json_records``).
        max_tokens: soft ceiling per chunk (records are never split to fit).
        count_tokens: callable ``str -> int`` token counter.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for record in records:
        record_tokens = count_tokens(record)
        separator = 1 if current else 0  # the "\n" that joins records within a chunk
        if current and current_tokens + separator + record_tokens > max_tokens:
            chunks.append("\n".join(current))
            current = [record]
            current_tokens = record_tokens
        else:
            current.append(record)
            current_tokens += separator + record_tokens
    if current:
        chunks.append("\n".join(current))
    return chunks
