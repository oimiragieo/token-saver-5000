"""Bounded-store regression tests for EvidenceStore + CompressionReplayLog (B5, A1 sibling).

See docs/plans/2026-08-24-a1-bounded-registries.md for the FIFO-eviction design
this mirrors, and src/constants.py::MAX_EVIDENCE_BUNDLES / MAX_REPLAY_LOG_ENTRIES
for the caps. Both stores are append-only sequences (not dict-keyed registries),
so they use collections.deque(maxlen=...) rather than BoundedDict.
"""

from src.compression_replay import CompressionReplayLog
from src.evidence_bundle import EvidenceBundle, EvidenceStore


def _make_bundle(tag: str) -> EvidenceBundle:
    return EvidenceBundle.create(
        operation="ingest",
        input_data=f"input-{tag}",
        output_data=f"output-{tag}",
        input_token_count=100,
        output_token_count=10,
        parameters={"tag": tag},
    )


class TestEvidenceStoreEvictsOldestBeyondCap:
    def test_evicts_oldest_beyond_cap(self, monkeypatch):
        cap = 20
        monkeypatch.setattr("src.evidence_bundle.MAX_EVIDENCE_BUNDLES", cap, raising=False)
        store = EvidenceStore()

        total = cap + 50
        for i in range(total):
            store.append(_make_bundle(str(i)))

        assert len(store) <= cap
        # The newest bundles must survive; the earliest ones must be gone.
        surviving_ops = [store[i].parameters["tag"] for i in range(len(store))]
        assert surviving_ops[-1] == str(total - 1)
        assert str(0) not in surviving_ops


class TestVerifyChainValidAfterEviction:
    def test_chain_stays_valid_after_eviction(self, monkeypatch):
        cap = 20
        monkeypatch.setattr("src.evidence_bundle.MAX_EVIDENCE_BUNDLES", cap, raising=False)
        store = EvidenceStore()

        total = cap + 50
        for i in range(total):
            store.append(_make_bundle(str(i)))

        assert len(store) <= cap
        valid, errors = store.verify_chain()
        assert valid is True
        assert errors == []


class TestReplayLogBounded:
    def test_replay_log_bounded(self, monkeypatch):
        cap = 20
        monkeypatch.setattr("src.compression_replay.MAX_REPLAY_LOG_ENTRIES", cap, raising=False)
        log = CompressionReplayLog()

        total = cap + 50
        for i in range(total):
            log.record(
                doc_id=f"doc-{i}",
                content_type="prose",
                input_tokens=100,
                output_tokens=10,
                ratio=0.1,
                fidelity_score=0.9,
            )

        assert len(log._log) <= cap
        remaining_ids = [e["doc_id"] for e in log._log]
        assert f"doc-{total - 1}" in remaining_ids
        assert "doc-0" not in remaining_ids
