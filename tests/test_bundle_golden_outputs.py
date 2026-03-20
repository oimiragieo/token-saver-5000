from src.bundle_distiller import distill_handoff_bundle


class _FakeTracker:
    def get_access_info(self, doc_id):
        return {"first_accessed": 1.0, "last_accessed": 2.0, "access_count": 2}


class _FakeReplay:
    def get_history(self, doc_id):
        return [{"ratio": 4.0}]


class _FakeTemporal:
    def get_active_facts(self, doc_id, as_of=None, include_invalidated=False):
        return [{"fact_id": f"{doc_id}_n0", "summary": "anchor", "active": True}]

    def search_timeline(self, **kwargs):
        return [{"event_type": "document_ingested", "timestamp": "2026-03-16T07:00:00Z"}]


class _FakeCompressor:
    graphs = {"auth_doc": object()}
    chunks = {
        "auth_doc_n0": type(
            "Node",
            (),
            {"importance": 0.71, "text": "Authentication entry point.", "metadata": {"tokens": 4}},
        )()
    }
    _access_tracker = _FakeTracker()
    _compression_replay = _FakeReplay()
    _temporal_graph = _FakeTemporal()

    def _generate_skeleton(self, file_id, query=None, exclude_node_ids=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            file_id=file_id,
            total_nodes=4,
            total_tokens=80,
            skeleton_tokens=20,
            compression_ratio=4.0,
            skeleton_text="AUTH SKELETON",
            node_map={"auth_doc_n0": "ANCHOR: Authentication entry point"},
        )

    def search_semantic_with_scores(self, query, file_id=None, top_k=5):
        return [("auth_doc_n0", 0.987)]

    def _generate_summary(self, text, max_length=100):
        return text[:max_length]

    def _count_tokens(self, text):
        return len(text.split())


def test_distilled_bundle_outputs_are_stable():
    payload = distill_handoff_bundle(
        compressor=_FakeCompressor(),
        visible_file_id="auth_doc",
        scoped_file_id="auth_doc",
        query="authentication flow",
        bundle_id="bundle-1",
        created_at="2026-03-16T07:00:00Z",
        metadata={"owner": "team-auth"},
    )

    assert payload["artifact"]["bundle_id"] == "bundle-1"
    assert payload["artifact"]["summary"] == "Distilled handoff for 'auth_doc'"
    assert payload["artifact"]["search_results"][0]["node_id"] == "auth_doc_n0"
    assert payload["artifact_toon"].startswith("handoff_bundle:")
    assert payload["token_estimate"]["json_tokens"] >= payload["token_estimate"]["toon_tokens"]
