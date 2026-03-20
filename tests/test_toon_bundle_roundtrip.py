from src.toon_serializer import TOONSerializer


def test_handoff_bundle_toon_roundtrip():
    serializer = TOONSerializer()
    artifact = {
        "bundle_id": "bundle-1",
        "doc_id": "auth_doc",
        "created_at": "2026-03-16T07:00:00Z",
        "query": "authentication flow",
        "summary": "Distilled handoff for 'auth_doc'",
        "skeleton": {
            "text": "AUTH SKELETON",
            "total_nodes": 4,
            "skeleton_tokens": 20,
            "compression_ratio": 4.0,
        },
        "search_results": [
            {
                "node_id": "auth_doc_n0",
                "similarity": 0.987,
                "importance": 0.712,
                "summary": "Authentication entry point",
            }
        ],
    }

    encoded = serializer.serialize_handoff_bundle(artifact)
    decoded = serializer.deserialize_handoff_bundle(encoded)

    assert encoded.startswith("handoff_bundle:")
    assert decoded["bundle_id"] == "bundle-1"
    assert decoded["doc_id"] == "auth_doc"
    assert decoded["search_results"][0]["node_id"] == "auth_doc_n0"
