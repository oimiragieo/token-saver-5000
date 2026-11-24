"""Wave 5 in-process smoke test for scripts/benchmarks/dogfood_2026_sota.py."""

from __future__ import annotations

import json

import pytest

from scripts.benchmarks.dogfood_2026_sota import main, run_dogfood_suite


def test_offline_dogfood_suite_passes_all_invariants(tmp_path) -> None:
    receipt = run_dogfood_suite(mode="offline", budget_usd=0.0, wheel_check=False)
    assert receipt["all_passed"] is True
    assert receipt["mode"] == "offline"
    assert receipt["budget_usd"] == 0.0
    assert receipt["waves"]["wave1"]["passed"] is True
    assert receipt["waves"]["wave2"]["passed"] is True
    assert receipt["waves"]["wave2"]["prefix_byte_identical_across_turns"] is True
    assert receipt["waves"]["wave2"]["validate_p95_ms"] < 0.5
    assert receipt["waves"]["wave3"]["passed"] is True
    assert receipt["waves"]["wave3"]["dtoc_sha256_roundtrip"] is True
    assert receipt["waves"]["wave4"]["passed"] is True
    assert receipt["waves"]["wave4"]["flow_delta_ms"] < 5.0
    paired = receipt["paired_non_inferiority"]
    assert paired["tokenizer"] == "tiktoken:cl100k_base"
    assert paired["baseline_tokens"] > paired["optimized_tokens"] > 0
    assert paired["optimized_cached_tokens"] > 0
    assert paired["recall_optimized"] >= paired["recall_baseline"]
    assert paired["non_inferiority_passed"] is True

    out_path = tmp_path / "dogfood_receipt.json"
    rc = main(["--mode", "offline", "--out", str(out_path)])
    assert rc == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["all_passed"] is True


def test_live_mode_fails_closed_without_budget_or_provider() -> None:
    with pytest.raises(ValueError, match="budget-usd"):
        run_dogfood_suite(mode="live", budget_usd=0.0)
    assert main(["--live", "--budget-usd", "1.0"]) == 1
