"""Tests for Wave 3 Acon failure-driven rule optimizer with omega-verified-rsi gate."""

from __future__ import annotations

from src.acon_rule_optimizer import (
    AconRuleOptimizer,
    CandidateRuleDiff,
    HeldOutFixture,
)
from src.compression_replay import CompressionReplayLog


def _sample_fixtures() -> list[HeldOutFixture]:
    return [
        HeldOutFixture(
            fixture_id="fix_1",
            content_type="code",
            text="def login():\n    assert verify_token()\n    return 200\n",
            required_terms=("verify_token", "return 200"),
            baseline_tokens=15,
            baseline_recall=0.5,
        )
    ]


class TestAconRuleOptimizer:
    def test_proposes_and_accepts_valid_improvement(self):
        log = CompressionReplayLog()
        log.record(
            doc_id="d1",
            content_type="code",
            input_tokens=400,
            output_tokens=120,
            ratio=0.5,
            fidelity_score=0.55,
            metadata={"recall_miss": True, "pattern_hint": "preserve_assertions"},
        )

        def good_evaluator(fixture: HeldOutFixture, cand: CandidateRuleDiff) -> tuple[float, int]:
            return 1.0, 14  # improves recall (1.0 >= 0.5) without inflating tokens (14 <= 15)

        opt = AconRuleOptimizer(min_fidelity=0.8, evaluator=good_evaluator)
        res = opt.optimize(log, _sample_fixtures())
        assert len(res.accepted) == 1
        assert len(res.rejected) == 0
        assert res.new_recall == 1.0
        assert res.new_tokens == 14

    def test_omega_verified_rsi_rejects_recall_regression(self):
        log = CompressionReplayLog()
        log.record("d1", "code", 400, 100, ratio=0.25, fidelity_score=0.4)

        def regressing_evaluator(
            fixture: HeldOutFixture, cand: CandidateRuleDiff
        ) -> tuple[float, int]:
            return 0.2, 10  # drops recall below 0.5

        opt = AconRuleOptimizer(min_fidelity=0.8, evaluator=regressing_evaluator)
        res = opt.optimize(log, _sample_fixtures())
        assert len(res.accepted) == 0
        assert len(res.rejected) == 1
        assert "held_out_recall_regression" in res.rejected[0][1]

    def test_omega_verified_rsi_rejects_token_inflation(self):
        log = CompressionReplayLog()
        log.record("d1", "code", 400, 100, ratio=0.25, fidelity_score=0.4)

        def inflating_evaluator(
            fixture: HeldOutFixture, cand: CandidateRuleDiff
        ) -> tuple[float, int]:
            return 1.0, 25  # inflates tokens above baseline 15

        opt = AconRuleOptimizer(min_fidelity=0.8, evaluator=inflating_evaluator)
        res = opt.optimize(log, _sample_fixtures())
        assert len(res.accepted) == 0
        assert len(res.rejected) == 1
        assert "net_token_inflation" in res.rejected[0][1]

    def test_default_evaluator_and_complete_ruleset_replay(self):
        log = CompressionReplayLog()
        log.record(
            doc_id="d_code",
            content_type="code",
            input_tokens=400,
            output_tokens=100,
            ratio=0.4,
            fidelity_score=0.4,
            metadata={"recall_miss": True, "pattern_hint": "preserve_assertions"},
        )
        fixtures = [
            HeldOutFixture(
                fixture_id="fix_default",
                content_type="code",
                text=(
                    "def step_one(): pass\n"
                    "def step_two(): pass\n"
                    "def step_three(): pass\n"
                    "assert verify_token() == 200\n"
                ),
                required_terms=("verify_token",),
                baseline_tokens=40,
                baseline_recall=0.0,
            )
        ]
        opt = AconRuleOptimizer(min_fidelity=0.8)
        res = opt.optimize(log, fixtures)
        assert len(res.accepted) == 1
        assert res.new_recall == 1.0
        assert res.new_tokens <= 40

    def test_combined_ruleset_evaluation_gates_joint_regression(self):
        log = CompressionReplayLog()
        log.record("d1", "code", 400, 100, ratio=0.3, fidelity_score=0.4)
        log.record("d2", "logs", 400, 100, ratio=0.3, fidelity_score=0.4)

        fixtures = [
            HeldOutFixture(
                "f_code", "code", "assert ok", ("ok",), baseline_tokens=10, baseline_recall=0.5
            ),
            HeldOutFixture(
                "f_logs", "logs", "ERROR fail", ("ERROR",), baseline_tokens=10, baseline_recall=0.5
            ),
        ]

        def tricky_eval(fixture: HeldOutFixture, cand: CandidateRuleDiff) -> tuple[float, int]:
            if cand.content_type == "code" and fixture.content_type == "code":
                return 1.0, 10
            if cand.content_type == "logs" and fixture.content_type == "logs":
                return (
                    1.0,
                    15,
                )  # individually <= total baseline 20, but combined with code (10+15=25) > 20!
            return fixture.baseline_recall, fixture.baseline_tokens

        opt = AconRuleOptimizer(min_fidelity=0.8, evaluator=tricky_eval)
        res = opt.optimize(log, fixtures)
        assert [c.content_type for c in res.accepted] == ["code"]
        assert len(res.rejected) == 1
        assert "net_token_inflation" in res.rejected[0][1]
