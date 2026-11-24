"""Acon failure-driven rule optimizer with omega-verified-rsi gate (Wave 3).

Consumes recall misses from CompressionReplayLog, proposes candidate rule diffs,
and gates every candidate against a held-out fixture set so no rule that drops
held-out recall or inflates net tokens can be accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable, Dict, List, Optional, Tuple

from .cli_output_optimizer import CLIOutputOptimizer, STRATEGY_MAP
from .compression_replay import CompressionReplayLog
from .token_estimation import TokenEstimator

_TOKEN_ESTIMATOR = TokenEstimator()


@dataclass(frozen=True)
class CandidateRuleDiff:
    """Proposed rule adjustment derived from replay recall misses."""

    rule_id: str
    content_type: str
    target_ratio: float
    pattern_hint: str
    description: str
    source_miss_count: int = 1


@dataclass(frozen=True)
class HeldOutFixture:
    """Held-out evaluation fixture for anti-reward-hacking verification."""

    fixture_id: str
    content_type: str
    text: str
    required_terms: Tuple[str, ...]
    baseline_tokens: int
    baseline_recall: float


@dataclass
class OptimizationResult:
    """Outcome of running AconRuleOptimizer with the omega-verified-rsi gate."""

    accepted: List[CandidateRuleDiff] = field(default_factory=list)
    rejected: List[Tuple[CandidateRuleDiff, str]] = field(default_factory=list)
    baseline_recall: float = 0.0
    new_recall: float = 0.0
    baseline_tokens: int = 0
    new_tokens: int = 0


EvaluatorFn = Callable[[HeldOutFixture, CandidateRuleDiff], Tuple[float, int]]


def _hint_keywords(pattern_hint: str) -> List[str]:
    tokens = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", pattern_hint) if len(t) >= 3]
    stop = {"preserve", "keep", "rule", "hint", "for", "the", "and", "with"}
    filtered = [t for t in tokens if t not in stop]
    if "assertions" in tokens or "assert" in tokens:
        filtered.append("assert")
    return filtered


def _apply_rules_to_fixture(
    fixture: HeldOutFixture, rules: List[CandidateRuleDiff]
) -> Tuple[float, int]:
    """Evaluate one or more candidate rules together on *fixture*."""
    matching = [r for r in rules if r.content_type == fixture.content_type]
    if not matching:
        return fixture.baseline_recall, fixture.baseline_tokens

    text = fixture.text
    # If content_type maps to a CLIOutputOptimizer strategy, run it as base pass
    if fixture.content_type in STRATEGY_MAP:
        text = (
            CLIOutputOptimizer()
            .filter(fixture.text, command_hint=fixture.content_type)
            .filtered_text
        )

    lines = text.splitlines() or [text]
    max_ratio = max(r.target_ratio for r in matching)
    keep_count = max(1, int(round(len(lines) * min(1.0, max(0.05, max_ratio)))))

    keywords: List[str] = []
    for r in matching:
        keywords.extend(_hint_keywords(r.pattern_hint))

    if keywords:
        matched_idx = [
            i for i, line in enumerate(lines) if any(kw in line.lower() for kw in keywords)
        ]
        selected = set(matched_idx[:keep_count])
        for i in range(len(lines)):
            if len(selected) >= keep_count:
                break
            selected.add(i)
        kept_lines = [lines[i] for i in sorted(selected)]
    else:
        kept_lines = lines[:keep_count]

    kept = "\n".join(kept_lines)
    if fixture.required_terms:
        hits = sum(1 for term in fixture.required_terms if term in kept)
        recall = hits / len(fixture.required_terms)
    else:
        recall = 1.0
    tokens = max(1, _TOKEN_ESTIMATOR.count_tokens(kept))
    return recall, tokens


def default_fixture_evaluator(
    fixture: HeldOutFixture, candidate: CandidateRuleDiff
) -> Tuple[float, int]:
    """Deterministic evaluator applying CLIOutputOptimizer + rule-pattern preservation."""
    return _apply_rules_to_fixture(fixture, [candidate])


class AconRuleOptimizer:
    """Failure-driven compression rule optimizer gated by held-out verification."""

    def __init__(
        self,
        min_fidelity: float = 0.8,
        ratio_step: float = 0.1,
        max_ratio: float = 0.9,
        evaluator: Optional[EvaluatorFn] = None,
    ) -> None:
        if not (0.0 < min_fidelity <= 1.0):
            raise ValueError("min_fidelity must be in (0, 1]")
        if ratio_step <= 0.0:
            raise ValueError("ratio_step must be positive")
        self.min_fidelity = min_fidelity
        self.ratio_step = ratio_step
        self.max_ratio = max_ratio
        self._evaluator = evaluator or default_fixture_evaluator

    def collect_recall_misses(self, replay_log: CompressionReplayLog) -> List[dict]:
        """Extract compression events representing recall misses."""
        misses: List[dict] = []
        for entry in replay_log.get_timeline(limit=10_000):
            meta = entry.get("metadata") or {}
            if entry.get("fidelity_score", 1.0) < self.min_fidelity or meta.get("recall_miss"):
                misses.append(entry)
        return misses

    def propose_candidates(self, replay_log: CompressionReplayLog) -> List[CandidateRuleDiff]:
        """Propose candidate rule diffs grouped by failing content_type."""
        misses = self.collect_recall_misses(replay_log)
        if not misses:
            return []

        by_type: Dict[str, List[dict]] = {}
        for entry in misses:
            by_type.setdefault(entry["content_type"], []).append(entry)

        candidates: List[CandidateRuleDiff] = []
        for idx, (content_type, entries) in enumerate(sorted(by_type.items()), start=1):
            worst = min(entries, key=lambda e: e.get("fidelity_score", 0.0))
            base_ratio = float(worst.get("ratio", 0.3))
            proposed_ratio = round(min(self.max_ratio, base_ratio + self.ratio_step), 4)
            hint = str((worst.get("metadata") or {}).get("pattern_hint", content_type))
            candidates.append(
                CandidateRuleDiff(
                    rule_id=f"acon_{content_type}_{idx}",
                    content_type=content_type,
                    target_ratio=proposed_ratio,
                    pattern_hint=hint,
                    description=(
                        f"Raise target_ratio for '{content_type}' from {base_ratio:.2f} "
                        f"to {proposed_ratio:.2f} after {len(entries)} recall miss(es)"
                    ),
                    source_miss_count=len(entries),
                )
            )
        return candidates

    def verify_candidate(
        self,
        candidate: CandidateRuleDiff,
        held_out_fixtures: List[HeldOutFixture],
    ) -> Tuple[bool, str, float, int]:
        """omega-verified-rsi gate: reject if held-out recall drops or net tokens inflate."""
        if not held_out_fixtures:
            return False, "empty_held_out_fixtures", 0.0, 0

        baseline_recall = sum(f.baseline_recall for f in held_out_fixtures) / len(held_out_fixtures)
        baseline_tokens = sum(f.baseline_tokens for f in held_out_fixtures)

        recalls: List[float] = []
        total_tokens = 0
        for fixture in held_out_fixtures:
            rec, tok = self._evaluator(fixture, candidate)
            recalls.append(rec)
            total_tokens += tok

        new_recall = sum(recalls) / len(recalls)
        if new_recall < baseline_recall:
            return (
                False,
                f"held_out_recall_regression:{new_recall:.4f}<{baseline_recall:.4f}",
                new_recall,
                total_tokens,
            )
        if total_tokens > baseline_tokens:
            return (
                False,
                f"net_token_inflation:{total_tokens}>{baseline_tokens}",
                new_recall,
                total_tokens,
            )
        return True, "accepted", new_recall, total_tokens

    def evaluate_ruleset(
        self,
        rules: List[CandidateRuleDiff],
        held_out_fixtures: List[HeldOutFixture],
    ) -> Tuple[bool, str, float, int]:
        """Evaluate the complete *rules* set together against *held_out_fixtures*."""
        if not held_out_fixtures:
            return False, "empty_held_out_fixtures", 0.0, 0

        baseline_recall = sum(f.baseline_recall for f in held_out_fixtures) / len(held_out_fixtures)
        baseline_tokens = sum(f.baseline_tokens for f in held_out_fixtures)

        recalls: List[float] = []
        total_tokens = 0
        for fixture in held_out_fixtures:
            matching = [r for r in rules if r.content_type == fixture.content_type]
            if not matching:
                recalls.append(fixture.baseline_recall)
                total_tokens += fixture.baseline_tokens
            elif self._evaluator is default_fixture_evaluator:
                rec, tok = _apply_rules_to_fixture(fixture, matching)
                recalls.append(rec)
                total_tokens += tok
            else:
                # Custom evaluator: evaluate each matching rule and take the latest/combined outcome
                rec, tok = fixture.baseline_recall, fixture.baseline_tokens
                for r in matching:
                    rec, tok = self._evaluator(fixture, r)
                recalls.append(rec)
                total_tokens += tok

        new_recall = sum(recalls) / len(recalls)
        if new_recall < baseline_recall:
            return (
                False,
                f"held_out_recall_regression:{new_recall:.4f}<{baseline_recall:.4f}",
                new_recall,
                total_tokens,
            )
        if total_tokens > baseline_tokens:
            return (
                False,
                f"net_token_inflation:{total_tokens}>{baseline_tokens}",
                new_recall,
                total_tokens,
            )
        return True, "accepted", new_recall, total_tokens

    def optimize(
        self,
        replay_log: CompressionReplayLog,
        held_out_fixtures: List[HeldOutFixture],
    ) -> OptimizationResult:
        """Propose candidates from *replay_log* and filter via *held_out_fixtures*."""
        baseline_recall = (
            sum(f.baseline_recall for f in held_out_fixtures) / len(held_out_fixtures)
            if held_out_fixtures
            else 0.0
        )
        baseline_tokens = sum(f.baseline_tokens for f in held_out_fixtures)

        result = OptimizationResult(
            baseline_recall=baseline_recall,
            new_recall=baseline_recall,
            baseline_tokens=baseline_tokens,
            new_tokens=baseline_tokens,
        )

        candidates = self.propose_candidates(replay_log)
        for cand in candidates:
            ok, reason, _, _ = self.verify_candidate(cand, held_out_fixtures)
            if not ok:
                result.rejected.append((cand, reason))
                continue

            combined_ok, combined_reason, combined_recall, combined_tokens = self.evaluate_ruleset(
                [*result.accepted, cand], held_out_fixtures
            )
            if combined_ok:
                result.accepted.append(cand)
                result.new_recall = combined_recall
                result.new_tokens = combined_tokens
            else:
                result.rejected.append((cand, combined_reason))

        return result
