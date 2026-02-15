"""Unit tests for skill-level composable compression pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / "skills" / "token-saver-context-compression" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from _pipeline import (  # noqa: E402
    PipelineExecutionError,
    PipelineStage,
    run_pipeline,
    run_pipeline_async,
)


def test_pipeline_runs_stages_in_order():
    calls: list[str] = []

    def stage_a(state):
        calls.append("a")
        state["value"] = 1
        return state

    def stage_b(state):
        calls.append("b")
        state["value"] += 2
        return state

    result = run_pipeline(
        {"value": 0},
        [
            PipelineStage(name="a", fn=stage_a),
            PipelineStage(name="b", fn=stage_b),
        ],
    )

    assert calls == ["a", "b"]
    assert result["value"] == 3


@pytest.mark.asyncio
async def test_pipeline_async_supports_sync_and_async_stages():
    async def stage_async(state):
        await asyncio.sleep(0)
        state["value"] += 3
        return state

    def stage_sync(state):
        state["value"] += 2
        return state

    result = await run_pipeline_async(
        {"value": 1},
        [
            PipelineStage(name="sync", fn=stage_sync),
            PipelineStage(name="async", fn=stage_async),
        ],
    )

    assert result["value"] == 6


def test_pipeline_error_contains_stage_name():
    def boom(state):
        raise ValueError("nope")

    with pytest.raises(PipelineExecutionError, match="stage='boom'"):
        run_pipeline({"ok": True}, [PipelineStage(name="boom", fn=boom)])
