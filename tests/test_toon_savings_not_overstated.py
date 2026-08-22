"""`estimate_token_savings` overstated TOON's benefit by ~31 percentage points.

WHY THIS IS A MONEY BUG, not a cosmetic one. `toon_encode` / `toon_decode` are
PUBLISHED MCP tools (both appear in the 193-tool catalogue), and this product
sells token savings. The estimator returned a `savings_percentage` and a
customer-facing string `"92.5% fewer tokens"` for a payload whose real saving is
**61.8%** — measured with tiktoken `cl100k_base`, the same encoder the engine's
own canonical `TokenEstimator` uses.

THE MECHANISM IS SELF-INFLICTED, WHICH IS WHY IT WAS SO LARGE. The old
implementation counted WORDS:

    json_tokens = int(len(json_str.split()) * 1.3)

TOON's entire compression trick is packing a record onto one line with commas
and NO spaces:

    n0,0.900,src/module_0.py,function,handler_0,100

That whole row is ONE "word" to `.split()`, while pretty-printed JSON is spaces
everywhere. So the estimator inflated exactly in proportion to how well TOON
worked — it counted the very thing TOON removes. The docstring even said "For
accurate counting, integrate tiktoken", so the shortcut was known; it just
shipped in a customer-facing surface anyway.

The real number is still GOOD (~62%). Nothing about TOON needed fixing — only
the claim about it. A public number can be wrong in either direction, and
overstatement is the dangerous one for a company selling savings.

`tiktoken` was already a declared dependency in BOTH `api/requirements.txt` and
this package, so the fix adds nothing.
"""

from __future__ import annotations

import json

import pytest

from src import toon_serializer as ts

tiktoken = pytest.importorskip("tiktoken", reason="tiktoken is a declared dependency here")


def _payload(n: int = 40) -> dict:
    """A realistic STRUCTURED tool result — the shape TOON targets."""
    return {
        "results": [
            {
                "node_id": f"n{i}",
                "score": round(0.9 - i * 0.01, 3),
                "path": f"src/module_{i}.py",
                "kind": "function",
                "name": f"handler_{i}",
                "line": 100 + i,
            }
            for i in range(n)
        ]
    }


def _real_savings_pct(json_str: str, toon_str: str) -> float:
    enc = tiktoken.get_encoding("cl100k_base")
    j, t = len(enc.encode(json_str)), len(enc.encode(toon_str))
    assert j > 0, "control: the JSON payload tokenised to nothing"
    return (j - t) / j * 100


def test_the_reported_saving_matches_a_real_tokenizer():
    """The discriminating property. The old word-count heuristic failed this by
    30.7 percentage points."""
    payload = _payload()
    json_str = json.dumps(payload, indent=2)
    toon_str = ts.format_response(payload, ts.OutputFormat.TOON)

    claimed = ts.estimate_token_savings(json_str, toon_str)["savings_percentage"]
    real = _real_savings_pct(json_str, toon_str)

    assert abs(claimed - real) <= 5.0, (
        f"reported {claimed}% saving vs a real tokenizer's {real:.1f}% "
        f"({claimed - real:+.1f}pp). This number is customer-facing — "
        "toon_encode is a published MCP tool and we sell token savings."
    )


def test_toon_really_does_save_a_lot():
    """The control. An estimator hardcoded to 0% would pass the test above while
    destroying the feature's whole claim."""
    payload = _payload()
    json_str = json.dumps(payload, indent=2)
    toon_str = ts.format_response(payload, ts.OutputFormat.TOON)

    real = _real_savings_pct(json_str, toon_str)
    assert real > 40.0, (
        f"TOON saved only {real:.1f}% on structured data — if this is genuinely "
        "true the feature's premise is gone, and the fix is not in the estimator."
    )
    assert ts.estimate_token_savings(json_str, toon_str)["savings_percentage"] > 40.0


def test_it_does_not_divide_by_zero_on_empty_input():
    out = ts.estimate_token_savings("", "")
    assert out["savings_percentage"] == 0


def test_toon_encode_handler_baseline_is_compact_json_not_pretty_printed():
    """The MCP tool's savings_percent must not be inflated by our own formatting.

    ``handle_toon_encode`` originally built its baseline with
    ``json.dumps(data, indent=2)`` and compared those CHARACTERS against compact
    TOON, reporting the result as ``savings_percent``. Two defects compounded,
    both in the flattering direction, on a published tool whose entire output is
    a savings claim. Measured on a 12-row payload: 46.9% reported, 31.6% once
    the baseline is compact, 23.7% once it is counted in tokens.

    Nobody pretty-prints JSON to save tokens, so the old baseline measured our
    indentation rather than TOON.
    """
    import asyncio

    from src.handlers.experimental_handlers import handle_toon_encode

    rows = [
        {
            "id": i,
            "title": f"Result {i}",
            "url": f"https://example.com/{i}",
            "score": round(0.9 - i * 0.01, 3),
            "snippet": "Semantic compression reduces token usage.",
        }
        for i in range(12)
    ]

    raw = handle_toon_encode(None, {"data": rows})
    if asyncio.iscoroutine(raw):
        raw = asyncio.run(raw)
    out = json.loads(raw)

    # Control: a handler that errored or returned an empty payload would make
    # every assertion below vacuous.
    assert out.get("original_tokens", 0) > 0, f"no baseline tokens to compare: {out}"
    assert out.get("toon_output"), "handler produced no TOON output"

    # The baseline must be compact JSON, byte-for-byte.
    assert out["original_chars"] == len(json.dumps(rows, separators=(",", ":"))), (
        "baseline is not compact JSON — if this is indent=2 again, savings_percent "
        "is inflated by our own whitespace"
    )
    assert out["original_chars"] < len(json.dumps(rows, indent=2)), (
        "compact baseline should be smaller than the pretty one; if not, this test "
        "cannot tell the two apart and proves nothing"
    )

    # savings_percent is TOKENS, so it must track the token counts, not the chars.
    expected = round(
        (out["original_tokens"] - out["toon_tokens"]) / out["original_tokens"] * 100, 1
    )
    assert out["savings_percent"] == expected, (
        f"savings_percent {out['savings_percent']} does not equal the token savings "
        f"{expected} — the label and the value have drifted apart again"
    )
    assert out["char_savings_percent"] != out["savings_percent"], (
        "char and token savings are identical, so this fixture cannot detect the two "
        "being conflated — pick a payload where they differ"
    )
