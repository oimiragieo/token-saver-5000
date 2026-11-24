"""CL 3c tests for CompactPrompt tabular quantization in TOONSerializer."""

from __future__ import annotations

from decimal import Decimal
import math

import pytest

from src.toon_serializer import dequantize_numeric_columns, quantize_numeric_columns


def test_quantize_dequantize_within_eps_max_across_bit_widths():
    rows = [
        {"id": "r1", "score": 0.1234, "latency_ms": 12.5, "const": 42.0, "note": None},
        {"id": "r2", "score": 0.5678, "latency_ms": 87.2, "const": 42.0, "note": "ok"},
        {"id": "r3", "score": 0.9912, "latency_ms": 45.0, "const": 42.0, "note": None},
    ]
    for bits in (2, 4, 8, 12, 16):
        payload = quantize_numeric_columns(rows, bits=bits)
        restored = dequantize_numeric_columns(payload)
        assert len(restored) == len(rows)

        for col, meta in payload["columns"].items():
            eps_max = meta["eps_max"]
            for orig, rec in zip(rows, restored):
                assert abs(float(orig[col]) - float(rec[col])) <= eps_max + 1e-12

        # Non-numeric and None fields remain untouched
        assert [r["id"] for r in restored] == ["r1", "r2", "r3"]
        assert [r["note"] for r in restored] == [None, "ok", None]


def test_quantize_handles_constant_none_bool_and_empty_rows():
    assert quantize_numeric_columns([]) == {"bits": 8, "columns": {}, "rows": []}

    rows = [
        {"flag": True, "const": 7.0, "x": None},
        {"flag": False, "const": 7.0, "x": 3.5},
    ]
    payload = quantize_numeric_columns(rows, bits=8)
    assert "flag" not in payload["columns"]
    assert payload["columns"]["const"]["eps_max"] == 0.0
    assert payload["columns"]["x"]["min"] == 3.5
    assert payload["columns"]["x"]["max"] == 3.5

    restored = dequantize_numeric_columns(payload)
    assert restored[0]["flag"] is True
    assert restored[0]["x"] is None
    assert restored[1]["x"] == 3.5
    assert restored[0]["const"] == 7.0


def test_quantize_rejects_non_finite_and_invalid_bits():
    with pytest.raises(ValueError, match="Non-finite"):
        quantize_numeric_columns([{"x": math.nan}])
    with pytest.raises(ValueError, match="Non-finite"):
        quantize_numeric_columns([{"x": math.inf}])

    for bad_bits in (1, 17, 0, -4, True):
        with pytest.raises(ValueError, match="bits"):
            quantize_numeric_columns([{"x": 1.0}], bits=bad_bits)


def test_quantize_preserves_53_bit_integer_precision_and_extreme_floats():
    int_rows = [{"v": 9007199254740992}, {"v": 9007199254740993}]
    int_payload = quantize_numeric_columns(int_rows, bits=8)
    int_restored = dequantize_numeric_columns(int_payload)
    assert int_payload["columns"]["v"]["min"] == 9007199254740992
    assert int_payload["columns"]["v"]["max"] == 9007199254740993
    assert int_restored[0]["v"] == 9007199254740992
    assert int_restored[1]["v"] == 9007199254740993
    assert Decimal(int_restored[1]["v"]) - Decimal(int_restored[0]["v"]) == Decimal(1)

    ext_rows = [{"v": -1e308}, {"v": 0.0}, {"v": 1e308}]
    ext_payload = quantize_numeric_columns(ext_rows, bits=8)
    ext_restored = dequantize_numeric_columns(ext_payload)
    for rec in ext_restored:
        assert math.isfinite(rec["v"])
    assert Decimal(str(ext_restored[0]["v"])) == Decimal("-1e+308")
    assert Decimal(str(ext_restored[2]["v"])) == Decimal("1e+308")
    assert math.isfinite(ext_payload["columns"]["v"]["eps_max"])


def test_quantize_mixed_large_int_and_float_preserves_precision():
    from decimal import Decimal

    rows = [{"id": 9007199254740993}, {"id": 9007199254740994.0}]
    q = quantize_numeric_columns(rows, bits=8)
    restored = dequantize_numeric_columns(q)
    eps = Decimal(str(q["columns"]["id"]["eps_max"]))
    for orig, rec in zip(rows, restored):
        err = abs(Decimal(str(orig["id"])) - Decimal(str(rec["id"])))
        assert err <= eps + Decimal("1e-12")
