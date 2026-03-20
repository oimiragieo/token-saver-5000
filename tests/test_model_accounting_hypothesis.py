from hypothesis import given, strategies as st

from src.metrics import compute_cost_savings


@given(
    original=st.integers(min_value=1, max_value=1_000_000),
    compressed=st.integers(min_value=0, max_value=1_000_000),
)
def test_cost_savings_never_negative_for_expansion(original, compressed):
    result = compute_cost_savings(
        original_tokens=original,
        compressed_tokens=compressed,
        model="claude-sonnet-4.6",
    )

    assert result.saved_tokens >= 0
    assert result.cost_savings_usd >= 0.0
