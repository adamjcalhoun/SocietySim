from societysim.metrics import gini


def test_gini_perfect_equality():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-6)


def test_gini_perfect_inequality():
    # One agent has everything
    result = gini([0, 0, 0, 10])
    assert result == pytest.approx(0.75, abs=1e-6)


def test_gini_empty():
    assert gini([]) == 0.0


def test_gini_all_zero():
    assert gini([0, 0, 0]) == 0.0


import pytest
