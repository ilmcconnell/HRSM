import numpy as np
import pandas as pd
import pytest

from hrsmpipe.obs.rootzone import compute_rootzone, layer_thicknesses

DEPTHS = [5, 10, 20, 50, 100]
W = [7.5, 7.5, 20, 40, 25]


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=[f"sm_{d}" for d in DEPTHS],
        index=pd.date_range("2024-01-01", periods=len(rows), freq="h"),
    )


def test_midpoint_thicknesses():
    assert layer_thicknesses(DEPTHS) == W
    assert sum(layer_thicknesses(DEPTHS)) == 100
    assert layer_thicknesses([5, 100]) == [52.5, 47.5]
    with pytest.raises(ValueError):
        layer_thicknesses([5, 120], 100)


def test_weighted_mean():
    df = _frame([[0.1, 0.2, 0.3, 0.4, 0.5]])
    expect = (0.1 * 7.5 + 0.2 * 7.5 + 0.3 * 20 + 0.4 * 40 + 0.5 * 25) / 100
    assert compute_rootzone(df, DEPTHS, W).iloc[0] == pytest.approx(expect)
    # default weights are the midpoint rule
    assert compute_rootzone(df, DEPTHS).iloc[0] == pytest.approx(expect)
    # the validated legacy array weights differ
    legacy = [2.5, 7.5, 20, 40, 25]
    assert compute_rootzone(df, DEPTHS, legacy).iloc[0] == pytest.approx(
        (0.1 * 2.5 + 0.2 * 7.5 + 0.3 * 20 + 0.4 * 40 + 0.5 * 25) / sum(legacy)
    )


def test_validity_rules():
    nan = np.nan
    df = _frame(
        [
            [nan, 0.2, 0.3, 0.4, 0.5],  # no surface
            [0.1, 0.2, 0.3, 0.4, nan],  # no deepest
            [0.1, nan, nan, nan, 0.5],  # no middle layer
            [0.1, nan, 0.3, nan, 0.5],  # one middle layer is enough
        ]
    )
    rz = compute_rootzone(df, DEPTHS, W)
    assert rz.iloc[:3].isna().all() and rz.iloc[3] == rz.iloc[3]


def test_interior_interpolation_along_depth():
    # 10 cm missing between 5 (0.1) and 20 (0.4): interpolated on the depth axis -> 0.2
    df = _frame([[0.1, np.nan, 0.4, 0.4, 0.4]])
    got = compute_rootzone(df, DEPTHS, W).iloc[0]
    expect = (0.1 * 7.5 + 0.2 * 7.5 + 0.4 * 20 + 0.4 * 40 + 0.4 * 25) / 100
    assert got == pytest.approx(expect)
