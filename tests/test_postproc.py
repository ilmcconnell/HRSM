import logging

import numpy as np
import pandas as pd
import pytest

from hrsmpipe.config import load_config
from hrsmpipe.ml.models import DROP_COLUMNS, FAMILY, ModelBundle, models_dir
from hrsmpipe.postproc import combine
from hrsmpipe.postproc.cdf import cdf_matching, empirical_cdf
from hrsmpipe.timeaxis import hourly_axis


def test_empirical_cdf():
    assert empirical_cdf([5, 1, 3]).tolist() == pytest.approx([1.0, 1 / 3, 2 / 3])
    assert empirical_cdf([1, 1]).tolist() == [0.75, 0.75]  # ties get the average rank


def test_cdf_matching_hand_computed():
    # source ranks/len = [1, 1/3, 2/3]; target quantiles = [.25, .5, .75, 1] of [10, 20, 30, 40]
    out = cdf_matching(np.array([3.0, 1.0, 2.0]), np.array([40.0, 10.0, 30.0, 20.0]))
    assert out.tolist() == pytest.approx(
        [40.0, 10 + (1 / 3 - 0.25) / 0.25 * 10, 20 + (2 / 3 - 0.5) / 0.25 * 10]
    )
    # the lowest source percentile below the lowest target quantile clamps to the target minimum
    assert cdf_matching(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([10.0, 20.0])).min() == 10.0


def _inf(rows):
    return pd.DataFrame(
        rows, columns=["site_id", "time", "sensor", "y_mean", "y_low", "y_high"]
    ).assign(time=lambda d: pd.to_datetime(d["time"], utc=True))


def test_reference_match_only_y_mean_and_unmatched_sites_warn(caplog):
    t = pd.date_range("2024-06-01", periods=3, freq="h", tz="UTC")
    rows = []
    for i, h in enumerate(t):
        rows += [
            (1, h, "sentinel2", 0.10 + 0.1 * i, 0.0, 0.2),
            (1, h, "sentinel1", 0.30 + 0.1 * i, 0.0, 0.2),
        ]
        rows += [(2, h, "sentinel1", 0.5 + 0.1 * i, 0.0, 0.2)]  # site 2 has no sentinel2
    inf = _inf(rows)
    with caplog.at_level(logging.WARNING):
        out = combine.reference_match(inf, "sentinel2")
    s1_site1 = (inf["site_id"] == 1) & (inf["sensor"] == "sentinel1")
    assert out[s1_site1].tolist() == pytest.approx(
        [0.10, 0.20, 0.30]
    )  # same distribution as the reference
    assert (
        out[inf["sensor"] == "sentinel2"].tolist()
        == inf.loc[inf["sensor"] == "sentinel2", "y_mean"].tolist()
    )
    site2 = inf["site_id"] == 2
    assert out[site2].tolist() == inf.loc[site2, "y_mean"].tolist()  # unmatched
    assert "[2]" in caplog.text


def test_sd_from_bounds_with_floor():
    inf = _inf(
        [
            (1, "2024-01-01", "hls", 0.2, 0.1, 0.1 + 3.29 * 0.05),
            (1, "2024-01-02", "hls", 0.2, 0.3, 0.3),
            (1, "2024-01-03", "hls", 0.2, 0.4, 0.3),
        ]
    )
    sd = combine.sd_from_bounds(inf, 3.29, 0.02)
    assert sd.tolist() == pytest.approx([0.05, 0.02, 0.02])  # <= 0 -> floor


def _values(rows):
    return pd.DataFrame(rows, columns=["site_id", "time", "sensor", "value"]).assign(
        time=lambda d: pd.to_datetime(d["time"], utc=True)
    )


AXIS = hourly_axis("2024-06-01", "2024-06-03")  # 48 h


def test_spread_order_and_precedence_with_consecutive_hours():
    v = _values(
        [(1, "2024-06-01 05:00", "sentinel2", 5.0), (1, "2024-06-01 06:00", "sentinel2", 6.0)]
    )
    grid, miu = combine.spread_and_grid(v, 1, AXIS, [1])
    g = grid[:, 0]
    # concat order is (hour-1 rows, originals, hour+1 rows) and the FIRST occurrence wins:
    assert g[4] == 5.0  # only hour-5 row shifted -1
    assert g[5] == 6.0  # the hour-6 row shifted -1 beats the hour-5 original
    assert g[6] == 6.0  # original beats the hour-5 row shifted +1
    assert g[7] == 6.0  # hour-6 row shifted +1
    assert np.isnan(g[3]) and np.isnan(g[8])
    assert miu[0] == pytest.approx(5.5)


def test_spread_does_not_cross_midnight():
    v = _values([(1, "2024-06-01 00:00", "hls", 1.0), (1, "2024-06-01 23:00", "hls", 2.0)])
    g = combine.spread_and_grid(v, 1, AXIS, [1])[0][:, 0]
    assert g[0] == 1.0 and g[1] == 1.0 and g[22] == 2.0 and g[23] == 2.0
    assert np.isnan(g[24])  # hour 24 / -1 rows are dropped, never moved to the next/previous day


def test_nanmean_over_sensors_and_missing_site():
    v = _values(
        [
            (1, "2024-06-01 10:00", "sentinel1", 0.2),
            (1, "2024-06-01 10:00", "sentinel2", 0.4),
            (1, "2024-06-01 10:00", "hls", 0.9),
            (1, "2024-06-01 12:00", "sentinel1", 0.3),
        ]
    )
    grid, miu = combine.spread_and_grid(v, 0, AXIS, [1, 2])
    assert grid[10, 0] == pytest.approx(0.5) and grid[12, 0] == 0.3  # mean over the sensors present
    assert np.isnan(grid[:, 1]).all() and np.isnan(miu[1])  # site without data
    assert miu[0] == pytest.approx(np.mean([0.2, 0.4, 0.9, 0.3]))


def test_postprocess_layer_end_to_end_small():
    cfg = load_config(None, {"run.start": "2024-06-01", "run.end": "2024-06-03"})
    t = pd.date_range("2024-06-01 08:00", periods=3, freq="h", tz="UTC")
    rows = []
    for i, h in enumerate(t):
        for sensor, base in (("sentinel1", 0.30), ("sentinel2", 0.10), ("hls", 0.20)):
            rows.append((1, h, sensor, base + 0.01 * i, base + 0.01 * i, 0.0, 0.0987))
    inf = pd.DataFrame(
        rows, columns=["site_id", "time", "sensor", "y_mean", "y_median", "y_low", "y_high"]
    )
    res = combine.postprocess_layer(inf, cfg, AXIS, [1])
    # after matching on sentinel2 every sensor equals the sentinel2 values 0.10, 0.11, 0.12
    assert res["ssm"][8, 0] == pytest.approx(
        0.11
    )  # hour 8: next hour's sentinel2 value wins (spread)
    assert res["ssm"][9, 0] == pytest.approx(0.12)
    assert res["sd"][8, 0] == pytest.approx(0.0987 / 3.29)
    assert res["ssm"].shape == (48, 1)  # N rows: no extra trailing row


def test_postprocess_layer_empty():
    cfg = load_config(None, {"run.start": "2024-06-01", "run.end": "2024-06-03"})
    res = combine.postprocess_layer(_inf([]).assign(y_median=[]), cfg, AXIS, [1, 2])
    assert np.isnan(res["ssm"]).all() and res["miu"].shape == (2,)


def test_all_18_models_load_and_drop_columns_match_scalers():
    bundle = ModelBundle.load(models_dir(load_config(None)))
    expected_inputs = {"ML1": 30, "ML2": 39, "ML3": 41}
    for sensor, fam in FAMILY.items():
        for layer in ("surface", "rootzone"):
            model = bundle.get(sensor, layer)
            assert model.scaler.n_features_in_ == expected_inputs[fam] - len(DROP_COLUMNS[fam])
