import numpy as np
import pandas as pd

from hrsmpipe.timeaxis import hour_index, hourly_axis, year_doy_hour


def test_axis_is_half_open_utc():
    ax = hourly_axis("2024-01-01", "2024-01-02")
    assert len(ax) == 24 and str(ax.tz) == "UTC"
    assert ax[0] == pd.Timestamp("2024-01-01", tz="UTC") and ax[-1].hour == 23


def test_study_period_length_is_n_not_n_plus_1():
    assert len(hourly_axis("2016-01-01", "2025-01-01")) == 78912  # rows of the saved study arrays


def test_hour_index_floors_including_negative_lead():
    start = "2024-01-01"
    times = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:59",
            "2024-01-01 01:00",
            "2023-12-31 23:00",
            "2023-12-31 23:30",
        ]
    )
    assert hour_index(times, start).tolist() == [0, 0, 1, -1, -1]
    lead = hourly_axis(pd.Timestamp(start) - pd.Timedelta(hours=144), "2024-01-02")
    idx = hour_index(lead, start)
    assert idx[0] == -144 and idx[144] == 0 and idx[-1] == 23


def test_hour_index_accepts_aware_times():
    t = pd.DatetimeIndex(["2024-06-01 02:30"], tz="US/Central")  # 07:30 UTC
    assert hour_index(t, "2024-06-01").tolist() == [7]


def test_year_doy_hour():
    y, d, h = year_doy_hour(pd.to_datetime(["2024-12-31 23:00", "2023-01-01 00:00"]))
    assert (y.tolist(), d.tolist(), h.tolist()) == ([2024, 2023], [366, 1], [23, 0])
    assert isinstance(y, np.ndarray)
