"""Live Earth Engine smoke test: one site, one day. Run with `pixi run test -m gee`."""

import numpy as np
import pytest

from hrsmpipe.config import load_config
from hrsmpipe.gee import static, timeseries
from hrsmpipe.sites import load_sites

pytestmark = pytest.mark.gee


def test_one_site_one_day(data_root):
    cfg = load_config(
        None,
        {
            "run.name": "smoke",
            "run.start": "2024-06-01",
            "run.end": "2024-06-02",
            "sites.include": [7],
        },
    )
    sites = load_sites(cfg)
    assert static.fetch_static(cfg, sites) == [7]
    st = static.load_static(cfg, sites)
    assert 200 < st.loc[7, "elevation"] < 400 and 0 <= st.loc[7, "sand_0_100"] <= 100
    assert st.loc[7, "ksat_0_100"] > 0
    assert static.fetch_static(cfg, sites) == []  # cached

    out = timeseries.fetch_rs(cfg, sites, sources=("era5",))["era5"]
    assert out.temperature_2m.shape == (24 + 144, 1)
    assert np.isfinite(out.temperature_2m.values).all()
    assert 250 < float(out.temperature_2m.mean()) < 320  # Kelvin
