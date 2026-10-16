import pandas as pd
import pytest

from hrsmpipe.config import load_config
from hrsmpipe.paths import resolve_pkg_path
from hrsmpipe.sites import SiteSet, load_sites


def test_packaged_site_sets():
    w = load_sites(load_config(None))
    assert len(w) == 79 and w.ids[:2] == [47, 7] and w.code(7) == "ALTN"  # CSV order is kept
    s = load_sites(load_config(None, {"sites.csv": "pkg:study_sites.csv"}))
    assert s.ids == [296, 1140, 2096]  # 2096, not site_info.csv's 2069


def test_include_preserves_csv_order_not_include_order():
    s = load_sites(load_config(None, {"sites.include": [7, 47]}))
    assert s.ids == [47, 7]
    assert s.index_of(7) == 1


def test_validation():
    with pytest.raises(ValueError):
        load_sites(load_config(None, {"sites.include": [999999]}))
    with pytest.raises(ValueError):
        SiteSet(
            pd.DataFrame({"site_id": [1, 1], "station_code": "a", "latitude": 0, "longitude": 0})
        )
    with pytest.raises(ValueError):
        SiteSet(pd.DataFrame({"site_id": [1]}))
    with pytest.raises(FileNotFoundError):
        resolve_pkg_path("pkg:nope.csv")
