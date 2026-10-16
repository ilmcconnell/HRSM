import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_legacy_obs_npy.py"


def _load():
    spec = importlib.util.spec_from_file_location("convert_legacy_obs_npy", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_legacy_conversion_localises_central_to_utc(tmp_path):
    from datetime import datetime

    rows = np.empty((4, 10), dtype=object)
    stamps = [
        datetime(2024, 1, 1, 0),
        datetime(2024, 7, 1, 0),
        datetime(2024, 11, 3, 1),
        datetime(2024, 3, 10, 2),
    ]
    for i, t in enumerate(stamps):
        rows[i] = [7, "ALTN", t, 0.1 + i, 0.2, 0.3, 0.4, 0.5, 0.25, float(i)]
    p = tmp_path / "legacy.npy"
    np.save(p, rows, allow_pickle=True)
    df, dropped = _load().convert(p)
    assert dropped == 2  # 2024-11-03 01:00 is ambiguous, 2024-03-10 02:00 does not exist
    assert df["time"].tolist() == [
        pd.Timestamp("2024-01-01 06:00", tz="UTC"),
        pd.Timestamp("2024-07-01 05:00", tz="UTC"),
    ]
    assert (
        df["sm_surface"].tolist() == df["sm_5"].tolist()
        and "sm_rootzone" in df
        and df["site_id"].tolist() == [7, 7]
    )


def test_config_reference_page_is_up_to_date():
    spec = importlib.util.spec_from_file_location(
        "gen_config_reference", SCRIPT.parent / "gen_config_reference.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.OUT.read_text() == mod.render(), (
        "run: pixi run python scripts/gen_config_reference.py"
    )
    assert "`gee.window_hours`" in mod.render()
