import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from hrsmpipe.evaluate import metrics, plots  # noqa: E402
from hrsmpipe.lgar import merge  # noqa: E402


def test_compute_metrics_values():
    obs = np.array([0.10, 0.20, 0.30, 0.40])
    pred = obs + 0.05
    bias, rmse, ubrmse, r = metrics.compute_metrics(obs, pred)
    assert bias == pytest.approx(0.05) and rmse == pytest.approx(0.05)
    assert ubrmse == pytest.approx(0.0, abs=1e-12) and r == pytest.approx(1.0)
    noisy = np.array([0.30, 0.20, 0.10, 0.40])
    bias, rmse, ubrmse, r = metrics.compute_metrics(obs, noisy)
    assert bias == pytest.approx(0.0, abs=1e-12)
    assert rmse == pytest.approx(np.sqrt(0.02)) and ubrmse == pytest.approx(rmse)  # zero bias
    assert r == pytest.approx(0.2)  # cov 0.0025 / var 0.0125


def _site_frame(n_days=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-06-01", periods=24 * n_days, freq="h", tz="UTC", name="time")
    truth = 0.25 + 0.05 * np.sin(np.arange(len(idx)) / 30)
    df = pd.DataFrame(index=idx, columns=merge.COLUMNS, dtype="float64")
    for layer, shift in (("sly", 0.0), ("rly", 0.02)):
        df[f"observation_{layer}"] = truth + shift
        df[f"analysis_sm_{layer}"] = truth + shift - 0.03 + rng.normal(0, 0.005, len(idx))
        df[f"analysis_sm_{layer}_cdf"] = truth + shift - 0.01
        df[f"original_sm_{layer}"] = truth + shift - 0.06
        df[f"original_sm_{layer}_cdf"] = truth + shift - 0.02
        df[f"analysis_sd_{layer}"] = 0.02
        df[f"analysis_sd_{layer}_cdf"] = 0.015
        ml = np.full(len(idx), np.nan)
        ml[::24] = truth[::24] + shift + 0.01  # one ML value per day
        df[f"{layer}_sm_ml_arr"] = ml
    df["precip_mm"] = 0.0
    df.loc[df.index[10], "precip_mm"] = 12.0
    df["precip_source"], df["pet_source"] = "era5", "era5"
    return df


def test_site_metrics_rows_series_and_values():
    rows = metrics.site_metrics(_site_frame(), 7, "ALTN")
    df = pd.DataFrame(rows)
    assert list(df.columns) == metrics.METRIC_COLUMNS
    assert set(df["series"]) == {
        f"{s}{suffix}" for s in metrics.SERIES for suffix in ("", "_daily")
    }
    assert (
        set(df["layer"]) == {"sly", "rly"}
        and set(df["site_id"]) == {7}
        and set(df["station_code"]) == {"ALTN"}
    )
    ml = df[(df.series == "ml") & (df.layer == "sly")].iloc[0]
    assert ml["n"] == 6 and ml["bias"] == pytest.approx(0.01)  # one ML value per day, +0.01
    lg = df[(df.series == "lgar_original") & (df.layer == "rly")].iloc[0]
    assert (
        lg["n"] == 144
        and lg["bias"] == pytest.approx(-0.06)
        and lg["ubrmse"] == pytest.approx(0.0, abs=1e-9)
    )
    daily = df[(df.series == "enkf_analysis_cdf_daily") & (df.layer == "sly")].iloc[0]
    assert daily["n"] == 6 and daily["bias"] == pytest.approx(-0.01, abs=1e-9)


def test_daily_pairing_matches_by_calendar_day():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    obs = pd.Series(np.r_[np.full(24, 0.2), np.full(24, 0.3)], index=idx)
    pred = pd.Series(
        np.r_[np.full(12, 0.1), np.full(12, 0.3), np.full(24, 0.5)], index=idx
    )  # day 1 mean 0.2
    pair = metrics._daily_pair(obs, pred)
    assert pair.iloc[:, 0].tolist() == pytest.approx([0.2, 0.3]) and pair.iloc[
        :, 1
    ].tolist() == pytest.approx([0.2, 0.5])
    assert metrics._row(pair) is None  # fewer than 3 matched points -> no row


def test_metrics_table_skips_failed_sites(tmp_path):
    from hrsmpipe.sites import SiteSet

    (tmp_path / "lgar").mkdir()
    _site_frame().to_parquet(tmp_path / "lgar" / "site_7.parquet")
    sites = SiteSet(
        pd.DataFrame(
            {
                "site_id": [7, 47],
                "station_code": ["ALTN", "ANGO"],
                "latitude": 43.0,
                "longitude": -89.0,
            }
        )
    )
    table = metrics.metrics_table(tmp_path, sites)
    assert set(table["site_id"]) == {7} and len(table) == 20


def test_paper_frame_daily_means_at_18h():
    d = plots.paper_frame(_site_frame())
    assert len(d) == 6 and set(d.index.hour) == {18}
    assert d["analysis_sm_sly_cdf_daily_avg"].iloc[0] == pytest.approx(
        _site_frame()["analysis_sm_sly_cdf"].iloc[:24].mean()
    )
    assert d["analysis_sd_sly_daily_avg"].iloc[0] == pytest.approx(0.02)  # RMS of a constant sd


def test_plots_render_headless(tmp_path):
    df = _site_frame()
    fig = plots.plot_site_paper(df, 7, "ALTN")
    assert len(fig.axes) >= 2  # two soil-moisture panels (+ twin precip axes)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("RMSE ML" in t for t in texts) and any("RMSE LGAR EnKF" in t for t in texts)
    fig.savefig(tmp_path / "paper.png", dpi=50)
    pairs = plots.validation_frame(df)
    fig2 = plots.plot_validation(pairs, 7, "ALTN")
    assert len(fig2.axes) == 4
    fig2.savefig(tmp_path / "validation.png", dpi=50)
    assert (tmp_path / "paper.png").stat().st_size > 1000


def test_paper_plot_without_observations_does_not_crash():
    df = _site_frame()
    df["observation_sly"] = np.nan
    df["observation_rly"] = np.nan
    fig = plots.plot_site_paper(df, 7, "ALTN")
    assert fig is not None
