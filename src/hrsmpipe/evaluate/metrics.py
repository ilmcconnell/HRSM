"""Metrics vs observations: ``metrics.csv`` and the comparison with the original study (cells 128, 136-137)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from hrsmpipe.config import RunConfig
from hrsmpipe.sites import SiteSet

log = logging.getLogger(__name__)

LAYERS = {"sly": "surface", "rly": "rootzone"}
# series name -> column template in lgar/site_<id>.parquet
SERIES = {
    "ml": "{layer}_sm_ml_arr",
    "lgar_original": "original_sm_{layer}",
    "enkf_analysis": "analysis_sm_{layer}",
    "enkf_analysis_cdf": "analysis_sm_{layer}_cdf",
    "lgar_original_cdf": "original_sm_{layer}_cdf",
}
METRIC_COLUMNS = ["site_id", "station_code", "layer", "series", "n", "bias", "rmse", "ubrmse", "r"]


def compute_metrics(obs, pred):
    """Bias, RMSE, unbiased RMSE and Pearson R of ``pred`` against ``obs`` (notebook cell 128)."""
    bias = np.mean(pred - obs)
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    # Unbiased RMSE (ubRMSE)
    pred_demeaned = pred - np.mean(pred)
    obs_demeaned = obs - np.mean(obs)
    ubrmse = np.sqrt(np.mean((pred_demeaned - obs_demeaned) ** 2))
    r_val, _ = pearsonr(obs, pred)
    return bias, rmse, ubrmse, r_val


def _hourly_pair(obs: pd.Series, pred: pd.Series) -> pd.DataFrame:
    return pd.concat([obs, pred], axis=1, join="inner").dropna()


def _daily_pair(obs: pd.Series, pred: pd.Series) -> pd.DataFrame:
    """Daily means of both, matched by calendar day (as ``_rmse_vs_obs`` in notebook cell 130)."""
    obs_daily = obs.resample("D").mean().dropna()
    pred_daily = pred.groupby(pred.index.normalize()).mean()
    return pd.concat([obs_daily, pred_daily], axis=1, join="inner").dropna()


def _row(pair: pd.DataFrame) -> dict | None:
    if len(pair) < 3:
        return None
    bias, rmse, ubrmse, r_val = compute_metrics(
        pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
    )
    return {"n": len(pair), "bias": bias, "rmse": rmse, "ubrmse": ubrmse, "r": r_val}


def site_metrics(df: pd.DataFrame, site_id: int, station_code: str) -> list[dict]:
    """Rows for one merged site frame: every series and its ``*_daily`` variant, both layers."""
    rows = []
    for layer in LAYERS:
        obs = df[f"observation_{layer}"]
        for name, template in SERIES.items():
            pred = df[template.format(layer=layer)]
            for suffix, pair_fn in (("", _hourly_pair), ("_daily", _daily_pair)):
                stats = _row(pair_fn(obs, pred))
                if stats is not None:
                    rows.append(
                        {
                            "site_id": site_id,
                            "station_code": station_code,
                            "layer": layer,
                            "series": name + suffix,
                            **stats,
                        }
                    )
    return rows


def load_site_frame(run_root: Path, site_id: int) -> pd.DataFrame | None:
    p = Path(run_root) / "lgar" / f"site_{site_id}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def metrics_table(run_root: Path, sites: SiteSet) -> pd.DataFrame:
    rows = []
    for sid in sites.ids:
        df = load_site_frame(run_root, sid)
        if df is None:
            log.warning("site %s has no lgar/site_%s.parquet: skipped in metrics", sid, sid)
            continue
        rows += site_metrics(df, sid, sites.code(sid))
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


# ---- comparison with the original study ---------------------------------------------------------


def _naive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    return out


def study_comparison(run_root: Path, sites: SiteSet, study_dir: Path) -> pd.DataFrame:
    """Our outputs against the original study's ``EnKF_LGAR_<id>_hourly.csv`` (notebook cells 136-137).

    ``table = "ml_stage"``: our combined ML vs the study's combined ML on the hours both have
    (study = reference). ``table = "vs_obs"``: study and our series against the study's observations.
    """
    rows = []
    for sid in sites.ids:
        csv = Path(study_dir) / f"EnKF_LGAR_{sid}_hourly.csv"
        ours = load_site_frame(run_root, sid)
        if not csv.exists() or ours is None:
            continue
        study = pd.read_csv(csv, parse_dates=["time"], index_col="time")
        ours = _naive(ours)
        base = {"site_id": sid, "station_code": sites.code(sid)}
        for layer in LAYERS:
            pair = _hourly_pair(study[f"{layer}_sm_ml_arr"], ours[f"{layer}_sm_ml_arr"])
            stats = _row(pair) or {"n": len(pair)}
            rows.append(
                {
                    "table": "ml_stage",
                    **base,
                    "layer": layer,
                    "series": "our combined ML vs study",
                    **stats,
                }
            )
            obs = study[f"observation_{layer}"]
            preds = {
                "study analysis": study[f"analysis_sm_{layer}"],
                "study analysis cdf": study[f"analysis_sm_{layer}_cdf"],
                "study combined ML": study[f"{layer}_sm_ml_arr"],
                "our analysis": ours[f"analysis_sm_{layer}"],
                "our combined ML": ours[f"{layer}_sm_ml_arr"],
            }
            for name, pred in preds.items():
                stats = _row(_hourly_pair(obs, pred))
                if stats is not None:
                    rows.append(
                        {"table": "vs_obs", **base, "layer": layer, "series": name, **stats}
                    )
    cols = ["table", *METRIC_COLUMNS]
    return pd.DataFrame(rows, columns=cols)


def evaluate_inputs(cfg: RunConfig) -> Path | None:
    return Path(cfg.evaluate.study_dir).expanduser() if cfg.evaluate.study_dir else None
