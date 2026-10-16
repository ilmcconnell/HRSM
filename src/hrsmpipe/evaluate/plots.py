"""Plots ported from the notebook: paper-style site figure (cell 130), validation (cell 129) and the
original-study comparison (cell 135). Functions return the matplotlib ``Figure``; saving is up to the caller."""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hrsmpipe.evaluate.metrics import compute_metrics

PAPER_PRECIP_COLOR = "#00A9FF"


def _naive_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    return out


def paper_frame(site_df: pd.DataFrame) -> pd.DataFrame:
    """Daily-mean columns of notebook cell 125, one row per day at 18:00 (``at_time("18:00")``).

    The ensemble ``sd`` daily value is the RMS of the hourly sd (sqrt of the mean variance).
    """
    d = _naive_index(site_df)
    day = d.index.date
    for layer in ("sly", "rly"):
        for col in (
            f"analysis_sm_{layer}",
            f"original_sm_{layer}",
            f"analysis_sm_{layer}_cdf",
            f"original_sm_{layer}_cdf",
        ):
            d[f"{col}_daily_avg"] = d.groupby(day)[col].transform("mean")
        for col in (f"analysis_sd_{layer}", f"analysis_sd_{layer}_cdf"):
            d[f"{col}_daily_avg"] = np.sqrt((d[col] ** 2).groupby(day).transform("mean"))
    return d.at_time("18:00")


def _rmse_vs_obs(obs_daily: pd.Series, estimate: pd.Series) -> tuple[float, int]:
    """RMSE of estimate vs daily-mean obs, matched by calendar day (estimate averaged per day), and the number of matched days"""
    estimate_daily = estimate.groupby(estimate.index.normalize()).mean()
    both = pd.concat([obs_daily, estimate_daily], axis=1, join="inner").dropna()
    if both.empty:
        return np.nan, 0
    return float(np.sqrt(np.mean((both.iloc[:, 1] - both.iloc[:, 0]) ** 2))), len(both)


def plot_paper_site_data(
    time_points_ml,
    ml_sly,
    ml_rly,
    time_points_obs,
    obs_sly,
    obs_rly,
    enkf_lgar_df: pd.DataFrame,
    time_points_precip=None,
    precip=None,
    precip_max_mm: float = 100,
    title: str = "",
    lgar_plot_cols: tuple[str, str] = ("analysis_sm_sly_daily_avg", "analysis_sm_rly_daily_avg"),
    lgar_rmse_cols: tuple[str, str] = ("analysis_sm_sly_daily_avg", "analysis_sm_rly_daily_avg"),
    lgar_sd_cols: tuple[str, str] | None = (
        "analysis_sd_sly_daily_avg",
        "analysis_sd_rly_daily_avg",
    ),
    lgar_original_cols: tuple[str, str] | None = (
        "original_sm_sly_daily_avg",
        "original_sm_rly_daily_avg",
    ),
    lgar_label_suffix: str = "",
    time_start: datetime | None = None,
    time_end: datetime | None = None,
    buffer_factor: float = 0.2,
    figsize: tuple[float, float] = (20, 8),
) -> plt.Figure:
    """
    2 panels (surface, rootzone):
    ML inference as small blue triangles, daily-mean in-situ observations as small red circles, LGAR EnKF analysis (daily mean) as a blue line
    with an optional +/- 1 sigma band (EnKF ensemble spread), optional open-loop LGAR (no assimilation) as a green line,
    optional daily total precipitation (mm) as inverted bars on a second y axis (0 - precip_max_mm), and
    RMSE vs daily-mean observations for ML and LGAR in the lower left (matched by calendar day)

    enkf_lgar_df: EnKF LGAR output with a datetime index; lgar_plot_cols are plotted, lgar_rmse_cols are used for RMSE,
    lgar_sd_cols (None: no band) are the ensemble sd for the +/- 1 sigma band,
    lgar_original_cols (None: not plotted) are the open-loop LGAR daily means, also used for its RMSE
    """
    ml = [
        pd.Series(np.asarray(v, dtype=float), index=pd.to_datetime(time_points_ml))
        for v in (ml_sly, ml_rly)
    ]
    # hourly in-situ observations -> daily mean
    obs = [
        pd.Series(np.asarray(v, dtype=float), index=pd.to_datetime(time_points_obs))
        .resample("D")
        .mean()
        .dropna()
        for v in (obs_sly, obs_rly)
    ]
    lgar_index = pd.to_datetime(enkf_lgar_df.index)

    # NOTE(hrsmpipe): without observations the window falls back to the model's time range
    if time_start is None:
        time_start = obs[0].index.min() if len(obs[0]) else lgar_index.min()
    if time_end is None:
        time_end = obs[0].index.max() if len(obs[0]) else lgar_index.max()

    lgar_lines_alpha = 0.8
    y_min = 0  # min(s.min() for s in in_window if len(s)) * (1 - buffer_factor)
    y_max = 0.5  # max(s.max() for s in in_window if len(s)) * (1 + buffer_factor)

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for (
        ax,
        layer_name,
        ml_s,
        obs_s,
        lgar_plot_col,
        lgar_rmse_col,
        lgar_sd_col,
        lgar_original_col,
    ) in zip(
        axes,
        ["Surface", "Rootzone"],
        ml,
        obs,
        lgar_plot_cols,
        lgar_rmse_cols,
        lgar_sd_cols or (None, None),
        lgar_original_cols or (None, None),
        strict=True,
    ):
        (obs_line,) = ax.plot(obs_s.index, obs_s.values, "o", color="red", markersize=2)
        (ml_line,) = ax.plot(ml_s.index, ml_s.values, "^", color="blue", markersize=4)
        lgar_mean = enkf_lgar_df[lgar_plot_col].to_numpy()
        enkf_band = None
        if lgar_sd_col is not None:
            lgar_sd = enkf_lgar_df[lgar_sd_col].to_numpy()
            enkf_band = ax.fill_between(
                lgar_index,
                lgar_mean - lgar_sd,
                lgar_mean + lgar_sd,
                color="tab:blue",
                alpha=0.2,
                linewidth=0,
            )
        (enkf_line,) = ax.plot(
            lgar_index, lgar_mean, "-", linewidth=1, color="tab:blue", alpha=lgar_lines_alpha
        )
        # one shared legend for both panels (built from the last panel; same artists styles in both)
        legend_items = [
            (obs_line, "in-situ observation (daily mean)"),
            (ml_line, "ML inference"),
            (
                (enkf_band, enkf_line)
                if enkf_band is not None
                else enkf_line,  # band drawn under the line in the legend icon
                (
                    "LGAR EnKF Analysis $\\pm 1\\sigma$"
                    if enkf_band is not None
                    else "LGAR EnKF Analysis (daily mean)"
                )
                + lgar_label_suffix,
            ),
        ]
        if lgar_original_col is not None:
            (original_line,) = ax.plot(
                lgar_index,
                enkf_lgar_df[lgar_original_col].to_numpy(),
                "-",
                color="green",
                alpha=lgar_lines_alpha,
            )
            legend_items.append((original_line, "LGAR simulation (daily mean)" + lgar_label_suffix))

        ml_rmse, ml_n = _rmse_vs_obs(obs_s, ml_s)
        lgar_rmse, lgar_n = _rmse_vs_obs(
            obs_s, pd.Series(enkf_lgar_df[lgar_rmse_col].to_numpy(), index=lgar_index)
        )
        rmse_text = (
            f"RMSE ML: {ml_rmse:.3f} m$^3$/m$^3$ (n={ml_n})\n"
            f"RMSE LGAR EnKF: {lgar_rmse:.3f} m$^3$/m$^3$ (n={lgar_n})"
        )
        if lgar_original_col is not None:
            original_rmse, original_n = _rmse_vs_obs(
                obs_s, pd.Series(enkf_lgar_df[lgar_original_col].to_numpy(), index=lgar_index)
            )
            rmse_text += f"\nRMSE LGAR simulation: {original_rmse:.3f} m$^3$/m$^3$ (n={original_n})"
        ax.text(0.01, 0.03, rmse_text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)

        ax.set_title(f"{title} {layer_name} Soil Moisture".strip())
        ax.set_xlabel("time")
        ax.set_ylabel("soil moisture ($m^3/m^3$)")
        ax.set_xlim(time_start, time_end)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)

        if precip is not None:
            ax2 = ax.twinx()
            # hourly precipitation -> daily total
            daily_precip = (
                pd.Series(np.asarray(precip, dtype=float), index=pd.to_datetime(time_points_precip))
                .resample("D")
                .sum()
            )
            precip_bars = ax2.bar(
                daily_precip.index,
                daily_precip.values,
                width=0.8,
                align="edge",
                color=PAPER_PRECIP_COLOR,
                alpha=0.5,
            )
            legend_items.append((precip_bars, "daily total precipitation"))
            ax2.set_ylim(
                precip_max_mm, 0
            )  # inverted: bars hang from the top, clear of the soil moisture
            ax2.set_ylabel("daily total precipitation (mm)")
            ax.set_zorder(
                ax2.get_zorder() + 1
            )  # soil moisture, legend and RMSE box above the precip bars
            ax.patch.set_visible(False)

    # shared time axis: no x tick labels (or x label) on the surface panel; bottom panel labels at 30 deg,
    # right end of each label anchored at its tick
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[0].set_xlabel("")
    axes[1].tick_params(axis="x", labelrotation=30, rotation_mode="anchor")
    plt.setp(axes[1].get_xticklabels(), horizontalalignment="right")

    # single legend across the top of the figure, panels laid out below it
    legend_ncol = 2
    legend_rows = int(np.ceil(len(legend_items) / legend_ncol))
    fig.legend(
        *zip(*legend_items, strict=True),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=legend_ncol,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.22 * legend_rows + 0.05) / figsize[1]))
    return fig


def plot_site_paper(
    site_df: pd.DataFrame, site_id: int, station_code: str, time_window=None, figsize=(20, 8)
) -> plt.Figure:
    """The paper figure for one merged ``lgar/site_<id>.parquet`` frame (LGAR outputs CDF-matched
    back to the ML climatology, as in the notebook's final call)."""
    d = _naive_index(site_df)
    daily = paper_frame(site_df)
    start, end = (pd.Timestamp(t) for t in time_window) if time_window else (None, None)
    return plot_paper_site_data(
        d.index,
        d["sly_sm_ml_arr"].to_numpy(),
        d["rly_sm_ml_arr"].to_numpy(),
        d.index,
        d["observation_sly"].to_numpy(),
        d["observation_rly"].to_numpy(),
        daily,
        time_points_precip=d.index,
        precip=d["precip_mm"].to_numpy(),
        time_start=start,
        time_end=end,
        title=f"{site_id}: {station_code}",
        figsize=figsize,
        lgar_plot_cols=("analysis_sm_sly_cdf_daily_avg", "analysis_sm_rly_cdf_daily_avg"),
        lgar_rmse_cols=("analysis_sm_sly_cdf_daily_avg", "analysis_sm_rly_cdf_daily_avg"),
        lgar_sd_cols=("analysis_sd_sly_cdf_daily_avg", "analysis_sd_rly_cdf_daily_avg"),
        lgar_original_cols=("original_sm_sly_cdf_daily_avg", "original_sm_rly_cdf_daily_avg"),
    )


def plot_validation(comparison_df: pd.DataFrame, site_id: int, station_code: str) -> plt.Figure:
    """2x2 time series + scatter of the EnKF analysis vs observations (notebook cell 129).

    ``comparison_df``: hourly, columns obs_surface, obs_rootzone, model_surface, model_rootzone (no NaN).
    """
    surface_metrics = compute_metrics(
        comparison_df["obs_surface"].values, comparison_df["model_surface"].values
    )
    rootzone_metrics = compute_metrics(
        comparison_df["obs_rootzone"].values, comparison_df["model_rootzone"].values
    )

    fig, axes = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [3, 1]})

    # Surface Time Series
    ax_ts_sf = axes[0, 0]
    ax_ts_sf.plot(
        comparison_df.index,
        comparison_df["obs_surface"],
        label="Observed Surface (5cm)",
        color="black",
        alpha=0.7,
    )
    ax_ts_sf.plot(
        comparison_df.index,
        comparison_df["model_surface"],
        label="EnKF Surface Analysis",
        color="tab:blue",
        linestyle="--",
    )
    ax_ts_sf.set_title(f"Surface (5cm) Soil Moisture Comparison: Site {site_id} ({station_code})")
    ax_ts_sf.set_ylabel("Volumetric Water Content (m³/m³)")
    ax_ts_sf.legend()
    ax_ts_sf.grid(True, linestyle=":", alpha=0.6)

    # Surface Scatter
    ax_sc_sf = axes[0, 1]
    ax_sc_sf.scatter(
        comparison_df["obs_surface"], comparison_df["model_surface"], alpha=0.4, color="tab:blue"
    )
    ax_sc_sf.plot([0, 0.5], [0, 0.5], "r--", alpha=0.8)
    ax_sc_sf.set_xlim(0, 0.5)
    ax_sc_sf.set_ylim(0, 0.5)
    ax_sc_sf.set_xlabel("Observed")
    ax_sc_sf.set_ylabel("EnKF Model")
    ax_sc_sf.set_title(f"R = {surface_metrics[3]:.3f}")
    ax_sc_sf.grid(True, linestyle=":", alpha=0.6)

    # Rootzone Time Series
    ax_ts_rz = axes[1, 0]
    ax_ts_rz.plot(
        comparison_df.index,
        comparison_df["obs_rootzone"],
        label="Observed Rootzone",
        color="black",
        alpha=0.7,
    )
    ax_ts_rz.plot(
        comparison_df.index,
        comparison_df["model_rootzone"],
        label="EnKF Rootzone Analysis",
        color="tab:orange",
        linestyle="--",
    )
    ax_ts_rz.set_title(f"Rootzone Soil Moisture Comparison: Site {site_id} ({station_code})")
    ax_ts_rz.set_ylabel("Volumetric Water Content (m³/m³)")
    ax_ts_rz.legend()
    ax_ts_rz.grid(True, linestyle=":", alpha=0.6)

    # Rootzone Scatter
    ax_sc_rz = axes[1, 1]
    ax_sc_rz.scatter(
        comparison_df["obs_rootzone"],
        comparison_df["model_rootzone"],
        alpha=0.4,
        color="tab:orange",
    )
    ax_sc_rz.plot([0, 0.5], [0, 0.5], "r--", alpha=0.8)
    ax_sc_rz.set_xlim(0, 0.5)
    ax_sc_rz.set_ylim(0, 0.5)
    ax_sc_rz.set_xlabel("Observed")
    ax_sc_rz.set_ylabel("EnKF Model")
    ax_sc_rz.set_title(f"R = {rootzone_metrics[3]:.3f}")
    ax_sc_rz.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig


def validation_frame(site_df: pd.DataFrame) -> pd.DataFrame:
    """Hourly obs/model pairs for ``plot_validation`` (rows with any NaN dropped)."""
    d = _naive_index(site_df)
    out = pd.DataFrame(
        {
            "obs_surface": d["observation_sly"],
            "obs_rootzone": d["observation_rly"],
            "model_surface": d["analysis_sm_sly"],
            "model_rootzone": d["analysis_sm_rly"],
        }
    )
    return out.dropna()


def plot_study_site(
    study_df: pd.DataFrame,
    site_id: int,
    station_code: str,
    ours_enkf_df: pd.DataFrame | None = None,
    precip: pd.Series | None = None,
    buffer_factor: float = 0.2,
) -> plt.Figure:
    """
    2 panels (surface, rootzone) in the style of plot_site_data
    study_df: original-study hourly csv for one site, time index
    ours_enkf_df: optional, our merged site frame with time index -> adds our LGAR daily mean
    precip: optional hourly precipitation (mm) series with time index -> daily totals as inverted bars
    """
    obs_any = study_df[["observation_sly", "observation_rly"]].dropna(how="all")
    time_start, time_end = obs_any.index[0], obs_any.index[-1]

    fig, axes = plt.subplots(2, 1, figsize=(20, 8))
    for ax, layer, layer_name in zip(axes, ["sly", "rly"], ["Surface", "Rootzone"], strict=True):
        obs = study_df[f"observation_{layer}"]
        ml = study_df[f"{layer}_sm_ml_arr"].dropna()
        study_daily = study_df[f"analysis_sm_{layer}"].resample("D").mean()
        series = [obs.dropna(), ml, study_daily]

        ax.plot(obs.index, obs, label="observation", color="black", alpha=0.7)
        ax.plot(ml.index, ml, "o", markersize=3, label="study combined ML inference")
        ax.plot(study_daily.index, study_daily, label="study LGAR daily mean")
        if ours_enkf_df is not None:
            ours_daily = ours_enkf_df[f"analysis_sm_{layer}"].resample("D").mean()
            ax.plot(ours_daily.index, ours_daily, label="our LGAR daily mean")
            series.append(ours_daily.dropna())

        y_min = min(s[time_start:time_end].min() for s in series)
        y_max = max(s[time_start:time_end].max() for s in series)
        ax.set_xlim(time_start, time_end)
        ax.set_ylim(y_min * (1 - buffer_factor), y_max * (1 + buffer_factor))
        ax.set_title(
            f"n_obs={obs.notna().sum()} n_ml={len(ml)} {layer_name} Soil Moisture for {site_id}: {station_code}"
        )
        ax.set_xlabel("time")
        ax.set_ylabel("soil moisture ($m^3/m^3$)")
        ax.legend(loc="upper left")
        ax.grid(True)

        if precip is not None:
            daily_precip = precip.resample("D").sum()[time_start:time_end]
            ax2 = ax.twinx()
            ax2.set_ylim(daily_precip.max() * 4, 0)  # Invert and scale precip - clear sm plots
            ax2.bar(daily_precip.index, daily_precip, color="tab:blue", alpha=0.5)
            ax2.set_ylabel("daily total precip (mm)")
            ax.set_zorder(ax2.get_zorder() + 1)  # soil moisture lines + legend above precip bars
            ax.patch.set_visible(False)
    fig.tight_layout()
    return fig
