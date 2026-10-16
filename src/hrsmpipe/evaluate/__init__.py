"""``hrsmpipe evaluate``: metrics vs observations, plots, optional comparison with the original study."""

from __future__ import annotations

import logging

import matplotlib
import pandas as pd

from hrsmpipe.config import RunConfig
from hrsmpipe.paths import RunPaths, resolve_data_root
from hrsmpipe.sites import SiteSet, load_sites

log = logging.getLogger(__name__)


def evaluate_paths(paths: RunPaths) -> dict:
    d = paths.root / "evaluate"
    return {
        "root": d,
        "metrics": d / "metrics.csv",
        "study": d / "study_comparison.csv",
        "plots": d / "plots",
    }


def evaluate(cfg: RunConfig, sites: SiteSet | None = None) -> None:
    from hrsmpipe.evaluate import metrics, plots

    matplotlib.use("Agg")  # headless: figures are written to disk, never shown
    import matplotlib.pyplot as plt

    sites = sites or load_sites(cfg)
    paths = RunPaths(resolve_data_root(cfg), cfg.run.name)
    out = evaluate_paths(paths)
    out["plots"].mkdir(parents=True, exist_ok=True)

    metrics.metrics_table(paths.root, sites).to_csv(out["metrics"], index=False)

    study_dir = metrics.evaluate_inputs(cfg)
    if study_dir is not None:
        metrics.study_comparison(paths.root, sites, study_dir).to_csv(out["study"], index=False)

    if not cfg.evaluate.plots:
        return
    for sid in sites.ids:
        df = metrics.load_site_frame(paths.root, sid)
        if df is None:
            continue
        code = sites.code(sid)
        fig = plots.plot_site_paper(df, sid, code, cfg.evaluate.time_window or None)
        fig.savefig(out["plots"] / f"site_{sid}_paper.png", dpi=300)
        plt.close(fig)
        pairs = plots.validation_frame(df)
        if len(pairs) >= 3:
            fig = plots.plot_validation(pairs, sid, code)
            fig.savefig(out["plots"] / f"site_{sid}_validation.png")
            plt.close(fig)
        else:
            log.warning("site %s: fewer than 3 hours with observations: no validation plot", sid)
        if study_dir is not None and (study_dir / f"EnKF_LGAR_{sid}_hourly.csv").exists():
            study = pd.read_csv(
                study_dir / f"EnKF_LGAR_{sid}_hourly.csv", parse_dates=["time"], index_col="time"
            )
            ours = plots._naive_index(df)
            if plots.study_obs_window(study, ours) is None:
                log.warning(
                    "site %s: run window does not overlap study observations: no study comparison plot",
                    sid,
                )
                continue
            fig = plots.plot_study_site(
                study, sid, code, ours_enkf_df=ours, precip=ours["precip_mm"]
            )
            fig.savefig(out["plots"] / f"study_{sid}_ours_vs_study.png")
            plt.close(fig)


def run(cfg: RunConfig) -> None:
    evaluate(cfg)
