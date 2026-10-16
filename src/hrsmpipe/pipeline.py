"""Stage orchestration: resolved config, provenance, fingerprints, skip/force logic."""

from __future__ import annotations

import datetime as dt
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hrsmpipe import manifest
from hrsmpipe.config import RunConfig, resolved_toml
from hrsmpipe.paths import RunPaths, resolve_data_root
from hrsmpipe.provenance import provenance_table
from hrsmpipe.sites import SiteSet, load_sites

log = logging.getLogger("hrsmpipe")

STAGES = (
    "fetch-static",
    "fetch-rs",
    "fetch-obs",
    "features",
    "infer",
    "postproc",
    "forcing",
    "lgar",
    "evaluate",
)


@dataclass
class Stage:
    name: str
    run: Callable[[RunConfig], None]
    outputs: Callable[[RunConfig, RunPaths], list[Path]]
    inputs: Callable[[RunConfig, SiteSet], list[Path]] = lambda cfg, sites: []
    extra: Callable[[], dict] = lambda: {}
    fingerprint_sites: bool = True


def _obs_inputs(cfg: RunConfig, sites: SiteSet) -> list[Path]:
    """Files the obs stage reads (sizes/mtimes enter the fingerprint). The API is cache-driven."""
    src, path = cfg.obs.source, cfg.obs.path
    if src == "table":
        return [Path(path.format(site_id=s)).expanduser() for s in sites.ids]
    if src == "wisconet_csv_dir" and path:
        return sorted(Path(path).expanduser().glob("*_all_hourly_*.csv"))
    return []


def _paths(cfg: RunConfig) -> RunPaths:
    return RunPaths(resolve_data_root(cfg), cfg.run.name)


def _study_inputs(cfg: RunConfig, sites: SiteSet) -> list[Path]:
    if not cfg.evaluate.study_dir:
        return []
    return [
        Path(cfg.evaluate.study_dir).expanduser() / f"EnKF_LGAR_{s}_hourly.csv" for s in sites.ids
    ]


def _registry() -> dict[str, Stage]:
    from hrsmpipe import evaluate as evaluate_mod
    from hrsmpipe import forcing as forcing_mod
    from hrsmpipe import obs
    from hrsmpipe.features import assemble
    from hrsmpipe.gee import static, timeseries
    from hrsmpipe.lgar import run as lgar_run
    from hrsmpipe.ml import infer as ml_infer
    from hrsmpipe.ml.models import models_dir
    from hrsmpipe.postproc import combine

    return {
        "fetch-static": Stage(
            "fetch-static",
            static.run,
            lambda cfg, rp: [rp.stage_dir("fetch-static") / "static.csv"],
            extra=lambda: {"assets": static.ASSETS},
        ),
        "fetch-rs": Stage(
            "fetch-rs",
            timeseries.run,
            lambda cfg, rp: [rp.rs(s) for s in timeseries.SOURCES],
        ),
        "fetch-obs": Stage(
            "fetch-obs",
            obs.run,
            lambda cfg, rp: [rp.obs],
            inputs=_obs_inputs,
        ),
        "features": Stage(
            "features",
            assemble.run,
            lambda cfg, rp: [
                assemble.features_path(rp, s, ly)
                for s in assemble.SENSORS
                for ly in assemble.LAYERS
            ],
            inputs=lambda cfg, sites: (
                [_paths(cfg).rs(s) for s in ("era5", "s1", "s2", "hls")]
                + [_paths(cfg).stage_dir("fetch-static") / "static.csv"]
            ),
        ),
        "infer": Stage(
            "infer",
            ml_infer.run,
            lambda cfg, rp: [ml_infer.inference_path(rp, ly) for ly in assemble.LAYERS],
            inputs=lambda cfg, sites: (
                [
                    assemble.features_path(_paths(cfg), s, ly)
                    for s in assemble.SENSORS
                    for ly in assemble.LAYERS
                ]
                + sorted(models_dir(cfg).glob("*.joblib"))
                + [_paths(cfg).obs]
            ),
        ),
        "postproc": Stage(
            "postproc",
            combine.run,
            lambda cfg, rp: (
                [
                    rp.root / "postproc" / f"{n}_{c}.nc"
                    for n in ("ssm", "sd")
                    for c in ("sly", "rly")
                ]
                + [rp.root / "postproc" / "miu.csv"]
            ),
            inputs=lambda cfg, sites: [
                ml_infer.inference_path(_paths(cfg), ly) for ly in assemble.LAYERS
            ],
        ),
        "forcing": Stage(
            "forcing",
            forcing_mod.run,
            lambda cfg, rp: list(forcing_mod.forcing_paths(rp).values()),
            inputs=lambda cfg, sites: [
                _paths(cfg).rs("era5"),
                _paths(cfg).obs,
                _paths(cfg).stage_dir("fetch-static") / "static.csv",
            ],
        ),
        "lgar": Stage(
            "lgar",
            lgar_run.run,
            lambda cfg, rp: [lgar_run.lgar_paths(rp)["failures"]],
            inputs=lambda cfg, sites: (
                list(forcing_mod.forcing_paths(_paths(cfg)).values())
                + [
                    _paths(cfg).root / "postproc" / f"{n}.nc"
                    for n in ("ssm_sly", "ssm_rly", "sd_sly", "sd_rly")
                ]
            ),
        ),
        "evaluate": Stage(
            "evaluate",
            evaluate_mod.run,
            lambda cfg, rp: [evaluate_mod.evaluate_paths(rp)["metrics"]],
            inputs=lambda cfg, sites: (
                [lgar_run.lgar_paths(_paths(cfg))["failures"]]
                + [
                    lgar_run.lgar_paths(_paths(cfg))["root"] / f"site_{s}.parquet"
                    for s in sites.ids
                ]
                + _study_inputs(cfg, sites)
            ),
        ),
    }


def setup_logging(paths: RunPaths) -> None:
    """Console plus ``<run>/log.txt``; safe to call repeatedly."""
    paths.ensure()
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    have_console = any(type(h) is logging.StreamHandler for h in log.handlers)
    if not have_console:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(fmt)
        log.addHandler(h)
    for h in [h for h in log.handlers if isinstance(h, logging.FileHandler)]:
        log.removeHandler(h)
        h.close()
    fh = logging.FileHandler(paths.log)
    fh.setFormatter(fmt)
    log.addHandler(fh)


def prepare_run(cfg: RunConfig) -> tuple[RunPaths, dict]:
    """Create the run dir, then write ``config.resolved.toml`` (with ``[provenance]``) and sites."""
    paths = RunPaths(resolve_data_root(cfg), cfg.run.name)
    setup_logging(paths)
    prov = provenance_table()
    paths.resolved_config.write_text(resolved_toml(cfg, prov))
    load_sites(cfg).df.to_csv(paths.sites_csv, index=False)
    log.info("run directory: %s", paths.root)
    return paths, prov


@dataclass
class RunReport:
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def run_stages(
    cfg: RunConfig,
    from_: str | None = None,
    to: str | None = None,
    force: str | None = None,
    sites: list[int] | None = None,
) -> RunReport:
    """Run ``STAGES[from_..to]`` in order, skipping stages whose fingerprint is unchanged.

    ``force`` reruns that stage and every stage after it. A stage that actually runs also forces
    all later stages in this invocation (their inputs changed).
    """
    if sites:
        cfg = cfg.model_copy(deep=True)
        cfg.sites.include = [int(s) for s in sites]
    for name, label in ((from_, "from_"), (to, "to"), (force, "force")):
        if name is not None and name not in STAGES:
            raise ValueError(f"unknown stage {name!r} for {label}; choose from {', '.join(STAGES)}")
    forced = manifest.stages_to_run(STAGES, force)
    paths, prov = prepare_run(cfg)
    site_set = load_sites(cfg)
    registry = _registry()
    lo = STAGES.index(from_) if from_ else 0
    hi = STAGES.index(to) if to else len(STAGES) - 1
    report = RunReport()
    dirty = False
    for name in STAGES[lo : hi + 1]:
        st = registry[name]
        stage_dir = paths.stage_dir(name)
        fp = manifest.fingerprint(
            name,
            cfg,
            st.inputs(cfg, site_set),
            site_set if st.fingerprint_sites else None,
            st.extra(),
        )
        outputs_ok = all(p.exists() for p in st.outputs(cfg, paths))
        if not (dirty or name in forced) and outputs_ok and manifest.is_done(stage_dir, fp):
            log.info("[%s] up to date, skipping", name)
            report.skipped.append(name)
            continue
        log.info("[%s] running", name)
        manifest.clear_done(stage_dir)
        started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        st.run(cfg)
        # inputs may only exist after the stage ran (e.g. table files fetched by it): refingerprint
        fp = manifest.fingerprint(
            name,
            cfg,
            st.inputs(cfg, site_set),
            site_set if st.fingerprint_sites else None,
            st.extra(),
        )
        manifest.mark_done(
            stage_dir, fp, started, dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), prov
        )
        report.ran.append(name)
        dirty = True
    return report
