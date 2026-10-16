"""Run LGAR (open loop) and LGAR+EnKF per site in parallel scratch directories (notebook cell 121)."""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from hrsmpipe.config import RunConfig
from hrsmpipe.forcing import forcing_paths
from hrsmpipe.forcing.soils import valid_soil_ids
from hrsmpipe.lgar import build, merge, settings
from hrsmpipe.paths import RunPaths, resolve_data_root
from hrsmpipe.postproc.cdf import cdf_matching
from hrsmpipe.sites import SiteSet, load_sites
from hrsmpipe.timeaxis import hourly_axis

log = logging.getLogger(__name__)

ENKF_HEADER = [
    "time",
    "forcast_sd_sly",
    "forcast_sd_rly",
    "analysis_sd_sly",
    "analysis_sd_rly",
    "forcast_sm_sly",
    "forcast_sm_rly",
    "analysis_sm_sly",
    "analysis_sm_rly",
    "ml_sd_sly",
    "ml_sd_rly",
    "ml_sm_sly",
    "ml_sm_rly",
]


@dataclass
class SiteJob:
    """Everything one worker needs; plain data so it can be pickled to a subprocess."""

    site_id: int
    scratch: Path
    original_dir: Path
    enkf_dir: Path
    lgar_bin: Path
    enkf_bin: Path
    settings: dict
    prec: np.ndarray
    pet: np.ndarray
    sm_sly: np.ndarray
    sd_sly: np.ndarray
    sm_rly: np.ndarray
    sd_rly: np.ndarray


@dataclass
class SiteResult:
    site_id: int
    status: str  # ok | truncated | failed
    message: str = ""


@dataclass
class LgarReport:
    results: list[SiteResult] = field(default_factory=list)

    @property
    def failures(self) -> dict[int, dict]:
        return {
            r.site_id: {"status": r.status, "message": r.message}
            for r in self.results
            if r.status != "ok"
        }


def _write_f64(path: Path, arr: np.ndarray, n: int, what: str) -> None:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{what}: expected {n} values, got shape {arr.shape}")
    if np.isnan(arr).any():
        raise ValueError(f"{what}: contains NaN")
    arr.tofile(path)


def _run(cmd: list, cwd: Path) -> None:
    res = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"{Path(cmd[0]).name} exited with {res.returncode}: {(res.stderr or res.stdout)[-500:]}"
        )


def run_site(job: SiteJob) -> SiteResult:
    """LGAR open loop -> CDF-match the ML onto it -> LGAR+EnKF. Never raises."""
    sid, scratch = job.site_id, job.scratch
    n = int(job.settings["num_hours"])
    try:
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        # the C++ reads these relative to nothing: absolute paths, cwd = scratch
        files = {
            k: scratch / f"{k}.bin" for k in ("prec", "PET", "sm_sly", "sd_sly", "sm_rly", "sd_rly")
        }
        settings_path = scratch / "settingInfo.txt"
        _write_f64(files["prec"], job.prec, n, "precipitation")
        _write_f64(files["PET"], job.pet, n, "PET")
        settings.write_settings(settings_path, job.settings)

        _run(
            [
                job.lgar_bin,
                "--filename_prec",
                files["prec"],
                "--filename_pet",
                files["PET"],
                "--settings_path",
                settings_path,
            ],
            scratch,
        )
        original = pd.read_csv(
            scratch / f"Original_LGAR_{sid}.csv", header=None, names=["hour", "sm_sly", "sm_rly"]
        )

        # put the ML in LGAR's climatology; NaN stays where there is no ML (not assimilated there)
        sm = {
            "sly": np.array(job.sm_sly, dtype=np.float64),
            "rly": np.array(job.sm_rly, dtype=np.float64),
        }
        for layer in ("sly", "rly"):
            finite = ~np.isnan(sm[layer])
            if finite.any():
                sm[layer][finite] = cdf_matching(
                    sm[layer][finite], original[f"sm_{layer}"].to_numpy()
                )
        sm["sly"].tofile(files["sm_sly"])
        np.asarray(job.sd_sly, dtype=np.float64).tofile(files["sd_sly"])
        sm["rly"].tofile(files["sm_rly"])
        np.asarray(job.sd_rly, dtype=np.float64).tofile(files["sd_rly"])

        _run(
            [
                job.enkf_bin,
                "--filename_prec",
                files["prec"],
                "--filename_pet",
                files["PET"],
                "--filename_sm_sly",
                files["sm_sly"],
                "--filename_sm_rly",
                files["sm_rly"],
                "--filename_sd_sly",
                files["sd_sly"],
                "--filename_sd_rly",
                files["sd_rly"],
                "--settings_path",
                settings_path,
            ],
            scratch,
        )
        enkf_csv = scratch / f"EnKF_LGAR_{sid}.csv"
        n_rows = len(pd.read_csv(enkf_csv))
        job.original_dir.mkdir(parents=True, exist_ok=True)
        job.enkf_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(
            str(scratch / f"Original_LGAR_{sid}.csv"), job.original_dir / f"Original_LGAR_{sid}.csv"
        )
        shutil.move(str(enkf_csv), job.enkf_dir / f"EnKF_LGAR_{sid}.csv")
        shutil.rmtree(scratch, ignore_errors=True)
        if n_rows < n:
            return SiteResult(
                sid,
                "truncated",
                f"EnKF wrote {n_rows} of {n} rows (the binary diverged and stopped)",
            )
        return SiteResult(sid, "ok")
    except Exception as exc:  # continue on failure; reported at the end
        return SiteResult(sid, "failed", f"{type(exc).__name__}: {exc}")


def lgar_paths(paths: RunPaths) -> dict[str, Path]:
    d = paths.root / "lgar"
    return {
        "root": d,
        "original": d / "original",
        "enkf": d / "enkf",
        "failures": d / "failures.json",
    }


def run_lgar(cfg: RunConfig, sites: SiteSet | None = None) -> LgarReport:
    sites = sites or load_sites(cfg)
    paths = RunPaths(resolve_data_root(cfg), cfg.run.name)
    out = lgar_paths(paths)
    out["root"].mkdir(parents=True, exist_ok=True)
    bins = build.build_binaries(cfg)  # builds only what is missing

    axis = hourly_axis(cfg.run.start, cfg.run.end)
    n = len(axis)
    fp = forcing_paths(paths)
    with xr.open_dataset(fp["prec"]) as ds:
        prec = ds["precip_mm"].load().to_numpy()
    with xr.open_dataset(fp["pet"]) as ds:
        pet = ds["pet_mm"].load().to_numpy()
    soils = pd.read_csv(fp["soils"])
    post = {}
    for name in ("ssm_sly", "ssm_rly", "sd_sly", "sd_rly"):
        with xr.open_dataset(paths.root / "postproc" / f"{name}.nc") as ds:
            post[name] = ds[name].load().to_numpy()

    report = LgarReport()
    soil_check = valid_soil_ids(soils, cfg.lgar.max_valid_soil_types)
    jobs = []
    for j, sid in enumerate(sites.ids):
        problem = soil_check.get(sid, "no soil classes")
        if problem:
            report.results.append(SiteResult(sid, "failed", problem))
            continue
        ids = (
            soils.loc[soils["site_id"] == sid].sort_values("layer")["soil_id"].astype(int).tolist()
        )
        jobs.append(
            SiteJob(
                site_id=sid,
                scratch=out["root"] / "_work" / str(sid),
                original_dir=out["original"],
                enkf_dir=out["enkf"],
                lgar_bin=bins["original"],
                enkf_bin=bins["enkf"],
                settings=settings.site_settings(cfg, sid, n, ids),
                prec=prec[:, j],
                pet=pet[:, j],
                sm_sly=post["ssm_sly"][:, j],
                sd_sly=post["sd_sly"][:, j],
                sm_rly=post["ssm_rly"][:, j],
                sd_rly=post["sd_rly"][:, j],
            )
        )

    workers = min(cfg.lgar.workers or os.cpu_count() or 1, max(len(jobs), 1))
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for res in tqdm(pool.map(run_site, jobs), total=len(jobs), desc="LGAR sites"):
            report.results.append(res)
            if res.status != "ok":
                log.error("site %s %s: %s", res.site_id, res.status, res.message)
    shutil.rmtree(out["root"] / "_work", ignore_errors=True)

    obs = pd.read_parquet(paths.obs) if paths.obs.exists() else None
    for res in report.results:
        if res.status in ("ok", "truncated"):
            try:
                merge.merge_site(cfg, sites, res.site_id, obs=obs)
            except Exception as exc:
                res.status, res.message = "failed", f"merge: {type(exc).__name__}: {exc}"
                log.error("site %s merge failed: %s", res.site_id, exc)
    out["failures"].write_text(json.dumps(report.failures, indent=2, sort_keys=True))
    if report.failures:
        log.warning("%d site(s) with problems: %s", len(report.failures), sorted(report.failures))
    return report


def run(cfg: RunConfig) -> None:
    run_lgar(cfg)
