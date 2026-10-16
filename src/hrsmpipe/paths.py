"""Filesystem layout: ``<data_root>/{bin,cache,runs}`` plus packaged data."""

from __future__ import annotations

import os
import platform as _platform
from importlib import resources
from pathlib import Path

from hrsmpipe.config import RunConfig


def resolve_data_root(cfg: RunConfig) -> Path:
    root = cfg.run.data_root or os.environ.get("HRSM_DATA_DIR", "")
    if not root:
        raise ValueError(
            "Set run.data_root in the config or the HRSM_DATA_DIR environment variable"
        )
    return Path(root).expanduser()


def platform_tag() -> str:
    system = _platform.system().lower()
    machine = _platform.machine().lower()
    return f"{system}-{machine}"


def resolve_pkg_path(spec: str) -> Path:
    """``pkg:wisconet_stations.csv`` -> the packaged file; any other string is a plain path."""
    if not spec.startswith("pkg:"):
        return Path(spec).expanduser()
    name = spec[len("pkg:") :]
    path = Path(str(resources.files("hrsmpipe") / "data" / "sites" / name))
    if not path.exists():
        raise FileNotFoundError(f"packaged data file not found: {spec}")
    return path


def repo_root() -> Path:
    """Checkout root (parent of ``src/``); only meaningful for editable installs."""
    return Path(__file__).resolve().parents[2]


class CachePaths:
    def __init__(self, data_root: Path):
        self.root = Path(data_root) / "cache"

    def gee_static(self, site_id: int) -> Path:
        return self.root / "gee" / "static" / f"{site_id}.json"

    def gee_chunk(self, source: str, site_id: int, month: str) -> Path:
        return self.root / "gee" / source / str(site_id) / f"{month}.nc"

    def obs_chunk(self, site_id: int, month: str) -> Path:
        return self.root / "obs" / "wisconet" / str(site_id) / f"{month}.parquet"


class RunPaths:
    def __init__(self, data_root: Path, run_name: str):
        self.data_root = Path(data_root)
        self.root = self.data_root / "runs" / run_name

    @property
    def resolved_config(self) -> Path:
        return self.root / "config.resolved.toml"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def log(self) -> Path:
        return self.root / "log.txt"

    @property
    def sites_csv(self) -> Path:
        return self.root / "sites.csv"

    @property
    def obs(self) -> Path:
        return self.root / "obs.parquet"

    def rs(self, source: str) -> Path:
        return self.root / "rs" / f"{source}.nc"

    def stage_dir(self, stage: str) -> Path:
        # NOTE(hrsmpipe): fetch-obs writes obs.parquet at the run root (layout in the prompt), so
        # its _done.json lives in an otherwise empty obs/ dir; fetch-static uses static/.
        name = {"fetch-static": "static", "fetch-rs": "rs", "fetch-obs": "obs"}.get(stage, stage)
        return self.root / name

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
