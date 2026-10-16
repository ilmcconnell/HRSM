"""Per-stage ``_done.json`` fingerprints: skip unchanged stages, ``--force`` semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from hrsmpipe.config import RunConfig, config_for_fingerprint
from hrsmpipe.sites import SiteSet

# Config sections (dotted names) that influence each stage. Provenance and run.name never do.
STAGE_SECTIONS: dict[str, list[str]] = {
    "fetch-static": ["gee.project", "gee.sample_scale_m", "gee.polaris_scale", "gee.terrain_scale"],
    "fetch-rs": [
        "run.start",
        "run.end",
        "gee.sample_scale_m",
        "gee.window_hours",
        "gee.era5_lead_hours",
    ],
    "fetch-obs": ["run.start", "run.end", "obs"],
    "features": ["run.start", "run.end", "features", "gee.era5_lead_hours"],
    "infer": ["run.start", "run.end", "models"],
    "postproc": ["run.start", "run.end", "postproc"],
    "forcing": ["run.start", "run.end", "gee.era5_lead_hours", "forcing"],
    "lgar": ["run.start", "run.end", "lgar"],
    "evaluate": ["run.start", "run.end", "evaluate"],
}


def _file_state(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def fingerprint(
    stage: str,
    cfg: RunConfig,
    input_paths: list[Path] | None = None,
    sites: SiteSet | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    payload = {
        "stage": stage,
        "config": config_for_fingerprint(cfg, STAGE_SECTIONS.get(stage, [])),
        "inputs": [_file_state(p) for p in (input_paths or [])],
        "sites": sites.df.to_dict(orient="list") if sites is not None else None,
        "extra": extra,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _done_path(stage_dir: Path) -> Path:
    return Path(stage_dir) / "_done.json"


def is_done(stage_dir: Path, fp: str) -> bool:
    p = _done_path(stage_dir)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("fingerprint") == fp
    except (OSError, ValueError):
        return False


def mark_done(
    stage_dir: Path, fp: str, started: str, finished: str, provenance: dict | None = None
) -> None:
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    doc = {"fingerprint": fp, "started": started, "finished": finished, "provenance": provenance}
    _done_path(stage_dir).write_text(json.dumps(doc, indent=2, default=str))


def clear_done(stage_dir: Path) -> None:
    _done_path(stage_dir).unlink(missing_ok=True)


def stages_to_run(stages: tuple[str, ...], force: str | None) -> set[str]:
    """``--force STAGE`` reruns that stage and every stage after it."""
    if force is None:
        return set()
    if force not in stages:
        raise ValueError(f"unknown stage {force!r}; choose from {', '.join(stages)}")
    return set(stages[stages.index(force) :])
