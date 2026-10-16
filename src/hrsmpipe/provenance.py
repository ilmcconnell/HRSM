"""Provenance table written to ``config.resolved.toml`` and every stage ``_done.json``."""

from __future__ import annotations

import datetime as dt
import logging
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path

import hrsmpipe

log = logging.getLogger(__name__)

_PACKAGES = {
    "wiscopy_version": "wiscopy",
    "numpy_version": "numpy",
    "pandas_version": "pandas",
    "xarray_version": "xarray",
    "scikit_learn_version": "scikit-learn",
    "lightgbm_version": "lightgbm",
    "mapie_version": "mapie",
    "earthengine_api_version": "earthengine-api",
}


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def git_info(cwd: Path | None = None) -> dict:
    """``git_*`` provenance fields; all None outside a git checkout."""
    cwd = Path(cwd) if cwd else Path(__file__).resolve().parent
    keys = (
        "git_describe",
        "git_tag",
        "git_commit",
        "git_commits_since_tag",
        "git_dirty",
        "git_branch",
    )
    if _git(["rev-parse", "--is-inside-work-tree"], cwd) != "true":
        log.warning("not inside a git checkout: git_* provenance fields are null")
        return dict.fromkeys(keys)
    info: dict = {
        "git_describe": _git(["describe", "--tags", "--long", "--dirty", "--always"], cwd),
        "git_commit": _git(["rev-parse", "HEAD"], cwd),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "git_tag": None,
        "git_commits_since_tag": None,
    }
    porcelain = _git(["status", "--porcelain"], cwd)
    info["git_dirty"] = None if porcelain is None else bool(porcelain)
    tag = _git(["describe", "--tags", "--abbrev=0"], cwd)  # fails when no tag exists
    if tag:
        info["git_tag"] = tag
        n = _git(["rev-list", "--count", f"{tag}..HEAD"], cwd)
        info["git_commits_since_tag"] = (
            int(n) if n is not None and re.fullmatch(r"\d+", n) else None
        )
    return info


def provenance_table(cwd: Path | None = None) -> dict:
    table = git_info(cwd)
    table["hrsmpipe_version"] = hrsmpipe.__version__
    for key, dist in _PACKAGES.items():
        table[key] = _version(dist)
    table["python"] = platform.python_version()
    table["platform"] = platform.platform()
    table["created_utc"] = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return table
