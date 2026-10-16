"""Compile the bundled LGAR and LGAR+EnKF C++ sources (``LGAR/build_macos.sh`` generalised).

``ensemble_size``, ``model_sd_sly``, ``model_sd_rly`` and the optional ``seed`` are build
parameters: they are patched into ``bmi_main_lgar.cxx`` and named in the binary's file name.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path

from hrsmpipe.config import RunConfig
from hrsmpipe.paths import platform_tag, resolve_data_root

log = logging.getLogger(__name__)

SOURCES = [
    "bmi_main_lgar.cxx",
    "src/aet.cxx",
    "src/bmi_lgar.cxx",
    "src/lgar.cxx",
    "src/linked_list.cxx",
    "src/mem_funcs.cxx",
    "src/soil_funcs.cxx",
    "src/util_funcs.cxx",
    "giuh/giuh.cxx",
]
INCLUDES = ["-Iinclude", "-Ibmi", "-Igiuh"]
DEFAULT_FLAGS = ["-std=c++17", "-O2"]


class BuildError(RuntimeError):
    pass


def bundled_source(kind: str) -> Path:
    """``original`` or ``enkf`` source tree shipped inside the package."""
    return Path(str(resources.files("hrsmpipe") / "data" / "lgar_src" / kind))


def default_compiler() -> str:
    return os.environ.get("CXX") or ("clang++" if platform.system() == "Darwin" else "g++")


def sd_tag(sd_sly: float, sd_rly: float) -> str:
    """``0.03, 0.02 -> '003002'``."""
    return f"{round(sd_sly * 100):03d}{round(sd_rly * 100):03d}"


def enkf_binary_name(cfg: RunConfig) -> str:
    """``EnKF_LGAR_<N>_<sdsly><sdrly>`` plus ``_seed<k>`` when seeded (20/0.03/0.02 -> EnKF_LGAR_20_003002)."""
    L = cfg.lgar
    name = f"EnKF_LGAR_{L.ensemble_size}_{sd_tag(L.model_sd_sly, L.model_sd_rly)}"
    return f"{name}_seed{L.seed}" if L.seed else name


def bin_dir(cfg: RunConfig) -> Path:
    if cfg.lgar.bin_dir:
        return Path(cfg.lgar.bin_dir).expanduser()
    return resolve_data_root(cfg) / "bin" / platform_tag()


def binary_paths(cfg: RunConfig) -> dict[str, Path]:
    d = bin_dir(cfg)
    return {"original": d / "LGAR", "enkf": d / enkf_binary_name(cfg)}


def _patch_once(text: str, pattern: str, repl: str, what: str) -> tuple[str, str]:
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise BuildError(
            f"patch '{what}' matched {len(matches)} time(s) in bmi_main_lgar.cxx, expected exactly 1"
        )
    m = matches[0]
    return text[: m.start()] + m.expand(repl) + text[
        m.end() :
    ], f"{m.group(0)}  ->  {m.expand(repl)}"


def patch_enkf_source(
    text: str, ensemble_size: int, sd_sly: float, sd_rly: float, seed: int
) -> tuple[str, list[str]]:
    """Apply the build parameters to the EnKF ``bmi_main_lgar.cxx``; each patch must hit exactly once."""
    patches = []
    text, p = _patch_once(
        text, r"int N_ensemble = \d+;", f"int N_ensemble = {int(ensemble_size)};", "ensemble size"
    )
    patches.append(p)
    for layer, sd in (("sly", sd_sly), ("rly", sd_rly)):
        text, p = _patch_once(
            text,
            rf"(std::normal_distribution<double>\s+model_error_{layer}\()0\.0,\s*[0-9.eE+-]+\)",
            rf"\g<1>0.0, {float(sd)!r})",
            f"model error sd {layer}",
        )
        patches.append(p)
    if seed:
        text, p = _patch_once(
            text,
            r"default_random_engine generator\(rd\(\)\);",
            f"default_random_engine generator({int(seed)});",
            "random seed",
        )
        patches.append(p)
    return text, patches


def _hash_tree(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _compile(src_dir: Path, out: Path, compiler: str, flags: list[str]) -> None:
    cmd = [compiler, *flags, *SOURCES, *INCLUDES, "-o", str(out)]
    log.info("building %s", out.name)
    res = subprocess.run(cmd, cwd=src_dir, capture_output=True, text=True)
    if res.returncode != 0:
        raise BuildError(f"{' '.join(cmd)} failed:\n{res.stderr[-4000:]}")
    out.chmod(out.stat().st_mode | 0o100)  # u+x


def build_binaries(cfg: RunConfig, force: bool = False) -> dict[str, Path]:
    """Compile ``LGAR`` and the EnKF binary into ``bin_dir``; existing binaries are kept unless ``force``."""
    paths = binary_paths(cfg)
    if not force and all(p.exists() for p in paths.values()):
        return paths
    out_dir = bin_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    compiler, flags = default_compiler(), list(DEFAULT_FLAGS)
    if shutil.which(compiler) is None:
        raise BuildError(f"C++ compiler '{compiler}' not found (set $CXX)")
    work = out_dir / "_build"
    shutil.rmtree(work, ignore_errors=True)
    L = cfg.lgar
    record: dict = {
        "compiler": compiler,
        "flags": flags,
        "platform": platform_tag(),
        "binaries": {},
    }

    if force or not paths["original"].exists():
        src = work / "original"
        shutil.copytree(bundled_source("original"), src)
        record["source_hash_original"] = _hash_tree(bundled_source("original"))
        _compile(src, paths["original"], compiler, flags)
        record["binaries"]["original"] = paths["original"].name
    if force or not paths["enkf"].exists():
        src = work / "enkf"
        shutil.copytree(bundled_source("enkf"), src)
        main = src / "bmi_main_lgar.cxx"
        text, patches = patch_enkf_source(
            main.read_text(), L.ensemble_size, L.model_sd_sly, L.model_sd_rly, L.seed
        )
        main.write_text(text)
        record["source_hash_enkf"] = _hash_tree(bundled_source("enkf"))
        record["patches"] = patches
        _compile(src, paths["enkf"], compiler, flags)
        record["binaries"]["enkf"] = paths["enkf"].name
    shutil.rmtree(work, ignore_errors=True)

    info = out_dir / "build.json"
    previous = json.loads(info.read_text()) if info.exists() else {}
    builds = previous.get("builds", {})
    builds[paths["enkf"].name] = record
    info.write_text(json.dumps({"builds": builds}, indent=2))
    return paths
