"""Static (time-invariant) layers: landcover, POLARIS soils, terrain. Cached as json per site."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from hrsmpipe.config import RunConfig
from hrsmpipe.gee import auth, collections
from hrsmpipe.paths import CachePaths, resolve_data_root
from hrsmpipe.sites import SiteSet, load_sites

log = logging.getLogger(__name__)

ASSETS = {
    "nlcd": collections.NLCD_ID,
    "polaris": collections.POLARIS_ROOT,
    **collections.TERRAIN_ASSETS,
}
_TERRAIN_BAND = "b1"


def _scale_spec(cfg: RunConfig) -> dict:
    return {
        "sample_scale_m": cfg.gee.sample_scale_m,
        "polaris_scale": cfg.gee.polaris_scale,
        "terrain_scale": cfg.gee.terrain_scale,
    }


def _sample(img, fc, scale, band_names: list[str]) -> dict[int, dict]:
    feats = collections.get_info(img.sampleRegions(collection=fc, scale=scale))["features"]
    return {
        int(f["properties"]["id_num"]): {b: f["properties"].get(b) for b in band_names}
        for f in feats
    }


def fetch_static(cfg: RunConfig, sites: SiteSet) -> list[int]:
    """Sample the static layers for every site not yet cached; return the ids fetched.

    The json stores raw sampled values plus asset ids and scales; a cache entry is reused only
    when its assets and scale settings equal the current ones.
    """
    cache = CachePaths(resolve_data_root(cfg))
    spec = {"assets": ASSETS, **_scale_spec(cfg)}
    missing = []
    for sid in sites.ids:
        p = cache.gee_static(sid)
        try:
            ok = p.exists() and json.loads(p.read_text()).get("spec") == spec
        except ValueError:
            ok = False
        if not ok:
            missing.append(sid)
    if not missing:
        log.info("static layers: all %d sites cached", len(sites))
        return []

    auth.initialize(cfg.gee.project)
    fc = collections.feature_collection(sites.subset(missing))
    imgs = collections.static_images()
    scales: dict[str, float] = {}
    values: dict[int, dict] = {sid: {} for sid in missing}

    def scale_of(img, configured):
        """(scale handed to sampleRegions, numeric scale recorded in the cache)."""
        if configured == "native":
            native = img.projection().nominalScale()
            return native, collections.get_info(native)
        return configured, configured

    # Terrain: each asset at its own nominalScale() (or a configured number), no reproject.
    for name in collections.TERRAIN_ASSETS:
        scale, scales[name] = scale_of(imgs[name], cfg.gee.terrain_scale)
        for sid, d in _sample(imgs[name], fc, scale, [_TERRAIN_BAND]).items():
            values[sid][name] = d[_TERRAIN_BAND]
    # NLCD landcover at sample_scale_m
    scales["landcover"] = cfg.gee.sample_scale_m
    for sid, d in _sample(imgs["landcover"], fc, cfg.gee.sample_scale_m, ["landcover"]).items():
        values[sid]["landcover"] = d["landcover"]
    # POLARIS on its native grid (bd_0_5 nominal scale, 30 m)
    p_scale, scales["polaris"] = scale_of(collections._polaris("bd", "0_5"), cfg.gee.polaris_scale)
    polaris_bands = collections.polaris_bands()
    ksat_bands = [f"ksat_{d}" for d in collections.POLARIS_DEPTHS]
    for sid, d in _sample(imgs["polaris"], fc, p_scale, polaris_bands).items():
        values[sid].update(d)
    for sid, d in _sample(imgs["polaris_ksat"], fc, p_scale, ksat_bands).items():
        values[sid].update(d)

    expected = len(collections.TERRAIN_ASSETS) + 1 + len(polaris_bands) + len(ksat_bands) - 1
    for sid in missing:
        if len(values[sid]) < expected:
            log.warning("site %s: incomplete static sample (site outside a raster?)", sid)
        p = cache.gee_static(sid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"site_id": sid, "spec": spec, "scales_m": scales, "values": values[sid]}, indent=1
            )
        )
    return missing


def load_static(cfg: RunConfig, sites: SiteSet) -> pd.DataFrame:
    """Per-site table (index = site_id, in site order) with the derived columns of cells 19-21."""
    cache = CachePaths(resolve_data_root(cfg))
    rows = []
    for sid in sites.ids:
        p = cache.gee_static(sid)
        if not p.exists():
            raise FileNotFoundError(f"static layers for site {sid} not cached; run fetch-static")
        rows.append({"site_id": sid, **json.loads(p.read_text())["values"]})
    raw = pd.DataFrame(rows).set_index("site_id").astype(float)
    out = pd.DataFrame(index=raw.index)
    out["latitude"] = sites.df["latitude"].to_numpy()
    out["longitude"] = sites.df["longitude"].to_numpy()
    elevation = raw["elevation"].to_numpy().copy()
    elevation[elevation < -1000] = np.nan
    aspect = raw["aspect"].to_numpy()
    sin_asp = np.sin(aspect / 180 * np.pi)
    sin_asp[aspect == -1] = 0
    cos_asp = np.cos(aspect / 180 * np.pi)
    cos_asp[aspect == -1] = 0
    out["elevation"] = elevation
    out["slope"] = raw["slope"].to_numpy()
    out["sin_asp"] = sin_asp
    out["cos_asp"] = cos_asp
    out["twi"] = raw["twi"].to_numpy()
    out["landcover"] = raw["landcover"].to_numpy()
    for col in raw.columns:
        if col.startswith(("bd_", "sand_", "clay_", "silt_")):
            out[col] = raw[col].to_numpy()
    # ksat: POLARIS stores log10(cm/h) -> cm/h
    k = {d: np.power(10, raw[f"ksat_{d}"].to_numpy()) for d in collections.POLARIS_DEPTHS}
    for d, v in k.items():
        out[f"ksat_{d}"] = v
    w = dict(zip(collections.POLARIS_DEPTHS, collections.POLARIS_WEIGHTS, strict=True))
    # harmonic mean over the 0-100 cm profile (total depth 1.0)
    out["ksat_0_100"] = sum(w.values()) / sum(w[d] / k[d] for d in w)
    return out


def run(cfg: RunConfig) -> None:
    sites = load_sites(cfg)
    fetch_static(cfg, sites)
    static = load_static(cfg, sites)
    from hrsmpipe.paths import RunPaths

    out = RunPaths(resolve_data_root(cfg), cfg.run.name).stage_dir("fetch-static")
    out.mkdir(parents=True, exist_ok=True)
    static.to_csv(out / "static.csv")
