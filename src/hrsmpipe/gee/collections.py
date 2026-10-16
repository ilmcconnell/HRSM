"""Earth Engine collections, copied from the notebook (cells 17, 27-30)."""

from __future__ import annotations

import ee
import pandas as pd

ERA5_ID = "ECMWF/ERA5_LAND/HOURLY"
HLS_ID = "NASA/HLS/HLSL30/v002"
S2_ID = "COPERNICUS/S2_SR_HARMONIZED"
S1_ID = "COPERNICUS/S1_GRD"

# Canonical variable order of rs/era5.nc: short, mid, long groups.
ERA5_SHORT = ["temperature_2m", "dewpoint_temperature_2m", "wind_10m", "surface_pressure"]
ERA5_MID = [
    "surface_solar_radiation_downwards_hourly",
    "surface_thermal_radiation_downwards_hourly",
]
ERA5_LONG = ["total_evaporation_hourly", "total_precipitation_hourly"]
ERA5_VARS = ERA5_SHORT + ERA5_MID + ERA5_LONG

HLS_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH", "angle"]

SOURCE_BANDS = {
    "era5": ERA5_VARS,
    "hls": HLS_BANDS,
    "s2": S2_BANDS,
    "s1": S1_BANDS,
}

POLARIS_ROOT = "projects/sat-io/open-datasets/polaris"
POLARIS_DEPTHS = ["0_5", "5_15", "15_30", "30_60", "60_100"]
POLARIS_WEIGHTS = [0.05, 0.1, 0.15, 0.3, 0.4]
NLCD_ID = "USGS/NLCD_RELEASES/2019_REL/NLCD"
TERRAIN_ASSETS = {
    "elevation": "projects/ee-scai62/assets/COUNS_dem100m",  # (sic) the asset name has this typo
    "twi": "projects/ee-scai62/assets/CONUS_dem100m_TWII",
    "aspect": "projects/ee-scai62/assets/CONUS_dem100m_aspect",
    "slope": "projects/ee-scai62/assets/CONUS_dem100m_slopeDegree",
}


def _d(t) -> str:
    """Naive-UTC ISO string for ``filterDate`` (end is exclusive)."""
    t = pd.Timestamp(t)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.strftime("%Y-%m-%dT%H:%M:%S")


# ---- ERA5-Land -------------------------------------------------------------------------------


def era5_short(roi, start, end) -> ee.ImageCollection:
    # wind_10m = hypot(u10, v10) is added server-side before select, as in the notebook
    return (
        ee.ImageCollection(ERA5_ID)
        .filterBounds(roi)
        .filterDate(_d(start), _d(end))
        .map(
            lambda img: img.addBands(
                img.select("u_component_of_wind_10m")
                .hypot(img.select("v_component_of_wind_10m"))
                .rename("wind_10m")
            )
        )
        .select(ERA5_SHORT)
    )


def era5_mid(roi, start, end) -> ee.ImageCollection:
    return (
        ee.ImageCollection(ERA5_ID)
        .filterBounds(roi)
        .filterDate(_d(start), _d(end))
        .select(ERA5_MID)
    )


def era5_long(roi, start, end) -> ee.ImageCollection:
    return (
        ee.ImageCollection(ERA5_ID)
        .filterBounds(roi)
        .filterDate(_d(start), _d(end))
        .select(ERA5_LONG)
    )


# ---- HLSL30 ----------------------------------------------------------------------------------


def _bitwise_extract(img, from_bit, to_bit):
    mask_size = ee.Number(1).add(to_bit).subtract(from_bit)
    mask = ee.Number(1).leftShift(mask_size).subtract(1)
    return img.rightShift(from_bit).bitwiseAnd(mask)


def _mask_hls(image):
    qc = image.select("Fmask")
    cloud = _bitwise_extract(qc, 1, 1).eq(0)
    shadow = _bitwise_extract(qc, 3, 3).eq(0)
    snow = _bitwise_extract(qc, 4, 4).eq(0)
    water = _bitwise_extract(qc, 5, 5).eq(0)
    aerosol = _bitwise_extract(qc, 6, 7).lte(2)
    mask = cloud.And(shadow).And(snow).And(water).And(aerosol)
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])


def hlsl30(roi, start, end) -> ee.ImageCollection:
    # no scale factor for the HLSL30 product
    return (
        ee.ImageCollection(HLS_ID)
        .filterBounds(roi)
        .filterDate(_d(start), _d(end))
        .map(_mask_hls)
        .select(HLS_BANDS)
    )


# ---- Sentinel-2 ------------------------------------------------------------------------------


def _mask_s2(img):
    scl = img.select("SCL")
    mask = scl.neq(8).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(mask).copyProperties(img, ["system:time_start"])


def sentinel2(roi, start, end) -> ee.ImageCollection:
    return (
        ee.ImageCollection(S2_ID)
        .filterDate(_d(start), _d(end))
        .filterBounds(roi)
        .map(_mask_s2)
        .select(S2_BANDS)
    )


# ---- Sentinel-1 ------------------------------------------------------------------------------


def _preprocess_vv(image):
    masked = image.updateMask(image.gt(-20).And(image.lt(-5)))
    return masked.convolve(ee.Kernel.gaussian(3))


def _preprocess_vh(image):
    masked = image.updateMask(image.gt(-30).And(image.lt(-10)))
    return masked.convolve(ee.Kernel.gaussian(3))


def _merge_bands(image):
    vv = _preprocess_vv(image.select("VV"))
    vh = _preprocess_vh(image.select("VH"))
    angle = image.select("angle")
    merged = vv.addBands(vh).addBands(angle.rename("angle"))
    return merged.copyProperties(image, ["system:time_start"])


def _to_float(image):
    return image.select(image.bandNames()).float().copyProperties(image, ["system:time_start"])


def sentinel1(roi, start, end) -> ee.ImageCollection:
    # No reproject before sampling.
    return (
        ee.ImageCollection(S1_ID)
        .filterDate(_d(start), _d(end))
        .filterBounds(roi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .sort("SLC_Processing_start")
        .map(_merge_bands)
        .map(_to_float)
    )


COLLECTIONS = {
    "era5": (era5_short, era5_mid, era5_long),
    "hls": (hlsl30,),
    "s2": (sentinel2,),
    "s1": (sentinel1,),
}


# ---- static images ---------------------------------------------------------------------------


def _polaris(prop: str, depth: str) -> ee.Image:
    name = f"{prop}_{depth}"
    return (
        ee.ImageCollection(f"{POLARIS_ROOT}/{prop}_mean")
        .filterMetadata("system:index", "equals", name)
        .first()
        .rename(name)
    )


def _weighted(prop: str) -> ee.Image:
    """0-100 cm depth-weighted mean of the five POLARIS layers."""
    layers = [_polaris(prop, d) for d in POLARIS_DEPTHS]
    out = layers[0].multiply(POLARIS_WEIGHTS[0])
    for layer, w in zip(layers[1:], POLARIS_WEIGHTS[1:], strict=True):
        out = out.add(layer.multiply(w))
    return out.rename(f"{prop}_0_100")


def static_images() -> dict[str, ee.Image]:
    """Constant images: ``landcover``, ``polaris``, ``polaris_ksat`` and the four terrain images.

    NOTE(hrsmpipe): the notebook added the ``ksat_0_5`` band to ``polaris`` twice (GEE renames
    the second to ``ksat_0_5_1``; never read). Not reproduced.
    """
    landcover = (
        ee.ImageCollection(NLCD_ID)
        .filterDate("2016-01-01", "2018-01-01")
        .select("landcover")
        .first()
    )
    polaris = (
        _polaris("bd", "0_5")
        .addBands(_weighted("bd"))
        .addBands(_polaris("ksat", "0_5"))
        .addBands(_polaris("clay", "0_5"))
        .addBands(_weighted("clay"))
    )
    for d in POLARIS_DEPTHS[1:]:
        polaris = polaris.addBands(_polaris("clay", d))
    polaris = polaris.addBands(_polaris("sand", "0_5")).addBands(_weighted("sand"))
    for d in POLARIS_DEPTHS[1:]:
        polaris = polaris.addBands(_polaris("sand", d))
    polaris = polaris.addBands(_polaris("silt", "0_5")).addBands(_weighted("silt"))
    polaris_ksat = _polaris("ksat", POLARIS_DEPTHS[0])
    for d in POLARIS_DEPTHS[1:]:
        polaris_ksat = polaris_ksat.addBands(_polaris("ksat", d))
    images = {
        "landcover": landcover,
        "polaris": polaris,
        "polaris_ksat": polaris_ksat,
    }
    images.update({k: ee.Image(v) for k, v in TERRAIN_ASSETS.items()})
    return images


def polaris_bands() -> list[str]:
    return (
        ["bd_0_5", "bd_0_100", "ksat_0_5"]
        + [f"clay_{d}" for d in ["0_5", *POLARIS_DEPTHS[1:], "0_100"]]
        + [f"sand_{d}" for d in ["0_5", *POLARIS_DEPTHS[1:], "0_100"]]
        + ["silt_0_5", "silt_0_100"]
    )


# ---- shared helpers --------------------------------------------------------------------------


def feature_collection(sites) -> ee.FeatureCollection:
    """Points carrying ``id_num`` (site_id) and ``constant = 1``, as in notebook cell 14."""
    feats = [
        ee.Feature(
            ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
            {"id_num": int(r.site_id), "constant": 1},
        )
        for r in sites
    ]
    return ee.FeatureCollection(feats)


def get_info(obj, attempts: int = 4, pause: float = 5.0):
    """``obj.getInfo()`` with exponential backoff (rate limits, transient EE errors)."""
    import time

    for i in range(attempts):
        try:
            return obj.getInfo()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(pause * 2**i)
