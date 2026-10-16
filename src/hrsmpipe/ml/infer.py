"""Run the models over the feature tables; write ``inference/<layer>.parquet``."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from hrsmpipe.config import RunConfig
from hrsmpipe.features.assemble import LAYERS, SENSORS, feature_columns, features_path
from hrsmpipe.ml.models import DROP_COLUMNS, FAMILY, ModelBundle, models_dir
from hrsmpipe.paths import RunPaths, resolve_data_root

log = logging.getLogger(__name__)

OBS_COLUMN = {"surface": "sm_surface", "rootzone": "sm_rootzone"}
INFERENCE_COLUMNS = ["site_id", "time", "sensor", "y_mean", "y_median", "y_low", "y_high", "obs"]


def inference_path(paths: RunPaths, layer: str):
    return paths.root / "inference" / f"{layer}.parquet"


def predict(model, X_feat: np.ndarray) -> dict[str, np.ndarray]:
    """scaler -> lgb mean and MAPIE-CQR interval for an already column-dropped feature matrix."""
    scaled = model.scaler.transform(X_feat)
    y_mean = model.lgb_mean.predict(scaled)
    median, pis = model.mapie_cqr.predict_interval(scaled)
    return {
        "y_mean": y_mean,
        "y_median": np.asarray(median),
        "y_low": pis[:, 0, 0],
        "y_high": pis[:, 1, 0],
    }


def infer_table(model, sensor: str, table: pd.DataFrame) -> pd.DataFrame:
    """Predictions for one (sensor, layer) feature table (columns f00..fNN, site_id, time)."""
    X = table[feature_columns(sensor)].to_numpy()
    X_feat = np.delete(X, DROP_COLUMNS[FAMILY[sensor]], axis=1)
    out = pd.DataFrame({"site_id": table["site_id"].to_numpy(), "time": table["time"].to_numpy()})
    out["sensor"] = sensor
    for k, v in predict(model, X_feat).items():
        out[k] = v
    return out


def join_obs(df: pd.DataFrame, obs: pd.DataFrame | None, column: str) -> pd.DataFrame:
    """Add ``obs`` by exact UTC hour and site (NaN where there is no observation)."""
    if obs is None or obs.empty or column not in obs:
        df["obs"] = np.nan
        return df
    o = obs[["site_id", "time", column]].rename(columns={column: "obs"})
    o = o.drop_duplicates(["site_id", "time"])
    o = o.assign(time=pd.to_datetime(o["time"], utc=True).dt.as_unit("ns"))
    df = df.assign(time=pd.to_datetime(df["time"], utc=True).dt.as_unit("ns"))
    return df.merge(o, on=["site_id", "time"], how="left")


def infer(cfg: RunConfig) -> dict[str, int]:
    paths = RunPaths(resolve_data_root(cfg), cfg.run.name)
    bundle = ModelBundle.load(models_dir(cfg))
    obs = pd.read_parquet(paths.obs) if paths.obs.exists() else None
    counts = {}
    for layer in LAYERS:
        parts = []
        for sensor in SENSORS:
            table = pd.read_parquet(features_path(paths, sensor, layer))
            if table.empty:
                continue
            parts.append(infer_table(bundle.get(sensor, layer), sensor, table))
        df = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=INFERENCE_COLUMNS)
        )
        if len(df):
            df = join_obs(df, obs, OBS_COLUMN[layer])
        p = inference_path(paths, layer)
        p.parent.mkdir(parents=True, exist_ok=True)
        df[INFERENCE_COLUMNS].to_parquet(p, index=False)
        counts[layer] = len(df)
        log.info("inference %s: %d rows", layer, len(df))
    return counts


def run(cfg: RunConfig) -> None:
    infer(cfg)
