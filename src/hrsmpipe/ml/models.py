"""Load the 18 pre-trained joblib files: ML{1,2,3}_{sly,rly}_{scaler,lgb_mean,mapie_cqr}."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from hrsmpipe.config import RunConfig
from hrsmpipe.paths import repo_root

# Model family per sensor: ML1 = Sentinel-1, ML2 = Sentinel-2, ML3 = HLSL30.
FAMILY = {"sentinel1": "ML1", "sentinel2": "ML2", "hls": "ML3"}
LAYER_CODE = {"surface": "sly", "rootzone": "rly"}

# Columns removed from the feature table before scaling (0-based; notebook cell 102).
DROP_COLUMNS = {
    "ML1": [0, 1, 3, 15],
    "ML2": [0, 1, 3, 15, 24, 25, 26],  # + B2, B3, B4
    "ML3": [0, 1, 3, 15, 24, 25, 26, 32, 33, 34, 35],  # + B2, B3, B4, SZA, SAA, VZA, VAA
}


@dataclass
class LayerModel:
    scaler: object
    lgb_mean: object
    mapie_cqr: object


class ModelBundle:
    def __init__(self, models: dict[tuple[str, str], LayerModel], directory: Path):
        self._models = models
        self.directory = directory

    @classmethod
    def load(cls, models_dir: str | Path) -> ModelBundle:
        d = Path(models_dir)
        models = {}
        for family in ("ML1", "ML2", "ML3"):
            for layer in ("sly", "rly"):
                parts = {
                    kind: joblib.load(d / f"{family}_{layer}_{kind}.joblib")
                    for kind in ("scaler", "lgb_mean", "mapie_cqr")
                }
                models[(family, layer)] = LayerModel(**parts)
        return cls(models, d)

    def get(self, sensor: str, layer: str) -> LayerModel:
        return self._models[(FAMILY[sensor], LAYER_CODE[layer])]


def models_dir(cfg: RunConfig) -> Path:
    return Path(cfg.models.dir).expanduser() if cfg.models.dir else repo_root() / "trainedML"
