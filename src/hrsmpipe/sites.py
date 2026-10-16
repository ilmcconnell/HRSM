"""Site sets. The row order of the sites CSV is the column order of every (time, site) array."""

from __future__ import annotations

import pandas as pd

from hrsmpipe.config import RunConfig
from hrsmpipe.paths import resolve_pkg_path

COLUMNS = ["site_id", "station_code", "latitude", "longitude"]


class SiteSet:
    """Ordered sites; ``site_id`` (int) is the canonical key, ``station_code`` is a label only."""

    def __init__(self, df: pd.DataFrame):
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"sites table is missing columns {missing}")
        df = df[COLUMNS].copy()
        df["site_id"] = df["site_id"].astype(int)
        if not df["site_id"].is_unique:
            dups = df.loc[df["site_id"].duplicated(), "site_id"].tolist()
            raise ValueError(f"duplicate site_id values: {dups}")
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self):
        return iter(self.df.itertuples(index=False))

    @property
    def ids(self) -> list[int]:
        return self.df["site_id"].tolist()

    def index_of(self, site_id: int) -> int:
        return self.ids.index(int(site_id))

    def subset(self, site_ids) -> SiteSet:
        """Keep the given sites, preserving this set's order."""
        keep = {int(s) for s in site_ids}
        unknown = keep - set(self.ids)
        if unknown:
            raise ValueError(f"unknown site_id(s): {sorted(unknown)}")
        return SiteSet(self.df[self.df["site_id"].isin(keep)])

    def code(self, site_id: int) -> str:
        return str(self.df.loc[self.df["site_id"] == int(site_id), "station_code"].iloc[0])


def load_sites(cfg: RunConfig, only: list[int] | None = None) -> SiteSet:
    sites = SiteSet(pd.read_csv(resolve_pkg_path(cfg.sites.csv)))
    if cfg.sites.include:
        sites = sites.subset(cfg.sites.include)
    if only:
        sites = sites.subset(only)
    return sites
