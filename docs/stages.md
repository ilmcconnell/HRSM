# Stages

Stages run in this order; `hrsmpipe <stage> <config.toml>` runs one of them, `hrsmpipe <config.toml>` all of them.
Everything uses **one hourly UTC axis `[start, end)`** (`pd.date_range(start, end, freq="h", inclusive="left")`);
its length is also LGAR's `num_hours`. (The notebook used N+1 rows; the original study's CSVs use N.)

| stage | reads | writes (under `<data_root>/runs/<name>/` unless noted) |
|---|---|---|
| `fetch-static` | Earth Engine | `<data_root>/cache/gee/static/<site>.json`, `static/static.csv` |
| `fetch-rs` | Earth Engine | `<data_root>/cache/gee/<era5\|s1\|s2\|hls>/<site>/<YYYY-MM>.nc`, `rs/{era5,s1,s2,hls}.nc` |
| `fetch-obs` | Wisconet API / CSV archive / table | `obs.parquet` (API: `<data_root>/cache/obs/wisconet/<site>/<YYYY-MM>.parquet`) |
| `features` | `rs/*.nc`, static layers | `features/<sensor>_<layer>.parquet` |
| `infer` | features, `trainedML/*.joblib`, `obs.parquet` | `inference/<layer>.parquet` |
| `postproc` | inference | `postproc/{ssm,sd}_{sly,rly}.nc`, `postproc/miu.csv` |
| `forcing` | `rs/era5.nc`, `obs.parquet`, static layers | `forcing/prec.nc`, `forcing/pet.nc`, `forcing/soil_classes.csv` |
| `lgar` | forcing, postproc | `lgar/original/*.csv`, `lgar/enkf/*.csv`, `lgar/site_<id>.parquet`, `lgar/failures.json`; binaries in `<data_root>/bin/<platform>/` |
| `evaluate` | `lgar/site_<id>.parquet` | `evaluate/metrics.csv`, `evaluate/study_comparison.csv`, `evaluate/plots/*.png` |

Site order: the order of the sites CSV is the column order of every `(time, site)` array. `site_id` (int) is the
key; `station_code` is only a label. Two site sets ship with the package: `wisconet_stations.csv` (79 stations) and
`study_sites.csv` (296, 1140, **2096**; the study's `site_info.csv` says 2069 but its data file is 2096).

## fetch-static

NLCD landcover (100 m), POLARIS soils at the native 30 m grid and terrain assets at each asset's own
`nominalScale()` (no reprojection). Derived on load: `elevation < -1000 → NaN`, `sin/cos(aspect)` (0 where
aspect = −1), `ksat = 10**ksat` (cm/h) and the harmonic `ksat_0_100`. The cache stores raw values with asset ids and
scales and is reused only when they match.

## fetch-rs

ERA5-Land (`wind_10m = hypot(u, v)` added server-side), HLSL30 (Fmask bits), Sentinel-2 (SCL) and Sentinel-1 (IW,
VV+VH, masked and smoothed) are sampled with `sampleRegions(scale=100)`. One request covers
`(source, month, all sites missing from the cache)` in `gee.window_hours` windows; per `(hour, site)` the mean of all
samples is kept. ERA5 spans `[start − era5_lead_hours, end)`, the satellites `[start, end)`. A cached chunk is
reused only if it covers the requested month range at the same scale.

## fetch-obs

`obs.source`:

* `wisconet_api`: through `wiscopy` with **UTC** bounds, cached per site-month. Field table in
  `hrsmpipe.obs.wisconet_api`. **The API returns soil moisture as a 0–1 fraction even though the field is named
  `pct`**, so it is not divided by 100 (a month whose maximum exceeds 1 is treated as percent).
* `wisconet_csv_dir`: the Wisconet CSV archive (`*_all_hourly_*.csv`, `gmt_timestamp`).
* `table`: any CSV/parquet with a column map (`[obs.columns]`); `{site_id}` in the path selects one file per site;
  `obs.timezone` applies to naive timestamps only.

Rootzone soil moisture is computed in the package: midpoint-rule weights `[7.5, 7.5, 20, 40, 25]` for 5/10/20/50/100 cm,
interior linear interpolation along depth, a row is valid only with the 5 cm, 100 cm and at least one middle depth.
(The legacy validated array used `[2.5, 7.5, 20, 40, 25]`.) The legacy `.npy` (Central local time) is only read by
`scripts/convert_legacy_obs_npy.py`.

## features, infer

Column tables: Sentinel-1 30, Sentinel-2 39, HLSL30 41 (see `hrsmpipe.features.assemble`). ERA5 lookbacks use weights
`1/[n..1]` over 24/72/120 h; in mode `raw` (default, the notebook) the first `n` rows keep the raw hourly value.
Rows with any NaN are dropped. Models: ML1 = Sentinel-1, ML2 = Sentinel-2, ML3 = HLSL30; columns are dropped per family
(`hrsmpipe.ml.models.DROP_COLUMNS`), then scaler → LightGBM mean and MAPIE-CQR interval.

## postproc

Per layer and per site present in the **reference** sensor (`postproc.reference_sensor`, default `sentinel2`, as the
notebook despite the function name `correct_2_ML3`) the other two sensors' `y_mean` is CDF-matched onto it.
`sd = (y_high − y_low)/3.29` from the *unmatched* bounds (≤ 0 → 0.02). Each value spreads to hour ± 1 of the same day
(concatenated in the order *hour−1, original, hour+1*; the first occurrence per sensor/site/time wins), then the three
sensors are averaged with `nanmean`.

## forcing

* Precipitation: mesonet where finite, else ERA5 (m → mm, values below 0.01 → 0), else 0; `precip_source`
  (`missing`/`mesonet`/`era5`) per hour. No interval shift by default (`forcing.interval_shift_hours = 0`).
* PET: Priestley–Taylor from ERA5 (or mesonet) radiation and temperature, values < 1e-6 → 0, gaps linearly
  interpolated per site; `pet_source` adds `interpolated`.
* Soil classes: the notebook's masks and Hydrus tables, per POLARIS depth layer. Every `soil_id` must be an int in
  1..12, otherwise the site is listed in `lgar/failures.json` and skipped. Note the notebook's assignment order: for
  `clay < 7 %` and `sand > 43 %` the later `sandy_loam` rule overrides `sand` and `loamy_sand`.

## lgar

`hrsmpipe build-lgar` (or the first `lgar` run) compiles the bundled sources: `ensemble_size`, `model_sd_sly`,
`model_sd_rly` and an optional `seed` are patched into the EnKF source (each patch must apply exactly once) and
named in the binary (`EnKF_LGAR_20_003002`, `..._seed7`). Sites run in a `ProcessPoolExecutor`, each in its own
scratch directory `lgar/_work/<site>/`: open-loop LGAR → ML CDF-matched onto it → LGAR + EnKF. A failing site does not
stop the others; failures (and `truncated` runs, where the EnKF binary stopped early) are written to `failures.json`.
The merged `lgar/site_<id>.parquet` has the columns of `hrsmpipe.lgar.merge.COLUMNS`.

## evaluate

`metrics.csv`: bias, RMSE, ubRMSE and R for the series `ml`, `lgar_original`, `enkf_analysis`, `enkf_analysis_cdf`,
`lgar_original_cdf` and their `*_daily` variants. With `evaluate.study_dir`, `study_comparison.csv` compares with the
original study's `EnKF_LGAR_<id>_hourly.csv`. Plots: `site_<id>_paper.png`, `site_<id>_validation.png`,
`study_<id>_ours_vs_study.png`.
