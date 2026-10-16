# Execution prompt: build the `hrsmpipe` package

You are implementing a Python package called **`hrsmpipe`** inside the repository at
`/Users/imcconnell/projects/HRSM` (branch `ilm-pipeline`). Work through the milestones in order. Every
design decision below was made by the repository owner in an interview; **do not revisit, simplify, or
"improve" them**. Where this prompt says "verbatim", copy the notebook code rather than re-deriving it.
When something is genuinely unspecified, choose the option that reproduces the notebook's behaviour and
leave a `# NOTE(hrsmpipe):` comment explaining the choice.

## 0. Orientation (read before writing code)

1. The source of truth for behaviour is the notebook
   `HRSM_MAPPING_ML_pipeline_multiplePoints_lgar_inputs.ipynb` (140 cells). Dump its cells to a text file
   and read it in slices; cell numbers below refer to this dump:
   ```bash
   python3 -c "import json; nb=json.load(open('HRSM_MAPPING_ML_pipeline_multiplePoints_lgar_inputs.ipynb'));
   [print(f'\n===== CELL {i} [{c[\"cell_type\"]}] =====\n'+''.join(c['source'])) for i,c in enumerate(nb['cells'])]" > /tmp/nb_cells.txt
   ```
2. Other inputs you will need:
   - `README.md`, `pyproject.toml` (pixi workspace, project currently named `hrsm`), `pixi.lock`.
   - `trainedML/*.joblib` (18 files: `ML{1,2,3}_{sly,rly}_{scaler,lgb_mean,mapie_cqr}.joblib`).
   - `LGAR/LGARTOKF_original_signalSite.zip`, `LGAR/LGARTOKF_EnKF_signalSite.zip` (C++ sources),
     `LGAR/build_macos.sh` (how they are built today), `LGAR/build/*` (already unzipped copies).
   - `HRSM_MAPPING_EnKF_pipeline.py` (original LGAR/EnKF driver script).
   - Data under `$HRSM_DATA_DIR` = `/Users/imcconnell/projects/HRSM/data/HRSM`:
     `shuohao_study_sites/` (saved notebook outputs for study sites 296, 1140, 2096: `*_arr_irriClimate.npy`,
     `ml_ready_datasets/`, `raw_model_inference_arrs/`, `post_processed_model_output/`, `LGAR/`,
     `lgar_output/{original,EnKF}/`), `shuohao_ground_truth_three_sites/` (`site_info.csv`,
     `EnKF_LGAR_{296,1140,2096}_hourly.csv` from the original study), `wisconet_soil_moisture_data/*.npy`
     (legacy obs array, **Central local time**, do not use as a package input).
   - `~/projects/wiscopy/src/wiscopy` (the owner's Wisconet API client; a dependency),
     `~/projects/wiscopy/src/wiscopy/static/cached_fields.json` (field names).
   - `~/projects/wiscosoil/src/wiscosoil/ingest.py` (Wisconet CSV-archive ingest + rootzone function to
     port by copy, with a docstring credit; do **not** import wiscosoil).
3. Environment: pixi (`pixi run python`, `pixi run test`). Python 3.13. Do not pip-install into the
   system Python.

## 1. Non-negotiable decisions

| Topic | Decision |
|---|---|
| Name / location | `hrsmpipe`, src layout `src/hrsmpipe/`; `[project] name = "hrsmpipe"`; console script `hrsmpipe = "hrsmpipe.cli:app"`; pixi task `hrsmpipe = "hrsmpipe"` |
| Interface | typer CLI **and** importable Python API (`hrsmpipe.pipeline.run_stages(config, ...)`, per-stage `run(config)` functions) |
| Entry points | `hrsmpipe <config.toml>` (bare positional path) runs the whole pipeline; `pixi run hrsmpipe <config.toml>` must work; subcommands: `init`, `config-schema`, `build-lgar`, `fetch-static`, `fetch-rs`, `fetch-obs`, `features`, `infer`, `postproc`, `forcing`, `lgar`, `evaluate`, `run` |
| Config | one TOML per run → pydantic v2 `RunConfig`; CLI flags override single fields; every key has a default equal to the notebook's value |
| Provenance | **one TOML completely defines a run**: before any stage, write `<run>/config.resolved.toml` with every key expanded (defaults + file + CLI overrides) plus a generated `[provenance]` table (see §5). It must be re-runnable verbatim |
| Platforms | macOS arm64, Linux x86-64, Google Colab |
| Sites | CSV (`site_id,station_code,latitude,longitude`); ship `wisconet_stations.csv` (79 rows, from cells 10 and 12) and `study_sites.csv` (296, 1140, **2096** — `site_info.csv` says 2069, the data file is 2096) as package data; config references them as `pkg:wisconet_stations.csv` |
| Observations | sources: `wisconet_api` (via `wiscopy`, cached per site-month in UTC), `wisconet_csv_dir` (archive with `gmt_timestamp`), `table` (generic CSV/parquet, `{site_id}` substitution allowed in the path); `obs.timezone` applies to naive table timestamps only, default `UTC`; rootzone computed in package; exact-hour (floor) matching |
| GEE | `gee.project` in TOML; init: service account if `GOOGLE_APPLICATION_CREDENTIALS`/`EE_SERVICE_ACCOUNT` set, else stored interactive credentials, else `ee.Authenticate()`; storage netCDF via xarray; cache per `(source, site, YYYY-MM)`; requests per `(source, month, all missing sites)` in 15 h windows |
| Fidelity | bit-for-bit with the notebook by default; verified by a data-marked parity test; placeholders and quirks are config values (§4) |
| Models | `models.dir` default `<repo>/trainedML`; `scikit-learn==1.7.2`, `mapie==1.1.0` |
| ML output | parquet long table per layer |
| Post-processing | `postproc.reference_sensor` default `"sentinel2"` (notebook matches ML1 and ML3 onto ML2 despite the function name `correct_2_ML3`) |
| Forcing | precip: mesonet with ERA5 filling gaps and pre-record hours, per-hour provenance; **no interval shift** (`forcing.interval_shift_hours = 0`, applied to both sources together); PET: `forcing.pet.source = "era5" | "mesonet"` with ERA5 fallback and provenance; all LGAR settings in TOML; per-site overrides |
| Time axis | one hourly UTC axis `[start, end)` everywhere (`pd.date_range(start, end, freq="h", inclusive="left")`); LGAR `num_hours` = its length. (The notebook used N+1 rows; the original study CSVs use N.) |
| Rootzone | midpoint-trapezoid weights for depths 5/10/20/50/100 cm = `[7.5, 7.5, 20, 40, 25]`; TOML comment records the legacy validated-array weights `[2.5, 7.5, 20, 40, 25]` |
| LGAR binaries | `hrsmpipe build-lgar` compiles the bundled sources on every platform; `ensemble_size`, `model_sd_sly`, `model_sd_rly`, optional `seed` are build parameters; binaries named by them |
| LGAR execution | per-site scratch dir, `ProcessPoolExecutor`, continue on failure, report failures at the end; keep raw CSVs **and** write merged `site_<id>.parquet`; write paper-style PNGs |
| Evaluate | `hrsmpipe evaluate`: `metrics.csv` + plots; optional `evaluate.study_dir` comparison |
| Re-runs | per-stage `_done.json` fingerprint; skip unchanged stages; `--force STAGE` reruns that stage and everything after |
| Libraries | pandas, xarray, numpy, netcdf4, pyarrow, scipy, lightgbm, joblib, earthengine-api, wiscopy>=0.2, typer, pydantic>=2, matplotlib, seaborn, tqdm. **Remove** from pyproject: folium, geemap, gdal, rasterio, geopandas, shapely, xee, permetrics, ipywidgets |
| Tooling | ruff (lint+format) + pytest in pyproject; mkdocs; GitHub Actions running ruff + unit tests only |
| Notebook | add thin `notebooks/hrsmpipe_walkthrough.ipynb`; leave the big notebook untouched until the parity test passes, then `git mv` it to `notebooks/legacy/` |
| Related code | `wiscosoil` is **not** a dependency (copy two functions with credit); the legacy `.npy` is converted only by `scripts/convert_legacy_obs_npy.py` |

## 2. Package layout (create exactly these modules)

```
pyproject.toml
src/hrsmpipe/__init__.py            __version__
src/hrsmpipe/cli.py                 typer app (see §1 entry points)
src/hrsmpipe/config.py              RunConfig + sub-models; load_config(path, overrides) -> RunConfig; resolved_toml(cfg) -> str
src/hrsmpipe/provenance.py          provenance_table() -> dict  (§5)
src/hrsmpipe/paths.py               CachePaths(data_root), RunPaths(data_root, run_name); resolve_pkg_path("pkg:...")
src/hrsmpipe/timeaxis.py            hourly_axis(start, end); hour_index(times, start) (floor division); year_doy_hour(times)
src/hrsmpipe/sites.py               SiteSet (ordered DataFrame; order = array column order); load_sites(cfg)
src/hrsmpipe/manifest.py            fingerprint(stage, cfg, input_paths); is_done(); mark_done(); force semantics
src/hrsmpipe/pipeline.py            STAGES tuple in order; run_stages(cfg, from_=None, to=None, force=None, sites=None)
src/hrsmpipe/gee/__init__.py
src/hrsmpipe/gee/auth.py            initialize(project)
src/hrsmpipe/gee/collections.py     era5_short/mid/long(roi, start, end), hlsl30(roi, start, end), sentinel2(...), sentinel1(...), static_images()
src/hrsmpipe/gee/static.py          fetch_static(cfg, sites) -> cache json per site; load_static(cfg, sites) -> DataFrame with derived columns
src/hrsmpipe/gee/timeseries.py      fetch_rs(cfg, sites); assemble_rs(cfg, sites, source) -> xr.Dataset (time, site)
src/hrsmpipe/obs/__init__.py        load_observations(cfg, sites) -> DataFrame
src/hrsmpipe/obs/wisconet_api.py    fetch via wiscopy, unit table, depth map, per site-month parquet cache
src/hrsmpipe/obs/wisconet_csv.py    archive-dir reader (ported from wiscosoil/ingest.py ingest_station/pivot_sm)
src/hrsmpipe/obs/table.py           generic reader with column map and timezone
src/hrsmpipe/obs/rootzone.py        layer_thicknesses(depths, rootzone_cm=100); compute_rootzone(df, depths)
src/hrsmpipe/features/era5.py       ERA5_VARS order; weighted_lookback(arr, n, mode)
src/hrsmpipe/features/satellite.py  s1_features, s2_features, hls_features
src/hrsmpipe/features/assemble.py   COLUMNS_S1 (30), COLUMNS_S2 (39), COLUMNS_HLS (41); build_features(cfg, ...) -> parquet per (sensor, layer)
src/hrsmpipe/ml/models.py           ModelBundle.load(models_dir); DROP_COLUMNS = {"ML1": [0,1,3,15], "ML2": [0,1,3,15,24,25,26], "ML3": [0,1,3,15,24,25,26,32,33,34,35]}
src/hrsmpipe/ml/infer.py            infer(cfg) -> inference/<layer>.parquet
src/hrsmpipe/postproc/cdf.py        empirical_cdf, cdf_matching  (verbatim from cell 114)
src/hrsmpipe/postproc/combine.py    reference_match(), sd_and_miu(), spread_and_grid() -> postproc/*.nc, miu.csv
src/hrsmpipe/forcing/pet.py         priestley_taylor(ssr_wm2, t2m_c, alpha, gamma); build_pet(cfg, ...)
src/hrsmpipe/forcing/precip.py      build_precip(cfg, ...) with provenance
src/hrsmpipe/forcing/soils.py       extract_soil_properties (verbatim masks + tables from cell 23); soil_classes(cfg, static) -> soil_classes.csv
src/hrsmpipe/lgar/build.py          build_binaries(cfg) -> bin paths + build.json
src/hrsmpipe/lgar/settings.py       write_settings(path, settings: dict)
src/hrsmpipe/lgar/run.py            run_site(...) ; run_lgar(cfg)
src/hrsmpipe/lgar/merge.py          merge_site(...) -> site_<id>.parquet
src/hrsmpipe/evaluate/metrics.py    compute_metrics(obs, pred) (verbatim cell 128); metrics_table(); study_comparison()
src/hrsmpipe/evaluate/plots.py      plot_paper_site_data (port of cell 130), plot_validation (cell 129)
src/hrsmpipe/data/sites/wisconet_stations.csv
src/hrsmpipe/data/sites/study_sites.csv
src/hrsmpipe/data/lgar_src/original/...   (unzipped LGARTOKF_original_signalSite)
src/hrsmpipe/data/lgar_src/enkf/...       (unzipped LGARTOKF_EnKF_signalSite)
configs/wisconet.toml
configs/study_sites.toml
scripts/convert_legacy_obs_npy.py
tests/conftest.py, tests/test_*.py (unit), tests/test_parity.py (@pytest.mark.data), tests/test_gee_smoke.py (@pytest.mark.gee)
notebooks/hrsmpipe_walkthrough.ipynb
docs/ + mkdocs.yml
.github/workflows/ci.yml
```

### On-disk layout (`data_root` defaults to `$HRSM_DATA_DIR`)

```
<data_root>/bin/<platform>/LGAR, EnKF_LGAR_<N>_<sdsly><sdrly>[_seed<k>], build.json
<data_root>/cache/gee/static/<site_id>.json
<data_root>/cache/gee/<era5|s1|s2|hls>/<site_id>/<YYYY-MM>.nc     attrs: time_min, time_max, scale_m, window_hours, asset ids
<data_root>/cache/obs/wisconet/<site_id>/<YYYY-MM>.parquet          UTC month chunks
<data_root>/runs/<run_name>/
  config.resolved.toml   manifest.json   log.txt   sites.csv   obs.parquet
  rs/era5.nc            spans [start - era5_lead_hours, end)
  rs/{s1,s2,hls}.nc     span [start, end)
  features/<sensor>_<layer>.parquet      columns f00..fNN + site_id + time
  inference/<layer>.parquet              site_id, time, sensor, y_mean, y_median, y_low, y_high, obs
  postproc/ssm_sly.nc ssm_rly.nc sd_sly.nc sd_rly.nc  (time, site)   miu.csv (site_id, miu_sly, miu_rly)
  forcing/prec.nc (precip_mm, precip_source)  pet.nc (pet_mm, pet_source)   soil_classes.csv
  lgar/original/Original_LGAR_<id>.csv   lgar/enkf/EnKF_LGAR_<id>.csv   lgar/site_<id>.parquet   lgar/failures.json
  evaluate/metrics.csv  evaluate/study_comparison.csv  evaluate/plots/site_<id>_paper.png  site_<id>_validation.png
```

## 3. Config schema (TOML; defaults are the notebook's values)

```toml
[run]
name = "wisconet_2021_2026"
data_root = ""                 # "" -> $HRSM_DATA_DIR
start = "2021-07-16"           # UTC date; axis is [start, end)
end = "2026-02-10"

[sites]
csv = "pkg:wisconet_stations.csv"
include = []                   # optional list of site_id

[gee]
project = "run-hrsm"
workers_era5 = 2
workers_satellite = 4
window_hours = 15
sample_scale_m = 100           # NLCD and all time series
polaris_scale = "native"       # bd_0_5.projection().nominalScale() (30 m)
terrain_scale = "native"       # each asset's nominalScale()
era5_lead_hours = 144          # 120 + 24, as the notebook's start_date_era5

[obs]
source = "wisconet_api"        # "wisconet_api" | "wisconet_csv_dir" | "table"
path = ""                      # csv dir / table path (may contain {site_id})
timezone = "UTC"               # naive table timestamps only
rootzone_depths_cm = [5, 10, 20, 50, 100]
rootzone_weights = [7.5, 7.5, 20, 40, 25]   # midpoint rule; the validated legacy array used [2.5, 7.5, 20, 40, 25]
[obs.columns]                  # table source only
time = "time"; site_id = "site_id"; sm_surface = "sm_surface"; sm_rootzone = "sm_rootzone"
precip_mm = "precip_mm"; air_temp_c = ""; solar_w = ""

[features]
smap_constant = 0.3            # X[15]
hls_angle_constant = 90.0      # HLS X[32..35]
s1_invalid_above = 1000.0      # cell 70
era5_lookback_warmup = "raw"   # "raw": rows 0..N-1 keep raw hourly values (notebook) | "lead": compute from era5_lead_hours

[models]
dir = ""                       # "" -> <repo>/trainedML

[postproc]
reference_sensor = "sentinel2" # "sentinel1" | "sentinel2" | "hls"
sd_divisor = 3.29
sd_floor = 0.02
hour_spread = 1

[forcing]
interval_shift_hours = 0
[forcing.precip]
source = "mesonet"             # "mesonet" | "era5"
fallback = "era5"              # "era5" | "none"
era5_zero_below_mm = 0.01
[forcing.pet]
source = "era5"                # "era5" | "mesonet"
fallback = "era5"
alpha = 1.3
gamma = 66.8
zero_below = 1e-6

[lgar]
bin_dir = ""                   # "" -> <data_root>/bin/<platform>
ensemble_size = 20
model_sd_sly = 0.03
model_sd_rly = 0.02
seed = 0                       # 0 keeps random_device (original behaviour)
workers = 0                    # 0 -> os.cpu_count()
verbosity = "none"
layer_thickness_cm = [5, 10, 15, 30, 40]
giuh_ordinates = [0.06, 0.51, 0.28, 0.12, 0.03]
timestep_h = 1
field_capacity_psi = 340.9
wilting_point_psi = 15495.0
initial_psi = 2000.0
initial_sm_perturb = 0
soil_params_file = ""
TO_enabled = false
adaptive_timestep = false
free_drainage_enabled = false
use_closed_form_G = false
root_zone_depth_cm = 60
mbal_tol = 80.0
ponded_depth_max_cm = 0
forcing_resolution_h = 1
max_valid_soil_types = 12
[lgar.site_overrides]          # optional: [lgar.site_overrides.7] layer_soil_type = [3,3,4,4,4]

[evaluate]
study_dir = ""                 # e.g. <data_root>/shuohao_ground_truth_three_sites
plots = true
time_window = []               # optional ["YYYY-MM-DD", "YYYY-MM-DD"] for plot x-limits
```

`site_ID`, `num_hours`, `endtime_s`, `layer_soil_type` are derived at run time and are **not** config keys.
`hrsmpipe config-schema` prints this with defaults; `hrsmpipe init <path>` writes it. A `[provenance]`
table in an input TOML is ignored and regenerated.

## 4. Exact behaviours to reproduce (by stage)

### 4.1 Sites
- Order of the sites CSV defines the column order of every `(time, site)` array and of `referIDArr`.
- `site_id` is an int and the canonical key; `station_code` is a label only.

### 4.2 GEE static (`cells 17–23`)
- Landcover: `USGS/NLCD_RELEASES/2019_REL/NLCD` filtered `2016-01-01..2018-01-01`, `landcover`, first; sampled at `scale=100`.
- POLARIS: `projects/sat-io/open-datasets/polaris/{bd,clay,ksat,sand,silt}_mean`, `system:index == "<prop>_<depth>"` for depths `0_5, 5_15, 15_30, 30_60, 60_100`; `*_0_100` = 0.05·d0_5 + 0.1·d5_15 + 0.15·d15_30 + 0.3·d30_60 + 0.4·d60_100 (bd, clay, sand, silt). Sampled at the native scale of `bd_0_5`. (The notebook adds the `ksat_0_5` band twice; GEE renames the second to `ksat_0_5_1` and it is never read — do not reproduce, just note it.)
- Terrain: `projects/ee-scai62/assets/COUNS_dem100m`, `CONUS_dem100m_TWII`, `CONUS_dem100m_aspect`, `CONUS_dem100m_slopeDegree`, band `b1`, each sampled at its own `projection().nominalScale()`, **no reproject**.
- Derived on load: `elevation[elevation < -1000] = NaN`; `sinAsp = sin(aspect·π/180)`, `cosAsp = cos(...)`, both 0 where `aspect == -1`; `ksat_* = 10**ksat_*` (cm/h); `ksat_0_100 = 1.0 / (0.05/k0_5 + 0.10/k5_15 + 0.15/k15_30 + 0.30/k30_60 + 0.40/k60_100)` (harmonic, total depth 1.0).
- Cache json stores raw sampled values plus asset ids and scales; the manifest fingerprint includes them.

### 4.3 GEE time series (`cells 25–50`)
- ROI = bounds of the site FeatureCollection; features carry `id_num` (site_id) and `constant = 1`.
- ERA5: `ECMWF/ERA5_LAND/HOURLY`. Three groups, each a separate `sampleRegions(scale=100)` per window:
  short `[temperature_2m, dewpoint_temperature_2m, wind_10m, surface_pressure]` where `wind_10m = u10.hypot(v10)` added **server-side** before select; mid `[surface_solar_radiation_downwards_hourly, surface_thermal_radiation_downwards_hourly]`; long `[total_evaporation_hourly, total_precipitation_hourly]`. Canonical variable order in `rs/era5.nc` = exactly that 8-list. ERA5 is fetched over `[start − era5_lead_hours, end)`.
- HLSL30: `NASA/HLS/HLSL30/v002`, mask from `Fmask` bits: cloud bit1==0, shadow bit3==0, snow/ice bit4==0, water bit5==0, aerosol bits6–7 ≤ 2; bands `[B2,B3,B4,B5,B6,B7,B9,B10,B11]` (9), no scale factor.
- Sentinel-2: `COPERNICUS/S2_SR_HARMONIZED`, `SCL` not in {8,9,10,11}; bands `[B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]` (10).
- Sentinel-1: `COPERNICUS/S1_GRD`, IW, VV and VH present, sorted by `SLC_Processing_start`; VV masked to (−20, −5), VH to (−30, −10), each `convolve(ee.Kernel.gaussian(3))`, plus `angle`; cast to float. Bands `[VV, VH, angle]` (3). No reproject before sampling.
- Satellites are fetched over `[start, end)`; `filterDate` end is exclusive.
- Windows: `window_hours` (15) consecutive, aligned to the month start inside each `(source, month)` job; each hour falls in exactly one window either way, so per-(hour, site) means equal the notebook's. Set `system:time_start` on every sampled feature (for ERA5 too — the notebook parsed the feature id `YYYYMMDDTHH`; equivalent) and compute `hour_idx = floor((t − start)/1h)`. Values per `(hour_idx, site)` are the **mean** of all samples (`groupby(...).mean()`), NaN where none.
- Thread pools: `workers_era5` (2) and `workers_satellite` (4). Per-site month chunks carry `time_min/time_max`; a chunk whose coverage does not contain the requested month range is refetched.

### 4.4 Observations
- Tidy frame columns: `time` (tz-aware UTC, floored to the hour), `site_id`, `sm_surface`, `sm_rootzone`, `precip_mm`; optional `sm_5, sm_10, sm_20, sm_50, sm_100, air_temp_c, solar_w`.
- `wisconet_api`: use `wiscopy.data.bulk_measures` (or the station interface) with **UTC** bounds and `tz_convert("UTC")` on the result — `WisconetStation.fetch_data` treats naive bounds as station-local and returns station-local timestamps. Fields and conversions (one documented table + unit test):
  `60min_soil_moisture_pct_avg@2in/@4in/@8in/@20in/@40in` → `sm_5/10/20/50/100` (divide by 100 **after** checking one month's range is 0–100; if the API already returns 0–1, do not divide — record which in the docstring),
  `60min_rain_in_tot × 25.4 → precip_mm`, `(60min_air_temp_f_avg − 32)·5/9 → air_temp_c`,
  `60min_solar_radiation_w_avg → solar_w` (already W/m²; `60min_solar_radiation_kwm2_tot × 1000` only as a documented fallback). Respect wiscopy's rate limiting; cache per site-month.
- `wisconet_csv_dir`: port `ingest_station`/`pivot_sm` from `wiscosoil/ingest.py` (files `*_all_hourly_*.csv`, `gmt_timestamp` seconds → UTC, metric columns `soil_moisture_{5,10,20,50,100}cm_m3m-3_avg`, `rain_mm_tot`, `air_temp_c_avg`, `solar_radiation_w_avg`). No QC filtering (the notebook had none).
- `table`: column map from `[obs.columns]`; naive times localized with `obs.timezone` then converted to UTC; `path` with `{site_id}` → one file per site (the study CSVs: `time`, `observation_sly` → sm_surface, `observation_rly` → sm_rootzone, no precip).
- Rootzone: `layer_thicknesses` midpoint rule (→ `[7.5,7.5,20,40,25]`), valid only if 5 cm, 100 cm and ≥1 middle depth present; interior linear interpolation along depth (`limit_area="inside"`); weighted mean / sum(weights). `sm_surface = sm_5`.

### 4.5 Features (`cells 33–34, 68–84`)
- Time columns from the UTC axis: `year`, `doy` (`tm_yday`), `hour`.
- ERA5 lookbacks: weights `w = 1/arange(n,0,-1); w /= w.sum()`; for row i ≥ n: `sum(x[i-n+1 .. i] · w)` with `np.sum` (NaN propagates). Short n=24 for d2m, t2m, wind, sp; mid n=72 for ssr, str; long n=120 for e, tp. Mode `"raw"`: rows `0..n−1` keep the raw hourly value (notebook: `range(24, N)`, so row 23 is raw). Mode `"lead"`: compute on the extended axis and slice to `[start, end)`.
- `tp` feature stays in **metres** (the mm conversion is forcing-only).
- S1: after assembly set values `> s1_invalid_above` to NaN, then features.
- Column tables (0-based; `static` = per-site constant):
  - **S1 (30)**: 0 site_id, 1 year, 2 doy, 3 hour, 4 lat, 5 lon, 6 elevation, 7 slope, 8 sinAsp, 9 cosAsp, 10 TWI, 11 bd_0_100, 12 sand_0_100, 13 clay_0_100, 14 ksat_0_100, 15 `smap_constant`, 16 d2m_lb, 17 t2m_lb, 18 ssr_lb, 19 str_lb, 20 e_lb, 21 wind_lb, 22 sp_lb, 23 tp_lb, 24 VV, 25 VH, 26 angle, 27 VV/VH, 28 (VV²+VH²)/√2, 29 (VV+VH)/(VV−VH).
  - **S2 (39)**: 0–23 as S1; 24–33 = B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 **× 1e-4**; 34 NDVI=(B8−B4)/(B8+B4); 35 NDWI=(B3−B8)/(B3+B8); 36 greenness = −0.3599·B2 −0.3533·B3 −0.4734·B4 +0.6633·B8 −0.0087·B11 −0.2856·B12; 37 brightness = 0.3510·B2 +0.3813·B3 +0.3437·B4 +0.7196·B8 +0.2396·B11 +0.1949·B12; 38 wetness = 0.2578·B2 +0.2305·B3 +0.0883·B4 +0.1071·B8 −0.7611·B11 −0.5308·B12 (all on the scaled bands).
  - **HLS (41)**: 0–23 as S1; 24 B2, 25 B3, 26 B4, 27 B5, 28 B6, 29 B7, 30 B10, 31 B11 (raw, **no** scale; B9 fetched but unused); 32–35 = `hls_angle_constant`; 36 NDVI=(B5−B4)/(B5+B4); 37 NDWI=(B3−B5)/(B3+B5); 38 greenness = −0.2941·B2 −0.243·B3 −0.5424·B4 +0.7276·B5 +0.0713·B6 −0.1608·B7; 39 brightness = 0.3510·B2 +0.3813·B3 +0.3437·B4 +0.7196·B5 +0.2396·B6 +0.1949·B7; 40 wetness = 0.2578·B2 +0.2305·B3 +0.0883·B4 +0.1071·B5 −0.7611·B6 −0.5308·B7.
- Rows with any NaN across all columns are dropped (`~np.any(isnan, axis=0)`). Surface and rootzone use identical feature tables per sensor (two parquet files with identical content are acceptable; or one file per sensor read twice — keep the public API per (sensor, layer)).
- Parquet columns: `f00..fNN` in table order plus `site_id`, `time`.

### 4.6 ML inference (`cells 100–112`)
- Families: ML1 = S1, ML2 = S2, ML3 = HLS; layers sly/rly. Drop columns per family (`DROP_COLUMNS` above) → `scaler.transform` → `lgb_mean.predict` → `mapie_cqr.predict_interval` giving `(median, pis)`; `y_low = pis[:,0,0]`, `y_high = pis[:,1,0]`.
- Output per layer: `site_id, time, sensor ∈ {sentinel1, sentinel2, hls}, y_mean, y_median, y_low, y_high, obs` (obs joined on exact UTC hour from the tidy obs frame, surface→`sm_surface`, rootzone→`sm_rootzone`).

### 4.7 Post-processing (`cell 114`, port verbatim then wrap)
- Per layer, per site present in the **reference** sensor: `cdf_matching(y_mean[other sensor, site], y_mean[reference, site])` for the two non-reference sensors; sites absent from the reference are left unmatched and a warning is logged. Only `y_mean` is matched.
- `sd = (y_high − y_low)/sd_divisor`, values ≤ 0 → `sd_floor` (bounds are the **unmatched** ones).
- Sensor code: sentinel1=1, sentinel2=2, hls=3 (only used for grouping).
- Spread: `concat(rows with hour−1, original rows, rows with hour+1)` in that order (`hour_spread`), then drop rows with hour ∉ [0,23], then per (sensor, site, time) keep the **first** occurrence, then place on the axis.
- `miu_<layer>` per site = mean of the spread values (as `calculate_sd_and_miu`).
- Grid: `(3 sensors, time, site)` → `nanmean` over sensors → `ssm_<layer>` and `sd_<layer>` on `[start, end)` (the notebook's extra trailing row, always NaN, is dropped).

### 4.8 Forcing (`cells 44–47`, new precip rule)
- ERA5 precip: `tp_m × 1000 → mm`, values `< era5_zero_below_mm → 0`; NaN kept as NaN at this point.
- Mesonet precip: tidy `precip_mm` placed on the axis per site.
- Blend when `source="mesonet"`: value = mesonet where finite, else ERA5 where finite, else 0; `precip_source` ∈ {`mesonet`, `era5`, `missing`} per hour (store as int codes with attrs). `source="era5"`: ERA5, NaN → 0 with `missing`. Apply `interval_shift_hours` (default 0) to both series identically.
- PET (Priestley–Taylor, cell 46 verbatim): `e_s(T)=611·exp(17.27T/(273.3+T))`, `delta=4098·e_s/(273.3+T)²`, `l_v=2500−2.36T`, `gamma` const, `E_r=R_n/(l_v·1000·rho_w)`, `E_PT=E_r·alpha·delta/(delta+gamma)`, `PET_mm_h = 3.6e6·E_PT`, with `R_n = ssr_J/3600` (W/m²) and `T = t2m_K − 273.15`. Then `PET < zero_below → 0`, then per-site linear interpolation `limit_direction="both"` (sites entirely NaN are skipped with a warning). `source="mesonet"`: `R_n = solar_w`, `T = air_temp_c`, ERA5 fallback hour-wise, `pet_source` provenance.
- Soils (`cell 23` verbatim): silt = 100 − sand − clay per depth; masks and the Hydrus tables (theta_r, theta_e, alpha, n, Ks, soilMap 1..12) copied exactly, assignments in the notebook order (sand, loamy_sand, sandy_loam, loam, silt, sandy_clay_loam, silt_loam, clay_loam, silty_clay_loam, sandy_clay, silty_clay, clay). Output `soil_classes.csv`: `site_id, layer (1..5), depth, soil_id, theta_r, theta_e, alpha, n, Ks`. **Validate**: every `soil_id` must be an int in 1..12; otherwise the site is recorded in `lgar/failures.json` and skipped (the C++ parser would cast NaN to int).
- Per-site overrides `[lgar.site_overrides.<id>].layer_soil_type` replace the five classes.

### 4.9 LGAR build (`LGAR/build_macos.sh` generalised)
- Copy `src/hrsmpipe/data/lgar_src/{original,enkf}` to a build dir; in `enkf/bmi_main_lgar.cxx` patch `int N_ensemble = 100;` → `ensemble_size` (verify with grep, fail otherwise), `std::normal_distribution<double> model_error_sly(0.0, 0.03);` / `model_error_rly(0.0, 0.02)` → configured SDs, and when `seed != 0` replace `default_random_engine generator(rd());` with `default_random_engine generator(<seed>);` (lines ~210, 238–239, 249–250 — locate by regex, verify each patch applied exactly once).
- Compile both trees with `$CXX` (default `clang++` on Darwin, `g++` elsewhere), `-std=c++17 -O2`, sources `bmi_main_lgar.cxx src/{aet,bmi_lgar,lgar,linked_list,mem_funcs,soil_funcs,util_funcs}.cxx giuh/giuh.cxx -Iinclude -Ibmi -Igiuh`.
- Outputs `<bin_dir>/LGAR` and `<bin_dir>/EnKF_LGAR_<N>_<sd>` where `<sd> = f"{round(sd_sly*100):03d}{round(sd_rly*100):03d}"` (20/0.03/0.02 → `EnKF_LGAR_20_003002`), suffix `_seed<k>` when seeded; `chmod u+x`; `build.json` with compiler, flags, patches, source hashes. `hrsmpipe lgar` auto-builds if the expected binary is missing.

### 4.10 LGAR + EnKF run (`cell 121`)
- Per site (ProcessPoolExecutor, `workers`): scratch dir `<run>/lgar/_work/<site_id>/`; arrays as float64 of length N = axis length: `prec.bin`, `PET.bin`; assert length N and no NaN before writing. Settings file: all `[lgar]` keys that the C++ parser reads (`verbosity, site_ID, num_hours, layer_thickness_cm, giuh_ordinates, layer_soil_type, timestep_h, endtime_s, field_capacity_psi, wilting_point_psi, initial_psi, initial_sm_perturb, soil_params_file, TO_enabled, adaptive_timestep, free_drainage_enabled, use_closed_form_G, root_zone_depth_cm, mbal_tol, ponded_depth_max_cm, forcing_resolution_h, max_valid_soil_types`), format `key: value`, lists as `[a, b, c]`, booleans lowercase; `num_hours = N`, `endtime_s = 3600·N`, `site_ID = site_id`.
- Run `LGAR --filename_prec --filename_pet --settings_path` with `cwd=scratch`; read `Original_LGAR_<id>.csv` (`header=None, names=["hour","sm_sly","sm_rly"]`).
- CDF-match: `sm_sly[finite] = cdf_matching(sm_sly[finite], lgar_sm_sly_full)` and same for rly; write `sm_sly.bin, sd_sly.bin, sm_rly.bin, sd_rly.bin` (NaN kept where ML absent — the binary assimilates only hours where all four are finite).
- Run EnKF with the seven `--filename_*`/`--settings_path` args, `cwd=scratch`; read `EnKF_LGAR_<id>.csv` (header: `time,forcast_sd_sly,forcast_sd_rly,analysis_sd_sly,analysis_sd_rly,forcast_sm_sly,forcast_sm_rly,analysis_sm_sly,analysis_sm_rly,ml_sd_sly,ml_sd_rly,ml_sm_sly,ml_sm_rly`, `time` is the hour index, ML columns blank when not assimilated). A CSV shorter than N means the binary diverged and broke out early: keep the file, mark the site `truncated` in `failures.json`.
- Move both CSVs to `lgar/original/` and `lgar/enkf/`; delete the scratch dir on success.
- Merge → `lgar/site_<id>.parquet` indexed by UTC `time` (N rows, both CSVs reindexed), columns in this order: `forcast_sd_sly, forcast_sd_rly, analysis_sd_sly, analysis_sd_rly, forcast_sm_sly, forcast_sm_rly, analysis_sm_sly, analysis_sm_rly, ml_sd_sly, ml_sd_rly, ml_sm_sly, ml_sm_rly, sly_sm_ml_arr, rly_sm_ml_arr, observation_sly, observation_rly, forcast_sm_sly_cdf, forcast_sm_rly_cdf, analysis_sm_sly_cdf, analysis_sm_rly_cdf, original_sm_sly, original_sm_rly, original_sm_sly_cdf, original_sm_rly_cdf, analysis_sd_sly_cdf, analysis_sd_rly_cdf, precip_mm, precip_source, pet_mm, pet_source`. `sly_sm_ml_arr/rly_sm_ml_arr` = post-processed ML before matching to LGAR; `observation_*` from the obs frame; `*_cdf` = `cdf_matching(series, ML non-NaN values of that site/layer)`; `analysis_sd_*_cdf` by the monotone band mapping of cell 125 (`to_ml = interp over sorted analysis → analysis_cdf`; `(to_ml(a+sd) − to_ml(a−sd))/2`).

### 4.11 Evaluate (`cells 125–137`)
- `compute_metrics` verbatim (bias, RMSE, ubRMSE, Pearson R). `metrics.csv` rows: `site_id, station_code, layer, series, n, bias, rmse, ubrmse, r` for series `ml` (post-processed ML vs obs on common hours), `lgar_original`, `enkf_analysis`, `enkf_analysis_cdf`, `lgar_original_cdf`, plus `*_daily` variants (daily means, as in `_rmse_vs_obs`).
- Plots: port `plot_paper_site_data` (cell 130) exactly (2 panels, obs daily-mean red circles, ML blue triangles, EnKF line ± 1σ band, open-loop green, inverted daily precip bars from `precip_mm`, RMSE text, shared legend) → `site_<id>_paper.png` (dpi 300); `plot_validation` (cell 129, 2×2) → `site_<id>_validation.png`.
- With `evaluate.study_dir`: load `EnKF_LGAR_<id>_hourly.csv` per study site, produce `study_comparison.csv` (ML-stage and vs-observation tables of cell 137) and `study_<id>_ours_vs_study.png` (cell 135's `plot_study_site`).

## 5. Provenance and manifests
- `provenance_table()` returns: `git_describe` (`git describe --tags --long --dirty --always`), `git_tag` (nearest tag or null), `git_commit` (full sha), `git_commits_since_tag` (int or null), `git_dirty` (bool from `git status --porcelain`), `git_branch`, `hrsmpipe_version`, `wiscopy_version`, versions of numpy/pandas/xarray/scikit-learn/lightgbm/mapie/earthengine-api, `python`, `platform`, `created_utc`. Outside a git checkout, `git_*` are null and a warning is logged.
- `config.resolved.toml` = full config dump (all keys, overrides applied) + `[provenance]`; written by every stage command and by `run` before any work. Stage fingerprints exclude `[provenance]` and `run.name`.
- `_done.json` per stage dir: fingerprint (hash of the relevant config sections + input file sizes/mtimes/hashes), timestamps, provenance snapshot. `run` skips a stage whose fingerprint matches; `--force STAGE` reruns it and all later stages.
- Logging: `log.txt` in the run dir plus console; `tqdm` progress for GEE windows and sites.

## 6. Milestones, in order (finish each, run its checks, then continue)

**M1 — skeleton + data in.** pyproject rename and dependency cleanup (keep the pixi workspace; add
`[tool.pixi.pypi-dependencies] hrsmpipe = { path = ".", editable = true }`, tasks `hrsmpipe`, `test`,
`lint`), `config.py`, `provenance.py`, `paths.py`, `timeaxis.py`, `sites.py`, `manifest.py`, `gee/*`,
`obs/*`, CLI with `fetch-static`, `fetch-rs`, `fetch-obs`, `init`, `config-schema`, `run` (stages so far),
bare-path invocation. Unit tests: config defaults/overrides/resolved dump, provenance (clean tag, dirty tree,
non-git), time axis + hour index (incl. negative lead), rootzone (weights, validity, interpolation), API unit
table, sites order. GEE smoke test (1 site, 1 day, `@gee`). **Check:** for `configs/study_sites.toml`
(`start=2016-01-01, end=2025-01-01`, `obs.source="table"`, `path="<study_dir>/EnKF_LGAR_{site_id}_hourly.csv"`),
`rs/<source>.nc` equals the first N hours of `shuohao_study_sites/{era5hour,sentinel1,Sentinel2,HLSL30}_arr_irriClimate.npy`
(feature order as in §4.3; `allclose` with NaN equality).

**M2 — features, ML, post-processing.** `features/*`, `ml/*`, `postproc/*`, CLI `features`, `infer`,
`postproc`. Unit tests with tiny fixtures: lookback boundary rows (23 raw, 24 computed) and NaN propagation,
each index/tasseled-cap formula, column tables length 30/39/41, drop columns, cdf_matching against a
hand-computed case, spread ordering with consecutive-hour rows, nanmean over sensors. **Parity test**
(`tests/test_parity.py`, `@pytest.mark.data`, skipped when `$HRSM_DATA_DIR/shuohao_study_sites` is absent):
features parquet vs `ml_ready_datasets/<sat>_<layer>_X_all_CDL16_24.npy` (rows matched on site_id, year,
doy, hour; all columns `atol=1e-6`), `y_mean` vs `raw_model_inference_arrs/*_y_CDL16_24.npy`, postproc
grids vs `post_processed_model_output/{ssm,sd}_{sl,rly}_arr2D.npy[:N]` and `miu_*`. Before pinning
scikit-learn 1.7.2 confirm predictions equal the saved (1.5.2) outputs; if they differ, report the max
difference and stop for a decision.

**M3 — forcing + LGAR.** `forcing/*`, `lgar/*`, CLI `forcing`, `build-lgar`, `lgar`. Unit tests:
Priestley–Taylor on known values, precip blend + provenance codes, soil masks on the 12 textbook textures
+ NaN rejection, settings writer format, binary name formula, patch verification. **Parity:** with
`forcing.precip.source="era5"`, `forcing/prec.nc` and `pet.nc` equal `shuohao_study_sites/LGAR/{Prec,PET}_arr.npy`
(transpose: saved arrays are `(site, hour)`), `soil_classes` equal `soilid_arr.npy`; after `build-lgar`,
`lgar/original/Original_LGAR_296.csv` first N rows equal `lgar_output/original/Original_LGAR_296.csv`;
EnKF: RMSE between our `analysis_sm_*` and the saved `EnKF_LGAR_296.csv` below 0.01 m³/m³ (random
ensemble). Then run `configs/wisconet.toml` for `--sites 7,47` with mesonet precip and inspect
`site_7.parquet`.

**M4 — evaluate, notebook, docs, CI.** `evaluate/*`, CLI `evaluate`; `notebooks/hrsmpipe_walkthrough.ipynb`
(load config → `run_stages` → show metrics and the paper plot for one site); `docs/` with mkdocs (stages,
config reference generated from the pydantic model via a small script, parity/provenance pages);
`.github/workflows/ci.yml` (ruff + `pytest -m "not data and not gee"`); README section; `git mv` the legacy
notebook to `notebooks/legacy/` only after the parity test passes; `scripts/convert_legacy_obs_npy.py`.

## 7. Final acceptance checklist
- [ ] `pixi run hrsmpipe configs/study_sites.toml` runs every stage from a clean cache (GEE credentials present) and produces `metrics.csv` consistent with the notebook's cell 136/137 tables.
- [ ] `pixi run hrsmpipe runs/<name>/config.resolved.toml` (with `run.name` changed) reproduces the outputs; `[provenance]` shows tag, commit, commits-since-tag, dirty flag.
- [ ] Re-running the same config skips all stages; `--force forcing` reruns forcing, lgar, evaluate only.
- [ ] `pixi run test` passes in seconds; `pixi run test -m data` passes on the owner's machine.
- [ ] `pixi run lint` clean; CI workflow green on push.
- [ ] `hrsmpipe --help` lists all subcommands; `hrsmpipe <config.toml>` and `hrsmpipe run -c <config.toml>` are equivalent.
- [ ] `hrsmpipe build-lgar` works on macOS arm64 and on Linux x86-64 (test in a Linux container if available).

## 8. Do not
- Do not change feature math, constants, masks, band lists, drop columns, soil tables, or CDF logic from the notebook; expose them as config only where §3 says so.
- Do not import `wiscosoil`; do not read the legacy `.npy` in the package.
- Do not use station-local time anywhere; all timestamps are UTC internally.
- Do not write outputs outside `<data_root>/{bin,cache,runs}`; LGAR binaries must run with `cwd` set to the per-site scratch dir.
- Do not commit data, binaries, or build artefacts; add `data_root` paths, `src/hrsmpipe/data/lgar_src/**/build`, and `*.bin` to `.gitignore`.
- Do not stop at a partially working milestone: run its checks, fix, and only then move on. If a parity check cannot be made to pass, report the exact mismatch (file, rows, max abs diff) rather than loosening the tolerance.
