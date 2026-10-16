# hrsmpipe

`hrsmpipe` turns the HRSM notebook (`notebooks/legacy/HRSM_MAPPING_ML_pipeline_multiplePoints_lgar_inputs.ipynb`)
into a package: ERA5-Land / Sentinel-1/2 / HLSL30 features → pre-trained LightGBM + MAPIE models →
post-processing → LGAR and LGAR + EnKF → evaluation. **One TOML file defines a run.**

```bash
pixi install
pixi run hrsmpipe init my_run.toml                 # every key at its default
pixi run hrsmpipe my_run.toml                      # all stages (same as `hrsmpipe run -c my_run.toml`)
pixi run hrsmpipe configs/wisconet.toml --sites 7,47 --start 2024-06-01 --end 2024-08-01 --name demo
pixi run hrsmpipe forcing my_run.toml              # one stage
pixi run hrsmpipe my_run.toml --force forcing      # forcing, lgar and evaluate again
pixi run hrsmpipe build-lgar my_run.toml           # compile the bundled LGAR / EnKF C++ sources
```

Python API:

```python
from hrsmpipe.config import load_config
from hrsmpipe.pipeline import run_stages

cfg = load_config("configs/study_sites.toml", {"run.name": "demo", "sites.include": [296]})
run_stages(cfg, from_="features", to="postproc")
```

## What you need

* [pixi](https://pixi.sh); Python 3.13 environments are solved for macOS arm64 and Linux x86-64
  (Colab: `pip install -e .` plus the dependencies of `pyproject.toml`).
* Google Earth Engine access for `fetch-static` / `fetch-rs`. Initialisation order: a service account if
  `GOOGLE_APPLICATION_CREDENTIALS` / `EE_SERVICE_ACCOUNT` is set, else stored interactive credentials, else
  `ee.Authenticate()`. The cloud project is `gee.project`.
* A C++ compiler (`clang++` on macOS, `g++` elsewhere; override with `$CXX`) for `build-lgar`.
* `HRSM_DATA_DIR` (or `run.data_root`): everything the package writes lives under
  `<data_root>/{bin,cache,runs}`.

## Pages

* [Stages](stages.md): what each stage reads and writes, and the exact behaviours reproduced from the notebook.
* [Config reference](config_reference.md): every key and default (generated from the pydantic model).
* [Provenance and re-runs](provenance.md): `config.resolved.toml`, `[provenance]`, fingerprints and `--force`.
* [Parity with the notebook](parity.md): the tests that tie the package to the notebook's saved outputs.
