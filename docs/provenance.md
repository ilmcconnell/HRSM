# Provenance and re-runs

## One TOML defines a run

Before any work every stage command (and `run`) writes `<run>/config.resolved.toml`: every key expanded (defaults +
file + command-line overrides) plus a generated `[provenance]` table:

```toml
[provenance]
git_describe = "v0.1.0-3-gabc1234-dirty"
git_tag = "v0.1.0"                 # nearest tag (absent when there is none)
git_commit = "abc1234…"
git_commits_since_tag = 3
git_dirty = true
git_branch = "ilm-pipeline"
hrsmpipe_version = "0.1.0"
wiscopy_version = "0.2.0"          # also numpy, pandas, xarray, scikit-learn, lightgbm, mapie, earthengine-api
python = "3.13.15"
platform = "macOS-…"
created_utc = "2026-10-15T21:00:00Z"
```

Outside a git checkout the `git_*` fields are omitted and a warning is logged. The resolved file can be re-run
verbatim (`hrsmpipe runs/<name>/config.resolved.toml --name other`): a `[provenance]` table in an input file is ignored
and regenerated.

## Skipping and forcing

Each stage writes `_done.json` in its stage directory: a fingerprint (hash of the config keys the stage depends on,
the stage's input files' sizes and mtimes, and the sites), timestamps and a provenance snapshot. `[provenance]` and
`run.name` never enter a fingerprint. A stage is skipped when its fingerprint matches and its outputs exist; when a
stage runs, every later stage runs too. `--force STAGE` reruns that stage and everything after it
(`--force forcing` → forcing, lgar, evaluate).

Caches under `<data_root>/cache/` (Earth Engine and Wisconet chunks) are independent of run names and shared
between runs; each chunk records the range it covers and, for Earth Engine, the scale and asset ids.

Logs go to `<run>/log.txt` and the console.
