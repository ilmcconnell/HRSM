# Parity with the notebook

The package reproduces the notebook's saved study-site outputs (`$HRSM_DATA_DIR/shuohao_study_sites`, sites 296, 1140,
2096, 2016-01-01 → 2025-01-01). The checks are in `tests/test_parity.py` (`pixi run test -m data`; the raw-series test
also needs Earth Engine: `-m "data and gee"`). The package never reads the legacy `.npy` files; the tests convert the
saved raw arrays into `rs/*.nc` so M2–M4 are checked independently of Earth Engine.

| what | result |
|---|---|
| `rs/<source>.nc` vs `*_arr_irriClimate.npy` (ERA5, Sentinel-1/2, HLSL30; a 21-day span crossing a month boundary, `HRSM_PARITY_SPAN` selects another) | equal (`allclose`, NaN pattern equal) |
| features vs `ml_ready_datasets/*_X_all_CDL16_24.npy` | equal, `atol=1e-6` |
| `y_mean` vs `raw_model_inference_arrs/*_y_*.npy` | max abs difference **0.0** (scikit-learn 1.7.2) |
| postproc grids and `miu` vs `post_processed_model_output/*` | max abs difference **0.0**, NaN patterns equal |
| `forcing/prec.nc`, `pet.nc` vs `LGAR/{Prec,PET}_arr.npy`; soil classes vs `soilid_arr.npy` | equal |
| open-loop LGAR vs `lgar_output/original/*.csv` (first N rows) | max abs difference **0.0** |
| EnKF vs `lgar_output/EnKF/*.csv` | see below |
| `study_comparison.csv` vs the tables printed in notebook cells 136–137 | equal to 4 decimals (the random EnKF rows within 0.01) |

## The EnKF threshold

The execution plan asked for an RMSE below 0.01 m³/m³ between our `analysis_sm_*` and the saved EnKF run. The 20-member
EnKF draws random model-error and observation noise, so two realizations of the *same* binary differ much more than
that: with seeds 1 and 2 the RMSE between our own runs is 0.014–0.0155 m³/m³, the same as ours against the saved run
(0.0135–0.0159). Seeded runs are exactly reproducible. The data test therefore asserts that the saved run is as close to
ours as our seeds are to each other (within a factor 1.25); the literal 0.01 check is kept as an `xfail` that documents
the measured numbers.

## scikit-learn

The models were pickled with scikit-learn 1.7.2. Predictions with 1.7.2 equal the notebook's saved 1.5.2 outputs
exactly, so the environment pins 1.7.2.
