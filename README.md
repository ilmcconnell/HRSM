## 💧 HRSM: High-Resolution Soil Moisture Mapping over US Croplands

This repository hosts the framework, code, and trained machine learning models for **High-Resolution Soil Moisture (HRSM)** mapping over U.S. croplands. The HRSM framework generates high-resolution (100 m, hourly) and spatiotemporally continuous soil moisture (SM) maps for both the surface and rootzone layers by integrating **multi-source remote sensing data, machine learning (ML)**, and the **Layered Green and Ampt Infiltration with Redistribution (LGAR) model**.

---

### ✨ Main Features

* **High Resolution:** Generates SM maps at $\mathbf{100\ m}$ spatial resolution.
* **High Temporality:** Provides $\mathbf{hourly}$ SM estimates.
* **Continuous Coverage:** Achieves spatiotemporally continuous SM across both **surface layer (SLY)** and **rootzone layer (RLY)** over cropland in the CONUS (Continental United States).
* **Hybrid Approach:** Combines the strengths of **data-driven (ML)** and **physically-based (LGAR)** modeling approaches, leveraging the **Ensemble Kalman Filter (EnKF)** for data assimilation.
* **Performance:** Preliminary evaluations demonstrate the framework's ability to capture spatial heterogeneity and human-managed field-scale patterns.

---

### 📄 Research Paper

Please cite the following paper if you use this framework or its data (APA Style):

Cai, S., Xu, Y., Yang, Z., Crow, W. T., Zhang, Z., Shang, J., Liu, J., La Follette, P., Reberg-Horton, C., ..., & Huang, J. (2026). High-resolution surface and rootzone soil moisture over US cropland: A novel framework assimilating multi-source remote sensing data, machine learning, and the Layered Green and Ampt Infiltration with Redistribution model. *Remote Sensing of Environment*, *334*, 115167. https://doi.org/10.1016/j.rse.2025.115167

---

### 🗺️ Framework Flowchart

The overall workflow of the HRSM framework is illustrated below:

![Flowchart](Figures/Figure02.jpeg)
---

### 📁 Repository Structure and Usage

This repository is organized into the following key folders:

#### 1. `LGAR` Folder

This folder contains the compiled C++ executables for the physically-based LGAR model and its implementation in the Ensemble Kalman Filter (EnKF) data assimilation scheme.

* **`LGAR`**: Compiled C++ program to run the original LGAR model.
    * **Example Command Line Run (Linux, siteId 01):**
        ```bash
        ./LGAR --filename_prec 'site01_prec.bin' --filename_pet 'site01_PET.bin' --settings_path 'site01_settingInfo.txt'
        ```

* **`EnKF_LGAR_20_003002`**: Compiled C++ program for data assimilation using EnKF with LGAR.
    * **Ensemble Size:** $\mathbf{20}$
    * **LGAR Model Structure Standard Deviations:** $\mathbf{0.03\ m^3/m^3}$ for SLY, $\mathbf{0.02\ m^3/m^3}$ for RLY.
    * **Example Command Line Run (Linux, siteId 01):**
        ```bash
        ./EnKF_LGAR_20_003002 --filename_prec 'site01_prec.bin' --filename_pet 'site01_PET.bin' --filename_sm_sly 'site01_sm_sly.bin' --filename_sm_rly 'site01_sm_rly.bin' --filename_sd_sly 'site01_sd_sly.bin' --filename_sd_rly 'site01_sd_rly.bin' --settings_path 'site01_settingInfo.txt'
        ```

* **Source Code (C++):**
    * `LGARTOKF_EnKF_signalSite.zip`
    * `LGARTOKF_original_signalSite.zip`
    * Source C++ code for LGAR and EnKF (for users who want to modify or recompile). For more details, see the official LGAR project: [https://github.com/NOAA-OWP/LGAR-C](https://github.com/NOAA-OWP/LGAR-C).

#### 2. `trainedML` Folder

This folder stores the trained machine learning (ML) models and scalers in `.joblib` format.

* **Usage in Python:** You can directly load these files using the `joblib` library:

    ```python
    import joblib

    # Load Scalers
    ML1_sly_scaler = joblib.load('ML1_sly_scaler.joblib')
    ML1_rly_scaler = joblib.load('ML1_rly_scaler.joblib')
    ML2_sly_scaler = joblib.load('ML2_sly_scaler.joblib')
    ML2_rly_scaler = joblib.load('ML2_rly_scaler.joblib')
    ML3_sly_scaler = joblib.load('ML3_sly_scaler.joblib')
    ML3_rly_scaler = joblib.load('ML3_rly_scaler.joblib')

    # Load ML Models (LightGBM Mean and MAPIE CQR for Uncertainty)
    ML1_sly_lgb_mean = joblib.load('ML1_sly_lgb_mean.joblib')
    ML1_sly_mapie_cqr = joblib.load('ML1_sly_mapie_cqr.joblib')
    # ... and so on for all other models
    ```

* **ML Pipeline Execution:**
    * Access the notebooks on Colab:
        * **`HRSM_MAPPING_ML_pipeline_area.ipynb`** (for a spatial area)
        * **`HRSM_MAPPING_ML_pipeline_multiplePoints.ipynb`** (for multiple points)
    * **Note:** Input data is accessed through **Google Earth Engine (GEE)**. You need to upload point coordinates for processing. Ensure trained ML models are stored and accessible on your Colab Google Compute Engine backend.

#### 3. Data Assimilation Pipeline

The data assimilation (DA) step is executed after obtaining ML results. Follow the procedures in the Python script **`HRSM_MAPPING_EnKF_pipeline.py`** (requires a Linux environment for C++ executables). The steps are:

1.  **LGAR Simulation:** Run the compiled **`LGAR`** program to get the base LGAR simulation.
2.  **CDF-Matching:** Match ML results to the LGAR simulation to remove systematic errors (essential for EnKF).
3.  **Data Assimilation (DA):** Run the compiled **`EnKF_LGAR_20_003002`** program to perform the EnKF-LGAR data assimilation.

---

### 📦 `hrsmpipe`: the pipeline as a package

The notebook pipeline (now in `notebooks/legacy/`) is also available as the installable package **`hrsmpipe`**
(`src/hrsmpipe/`): ERA5-Land / Sentinel-1/2 / HLSL30 features → the pre-trained models in `trainedML/` → post-processing →
LGAR and LGAR + EnKF → evaluation, driven by **one TOML file per run** and runnable headlessly on macOS arm64, Linux
x86-64 and Colab. Every run writes `config.resolved.toml` (all keys plus git provenance) so it can be repeated verbatim.

```bash
pixi install
pixi run hrsmpipe init my_run.toml          # config with every key at its default
pixi run hrsmpipe my_run.toml               # run all stages (unchanged stages are skipped)
pixi run hrsmpipe configs/study_sites.toml  # the original study's three sites
pixi run hrsmpipe build-lgar my_run.toml    # compile LGAR / EnKF_LGAR from the bundled C++ sources
pixi run test                               # unit tests; `pixi run test -m data` checks parity with the notebook
```

`notebooks/hrsmpipe_walkthrough.ipynb` shows the Python API, and `pixi run docs` builds the documentation in `docs/`
(stages, generated config reference, provenance, parity with the notebook).
The compiled binaries in `LGAR/` remain for the notebook workflow; the package builds its own into
`<data_root>/bin/<platform>/`.

---

### 🔜 Future Plans

This is an ongoing project. We are working toward publishing the **CONUS cropland soil moisture dataset**.

---

### ✉️ Contact

Please reach out to us for more information or if you encounter any problems when running the Colab pipeline.
