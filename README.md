# KISIK ICU Length-of-Stay Prediction

Code for predicting **intensive-care length of stay (LOS)** from the **first 24 hours**
after ICU admission — with strict data-leakage control and a prospective benchmark
against senior-physician estimates. Companion code to the manuscript for the Frontiers
Research Topic *"MedicineAI: Advancing the Synergy of Medicine and AI — From Data to
Clinical Impact."*

> ⚠️ **No patient data is in this repository.** All `.csv` / `.parquet` / `.json` inputs,
> notebooks and generated documents are excluded via `.gitignore`. The scripts expect a
> local KISIK extract (see [Raw data inputs](#raw-data-inputs)).

---

## What happens, with which data, where

```mermaid
flowchart TD
    R1["Retrospective KISIK CSVs<br/>stays · diagnoses · labs · vitals<br/>procedures · access<br/>(key: fallid, ISO dates)"]
    R2["OP / anaesthesia data<br/>op_an.csv · op_zeitintervalle.csv"]
    P1["Prospective OLD snapshots<br/>daily folders<br/>(key: fallnr, DE dates)"]
    S1["Senior-physician LOS estimates<br/>(CSV)"]

    R1 --> A["pipeline/retrospective_dataset_pipeline.py"]
    A --> D1[("kisik2_icu_ml_dataset.parquet<br/>full-stay aggregates")]
    D1 --> B["pipeline/add_24h_features.py"]
    R1 --> B
    B --> D2[("kisik2_icu_ml_dataset_24h.parquet<br/>first-24h features · leakage-free")]
    D2 --> L["pipeline/check_leakage.py<br/>(diagnostics)"]

    P1 --> C["pipeline/prospective_dataset_pipeline.py"]
    C --> D3[("kisik2_prospektiv_ml_dataset.parquet")]

    D2 --> T["modeling/train_los_model_24h.py"]
    T --> M{{"Trained LoS model<br/>XGBoost · log1p target"}}

    D2 --> CMP["Evaluation & tail methods<br/>oberarzt_vs_ml_extended.py<br/>quantile_op_prospective.py<br/>tweedie_hazard.py"]
    D3 --> CMP
    R2 --> CMP
    S1 --> CMP

    CMP --> O1["Result CSVs + figures (300 DPI)"]
    O1 --> REP["reporting/<br/>Frontiers tables + manuscript (.docx)"]
    D2 --> DASH["dashboard/<br/>per-patient SHAP + interactive HTML"]
```

---

## Pipeline stages (step by step)

| # | Script | Reads | Produces | What it does |
|---|--------|-------|----------|--------------|
| 1 | `pipeline/retrospective_dataset_pipeline.py` | raw retrospective KISIK CSVs (stays, diagnoses, labs, vitals, procedures, access) | `kisik2_icu_ml_dataset.parquet` | Links all modalities by `fallid`, reconstructs ICU stays/episodes, applies the ward filter, builds the base feature matrix (whole-stay aggregates). |
| 2 | `pipeline/add_24h_features.py` | base parquet + raw lab/vital/procedure/access CSVs | `kisik2_icu_ml_dataset_24h.parquet` | Recomputes labs/vitals/procedures/access **only within the first 24 h** (`planbegin → +24 h`) → `lab24_ / vital24_ / proc24_ / zugang24_` columns. This is the leakage-free dataset. |
| 3 | `pipeline/check_leakage.py`, `check_features_24h.py` | 24h parquet + selected-feature list | console report | Confirms predictors use the 24 h window, not whole-stay summaries; verifies the selected features exist. |
| 4 | `pipeline/prospective_dataset_pipeline.py` | daily OLD live snapshots | `kisik2_prospektiv_ml_dataset.parquet` | Loads each day's snapshot (`pros_load_day_csv`), detects still-open stays (`pros_detect_open_stay`), assembles the prospective dataset. Key `fallnr`, German dates. |
| 5 | `modeling/train_los_model_24h.py` | 24h parquet + `los_selected_features_ain_24h_compact.csv` | trained model + hold-out metrics | Trains the LOS regressor (`TransformedTargetRegressor`, log1p target), patient-level train/test split, then applies it to the prospective dataset. |
| 6 | `modeling/oberarzt_vs_ml_extended.py` | 24h + prospective parquet + senior-estimates CSV | head-to-head CSVs + figures | Trains RF / ExtraTrees / XGBoost / Ridge and benchmarks them against the **senior physician** (matched cohort, Wilcoxon, subgroups, calibration). |
| 7 | `modeling/experiment_op_features.py` | 24h parquet + `op_an.csv` + `op_zeitintervalle.csv` | experiment CSV | Adds perioperative features (ASA, surgery/anaesthesia/bypass time) and tests an asymmetric loss for long-stayers. |
| 8 | `modeling/quantile_op_prospective.py` | retro + prospective + OP files + senior CSV | quantile head-to-head CSVs + figure | XGBoost **quantile** regression (P50/P80) with OP features; prospective benchmark + P80 coverage. |
| 9 | `modeling/tweedie_hazard.py` | retro + prospective + OP + senior CSV | retro/prospective CSVs + figure | **Tweedie/Gamma** objectives and a **discrete-time hazard** model for the long-stay tail. |
| 10 | `reporting/build_frontiers_tables.py`, `build_frontiers_manuscript.py` | result CSVs + figures | `.docx` tables + manuscript | Generates publication-ready Word tables and the manuscript draft. |
| 11 | `dashboard/build_dashboard_data.py` → `build_dashboard_html.py` | 24h parquet + selected features | JSON → standalone HTML | Per-day ward view: predicted LOS per bed + **per-patient SHAP** (XGBoost `pred_contribs`). |

---

## Raw data inputs

Not included — provide a local KISIK extract. Expected tables (linked by case ID):

| Modality | Retrospective file(s) | Prospective source | Notes |
|----------|----------------------|--------------------|-------|
| ICU stays / episodes | stay/episode export | OLD daily snapshot | basis for cohort & target (ICU LOS) |
| Diagnoses (ICD-10) | diagnoses export | OLD snapshot | `diag_main_*` binary features |
| Laboratory | `lab.csv` | OLD snapshot | `lab24_*` (first/mean/min/max/last/count) |
| Vital signs | `vitalzeichen.csv` | OLD snapshot | `vital24_*` (e.g. SpO₂) |
| Procedures (OPS) | `prozeduren.csv` | OLD snapshot | `proc24_*` presence + count |
| Vascular access | `zugaenge.csv` | OLD snapshot | `zugang24_*` presence + count |
| OP / anaesthesia | `op_an.csv`, `op_zeitintervalle.csv` | OLD `op_*` snapshots | ASA, surgery/anaesthesia/HLM-bypass times |
| Senior estimates | senior-estimates CSV (`best_senior_estimate_days`) | — | benchmark for the prospective comparison |

### Key data facts the code relies on
- **Join key differs by cohort:** retrospective `fallid`; prospective/OLD `fallnr`.
  Senior estimates match `tages_stay_id` ↔ prospective `stay_id`.
- **Date formats differ:** retrospective ISO `YYYY-MM-DD HH:MM:SS`; prospective OLD CSVs
  German `DD.MM.YYYY HH:MM:SS` → parse with `COALESCE(TRY_CAST(...), TRY_STRPTIME(..., '%d.%m.%Y %H:%M:%S'))`.
- **Ward filter** tuple order is `(wardshort, oebenekurz)`.
- **Perioperative window** for OP features: `[planbegin − 1 day, planbegin + 24 h]`.
- Paths are hard-coded to a local `D:\Ausgangsdaten\KISIK Projekt` layout — adjust the
  path constants at the top of each script.

---

## Results at a glance

| Setting | Finding |
|---------|---------|
| Leakage check | Whole-stay aggregates inflate apparent fit (R² ≈ 0.61); strict 24 h window removes it. |
| Retrospective hold-out (n ≈ 3,429) | Best model **XGBoost**: MAE ≈ 2.1 d, R² ≈ 0.57. |
| Prospective vs. senior physician (n = 359) | **Physician wins overall** (MAE 2.60 vs 3.65 d; R² 0.25 vs −0.20), gap concentrated in long stays. |
| Long-stayers (> 7 d) | Tweedie (p≈1.3) & discrete hazard cut MAE ~10–12 % and remove the underestimation bias. |
| Short stays (1–7 d) | Quantile-P50 / hazard-median **beat the physician** (MAE ≈ 1.2 vs 1.4 d). |
| Capacity planning | P80 quantile is well-calibrated (≈ 77 % coverage) — a "discharged-by day X" bound the point estimate can't give. |

---

## Repository layout

```
pipeline/    data pipelines, 24h feature engineering & leakage diagnostics
  retrospective_dataset_pipeline.py   build retrospective ML dataset from raw CSVs (key: fallid)
  prospective_dataset_pipeline.py     build prospective ML dataset from daily OLD snapshots (key: fallnr)
  add_24h_features.py                 first-24h windowed features (leakage-free)
  check_leakage.py                    leakage diagnostics
  check_features_24h.py               verify selected features exist
modeling/    model training & evaluation
  train_los_model_24h.py              train LoS model (log1p) + prospective application
  oberarzt_vs_ml_extended.py          RF/ExtraTrees/XGBoost/Ridge vs senior physician
  experiment_op_features.py           OP/anaesthesia features + asymmetric-loss tail model
  quantile_op_prospective.py          quantile (P50/P80) + OP features, prospective head-to-head
  tweedie_hazard.py                   Tweedie/Gamma + discrete-time hazard
figures/     publication figures (matplotlib, 300 DPI)
reporting/   Word tables & manuscript draft (python-docx)
dashboard/   interactive per-day ward dashboard with per-patient SHAP
```

---

## Requirements & how to run

Python 3.12 — `pip install -r requirements.txt`
(`duckdb`, `xgboost>=2.0`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `python-docx`, `Pillow`).

```bash
# 1) build datasets
python pipeline/retrospective_dataset_pipeline.py
python pipeline/add_24h_features.py
python pipeline/prospective_dataset_pipeline.py
python pipeline/check_leakage.py
# 2) train & evaluate
python modeling/train_los_model_24h.py
python modeling/oberarzt_vs_ml_extended.py
python modeling/quantile_op_prospective.py
python modeling/tweedie_hazard.py
# 3) outputs
python figures/fig_tweedie_hazard.py
python reporting/build_frontiers_tables.py
python dashboard/build_dashboard_data.py
python dashboard/build_dashboard_html.py
```

> The two extracted pipeline files (`retrospective_/prospective_dataset_pipeline.py`)
> and `train_los_model_24h.py` are source-only extracts of the original Jupyter notebooks
> (`# %% [cell N]` markers, shared state, top-to-bottom execution). Complete as
> documentation/reference; for a clean script run, adjust paths and cell order.

---

## Privacy & ethics

This repository contains **only code**. Never commit patient-level data. Any sharing of
model outputs requires the originating institution's ethics approval and data-protection
clearance. Before public release, add a license and complete the manuscript's
author / affiliation / ethics fields.
