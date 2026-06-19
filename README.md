# KISIK ICU Length-of-Stay Prediction

Code for predicting **intensive-care length of stay (LOS)** from the first 24 hours
after ICU admission, with strict data-leakage control and a prospective benchmark
against senior-physician estimates. Companion code to the manuscript prepared for the
Frontiers Research Topic *"MedicineAI: Advancing the Synergy of Medicine and AI – From
Data to Clinical Impact."*

> **No patient data is included in this repository.** All `.csv` / `.parquet` / `.json`
> inputs and generated documents are excluded via `.gitignore`. The scripts expect a
> local KISIK data extract (see *Data* below).

## Scientific summary

1. **Leakage control.** Features summarising the *whole* stay inflate apparent
   performance (R² ≈ 0.61). Restricting predictors to a strict first-24h window
   removes this and reveals a much harder task.
2. **Models.** XGBoost, random forest, extra trees, ridge — log1p target.
   Best retrospective hold-out: XGBoost (MAE ≈ 2.1 d, R² ≈ 0.57).
3. **Prospective clinician benchmark.** Against documented senior-physician estimates
   (n = 359 matched stays) the physician outperforms the models overall, with the gap
   concentrated in long stays (> 7 d).
4. **Addressing long-stayers.** Asymmetric loss, quantile regression (P50/P80),
   Tweedie/Gamma objectives and a **discrete-time hazard** model each reduce long-stay
   error by ~10–12 % and remove the systematic underestimation; the hazard-median even
   beats the physician on short stays. The extreme tail (> 14 d) stays unsolved from
   24h data alone.

## Repository structure

```
pipeline/    24h feature engineering & leakage diagnostics
  add_24h_features.py      build true first-24h windowed features (labs, vitals, procedures, access)
  check_leakage.py         correlation/column-mapping leakage test
  check_features_24h.py    verify selected features exist in the 24h dataset
modeling/    model training & evaluation
  oberarzt_vs_ml_extended.py   RF/ExtraTrees/XGBoost/Ridge + prospective senior comparison
  experiment_op_features.py    add OP/anaesthesia features + asymmetric-loss tail model
  quantile_op_prospective.py   XGBoost quantile (P50/P80) + OP features, prospective head-to-head
  tweedie_hazard.py            Tweedie/Gamma objectives + discrete-time hazard model
figures/     publication figures (matplotlib, 300 DPI)
reporting/   Word tables & manuscript draft (python-docx)
dashboard/   interactive per-day ward dashboard with per-patient SHAP
  build_dashboard_data.py  trains model, computes per-patient SHAP (XGBoost pred_contribs), exports JSON
  build_dashboard_html.py  renders a standalone interactive HTML dashboard
```

## Data (not included)

Single-centre KISIK extract. Key facts the code relies on:

- **Join key differs by cohort:** retrospective uses `fallid`; the prospective/OLD
  extract uses `fallnr`. Senior estimates match `tages_stay_id` ↔ `stay_id`.
- **Date formats differ:** retrospective ISO `YYYY-MM-DD HH:MM:SS`; prospective OLD CSVs
  German `DD.MM.YYYY HH:MM:SS` (parse with both, see `tweedie_hazard.py`).
- **Ward filter** tuple order is `(wardshort, oebenekurz)`.
- **OP data:** `op_an.csv` (ASA, procedure, planned duration) and
  `op_zeitintervalle.csv` (Schnitt-Naht / anaesthesia / HLM bypass durations);
  perioperative window `[planbegin-1d, planbegin+24h]`.

Paths are currently hard-coded to a local `D:\Ausgangsdaten\KISIK Projekt` layout —
adjust the path constants at the top of each script for your environment.

## Requirements

Python 3.12. See `requirements.txt`:

```
duckdb>=1.0
xgboost>=2.0
scikit-learn>=1.3
scipy
pandas
numpy
matplotlib
python-docx
Pillow
```

## How to run (typical order)

```bash
python pipeline/add_24h_features.py        # build 24h feature parquet
python pipeline/check_leakage.py           # confirm no leakage
python modeling/oberarzt_vs_ml_extended.py # baseline models + senior benchmark
python modeling/quantile_op_prospective.py # quantile + OP features
python modeling/tweedie_hazard.py          # Tweedie/Gamma + discrete hazard
python figures/fig_tweedie_hazard.py       # figure
python reporting/build_frontiers_tables.py # Word tables
python dashboard/build_dashboard_data.py   # dashboard data (per-patient SHAP)
python dashboard/build_dashboard_html.py   # standalone interactive dashboard
```

## Privacy & ethics

This repository contains **only code**. It must never be used to commit patient-level
data. Any deployment or sharing of model outputs requires the appropriate ethics
approval and data-protection clearance of the originating institution.

## License

Research code — add a license before public release (e.g. MIT) and complete the
manuscript's author/affiliation/ethics fields.
