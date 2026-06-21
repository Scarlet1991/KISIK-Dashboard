# MIMIC-IV external validation of the KISIK ICU-LOS methodology

External validation of the KISIK first-24-hour ICU length-of-stay (LOS) prediction
**methodology** on the independent MIMIC-IV 3.1 critical-care database
(Beth Israel Deaconess Medical Center, US). The frozen KISIK model itself cannot be
transferred (German OPS/ICD codes and item names absent from MIMIC); instead the full
**design** is replicated on MIMIC-native first-24-hour features. This provides the
genuine external-validation evidence the manuscript reviewer requested (KISIK is a
single-centre study).

## Data
MIMIC-IV 3.1 at `F:\Mimic\mimic-iv-3.1\mimic-iv-3.1` (`hosp/` + `icu/` modules, CSV.gz).

## Design (mirrors the KISIK pipeline)
- **Cohort:** ICU stays with LOS > 1 day (24-hour landmark). Outcome = `icustays.los` (days).
- **Analysis unit:** ICU stay; **grouping:** `subject_id` (patient) for all splits/CV.
- **Predictors — first 24 h after ICU `intime` only** (leakage-free prediction time point):
  - Labs (`labevents`): 30 most frequent itemids × {first, last, min, max, mean, count}
  - Vitals (`chartevents`): curated vital itemids × {first, last, min, max, mean, count}
  - Procedures (`procedureevents`): 15 most frequent itemids as binary presence + total count
  - Demographics/context: age, gender, ICU care-unit, ICU stay number, admission type
  - **Excluded:** `diagnoses_icd` — billed at discharge, not available at 24 h (would leak).
- **Target transform:** log1p; predictions back-transformed with expm1 and clipped at 0.
- **Split:** patient-grouped 80/20 (GroupShuffleSplit, seed 42).
- **Models:** Ridge, Random Forest, Extra Trees, XGBoost; final model selected by lowest
  4-fold GroupKFold CV-MAE on the training set.
- **Evaluation (holdout):** MAE / median AE / RMSE / R² / bias with paired bootstrap 95% CIs;
  null-model baseline (training-set median, applied unchanged); calibration (slope + CI, CITL,
  by-actual-LoS-group model-vs-null); permutation importance.

## Scripts
1. `01_features.py` — builds `mimic_features.parquet` (+ `mimic_feature_dict.csv`).
2. `02_model.py` — trains/selects/evaluates; writes metrics, bootstrap CIs, calibration,
   importance, `mimic_summary.json`, figures, and `kisik_vs_mimic.csv`.

## Key question
Does an analogous 24-hour model generalise to a different ICU population — i.e. does it
reach comparable MAE/R², and does it (as in KISIK) beat a trivial null model retrospectively
while showing the same compressed-prediction/calibration pattern at the LoS extremes?

_No patient-level data is committed; only aggregate outputs/figures._
