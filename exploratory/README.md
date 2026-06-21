# Exploratory extensions — NOT part of the manuscript

> ⚠️ **Separation note.** Everything under `exploratory/` is **post-manuscript exploratory
> analysis**. It is **not** part of the manuscript and was **not** used to produce any number,
> table or figure in `reporting/KISIK_Frontiers_Manuskript_v2.docx`. The manuscript and its
> reproducible pipeline live entirely in the other top-level folders (`modeling/`, `pipeline/`,
> `reporting/`, `figures/`, `dashboard/`) and are unchanged by this folder.

These analyses extend the study in directions a reviewer might ask about (external validation,
alternative modelling objectives, model routing). They are kept here so the manuscript state stays
clean and clearly identifiable.

Only **aggregate** outputs are included (metrics, coverage, importance, AUC tables, figures).
No patient-level data, no per-stay prediction files, and no feature matrices are committed.

---

## 1 · `mimic_external/` — external validation on MIMIC-IV 3.1
Replicates the KISIK *methodology* (the frozen KISIK model cannot transfer — German OPS/ICD codes
are absent from MIMIC) on the independent MIMIC-IV 3.1 ICU database (74,827 stays, LOS > 1 d).
Same design: first-24-hour features, patient-grouped split, log1p target, 4 model families,
CV selection, bootstrap CIs, null baseline, calibration.

- `01_features.py` → first-24h feature matrix · `02_model.py` → train/select/evaluate ·
  `03_fix_ridge.py` → robust linear baseline (winsorised) · `04_alt_models.py` → Tweedie/Hazard/Quantile.
- **Headline:** the method generalises (XGBoost holdout MAE 2.36, R² 0.18; beats null 2.79) and
  shows the *same* compressed-prediction pathology as KISIK (slope 1.39, severe long-stay
  under-prediction). See `kisik_vs_mimic.csv` and `fig_mimic_*.png`.

## 2 · `kisik_alternatives/` — alternative objectives on the AIN cohort
Tweedie (1.3/1.5/1.7), Gamma, discrete-time hazard, and quantile P50/P80, applied to the same
AIN cohort (n = 12,884) as the manuscript — for an apples-to-apples comparison with MIMIC.

- `tweedie_hazard.py`, `quantile_op_prospective.py` — **AIN-restricted variants** (these differ from
  the manuscript-era `modeling/` copies, which were never part of the manuscript proper).
- `05_kisik_alt_prospective.py` — alternative objectives on the **reconstructed** prospective
  193-feature matrix (same features as the manuscript's Extra Trees).
- **Headline:** alternatives give only marginal point-accuracy gains over Extra Trees and never beat
  the senior physician; quantile-P80 is the only ML approach competitive with the physician on long
  stays (≈ 6.4 d) and reaches ~77 % capacity-planning coverage. See `los_alt_prospective_193*.csv`.

## 3 · `routing/` — gated ensemble & physician-as-detector
- `06_routing_experiment.py` — "best tool per region" routing. Train/Dev/Test (patient-grouped).
  Routing by a regime classifier or a predicted-LoS threshold gives only ~3–4 % MAE improvement
  (2.78 → 2.70) vs an oracle ceiling of ~21 % (2.20), because the compressed predictions cannot
  separate the regimes at prediction time (classifier recall: 1–2 d 72 %, 2–4 d 22 %, 4–7 d 32 %,
  > 7 d 75 %). Forcing or confidence-gating quantile-P80 for long stays does not help.
- `07_physician_detector.py` — the **senior-physician estimate is a strong long-stay detector**
  (ROC-AUC 0.93 for > 7 d and > 14 d, vs 0.71/0.79 for the model): it carries exactly the regime
  signal the compressed model lacks. See `los_routing_experiment.csv`, `los_physician_detector.csv`.

---

## Reproducibility
- Data are **not** included (data-protection). MIMIC-IV 3.1 is accessed locally; KISIK uses the
  internal parquet/snapshots referenced by the scripts.
- `05_kisik_alt_prospective.py` reads two cached matrices (`canonical/alt_matrices/`) written by a
  matrix-saving variant of `modeling/prospective_24h_rebuild.py`. That variant adds, right after the
  prospective feature frame `Xp` is built:
  ```python
  ALT = AN/"canonical"/"alt_matrices"; ALT.mkdir(parents=True, exist_ok=True)
  X.iloc[tr].assign(__y__=y[tr]).to_parquet(ALT/"retro_train.parquet")
  Xp.assign(__los__=mg2["los_days"].values, __arzt__=mg2["arzt"].values,
            __stay_id__=mg2["stay_id"].astype(str).values).to_parquet(ALT/"prospective_rebuilt_193.parquet")
  import json; json.dump({"present":present,"numc":numc,"cat":cat}, open(ALT/"feature_lists.json","w"))
  ```
  The matrices themselves are patient-level and are **not** committed.
