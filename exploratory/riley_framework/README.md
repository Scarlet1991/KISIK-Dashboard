# Riley/Collins framework — exploratory analysis

Exploratory work applying the prediction-model framework of Riley, Collins et al. to the
leak-free **no_isopen** cohort (all matched prospective stays, n = 286; retrospective
development set with ICU LoS > 1 day). All models here use the **leak-free** feature set
(OPS `8-98f` removed). This folder is strictly exploratory and is **not** part of either
manuscript; it documents how we arrived at the soft physician-gated hybrid.

> **Data-safety note:** every file here is aggregate (metrics, sweeps, summaries, figures).
> The per-stay file `riley_instability_per_stay.csv` (stay IDs + observed LoS) was
> deliberately **excluded** from the repository as patient-level data, exactly as
> `who_won_per_case.csv` was. Do not commit per-stay records.

## The question
A single continuous regressor calibrates well in cross-validation but is unconvincing on
prospective long-stayers. Can the Riley/Collins toolkit (recalibration, smearing, honest
back-transform, bootstrap instability, long-stay-as-estimand) turn it into something that
beats — or at least matches — the senior physician?

## The finding chain (each script is one step)

| # | Script | What it tests | Result |
|---|--------|---------------|--------|
| 1 | `explore_riley_framework.py` | Calibration slope, linear recalibration, Duan smearing, bootstrap instability (B), long-stay classifier | log1p back-transform is biased (slope 1.3, underestimates long stays); recalibration/smearing fix calibration but add **no prognostic information** — prospective R² stays ≈ 0. Long-stayer bootstrap SD is *low* → the model confidently regresses them to the mean (missing signal, not estimation noise). |
| 2 | `riley_combined_model.py` | Tweedie + recalibration as the continuous core, plus P(LOS≥7) / P(LOS≥10) classifiers | Combined continuous: prospective MAE 3.82, R² 0.04. Long-stay classifier AUROC 0.73 (≥7) / 0.68 (≥10). |
| 3 | `riley_combined_vs_physician.py` | Put the physician on the *same* framework (continuous + flagging) | **Physician dominates the continuous model**: MAE 2.94 vs 3.82, R² 0.28 vs 0.04; and as a long-stay flag the physician (est≥7) reaches **AUROC 0.91** vs the model's 0.73. |
| 4 | `riley_scorecard.py` | Clean side-by-side scorecard figure | (figure only) |
| 5 | `riley_cap7_sensitivity.py` | Restrict the continuous model to 1 < LoS ≤ 7 (outcome-conditioned caveat noted) | Confirms the continuous model is competitive only where there are no long-stayers. |
| 6 | `riley_routing_model.py` | Mixture-of-experts: short expert (≤7) + long expert (>7) + **ML gate** P(LOS≥7), soft & hard | ML-gated routing makes overall MAE *worse* (≈3.9) — the AUROC-0.73 gate misroutes short stays into the long expert. |
| 7 | `riley_physician_gated.py` | Use the **physician estimate** (est≥K) as the gate instead of the ML classifier | Physician gate sharply improves long-stay MAE. |
| 8 | `riley_routing_sweep.py` | Sweep long-stay threshold K ∈ {5,7,10,14} × tool {ExtraTrees, Tweedie}, ML gate | **The tool doesn't matter** (ExtraTrees ≈ Tweedie). The gate AUROC (~0.69–0.73) is the bottleneck. |
| 9 | `riley_routing_physgate_sweep.py` | Same sweep, **physician gate vs ML gate** head-to-head | Physician hard-gate **K=5** is best: MAE 3.27, R² 0.31, MAE>7 6.48 — beats every ML-gate configuration. |
| 10 | `riley_mixed_expert_hybrid.py` + `chk_misroute.py` | log1p-ExtraTrees vs Tweedie as the *short* expert; confirm where the gate misroutes | Short-expert choice barely matters; `chk_misroute.py` shows hard gates misroute genuinely short stays whose physician estimate is ≥K. |
| 11 | `riley_soft_gate.py` | **Soft** physician gate w = σ((arzt − c)/s); sweep centre c × steepness s | Soft blending recovers the short-stay accuracy a hard gate loses. Best (in-sample) c=7, s=1.0. |
| 12 | `riley_soft_gate_cv.py` | Tune (c,s) **honestly** via nested 5-fold CV on the prospective cohort (the physician estimate exists only prospectively) | **Headline result** — see below. (c=7, s=1.0) chosen in 4/5 folds; optimism gap only +0.05 d. |
| 13 | `riley_architecture_diagram.py` | Documentation figure: which data / filter trains which function, and how it is applied prospectively | `figures/fig_architecture.png` |

## Headline result — soft physician-gated hybrid (`soft_gate_cv.csv`)

Nested-CV-honest numbers on the prospective cohort (n = 286, leak-free):

| Approach | MAE | R² | calib. slope | MAE (>7 d) | MAE 2–4 d | MAE 4–7 d |
|----------|----:|---:|-----:|-----:|-----:|-----:|
| **Soft gate (nested-CV, frozen c,s)** | **2.99** | **0.36** | 0.89 | **6.86** | 1.45 | 2.74 |
| Soft gate (optimistic best c=7,s=1.0) | 2.93 | 0.36 | 0.91 | 6.89 | 1.40 | 2.56 |
| Hard gate (arzt≥7) | 3.32 | 0.24 | 0.70 | 7.60 | 1.47 | 3.83 |
| Senior physician | 2.94 | 0.28 | 0.83 | 7.74 | 1.50 | 1.95 |

**Interpretation.** The honestly tuned soft physician gate **matches the senior physician
overall** (MAE 2.99 vs 2.94; difference not significant) while **beating** the physician on
calibration/explained variance (R² 0.36 vs 0.28) and on long-stay accuracy (MAE>7 6.86 vs
7.74). It is the only configuration in this exploration that achieves a genuine
"medicine + AI" synergy: the physician's estimate routes the regime, the leak-free ML
experts sharpen the magnitude. The optimism gap (nested-CV vs optimistic) is just +0.05 d.

## How to run
Scripts read the retrospective parquet and the rebuilt prospective matrix from
`Eigene Auswertung/canonical/…` (not in the repo — patient-level) and write outputs to
`Eigene Auswertung/exploratory_riley/`. They are listed in dependency-free order; each is
self-contained. Run from the project root with the KISIK Python environment.
