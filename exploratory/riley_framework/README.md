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
| 14 | `riley_quantile_longstay.py` | Does a quantile approach help long-stayers? Standalone P50–P90 + quantile as the long expert in the gate | **No for accuracy** — the recalibrated mean long expert already gives the best MAE>7 (6.89), better than any quantile. A high quantile only buys **coverage** (capacity planning): P80/P90 raise long-stay coverage 26 %→43–53 % at the cost of MAE. |
| 15 | `riley_midband_4_7.py` | Can the 4–7 d band be improved? 3-component blend with a mid component (physician number or a 3–7 d mid-expert), nested-CV | The physician is genuinely best at 4–7 d (1.95); ML can only approach it at a net cost. A mid-expert (3–7 d) gives a tiny Pareto gain (2.56→2.48). **Bonus:** mixing the physician *number* (not just as gate) yields the best overall hybrid. |
| 16 | `riley_3comp_full.py` | Full evaluation of the best variant — 3-component blend, MID = physician number, nested-CV — with significance testing vs the physician (paired bootstrap B=5000 + Wilcoxon) and hexbins | **Best hybrid.** See below. |
| 17 | `riley_hexbins_prospective.py` | Three standalone prospective hexbins (predicted vs *actual* LoS): best ML model alone, physician alone, best hybrid | `figures/fig_hexbin_pros_{ml_alone,physician,hybrid}.png` |

### Later steps — best standalone model & the manuscript's hybrid choice

| # | Script | What it tests | Result |
|---|--------|---------------|--------|
| 18 | `riley_clean_ceiling.py` | **Leak-free retrospective information ceiling** — every legitimate 24-h feature (all `lab24_/vital24_/proc24_/zugang24_/diag_` + demographics + genuine 24-h SAPS II / TISS-28 / SOFA), only true leaks removed (8-98f family, whole-stay aggregates, leaky window-scores) | The best leak-free retro model reaches only **C-index ≈ 0.71** (MAE 3.21) — still **below the physician's prospective 0.766**. No clean structured-data model beats the clinician; the earlier 0.83–0.84 "above-physician" figures were all leaks. |
| 19 | `riley_scores_consistent.py` | **Best deployable standalone** — add *train/serve-consistent* genuine 24-h severity scores (SAPS II + TISS-28 within `[planbegin, +24 h)` retro; `score_all.csv` prospectively) to the deployment-aware model | **Current best standalone ML.** Scores lift it from C-index 0.602 → **0.680**, Spearman 0.30 → **0.52**, extreme-AUROC (≤3 vs >10 d) 0.69 → **0.86**, MAE 3.75 → **3.40 d** — the single real feature lever — but it still trails the physician (0.766 / 0.72). |
| 20 | `riley_hybrid_longonly.py`, `riley_best_hybrid_and_fio2.py` | The **parsimonious long-only** hybrid selected for the manuscript: `pred = (1−p)·â + p·L̂(x)`, `p = σ((â−7)/1)` — one ML expert (retro > 7 d), one threshold. FiO₂ tested as a long-expert feature | **Manuscript hybrid.** Honest OOF prospective **MAE 2.86, R² 0.40, slope 0.87**; > 7 d 6.18 vs physician 7.74 (**ΔMAE +1.56 [0.84, 2.31], p < 0.001**); overall non-inferior (+0.08 [−0.15, +0.32], p = 0.54). Chosen over the 3-component (equivalent overall but degrades very-short stays). FiO₂ adds nothing (poorly populated, no honest gain) and is dropped. |
| 21 | `riley_hybrid_scores.py` | Add the genuine 24-h scores to the hybrid's **long** expert | Hybrid-long with scores: **MAE 2.88, C-index 0.766 (= physician)**, > 7 d ΔMAE +1.43 [0.67, 2.16] p = 0.0002; overall still n.s. (p = 0.65). Scores must feed the *long* expert — a general score-ML hybrid is significantly *worse* on > 7 d (it compresses). |
| 22 | (analysis in `riley_best_hybrid_and_fio2.py` / `riley_hybrid_longonly.py`) | Is the hybrid's long expert really "ML"? Replace L̂ with a fixed retro long-stay **mean / median** anchor | The long expert is effectively **constant** in the tail (prospective mean 14.1 d, SD 1.3; within-long ranking ≈ chance). A fixed population anchor reproduces the hybrid (MAE 2.84). **The hybrid's gain is a bias-correction of the physician's anchor by a population constant, not an ML discrimination signal** — the physician supplies discrimination, the constant supplies level. |

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

## Best variant — 3-component blend with the physician *number* (`3comp_metrics.csv`, `3comp_superiority.csv`)

Steps 14–16 push further: instead of using the physician estimate only as a *gate* between two
ML experts, the best variant **blends the physician's number itself** as a third (mid-band)
component: `pred = p_short·SHORT + p_mid·ARZT + p_long·LONG`, with the regime weights derived
from the physician estimate (c_lo, s tuned by nested CV; (c_lo=3, s=1.0) chosen in all 5 folds).
Nested-CV-honest prospective numbers (n = 286, leak-free):

| Model | MAE | R² | slope | 1–2 d | 2–4 d | 4–7 d | >7 d |
|-------|----:|---:|-----:|-----:|-----:|-----:|-----:|
| **3-component hybrid (MID = physician)** | **2.89** | **0.40** | 0.90 | 1.42 | 1.72 | 2.69 | **6.24** |
| Soft 2-component gate | 2.93 | 0.36 | 0.89 | 1.41 | 1.40 | 2.56 | 6.89 |
| Senior physician | 2.94 | 0.28 | 0.83 | 0.97 | 1.50 | 1.95 | 7.74 |

**Significance vs the physician** (paired bootstrap B=5000 ΔMAE = MAE_phys − MAE_model + Wilcoxon):

| Subgroup | n | ΔMAE [95% CI] | Wilcoxon p | verdict |
|----------|--:|---------------|----:|---------|
| overall | 286 | +0.05 [−0.18, +0.30] | 0.18 | **n.s. (equivalent)** |
| 1–2 d | 86 | −0.45 [−0.65, −0.20] | <0.0001 | physician better |
| 2–4 d | 84 | −0.22 [−0.48, +0.01] | 0.42 | n.s. |
| 4–7 d | 46 | −0.74 [−1.26, −0.24] | 0.33 | physician better* |
| **>7 d** | 70 | **+1.50 [+0.77, +2.25]** | **0.0001** | **hybrid better** |

\* At 4–7 d the mean-based bootstrap CI favours the physician, but the rank-based Wilcoxon test is
non-significant (p=0.33) — the gap is driven by a few large errors, not a consistent case-by-case
advantage. The hybrid is **statistically equivalent to the physician overall**, **significantly
better on long-stayers (>7 d)** — the capacity-planning–relevant group — and better calibrated
(R² 0.40 vs 0.28). The physician retains a significant edge only on short stays (1–2 d). The
prospective hexbins (`figures/fig_hexbin_pros_*.png`) show the ML model alone collapsing to a
vertical band (R² 0.01), the physician spreading along the diagonal (R² 0.28), and the hybrid
tightest of all (R² 0.40).

## Current best standalone ML — genuine 24-h severity scores (`riley_scores_consistent.py`)

The best *standalone* model (no physician input) is the deployment-aware 24-h Extra Trees
regressor **augmented with train/serve-consistent 24-h severity scores** (SAPS II + TISS-28,
computed within `[planbegin, +24 h)` retrospectively and taken from `score_all.csv`
prospectively). Prospective cohort (n ≈ 276–286, leak-free):

| Standalone model | C-index | Spearman | MAE | extreme-AUROC (≤3 vs >10 d) |
|------------------|:------:|:-------:|:---:|:--------------------------:|
| Deployment-aware 24-h (no scores) | 0.602 | 0.30 | 3.75 | 0.69 |
| **+ genuine 24-h scores (best standalone)** | **0.680** | **0.52** | **3.40** | **0.86** |
| Leak-free retro information ceiling (`riley_clean_ceiling.py`) | ~0.71 | — | 3.21 | — |
| Senior physician | **0.766** | **0.72** | **2.94** | 0.96 |

**Interpretation.** Adding the genuine 24-h scores is the single real feature lever — it roughly
halves the ranking gap to the physician — but even the leak-free retrospective *ceiling* (every
legitimate 24-h feature) lands at C-index ≈ 0.71, still short of the clinician's 0.766. Every
earlier "standalone beats the physician" number (C-index 0.83–0.84) was a leak (the 8-98f
intensive-care-days family, or window-aggregated score columns). **Structured 24-h data does not
beat the clinician's gestalt** — which is exactly why the value is combinatorial (the hybrid), not
standalone.

## Manuscript decision — the parsimonious long-only physician-gated hybrid

Among all explored variants (soft 2-component gate, 3-component physician-number blend, hard gate),
the manuscript keeps the **single long-only** hybrid for parsimony and robustness:

> `pred = (1 − p)·â + p·L̂(x)`,  `p = σ((â − 7) / 1)`  — one ML long-stay expert (retro > 7 d), one
> physician-driven threshold.

Honest OOF prospective numbers (n = 286, leak-free):

| Model | MAE | R² | slope | > 7 d MAE | ΔMAE vs physician (overall) | ΔMAE vs physician (> 7 d) |
|-------|----:|---:|-----:|-----:|-----|-----|
| **Long-only hybrid (manuscript)** | **2.86** | **0.40** | 0.87 | **6.18** | +0.08 [−0.15, +0.32] p = 0.54 (non-inferior) | **+1.56 [0.84, 2.31] p < 0.001** |
| 3-component (MID = physician) | 2.89 | 0.40 | 0.90 | 6.24 | +0.05 n.s. | +1.50 [0.77, 2.25] p = 0.0001 |
| + genuine 24-h scores in long expert | 2.88 | — | — | — | +0.06 n.s. (p = 0.65); **C-index 0.766 = physician** | +1.43 [0.67, 2.16] p = 0.0002 |
| Senior physician | 2.94 | 0.28 | 0.83 | 7.74 | — | — |

Two honest caveats carried into the manuscript:

- **The long expert is effectively a constant.** In the tail L̂ collapses to ≈ 14 d (SD 1.3;
  within-long ranking ≈ chance), and replacing it with a fixed retro long-stay mean/median anchor
  reproduces the hybrid (MAE 2.84). The hybrid's benefit is therefore a **bias-correction of the
  physician's long-stay under-estimation by a population constant**, not an ML discrimination signal
  — transparent and robust, but not "AI cleverness".
- **Overall superiority is power-limited, not model-limited.** The overall ΔMAE (+0.08) would need
  n ≈ 5,000+ for 80 % power; the effect lives in the > 7 d subgroup, which is already significant at
  n = 70. The manuscript pre-specifies the **> 7 d long-stayers as the primary endpoint** (the
  capacity-planning–relevant group) and overall non-inferiority as secondary.

## How to run
Scripts read the retrospective parquet and the rebuilt prospective matrix from
`Eigene Auswertung/canonical/…` (not in the repo — patient-level) and write outputs to
`Eigene Auswertung/exploratory_riley/`. The score-based steps (19, 21) additionally read the
retrospective `kisik2/score.csv` and the prospective `…/OLD/Entlassdaten/score_all.csv` (genuine
24-h SAPS II / TISS-28). They are listed in dependency-free order; each is self-contained. Run from
the project root with the KISIK Python environment.

Only aggregate metric CSVs are committed for steps 1–17; the later steps' outputs
(`clean_ceiling*.csv`, `scores_consistent*.csv`, `hybrid_longonly.csv`, `best_hybrid.csv`,
`fio2_test.csv`, `hybrid_scores*.csv`) live in the analysis workspace and are not committed
(same data-safety policy).
