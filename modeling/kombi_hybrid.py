# -*- coding: utf-8 -*-
"""
KOMBI hybrid: recalibrated-physician + long-stay ML expert, physician-gated.
================================================================================

Final prospective model. It combines the senior-physician LOS estimate with a leakage-free
retrospective ML expert, so that each contributes where it is strongest:

    LOS_hat(x, a) = (1 - p) * recal(a)  +  p * L(x),      p = sigma((a - c) / s)

where ``a`` is the senior-physician estimate (days), ``x`` the first-24h features, and the three
components are:

1. recal(a) — RECALIBRATED PHYSICIAN.  KNN-median of the observed LOS in the neighbourhood of the
   physician estimate ``a`` (k = 35). This removes the clinician's systematic short-stay
   over-estimation while preserving the physician's (strong) ranking. It uses ONLY the physician
   estimate and observed LOS — no ML features, no imputation. Fit per cross-validation fold on the
   training fold and applied to the held-out fold.

2. L(x) — LONG-STAY ML EXPERT.  ExtraTrees (log1p target + linear recalibration of the out-of-fold
   predictions) trained on the retrospective cohort restricted to LOS > 7 days, then FROZEN. It runs
   on an ENRICHED, deployable 24h feature set: the leakage-free base features + rebuilt 24h vitals
   (HR/RR/GCS/temp/ABP/SpO2/FiO2/PEEP) + peri-admission OP context (surgery/anaesthesia/planned
   duration, admission type, referring specialty), restricted to features with >= 50 % prospective
   coverage, and MISSING VALUES IMPUTED BY KNN (k = 10) fit on the retrospective data. The expert
   never sees the prospective labels.

3. gate p = sigma((a - c) / s) — PHYSICIAN GATE.  For a short physician estimate the prediction is
   essentially recal(a); as ``a`` grows past the gate centre ``c`` the long-stay expert L(x) is
   blended in. This routes the ML expert to exactly the long-stay region where it adds value while
   the recalibrated physician anchors the (majority) short/mid stays.

GATE SELECTION — NESTED, NOT IN-SAMPLE.  The gate parameters (c, s) are tuned INSIDE a
cross-validation loop: for every outer fold they are chosen on the TRAINING fold only (minimising a
pre-specified objective — MAE or RMSE — of the blended prediction), then applied to the held-out
fold. The reported metrics are therefore honest out-of-fold estimates in which the gate never saw
the case it predicts. Repeating over several seeds quantifies fold-split variance.

KEY RESULT (prospective n = 286, nested CV, RMSE-objective gate, mean over 20 seeds): the hybrid is
statistically EQUIVALENT to the senior physician on overall MAE (2.96 vs 2.94 d) and ranking
(C-index 0.762 vs 0.766) but SUPERIOR on calibration and the long-stay tail (R^2 0.405 vs 0.276;
RMSE 4.93 vs 5.44 d; MAE for LOS > 7 d: 6.16 vs 7.74 d). The long-stay advantage is robust — it is
also significant for the simpler canonical gate.

NOTE: the recalibration and gate use the physician estimate, which exists only in the prospective
cohort; the hybrid therefore has no retrospective counterpart. The ML expert (component 2) is trained
purely retrospectively. Hard-wired local paths -> adapt CONFIG. No patient data included.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.impute import KNNImputer
from sklearn import config_context
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------ CONFIG ----
RS = 42
KNN_K = 10            # feature imputation neighbours
RECAL_K = 35          # KNN-median physician recalibration neighbours
LONG_THRESHOLD = 7    # retro LOS (days) above which the long-stay expert is trained
COV_MIN = 0.50        # keep features with >= 50 % prospective coverage
GATE_C = [round(c, 2) for c in np.arange(2, 12.01, 0.5)]     # gate centre grid (days)
GATE_S = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]           # gate steepness grid
GATE_OBJECTIVE = "RMSE"   # pre-specified a priori: "MAE" (parsimonious) or "RMSE" (tail/variance)
N_SEEDS = 20


def _et_pipeline(bpet: dict) -> Pipeline:
    """KNN-imputed, log1p-target ExtraTrees — the long-stay expert's estimator."""
    return Pipeline([
        ("imp", KNNImputer(n_neighbors=KNN_K)),
        ("mdl", TransformedTargetRegressor(
            ExtraTreesRegressor(**bpet, random_state=RS, n_jobs=4),
            func=np.log1p, inverse_func=np.expm1)),
    ])


def fit_long_expert(Xr, yr, groups_r, Xp, bpet):
    """Train the frozen long-stay ML expert on retro LOS>7d; return its recalibrated prospective
    prediction L(x) for every prospective case. Xr/Xp already restricted to >=50 %-coverage features
    (NaN preserved; KNN imputes inside the pipeline)."""
    long = (yr > LONG_THRESHOLD).nonzero()[0]
    with config_context(working_memory=128):     # chunk KNN distances -> bounded memory
        est = _et_pipeline(bpet).fit(Xr[long], yr[long])
        oof = np.clip(cross_val_predict(_et_pipeline(bpet), Xr[long], yr[long],
                                        groups=groups_r[long], cv=GroupKFold(3), n_jobs=1), 0, None)
        b, a = np.polyfit(oof, yr[long], 1)       # linear recalibration of OOF predictions
        return np.clip(a + b * np.clip(est.predict(Xp), 0, None), 0, None)


def _knn_median(a_train, los_train, a_query, k=RECAL_K):
    out = np.empty(len(a_query))
    for i, v in enumerate(a_query):
        out[i] = np.median(los_train[np.argsort(np.abs(a_train - v))[:k]])
    return out


def _objective(name, y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) if name == "MAE" else mean_squared_error(y_true, y_pred) ** 0.5


def nested_kombi(arzt, los, LONG, objective=GATE_OBJECTIVE, seeds=N_SEEDS):
    """Honest out-of-fold hybrid prediction with the gate tuned per fold on the training fold only.
    Returns (mean_metrics, std_metrics, modal_gate) over ``seeds`` repeats of 5-fold CV."""
    N = len(los); res = []; gates = []
    for sd in range(seeds):
        pred = np.full(N, np.nan)
        for trn, tst in KFold(5, shuffle=True, random_state=sd).split(np.arange(N)):
            rec_trn = np.clip(_knn_median(arzt[trn], los[trn], arzt[trn]), 0, None)   # tuning only
            rec_tst = np.clip(_knn_median(arzt[trn], los[trn], arzt[tst]), 0, None)   # applied to test
            best = None
            for c in GATE_C:
                for s in GATE_S:
                    p = 1 / (1 + np.exp(-(arzt[trn] - c) / s))
                    bl = np.clip((1 - p) * rec_trn + p * LONG[trn], 0, None)
                    v = _objective(objective, los[trn], bl)
                    if best is None or v < best[0]:
                        best = (v, c, s)
            _, c, s = best; gates.append((c, s))
            p = 1 / (1 + np.exp(-(arzt[tst] - c) / s))
            pred[tst] = np.clip((1 - p) * rec_tst + p * LONG[tst], 0, None)
        res.append(_all_metrics(los, pred))
    R = np.array(res)
    return R.mean(0), R.std(0), Counter(gates).most_common(1)[0]


def _cindex(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); comp = y[:, None] < y[None, :]; d = comp.sum()
    return float(((p[:, None] < p[None, :]) & comp).sum() + 0.5 * ((p[:, None] == p[None, :]) & comp).sum()) / d if d else 0.5


def _all_metrics(y, pred):
    m7 = y > LONG_THRESHOLD
    return [mean_absolute_error(y, pred), mean_squared_error(y, pred) ** 0.5,
            r2_score(y, pred), _cindex(y, pred), np.abs(y[m7] - pred[m7]).mean()]


if __name__ == "__main__":
    # Expects pre-built enriched matrices (see pipeline/build_24h_measurement_features.py +
    # build_perioperative_features.py) restricted to >=50 %-coverage features, plus the physician
    # estimate and observed LOS for the prospective cohort. Wire up the loaders for your environment.
    print(__doc__.split("KEY RESULT")[0])
