# -*- coding: utf-8 -*-
"""
Experiment: Verbessern OP-/Anaesthesie-Features (op_an, op_zeitintervalle)
die LoS-Vorhersage, insbesondere fuer Langlieger (>7 Tage)?

Vergleich:
  A) XGBoost Baseline  (24h-compact-Features)
  B) XGBoost + OP-Features
  C) XGBoost + OP-Features + asymmetrischer Loss (Unterschaetzung 3x bestraft)

Alle perioperativen OP-Daten nur aus dem Fenster [planbegin-1d, planbegin+24h]
-> 24h-konform, kein Leakage durch spaetere Re-Operationen.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE   = Path(r"D:\Ausgangsdaten\KISIK Projekt")
PQ     = BASE / "kisik2" / "kisik2_icu_ml_dataset_24h.parquet"
OPAN   = BASE / "kisik2" / "op_an.csv"
OPZ    = BASE / "kisik2" / "op_zeitintervalle.csv"
FEAT   = BASE / "Eigene Auswertung" / "los_selected_features_ain_24h_compact.csv"
OUT    = BASE / "Eigene Auswertung" / "los_op_feature_experiment_results.csv"
RS = 42

allowed = [("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31"),("AIN","IZ01"),
           ("AUG","IZ01"),("AVT","IZ01"),("GCH","IZ01"),("GYN","IZ01"),
           ("HNO","IZ01"),("HTC","IZ01"),("IZPV","IZ01"),("MKG","IZ01"),
           ("NCH","IZ01"),("NUK","IZ01"),("STR","IZ01"),("UCH","IZ01"),("URO","IZ01")]
allowed_sql = ", ".join(f"('{w}','{o}')" for w,o in allowed)

con = duckdb.connect()
def rc(p): return f"read_csv_auto('{p.as_posix()}', delim=';', header=true, all_varchar=true, ignore_errors=true)"

# ---------------------------------------------------------------- Kohorte
print("Lade Kohorte ...")
df = con.execute(f"""
    SELECT * FROM read_parquet('{PQ.as_posix()}')
    WHERE (wardshort, oebenekurz) IN ({allowed_sql})
      AND icu_duration_h/24.0 > 1
""").df()
df["planbegin"] = pd.to_datetime(df["planbegin"], errors="coerce")
df["los_days"]  = df["icu_duration_h"] / 24.0
print(f"  {len(df):,} Stays")

# ---------------------------------------------------------------- OP-Features
print("Berechne OP-Features (perioperatives Fenster) ...")
con.register("stays_df", df[["stay_id","fallid","planbegin"]])
con.execute("CREATE TEMP TABLE s AS SELECT stay_id, fallid, planbegin AS pb FROM stays_df")

# dauer kann 'HH:MM:SS' oder '1 day 05:23:00' sein -> robust ueber INTERVAL
min2num = "COALESCE(EXTRACT(EPOCH FROM TRY_CAST(z.dauer AS INTERVAL))/60.0, 0)"

# aus op_zeitintervalle (dedupliziert, gefenstert)
opz = con.execute(f"""
WITH z AS (
  SELECT DISTINCT fallid, zeitintbez, beginn, ende, dauer FROM {rc(OPZ)}
),
zw AS (
  SELECT s.stay_id, z.zeitintbez,
         {min2num} AS min
  FROM s JOIN z ON z.fallid = s.fallid
  WHERE TRY_CAST(z.beginn AS TIMESTAMP) BETWEEN s.pb - INTERVAL 1 DAY AND s.pb + INTERVAL 24 HOURS
)
SELECT stay_id,
  SUM(CASE WHEN zeitintbez='Schnitt-Naht'        THEN min ELSE 0 END) AS op_schnittnaht_min,
  SUM(CASE WHEN zeitintbez='Reine Anästhesiezeit' THEN min ELSE 0 END) AS op_anaesth_min,
  SUM(CASE WHEN zeitintbez='Anästhesiepräsenz'    THEN min ELSE 0 END) AS op_anaesth_praesenz_min,
  SUM(CASE WHEN zeitintbez='HLM'                 THEN min ELSE 0 END) AS op_hlm_min,
  MAX(CASE WHEN zeitintbez='HLM'                 THEN 1 ELSE 0 END) AS op_hlm_flag,
  COUNT(*) AS op_n_intervalle
FROM zw GROUP BY stay_id
""").df()

# aus op_an (ASA, geplante Dauer, Anzahl Eingriffe; gefenstert ueber opplandatum)
opan = con.execute(f"""
WITH a AS (
  SELECT DISTINCT fallid, eingrbez, asaid, opplandatum, indivdauer FROM {rc(OPAN)}
),
aw AS (
  SELECT s.stay_id,
         TRY_CAST(a.asaid AS INT) AS asa,
         TRY_CAST(a.indivdauer AS DOUBLE) AS plan_min
  FROM s JOIN a ON a.fallid = s.fallid
  WHERE TRY_CAST(a.opplandatum AS TIMESTAMP) BETWEEN s.pb - INTERVAL 1 DAY AND s.pb + INTERVAL 24 HOURS
)
SELECT stay_id,
  MAX(asa) AS op_asa_max,
  SUM(plan_min) AS op_plandauer_sum,
  COUNT(*) AS op_n_eingriffe
FROM aw GROUP BY stay_id
""").df()

op = pd.merge(opz, opan, on="stay_id", how="outer")
op_cols = [c for c in op.columns if c != "stay_id"]
df = df.merge(op, on="stay_id", how="left")
# op_any-Flag + fehlende Werte fuellen
df["op_any"] = df["op_n_intervalle"].notna().astype(int) | df["op_n_eingriffe"].notna().astype(int)
df["op_any"] = df["op_any"].astype(int)
for c in op_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
op_cols = op_cols + ["op_any"]

n_with_op = int((df["op_any"] == 1).sum())
print(f"  OP-Features fuer {n_with_op:,} / {len(df):,} Stays ({round(100*n_with_op/len(df),1)}%)")
print(f"  davon HLM/Bypass (Kardio): {int(df['op_hlm_flag'].sum()):,}")

# ---------------------------------------------------------------- Feature-Matrix
feat = pd.read_csv(FEAT, sep=";")["Feature"].tolist()
present = [f for f in feat if f in df.columns]
cat_cols = [c for c in present if c == "oebenekurz"]
num_base = [c for c in present if c not in cat_cols]

def build_X(use_op):
    parts = [df[num_base].apply(pd.to_numeric, errors="coerce")]
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols).astype(float))
    if use_op:
        parts.append(df[op_cols].apply(pd.to_numeric, errors="coerce"))
    X = pd.concat(parts, axis=1)
    X.columns = [str(c) for c in X.columns]
    # pandas-NA -> echtes float64 (NaN), das XGBoost nativ versteht
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.astype(np.float64)

y = np.log1p(df["los_days"].clip(lower=0).values)
groups = df["pid"].fillna("unknown").astype(str).values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RS)
tr_idx, te_idx = next(gss.split(df, y, groups))
print(f"  Train {len(tr_idx):,} | Test {len(te_idx):,}")
los_test = df["los_days"].values[te_idx]

# ---------------------------------------------------------------- Metriken
def metrics(y_true_days, y_pred_days, label, subset=None):
    if subset is not None:
        y_true_days, y_pred_days = y_true_days[subset], y_pred_days[subset]
    ae = np.abs(y_true_days - y_pred_days)
    return {"Modell": label, "n": len(y_true_days),
            "MAE": round(float(ae.mean()),3),
            "Median_AE": round(float(np.median(ae)),3),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true_days, y_pred_days))),3),
            "R2": round(float(r2_score(y_true_days, y_pred_days)),3),
            "Bias": round(float((y_pred_days - y_true_days).mean()),3)}

XGB = dict(n_estimators=600, max_depth=6, learning_rate=0.05,
           subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
           random_state=RS, n_jobs=-1)

def asym_obj(alpha=3.0):
    # sklearn-XGBRegressor ruft obj(y_true, y_pred) mit numpy-Arrays auf
    def obj(y_true, y_pred):
        resid = y_pred - y_true
        w = np.where(resid < 0, alpha, 1.0)   # Unterschaetzung staerker bestrafen
        grad = 2.0 * w * resid
        hess = 2.0 * w
        return grad, hess
    return obj

def run_model(use_op, asym=False, label=""):
    X = build_X(use_op)
    Xtr, Xte = X.values[tr_idx], X.values[te_idx]
    ytr = y[tr_idx]
    if asym:
        m = XGBRegressor(objective=asym_obj(3.0), **XGB)
    else:
        m = XGBRegressor(objective="reg:squarederror", **XGB)
    m.fit(Xtr, ytr)
    pred = np.expm1(m.predict(Xte))
    pred = np.clip(pred, 0, None)
    rows = [metrics(los_test, pred, f"{label} | gesamt")]
    for name, mask in [(">1d", los_test>1), ("1-7d", (los_test>1)&(los_test<=7)),
                       (">7d", los_test>7), (">14d", los_test>14)]:
        if mask.sum() > 5:
            r = metrics(los_test, pred, f"{label} | {name}", subset=mask)
            rows.append(r)
    return rows, X.shape[1]

print("\n" + "="*70)
print("ERGEBNISSE  (Holdout-Testset, Tage)")
print("="*70)
all_rows = []
for use_op, asym, label in [(False,False,"A_Baseline"),
                            (True, False,"B_plusOP"),
                            (True, True, "C_plusOP_asym")]:
    rows, ncols = run_model(use_op, asym, label)
    all_rows += rows
    print(f"\n--- {label}  ({ncols} Features) ---")
    print(pd.DataFrame(rows)[["Modell","n","MAE","Median_AE","RMSE","R2","Bias"]].to_string(index=False))

res = pd.DataFrame(all_rows)
res.to_csv(OUT, sep=";", index=False)
print(f"\nGespeichert: {OUT}")

# ---------------------------------------------------------------- Feature-Importance OP
print("\n" + "="*70)
print("OP-Feature-Wichtigkeit (Modell B)")
print("="*70)
X = build_X(True)
m = XGBRegressor(objective="reg:squarederror", **XGB)
m.fit(X.values[tr_idx], y[tr_idx])
imp = pd.Series(m.feature_importances_, index=X.columns)
print(imp[[c for c in op_cols if c in imp.index]].sort_values(ascending=False).round(4).to_string())
