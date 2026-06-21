# -*- coding: utf-8 -*-
"""
Korrigiert Tabelle 9 (Null-Modell + Verteilung) gemaess Reviewer:
- Retrospektive Zeile = TATSAECHLICHER Holdout (n=2.601), nicht die Gesamtkohorte.
- Null-Modell = Median NUR aus dem Trainingsset, danach unveraendert auf Holdout + prospektiv.
- Reproduziert den exakten Split aus canonical_analysis.py (GroupShuffleSplit, seed 42, pid-gruppiert).
Schreibt: canonical/prospective_null_baseline.csv (mit Spalte train_median_used)
          und gibt Featurezahlen (gesamt / numerisch / kategorial) aus.
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"
FEAT=AN/"los_selected_features_ain_24h_compact.csv"
asql="('AIN','IZ32'),('AIN','IZ21'),('AIN','IZ31')"; RS=42
con=duckdb.connect()
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
y=(df["icu_duration_h"]/24.0).values
groups=df["pid"].fillna("unknown").astype(str).values
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(np.zeros((len(df),1)),y,groups))
ytr,yte=y[tr],y[te]
train_median=float(np.median(ytr))
print(f"n_train={len(tr)}  n_holdout={len(te)}  train_median={train_median:.4f}  holdout_median={np.median(yte):.4f}")

# Feature-Zahlen (identische Logik wie canonical_analysis.py)
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns]
present=[f for f in present if not f.startswith(("lab_","vital_","proc_","zugang_"))]
cat=[c for c in present if c=="oebenekurz"]; numc=[c for c in present if c not in cat]
print(f"FEATURES: total={len(present)}  categorical={len(cat)} ({cat})  numeric={len(numc)}")

# prospektive Matches
p=pd.read_csv(CAN/"metrics_prospective_fair24h_predictions.csv",sep=";")["los_obs"].values
def mae(x,c): return float(np.mean(np.abs(x-c)))
# Modell-MAE Holdout aus metrics_retrospective.csv (ExtraTrees, auf yte berechnet)
et_hold=float(pd.read_csv(CAN/"metrics_retrospective.csv",sep=";").set_index("Modell").loc["ExtraTrees","MAE_days"])
et_pros=float(pd.read_csv(CAN/"metrics_prospective_fair24h.csv",sep=";").set_index("Modell").loc["ExtraTrees","MAE"])

rows=[]
for nm,x,mdl in [("Retrospective holdout (AIN)",yte,et_hold),("Prospective (n=193)",p,et_pros)]:
    rows.append({"Cohort":nm,"n":len(x),"LoS_mean":round(x.mean(),2),"LoS_median":round(float(np.median(x)),2),
                 "LoS_std":round(x.std(),2),"LoS_max":round(float(x.max()),1),
                 "pct_gt7d":round(100*(x>7).mean(),1),"pct_gt14d":round(100*(x>14).mean(),1),
                 "Null_MAE_trainMed":round(mae(x,train_median),2),"Model_MAE":round(mdl,3),
                 "train_median_used":round(train_median,2)})
T=pd.DataFrame(rows); T.to_csv(CAN/"prospective_null_baseline.csv",sep=";",index=False)
print("\n=== Tabelle 9 (korrigiert: Holdout + Trainings-Median-Nullmodell) ===")
print(T.to_string(index=False))
