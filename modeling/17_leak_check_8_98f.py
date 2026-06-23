# -*- coding: utf-8 -*-
"""LEAK-CHECK fuer die OPS-Familie 8-98f (intensivmed. Komplexbehandlung).
Verdacht: die OPS-Suffixe (.0/.10/.../.60) kodieren Behandlungstage-Baender -> wenn im
ersten-24h-Fenster vorhanden, koennten sie die finale LoS vorwegnehmen (Leakage).
Tests: (1) bedingte LoS pro Suffix (monotone Treppe = verdaechtig),
       (2) Ablation ExtraTrees mit vs ohne 8-98f (und 8-98-Familie),
       (3) Modell NUR mit 8-98f-Features (wieviel Signal allein?)."""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"
FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"
bp=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["ExtraTrees"]

con=duckdb.connect()
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>2").df()
df["los_days"]=df["icu_duration_h"]/24.0
y=df["los_days"].values; groups=df["pid"].fillna("unknown").astype(str).values
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns and not f.startswith(("lab_","vital_","proc_","zugang_"))]
f98=[c for c in present if c.startswith("proc24_8_98f")]
fam98=[c for c in present if c.startswith("proc24_8_98")]
print(f"n={len(df)} Stays | {len(present)} Modell-Features | davon 8-98f: {f98} | ganze 8-98-Familie: {len(fam98)}")

# ---------- (1) bedingte LoS pro 8-98f-Suffix ----------
print("\n=== (1) Bedingte LoS, wenn 8-98f-Code im ERSTEN 24h vorhanden ===")
print(f"{'Code (OPS 8-98f.x)':<22}{'n':>6}{'Praevalenz':>11}{'Median LoS':>12}{'IQR':>16}{'Mittel':>9}")
allcols=[c for c in df.columns if c.startswith("proc24_8_98f")]
rows=[]
for c in sorted(allcols, key=lambda s:int(s.split("_")[-1])):
    v=pd.to_numeric(df[c],errors="coerce").fillna(0).values>0
    if v.sum()<5: continue
    lo=y[v]; suf=c.replace("proc24_8_98f_","8-98f.")
    q1,q3=np.percentile(lo,[25,75])
    print(f"{suf:<22}{int(v.sum()):>6}{100*v.mean():>10.1f}%{np.median(lo):>11.1f}d{f'[{q1:.0f}-{q3:.0f}]':>16}{lo.mean():>8.1f}d")
    rows.append({"code":suf,"n":int(v.sum()),"median_los":round(float(np.median(lo)),1),"mean_los":round(float(lo.mean()),1)})
los_base=np.median(y); print(f"{'(Gesamtkohorte)':<22}{len(y):>6}{'100.0%':>11}{los_base:>11.1f}d")
pd.DataFrame(rows).to_csv(AN/"leak_8_98f_conditional_los.csv",sep=";",index=False)

# ---------- Pipeline ----------
cat=[c for c in present if c=="oebenekurz"]
def model_for(cols):
    numc=[c for c in cols if c not in cat]; thiscat=[c for c in cat if c in cols]
    pre=ColumnTransformer([("num",SimpleImputer(strategy="median"),numc)]+
        ([("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),thiscat)] if thiscat else []))
    return TransformedTargetRegressor(Pipeline([("pre",pre),("mdl",ExtraTreesRegressor(**bp,random_state=RS,n_jobs=-1))]),func=np.log1p,inverse_func=np.expm1)
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(df,y,groups))
def Xf(cols):
    X=df.reindex(columns=cols).copy()
    for c in cols:
        if c!="oebenekurz": X[c]=pd.to_numeric(X[c],errors="coerce")
        else: X[c]=X[c].astype(str)
    return X
def evalu(cols,label):
    m=model_for(cols); X=Xf(cols); m.fit(X.iloc[tr],y[tr]); p=np.clip(m.predict(X.iloc[te]),0,None)
    r2=r2_score(y[te],p); mae=mean_absolute_error(y[te],p)
    print(f"  {label:<46} R²={r2:.3f}  MAE={mae:.2f} d  ({len(cols)} Features)")
    return r2,mae

# ---------- (2) Ablation ----------
print("\n=== (2) Ablation (gleicher patient-grouped Holdout, ExtraTrees) ===")
full=evalu(present,"ALLE Features (Referenz)")
no_f=evalu([c for c in present if c not in f98],"OHNE 8-98f.* (.0/.10/...)")
no_fam=evalu([c for c in present if c not in fam98],"OHNE ganze 8-98-Familie (980/987/98e/98f/98g)")
only_f=evalu(f98 if f98 else ["alter"],"NUR 8-98f.*-Features")
print(f"\nΔ durch 8-98f.*:   R² {full[0]:.3f} -> {no_f[0]:.3f} (Δ {no_f[0]-full[0]:+.3f}) | MAE {full[1]:.2f} -> {no_f[1]:.2f} (Δ {no_f[1]-full[1]:+.2f} d)")
print(f"8-98f.* allein:    R² {only_f[0]:.3f}  MAE {only_f[1]:.2f} d  (vs Gesamtkohorte-Median-Baseline)")
base_mae=mean_absolute_error(y[te],np.full(len(te),np.median(y[tr])))
print(f"Median-Baseline:   MAE {base_mae:.2f} d, R²=0")
