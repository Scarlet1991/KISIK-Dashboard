# -*- coding: utf-8 -*-
"""Schlaegt das LECKFREIE Modell (8-98f entfernt) eine Konstant-/Mittelwertvorhersage?
Vergleich gegen Null = Trainings-Mittelwert und Trainings-Median (MAE-optimal),
retrospektiv (Hold-out n=2601) und prospektiv (no_isopen n=286), gepaarter Bootstrap."""
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
from sklearn.metrics import r2_score

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; B=5000; rng=np.random.default_rng(RS)
bp=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["ExtraTrees"]

con=duckdb.connect()
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
y=(df["icu_duration_h"]/24.0).values; groups=df["pid"].fillna("unknown").astype(str).values
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns and not f.startswith(("lab_","vital_","proc_","zugang_"))]
present_lf=[c for c in present if not c.startswith("proc24_8_98f")]
cat=[c for c in present_lf if c=="oebenekurz"]; numc=[c for c in present_lf if c not in cat]
pre=ColumnTransformer([("num",SimpleImputer(strategy="median"),numc)]+
    ([("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat)] if cat else []))
mdl=TransformedTargetRegressor(Pipeline([("pre",pre),("mdl",ExtraTreesRegressor(**bp,random_state=RS,n_jobs=-1))]),func=np.log1p,inverse_func=np.expm1)
X=df.reindex(columns=present_lf).copy()
for c in present_lf: X[c]=(X[c].astype(str) if c=="oebenekurz" else pd.to_numeric(X[c],errors="coerce"))
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(X,y,groups))
mdl.fit(X.iloc[tr],y[tr])
TRAIN_MEAN=float(np.mean(y[tr])); TRAIN_MED=float(np.median(y[tr]))
print(f"Trainings-Mittelwert={TRAIN_MEAN:.2f} d | Trainings-Median={TRAIN_MED:.2f} d  (leckfrei, {len(present_lf)} Features)")

def boot(yt,err_model,err_null):
    n=len(yt); idx=rng.integers(0,n,size=(B,n))
    d=err_null[idx].mean(1)-err_model[idx].mean(1)   # >0 => Modell besser als Null
    return float(err_null.mean()-err_model.mean()), tuple(np.percentile(d,[2.5,97.5]))
def compare(yt,pred,label):
    pm=np.clip(pred,0,None); em=np.abs(yt-pm)
    print(f"\n--- {label} (n={len(yt)}) ---")
    print(f"  Modell:        MAE {em.mean():.2f} d | R²={r2_score(yt,pm):+.3f}")
    for nm,c in [("Mittelwert",TRAIN_MEAN),("Median",TRAIN_MED)]:
        en=np.abs(yt-c); d,(lo,hi)=boot(yt,em,en)
        verdict="Modell BESSER" if lo>0 else ("Null BESSER" if hi<0 else "n.s. (gleichwertig)")
        print(f"  vs Null={nm} ({c:.2f} d): Null-MAE {en.mean():.2f} | ΔMAE(Null-Modell) {d:+.2f} [{lo:+.2f},{hi:+.2f}] -> {verdict}")

# ---- retrospektiv Hold-out ----
compare(y[te], mdl.predict(X.iloc[te]), "RETROSPEKTIV Hold-out (leckfrei)")

# ---- prospektiv no_isopen ----
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float)
Xp=pd.DataFrame(index=PR.index)
for c in present_lf: Xp[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
pred_p=mdl.predict(Xp[present_lf])
compare(los, pred_p, "PROSPEKTIV no_isopen (leckfrei)")
# zum Vergleich: Arzt
ea=np.abs(los-arzt); print(f"  [Referenz Oberarzt: MAE {ea.mean():.2f} d]")

# ---- prospektiv nach Subgruppe: Modell vs Median-Null ----
print("\n=== PROSPEKTIV nach Subgruppe: leckfreies Modell vs Median-Null ===")
print(f"{'bin':<8}{'n':>5}{'Modell-MAE':>12}{'Null(Med)-MAE':>15}{'ΔMAE[CI]':>22}  verdict")
em_p=np.abs(los-np.clip(pred_p,0,None))
for lab,lo_,hi_ in [("1-2 d",1,2),("2-4 d",2,4),("4-7 d",4,7),(">7 d",7,999)]:
    m=(los>lo_)&(los<=hi_) if hi_<999 else los>lo_
    if m.sum()<5: continue
    en=np.abs(los[m]-TRAIN_MED); d,(lo,hi)=boot(los[m],em_p[m],en)
    v="Modell besser" if lo>0 else ("Null besser" if hi<0 else "n.s.")
    print(f"{lab:<8}{int(m.sum()):>5}{em_p[m].mean():>12.2f}{en.mean():>15.2f}{f'  {d:+.2f}[{lo:+.2f},{hi:+.2f}]':>22}  {v}")
