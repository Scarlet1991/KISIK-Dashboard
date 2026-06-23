# -*- coding: utf-8 -*-
"""Formale Signifikanztests: leckfreies Extra-Trees-Modell vs Konstant-/Mittelwertvorhersage.
Gepaarter Wilcoxon (einseitig) + gepaarter t-Test auf Fehlerdifferenzen, Bootstrap-p, R²-Bootstrap-CI.
Getrennt fuer absolute Fehler (MAE) und quadratische Fehler (R²/MSE)."""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from scipy import stats
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
mdl.fit(X.iloc[tr],y[tr]); MEAN=float(np.mean(y[tr])); MED=float(np.median(y[tr]))

ROWS=[]
def pstr(p): return "<0.001" if p<0.001 else f"{p:.3f}"
def tests(yt,pred,const,kind):
    pm=np.clip(pred,0,None)
    if kind=="abs": em,en=np.abs(yt-pm),np.abs(yt-const)
    else:           em,en=(yt-pm)**2,(yt-const)**2
    d=en-em                      # >0 => Modell besser (kleinerer Fehler)
    w=stats.wilcoxon(d,alternative="greater").pvalue        # H1: Modellfehler < Nullfehler
    t=stats.ttest_rel(en,em,alternative="greater").pvalue
    n=len(yt); idx=rng.integers(0,n,size=(B,n)); bd=d[idx].mean(1)
    lo,hi=np.percentile(bd,[2.5,97.5]); bp_=float((bd<=0).mean())   # einseitiges Bootstrap-p
    return em.mean(),en.mean(),float(d.mean()),lo,hi,w,t,bp_

def block(yt,pred,title):
    print(f"\n{'='*92}\n{title} (n={len(yt)})\n{'='*92}")
    print(f"Modell-MAE={np.abs(yt-np.clip(pred,0,None)).mean():.3f} d | R²(vs eval-mean)={r2_score(yt,np.clip(pred,0,None)):+.3f}")
    # R²-Bootstrap-CI
    n=len(yt); idx=rng.integers(0,n,size=(B,n)); r2b=[r2_score(yt[i],np.clip(pred,0,None)[i]) for i in idx[:2000]]
    print(f"R²-Bootstrap 95%-CI: [{np.percentile(r2b,2.5):+.3f}, {np.percentile(r2b,97.5):+.3f}]")
    print(f"\n{'Vergleich':<34}{'MAE M/N':>14}{'ΔMAE':>8}{'Wilcoxon p':>13}{'t-Test p':>11}{'Boot-p':>9}  Urteil")
    r2=r2_score(yt,np.clip(pred,0,None)); r2lo,r2hi=np.percentile(r2b,[2.5,97.5])
    for nm,const in [("Trainings-Mittelwert",MEAN),("Trainings-Median",MED),("Eval-Mittelwert",float(np.mean(yt)))]:
        em,en,dm,lo,hi,w,t,bpv=tests(yt,pred,const,"abs")
        verdict=("model better" if (w<0.05 and dm>0) else ("null better" if (dm<0 and hi<0) else "n.s."))
        print(f"  abs vs {nm:<22}{em:>6.2f}/{en:<6.2f}{dm:>+8.2f}{pstr(w):>13}{pstr(t):>11}{pstr(bpv):>9}  {verdict}")
        ROWS.append({"cohort":title.split(" (")[0],"error":"MAE","baseline":nm,"MAE_model":round(em,3),"MAE_null":round(en,3),
                     "dMAE":round(dm,3),"wilcoxon_p":pstr(w),"ttest_p":pstr(t),"boot_p":pstr(bpv),"R2_model":round(r2,3),
                     "R2_CI_low":round(r2lo,3),"R2_CI_high":round(r2hi,3),"verdict":verdict})
    # quadratische Fehler vs eval-mean (entspricht R²>0)
    em,en,dm,lo,hi,w,t,bpv=tests(yt,pred,float(np.mean(yt)),"sq")
    verdict=("explains variance (R2>0)" if (w<0.05 and dm>0 and r2lo>0) else ("worse than mean (R2<0)" if r2hi<0 else "n.s."))
    print(f"  sq  vs {'Eval-Mittelwert (R²-Test)':<22}{'':>13}{dm:>+8.2f}{pstr(w):>13}{pstr(t):>11}{pstr(bpv):>9}  {verdict}")
    ROWS.append({"cohort":title.split(" (")[0],"error":"MSE/R2","baseline":"Eval-mean (R2 test)","MAE_model":"","MAE_null":"",
                 "dMSE":round(dm,3),"wilcoxon_p":pstr(w),"ttest_p":pstr(t),"boot_p":pstr(bpv),"R2_model":round(r2,3),
                 "R2_CI_low":round(r2lo,3),"R2_CI_high":round(r2hi,3),"verdict":verdict})

block(y[te], mdl.predict(X.iloc[te]), "RETROSPEKTIV Hold-out (leckfrei)")

PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float)
Xp=pd.DataFrame(index=PR.index)
for c in present_lf: Xp[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
block(los, mdl.predict(Xp[present_lf]), "PROSPEKTIV no_isopen (leckfrei)")
print("\nLegende: ΔMAE = MAE(Null) − MAE(Modell), >0 => Modell besser. Wilcoxon/t einseitig (H1: Modellfehler<Nullfehler).")
pd.DataFrame(ROWS).to_csv(AN/"leakfree"/"null_baseline_stats_lf.csv",sep=";",index=False)
print(f"Gespeichert: leakfree/null_baseline_stats_lf.csv")
