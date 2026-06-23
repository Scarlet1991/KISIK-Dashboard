# -*- coding: utf-8 -*-
"""Superioritaet Extra Trees vs Oberarzt unter ALTERNATIVEN LoS-Bin-Definitionen.
Prospektiv no_isopen (n=286, LoS>1), fuer beide Modelle: mit 8-98f (Leck) und leckfrei."""
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

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; B=5000; rng=np.random.default_rng(RS)
bp=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["ExtraTrees"]

con=duckdb.connect()
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
y=(df["icu_duration_h"]/24.0).values; groups=df["pid"].fillna("unknown").astype(str).values
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns and not f.startswith(("lab_","vital_","proc_","zugang_"))]
def fit_pred(cols, Xp):
    cat=[c for c in cols if c=="oebenekurz"]; numc=[c for c in cols if c not in cat]
    pre=ColumnTransformer([("num",SimpleImputer(strategy="median"),numc)]+
        ([("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat)] if cat else []))
    m=TransformedTargetRegressor(Pipeline([("pre",pre),("mdl",ExtraTreesRegressor(**bp,random_state=RS,n_jobs=-1))]),func=np.log1p,inverse_func=np.expm1)
    X=df.reindex(columns=cols).copy()
    for c in cols: X[c]=(X[c].astype(str) if c=="oebenekurz" else pd.to_numeric(X[c],errors="coerce"))
    tr,_=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(X,y,groups))
    m.fit(X.iloc[tr],y[tr]); return np.clip(m.predict(Xp),0,None)

PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float)
pred_leak=np.clip(PR["__pred_ExtraTrees__"].to_numpy(float),0,None)   # mit 8-98f (aus 08)
present_lf=[c for c in present if not c.startswith("proc24_8_98f")]
def Xp_for(cols):
    X=pd.DataFrame(index=PR.index)
    for c in cols: X[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
    return X[cols]
pred_lf=fit_pred(present_lf, Xp_for(present_lf))   # leckfrei

def sup(mask,pred):
    yt=los[mask]; a=arzt[mask]; p=pred[mask]; ea=np.abs(yt-a); ee=np.abs(yt-p); n=len(yt)
    if n<5: return None
    idx=rng.integers(0,n,size=(B,n)); boot=ea[idx].mean(1)-ee[idx].mean(1); lo,hi=np.percentile(boot,[2.5,97.5])
    d=ea.mean()-ee.mean(); v="model" if lo>0 else ("physician" if hi<0 else "n.s.")
    return n,round(a_:=float(np.abs(yt-a).mean()),2),round(float(np.abs(yt-p).mean()),2),round(float(d),2),round(float(lo),2),round(float(hi),2),v

def edges_to_bins(edges):
    out=[]
    for i in range(len(edges)-1):
        lo,hi=edges[i],edges[i+1]
        lab=f"{lo}-{hi}d" if hi<999 else f">{lo}d"
        m=(los>lo)&(los<=hi) if hi<999 else los>lo
        out.append((lab,m))
    return out

schemes={
 "A. 1-2 / 2-10 / >10":      [1,2,10,999],
 "B. 1-3 / 3-7 / >7":        [1,3,7,999],
 "C. 1-2 / 2-4 / 4-7 / >7 (current)":[1,2,4,7,999],
 "D. 1-3 / >3":              [1,3,999],
 "E. 1-2 / >2":              [1,2,999],
 "F. 1-2 / 2-5 / >5":        [1,2,5,999],
}
for name,edges in schemes.items():
    print(f"\n=== {name} ===")
    print(f"{'bin':<9}{'n':>5} | {'mit 8-98f: Arzt/ET dMAE[CI] verdict':<46} | leckfrei: Arzt/ET dMAE[CI] verdict")
    for lab,m in edges_to_bins(edges):
        rL=sup(m,pred_leak); rF=sup(m,pred_lf)
        if rL is None: continue
        sL=f"{rL[1]}/{rL[2]} {rL[3]:+.2f}[{rL[4]:+.2f},{rL[5]:+.2f}] {rL[6]}"
        sF=f"{rF[1]}/{rF[2]} {rF[3]:+.2f}[{rF[4]:+.2f},{rF[5]:+.2f}] {rF[6]}"
        print(f"{lab:<9}{rL[0]:>5} | {sL:<46} | {sF}")
