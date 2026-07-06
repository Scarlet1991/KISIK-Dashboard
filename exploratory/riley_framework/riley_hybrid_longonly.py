# -*- coding: utf-8 -*-
"""Modell neu aufrufen + Variante testen:
  A) Voll 3-Komponenten-Hybrid:  pred = p_short*SHORT + p_mid*ARZT + p_long*LONG   (Referenz, MAE 2.89)
  B) Long-only Hybrid:           pred = (p_short+p_mid)*ARZT + p_long*LONG          (Short-Experte weg -> Arzt)
SHORT/LONG = log1p-ExtraTrees+recal (retro 1<LoS<=7 bzw >7). Gate (c_lo,s) per nested-5-fold-CV.
Vergleich vs Oberarzt + A vs B, Signifikanz (Bootstrap B=5000 + Wilcoxon). Prospektiv n=286.
Ausgabe: exploratory_riley/hybrid_longonly.csv
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
from collections import Counter
import duckdb, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score
BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; K=BASE/"kisik2"; OUT=AN/"exploratory_riley"
RETRO=K/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"; RS=42; B=5000; rng=np.random.default_rng(RS)
asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; bpet=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["ExtraTrees"]
con=duckdb.connect()
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
allcols=list(con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') LIMIT 0").df().columns)
present=[f for f in feat if f in allcols and not f.startswith(("lab_","vital_","proc_","zugang_")) and not f.startswith("proc24_8_98f")]
meta=["icu_duration_h","oebenekurz","wardshort","pid"]; selc=list(dict.fromkeys(meta+present)); colstr=", ".join('"'+c+'"' for c in selc)
df=con.execute(f"SELECT {colstr} FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet"); los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float); N=len(los)
y=(df["icu_duration_h"]/24.0).values; groups=df["pid"].fillna("unknown").astype(str).values
cat=["oebenekurz"]; numc=[c for c in present if c!="oebenekurz"]
def Xf(frame):
    X=frame.reindex(columns=present).copy()
    for c in present: X[c]=(X[c].astype(str) if c=="oebenekurz" else pd.to_numeric(X[c],errors="coerce"))
    return X
X=Xf(df); tr,_=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(X,y,groups)); Xtr,ytr,gtr=X.iloc[tr],y[tr],groups[tr]
Xp=pd.DataFrame(index=PR.index)
for c in present: Xp[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
Xp=Xp[present]
def pre(): return ColumnTransformer([("num",SimpleImputer(strategy="median"),numc),("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat)])
def et(): return TransformedTargetRegressor(Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=2))]),func=np.log1p,inverse_func=np.expm1)
def fit_recal(Xt,yt,gt):
    m=et(); m.fit(Xt,yt); oof=np.clip(cross_val_predict(et(),Xt,yt,groups=gt,cv=GroupKFold(3),n_jobs=1),0,None); b,a=np.polyfit(oof,yt,1); return m,a,b
def ap(m,a,b,Xs): return np.clip(a+b*np.clip(m.predict(Xs),0,None),0,None)
ms=(ytr>1)&(ytr<=7); ml=ytr>7
Ms,as_,bs=fit_recal(Xtr.iloc[ms.nonzero()[0]],ytr[ms],gtr[ms]); SHORT=ap(Ms,as_,bs,Xp)
Ml,al,bl=fit_recal(Xtr.iloc[ml.nonzero()[0]],ytr[ml],gtr[ml]); LONG=ap(Ml,al,bl,Xp)
def gates(idx,c_lo,s,c_hi=7.0):
    a=arzt[idx]; pl=1/(1+np.exp(-(a-c_hi)/s)); psh=1/(1+np.exp(-(c_lo-a)/s)); pm=np.clip(1-pl-psh,0,None); tot=pl+psh+pm
    return psh/tot,pm/tot,pl/tot,a
def blend_full(idx,c_lo,s): psh,pm,pl,a=gates(idx,c_lo,s); return psh*SHORT[idx]+pm*a+pl*LONG[idx]
def blend_long(idx,c_lo,s): psh,pm,pl,a=gates(idx,c_lo,s); return (psh+pm)*a+pl*LONG[idx]   # Short-Anteil -> Arzt
def nested(blend):
    oof=np.full(N,np.nan); picks=[]
    for trn,tst in KFold(5,shuffle=True,random_state=RS).split(np.arange(N)):
        best=None
        for c_lo in [3,4,5]:
            for s in [1.0,1.5,2.0]:
                v=mean_absolute_error(los[trn],np.clip(blend(trn,c_lo,s),0,None))
                if best is None or v<best[0]: best=(v,c_lo,s)
        _,c_lo,s=best; picks.append((c_lo,s)); oof[tst]=np.clip(blend(tst,c_lo,s),0,None)
    return oof,Counter(picks).most_common(1)[0][0]
HFULL,gf=nested(blend_full); HLONG,gl=nested(blend_long)
SG=[("1-2 d",1,2),("2-4 d",2,4),("4-7 d",4,7),(">7 d",7,1e9)]
def met(p):
    p=np.clip(p,0,None); m=los>7; sb,_=np.polyfit(p,los,1); d={}
    for l,lo,hi in SG:
        mm=(los>lo)&(los<=hi) if hi<1e8 else los>7; d[l]=round(float(np.abs(los[mm]-p[mm]).mean()),2)
    return dict(MAE=round(float(mean_absolute_error(los,p)),3),R2=round(float(r2_score(los,p)),3),slope=round(float(sb),3),gt7=round(float(np.abs(los[m]-p[m]).mean()),2),**d)
mF,mL,mP=met(HFULL),met(HLONG),met(arzt)
def supr(a_pred,b_pred,mask):  # dMAE=MAE_b-MAE_a; >0 => a besser
    yt=los[mask]; ea=np.abs(yt-a_pred[mask]); eb=np.abs(yt-b_pred[mask]); n=len(yt)
    idx=rng.integers(0,n,size=(B,n)); dd=eb[idx].mean(1)-ea[idx].mean(1); lo,hi=np.percentile(dd,[2.5,97.5])
    try: p=float(wilcoxon(ea,eb).pvalue)
    except: p=float("nan")
    return round(float(ea.mean()-eb.mean()),3),round(float(lo),2),round(float(hi),2),round(p,4),("a besser" if lo>0 else "b besser" if hi<0 else "n.s.")
print("=== Modell neu aufgerufen — prospektiv n=%d ==="%N)
print(pd.DataFrame([{"Modell":"Oberarzt",**mP},{"Modell":"A) Voll 3-Komp",**mF},{"Modell":"B) Long-only (Short->Arzt)",**mL}])[
    ["Modell","MAE","R2","slope","gt7","1-2 d","2-4 d","4-7 d",">7 d"]].to_string(index=False))
print(f"\nGate-Wahl: voll {gf} | long-only {gl}")
allm=np.ones(N,bool); m7=los>7
print("\n=== Signifikanz ===")
for lab,a_pred,b_pred,mask in [
    ("A (voll) vs Arzt — gesamt",HFULL,arzt,allm),("A (voll) vs Arzt — >7d",HFULL,arzt,m7),
    ("B (long) vs Arzt — gesamt",HLONG,arzt,allm),("B (long) vs Arzt — >7d",HLONG,arzt,m7),
    ("B (long) vs A (voll) — gesamt",HLONG,HFULL,allm),("B (long) vs A (voll) — >7d",HLONG,HFULL,m7)]:
    d,lo,hi,p,v=supr(a_pred,b_pred,mask)
    print(f"  {lab:<30} dMAE {d:+.3f} [{lo:+.2f},{hi:+.2f}] p={p:.4f}  ({v.replace('a','1.').replace('b','2.')})")
pd.DataFrame([{"model":"physician",**mP},{"model":"full_3comp",**mF},{"model":"long_only",**mL}]).to_csv(OUT/"hybrid_longonly.csv",sep=";",index=False)
print("\nGespeichert: hybrid_longonly.csv")
