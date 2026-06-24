# -*- coding: utf-8 -*-
"""Drei prospektive Hexbins (predicted vs TATSAECHLICHE LoS, no_isopen n=286, leckfrei):
 (1) bestes ML-Tool ALLEIN  = log1p-ExtraTrees + recal (volle Kohorte)
 (2) Oberarzt ALLEIN
 (3) bester Hybrid          = 3-Komponenten-Blend (MID=Arzt, nested-CV OOF)
Jede als eigenstaendige Vollgrafik. Ausgabe: Eigene Auswertung/exploratory_riley/
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
from collections import Counter
import duckdb, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; OUT=AN/"exploratory_riley"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"
bpet=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["ExtraTrees"]

con=duckdb.connect()
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
allcols=list(con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') LIMIT 0").df().columns)
meta=["icu_duration_h","wardshort","oebenekurz","pid"]
present=[f for f in feat if f in allcols and not f.startswith(("lab_","vital_","proc_","zugang_")) and not f.startswith("proc24_8_98f")]
selc=meta+[f for f in present if f not in meta]; colstr=", ".join('"'+c+'"' for c in selc)
df=con.execute(f"SELECT {colstr} FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
y=(df["icu_duration_h"]/24.0).values; groups=df["pid"].fillna("unknown").astype(str).values
cat=[c for c in present if c=="oebenekurz"]; numc=[c for c in present if c not in cat]
def Xf(frame):
    X=frame.reindex(columns=present).copy()
    for c in present: X[c]=(X[c].astype(str) if c=="oebenekurz" else pd.to_numeric(X[c],errors="coerce"))
    return X
X=Xf(df); tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(X,y,groups))
Xtr,ytr,gtr=X.iloc[tr],y[tr],groups[tr]
def pre(): return ColumnTransformer([("num",SimpleImputer(strategy="median"),numc),
    ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat)])
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float)
Xp=pd.DataFrame(index=PR.index)
for c in present: Xp[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
Xp=Xp[present]

def et(): return TransformedTargetRegressor(Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=2))]),func=np.log1p,inverse_func=np.expm1)
def fit_recal(Xt,yt,gt):
    m=et(); m.fit(Xt,yt); oof=np.clip(cross_val_predict(et(),Xt,yt,groups=gt,cv=GroupKFold(3),n_jobs=1),0,None); b,a=np.polyfit(oof,yt,1); return m,a,b
def ap(m,a,b,Xs): return np.clip(a+b*np.clip(m.predict(Xs),0,None),0,None)

# (1) bestes ML-Tool allein: volle Kohorte
Mf,af,bf=fit_recal(Xtr,ytr,gtr); ML_ALONE=ap(Mf,af,bf,Xp)
# Experten fuer Hybrid
ms=(ytr>1)&(ytr<=7); ml=ytr>7
Ms,as_,bs=fit_recal(Xtr.iloc[ms.nonzero()[0]],ytr[ms],gtr[ms]); SHORT=ap(Ms,as_,bs,Xp)
Ml,al,bl=fit_recal(Xtr.iloc[ml.nonzero()[0]],ytr[ml],gtr[ml]); LONG=ap(Ml,al,bl,Xp)
def soft3(idx,c_lo,s,c_hi=7.0):
    a=arzt[idx]; pl=1/(1+np.exp(-(a-c_hi)/s)); psh=1/(1+np.exp(-(c_lo-a)/s)); pm=np.clip(1-pl-psh,0,None)
    tot=pl+psh+pm; return (psh*SHORT[idx]+pm*a+pl*LONG[idx])/tot
CLO=[3,4,5]; SS=[1.0,1.5,2.0]; kf=KFold(5,shuffle=True,random_state=RS); N=len(los); HYB=np.full(N,np.nan)
for trn,tst in kf.split(np.arange(N)):
    best=None
    for c_lo in CLO:
        for s in SS:
            v=mean_absolute_error(los[trn],np.clip(soft3(trn,c_lo,s),0,None))
            if best is None or v<best[0]: best=(v,c_lo,s)
    _,c_lo,s=best; HYB[tst]=np.clip(soft3(tst,c_lo,s),0,None)

CAP=30
def hexfig(pred,title,fname):
    fig,ax=plt.subplots(figsize=(6.6,6.4))
    p=np.clip(pred,0,CAP); o=np.clip(los,0,CAP)
    hb=ax.hexbin(p,o,gridsize=26,cmap="viridis",mincnt=1,extent=(0,CAP,0,CAP))
    ax.plot([0,CAP],[0,CAP],"--",color="#e74c3c",lw=1.6,label="perfect prediction")
    mae=mean_absolute_error(los,np.clip(pred,0,None)); r2=r2_score(los,np.clip(pred,0,None))
    sb,_=np.polyfit(np.clip(pred,0,None),los,1); m=los>7; mae7=float(np.abs(los[m]-np.clip(pred,0,None)[m]).mean())
    ax.set_xlim(0,CAP); ax.set_ylim(0,CAP); ax.set_aspect("equal","box")
    ax.set_xlabel("Predicted LoS (days)",fontsize=11); ax.set_ylabel("Actual (observed) LoS (days)",fontsize=11)
    ax.set_title(title,weight="bold",fontsize=12)
    ax.text(0.045,0.955,f"MAE {mae:.2f} d\nMAE >7 d  {mae7:.2f} d\nR²  {r2:.2f}\nslope  {sb:.2f}\nn = {N}",transform=ax.transAxes,
            va="top",ha="left",fontsize=10,bbox=dict(boxstyle="round",fc="white",ec="#bbb",alpha=.9))
    ax.legend(loc="lower right",fontsize=9.5)
    cb=fig.colorbar(hb,ax=ax,fraction=0.046,pad=0.03); cb.set_label("cases per bin",fontsize=10)
    fig.tight_layout(); fig.savefig(str(OUT/fname),dpi=300,bbox_inches="tight"); plt.close(fig)
    print(f"  {fname}: MAE {mae:.2f}, MAE>7 {mae7:.2f}, R2 {r2:.2f}, slope {sb:.2f}")

print("=== Prospektive Hexbins (predicted vs tatsaechliche LoS, n=%d) ==="%N)
hexfig(ML_ALONE,"Best ML model alone (prospective)\nlog1p-ExtraTrees + recalibration","fig_hexbin_pros_ml_alone.png")
hexfig(arzt,"Senior physician alone (prospective)","fig_hexbin_pros_physician.png")
hexfig(HYB,"Best hybrid (prospective)\n3-component blend (ML experts + physician gate)","fig_hexbin_pros_hybrid.png")
print("\nGespeichert: fig_hexbin_pros_ml_alone.png, fig_hexbin_pros_physician.png, fig_hexbin_pros_hybrid.png")
