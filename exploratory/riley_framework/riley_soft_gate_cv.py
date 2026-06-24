# -*- coding: utf-8 -*-
"""EXPLORATIV: WEICHE Arzt-Weiche mit SAUBERER (c,s)-Festlegung via nested CV.
Experten (Kurz<=7 / Lang>7, Tweedie+recal) auf RETRO trainiert (kein Leck).
Die Gate-Parameter c (Zentrum) und s (Steilheit) haengen von der ARZT-Schaetzung ab,
die es nur prospektiv gibt -> daher nested 5-fold CV auf der prospektiven Kohorte:
c,s werden in den Trainings-Folds nach MAE gewaehlt, eingefroren, im Holdout-Fold geprueft.
Vergleich: CV-ehrlicher Soft-Gate vs harte Weiche vs Arzt vs (optimistisch) auf-allen-bestes c,s.
Ausgabe: Eigene Auswertung/exploratory_riley/
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
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
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; bpet=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["ExtraTrees"]

con=duckdb.connect()
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
allcols=list(con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') LIMIT 0").df().columns)
meta=["icu_duration_h","wardshort","oebenekurz","pid"]
present=[f for f in feat if f in allcols and not f.startswith(("lab_","vital_","proc_","zugang_")) and not f.startswith("proc24_8_98f")]
selc=meta+[f for f in present if f not in meta]
colstr=", ".join('"'+c+'"' for c in selc)
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
def tw(): return TransformedTargetRegressor(Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=2))]),func=np.log1p,inverse_func=np.expm1)
def fit_recal(Xt,yt,gt):
    m=tw(); m.fit(Xt,yt); oof=np.clip(cross_val_predict(tw(),Xt,yt,groups=gt,cv=GroupKFold(3),n_jobs=1),0,None); b,a=np.polyfit(oof,yt,1); return m,a,b
def ap(m,a,b,Xs): return np.clip(a+b*np.clip(m.predict(Xs),0,None),0,None)
ms=(ytr>1)&(ytr<=7); ml=ytr>7
Ms,as_,bs=fit_recal(Xtr.iloc[ms.nonzero()[0]],ytr[ms],gtr[ms]); SHORT=ap(Ms,as_,bs,Xp)
Ml,al,bl=fit_recal(Xtr.iloc[ml.nonzero()[0]],ytr[ml],gtr[ml]); LONG=ap(Ml,al,bl,Xp)
def soft(idx,c,s): w=1/(1+np.exp(-(arzt[idx]-c)/s)); return (1-w)*SHORT[idx]+w*LONG[idx]

CS=[6,7,8,9,10,12]; SS=[0.5,1.0,2.0,3.0]
# ---- nested 5-fold CV: c,s in Trainfolds nach MAE waehlen, im Holdout pruefen ----
kf=KFold(5,shuffle=True,random_state=RS); idxall=np.arange(len(los))
oof=np.full(len(los),np.nan); picks=[]
for trn,tst in kf.split(idxall):
    best=None
    for c in CS:
        for s in SS:
            mae=mean_absolute_error(los[trn],np.clip(soft(trn,c,s),0,None))
            if best is None or mae<best[0]: best=(mae,c,s)
    _,c,s=best; picks.append((c,s)); oof[tst]=np.clip(soft(tst,c,s),0,None)
def met(obs,p):
    p=np.clip(p,0,None); sb,_=np.polyfit(p,obs,1); m=obs>7; sg={}
    for l,lo_,hi_ in [("1-2",1,2),("2-4",2,4),("4-7",4,7),(">7",7,999)]:
        mm=(obs>lo_)&(obs<=hi_) if hi_<999 else obs>7; sg["MAE_"+l]=round(float(np.abs(obs[mm]-p[mm]).mean()),2)
    return dict(MAE=round(float(mean_absolute_error(obs,p)),3),R2=round(float(r2_score(obs,p)),3),
                slope=round(float(sb),3),MAE_gt7=round(float(np.abs(obs[m]-p[m]).mean()),3),**sg)
cv=met(los,oof); phys=met(los,arzt); hard=met(los,np.where(arzt>=7,LONG,SHORT))
# auf-allen-bestes (optimistisch) zum Vergleich
allbest=None
for c in CS:
    for s in SS:
        mae=mean_absolute_error(los,np.clip(soft(idxall,c,s),0,None))
        if allbest is None or mae<allbest[0]: allbest=(mae,c,s)
opt=met(los,np.clip(soft(idxall,allbest[1],allbest[2]),0,None))
from collections import Counter
print("Gewaehlte (c,s) je Fold:",picks,"  haeufigste:",Counter(picks).most_common(1)[0])
out=pd.DataFrame([{"approach":"Soft gate (nested-CV, frozen c,s)",**cv},
                  {"approach":f"Soft gate (optimistic best c={allbest[1]},s={allbest[2]})",**opt},
                  {"approach":"Hard gate (arzt>=7)",**hard},
                  {"approach":"Senior physician",**phys}])
out.to_csv(OUT/"soft_gate_cv.csv",sep=";",index=False)
cols=["approach","MAE","R2","slope","MAE_gt7","MAE_2-4","MAE_4-7","MAE_>7"]
print("\n=== SAUBER (nested-CV) vs optimistisch vs Baselines (prospektiv n=%d) ==="%len(los))
print(out[cols].to_string(index=False))
print(f"\nOptimismus-Luecke (MAE): nested-CV {cv['MAE']:.3f} vs optimistisch {opt['MAE']:.3f} (Δ {cv['MAE']-opt['MAE']:+.3f})")

# ===== Figur =====
fig,ax=plt.subplots(1,2,figsize=(13.5,5.4))
sel=[("Soft gate (nested-CV)",oof,"#1b7f3b"),("Hard gate (≥7)",np.where(arzt>=7,LONG,SHORT),"#7f8c8d"),("Physician",arzt,"#c0392b")]
binsl=["1-2","2-4","4-7",">7"]; xb=np.arange(4); w=0.27
for i,(lab,p,c) in enumerate(sel):
    mm=met(los,p); vals=[mm["MAE_"+b] for b in binsl]; bb=ax[0].bar(xb+(i-1)*w,vals,w,label=lab,color=c); ax[0].bar_label(bb,fmt="%.1f",fontsize=6.5,padding=1)
ax[0].set_xticks(xb); ax[0].set_xticklabels([b+" d" for b in binsl]); ax[0].set_ylabel("MAE (days)")
ax[0].set_title("(A) Subgroup MAE — honest soft gate vs hard vs physician",weight="bold",fontsize=10.5); ax[0].legend(fontsize=8.5)
ov=[("Soft\nnested-CV",cv,"#1b7f3b"),("Soft\noptimistic",opt,"#9fdab8"),("Hard\ngate",hard,"#7f8c8d"),("Physician",phys,"#c0392b")]
x2=np.arange(len(ov));
b1=ax[1].bar(x2-0.2,[o[1]["MAE"] for o in ov],0.4,label="MAE (↓)",color="#185fa5")
ax[1].bar_label(b1,fmt="%.2f",fontsize=8)
axb=ax[1].twinx(); b2=axb.bar(x2+0.2,[o[1]["R2"] for o in ov],0.4,label="R² (↑)",color="#d99316")
axb.bar_label(b2,fmt="%.2f",fontsize=8); axb.set_ylabel("R²"); axb.set_ylim(0,0.45)
ax[1].set_xticks(x2); ax[1].set_xticklabels([o[0] for o in ov],fontsize=8.5); ax[1].set_ylabel("MAE (days)"); ax[1].set_ylim(0,4.2)
ax[1].set_title("(B) Overall MAE and R²",weight="bold",fontsize=10.5)
h1,l1=ax[1].get_legend_handles_labels(); h2,l2=axb.get_legend_handles_labels(); ax[1].legend(h1+h2,l1+l2,fontsize=8,loc="upper center")
fig.suptitle("Soft physician gate with honestly tuned (c,s): nested-CV on prospective cohort (n=286, leak-free)",weight="bold",fontsize=11.5)
fig.tight_layout(); fig.savefig(str(OUT/"fig_soft_gate_cv.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print(f"\nGespeichert: fig_soft_gate_cv.png + soft_gate_cv.csv")
