# -*- coding: utf-8 -*-
"""EXPLORATIV: WEICHE Arzt-Weiche (graduelles Blending) statt hartem Routing.
w = sigmoid((arzt - c)/s) ; pred = (1-w)*Kurz-Experte + w*Lang-Experte.
Experten fix: Kurz=Tweedie+recal (1<LoS<=7), Lang=Tweedie+recal (LoS>7). Arzt-basierte Weiche.
Sweep ueber Zentrum c und Steilheit s. Ziel: Kurzlieger-Genauigkeit + Langlieger-Erkennung vereinen.
Evaluierung prospektiv no_isopen (alle n=286). Leckfrei. Ausgabe: Eigene Auswertung/exploratory_riley/
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; OUT=AN/"exploratory_riley"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; bptw=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["Tweedie"]

con=duckdb.connect()
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
y=(df["icu_duration_h"]/24.0).values; groups=df["pid"].fillna("unknown").astype(str).values
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns and not f.startswith(("lab_","vital_","proc_","zugang_")) and not f.startswith("proc24_8_98f")]
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
def tw(): return Pipeline([("pre",pre()),("mdl",XGBRegressor(objective="reg:tweedie",**bptw,random_state=RS,n_jobs=2,tree_method="hist"))])
def fit_recal(Xt,yt,gt):
    m=tw(); m.fit(Xt,yt); oof=np.clip(cross_val_predict(tw(),Xt,yt,groups=gt,cv=GroupKFold(3),n_jobs=1),0,None); b,a=np.polyfit(oof,yt,1); return m,a,b
def ap(m,a,b,Xs): return np.clip(a+b*np.clip(m.predict(Xs),0,None),0,None)
ms=(ytr>1)&(ytr<=7); ml=ytr>7
Ms,as_,bs=fit_recal(Xtr.iloc[ms.nonzero()[0]],ytr[ms],gtr[ms]); short=ap(Ms,as_,bs,Xp)
Ml,al,bl=fit_recal(Xtr.iloc[ml.nonzero()[0]],ytr[ml],gtr[ml]); long=ap(Ml,al,bl,Xp)

def met(obs,p):
    p=np.clip(p,0,None); sb,_=np.polyfit(p,obs,1); m=obs>7
    sg={}
    for l,lo_,hi_ in [("1-2",1,2),("2-4",2,4),("4-7",4,7),(">7",7,999)]:
        mm=(obs>lo_)&(obs<=hi_) if hi_<999 else obs>7; sg["MAE_"+l]=round(float(np.abs(obs[mm]-p[mm]).mean()),2)
    return dict(MAE=round(float(mean_absolute_error(obs,p)),3),R2=round(float(r2_score(obs,p)),3),
                slope=round(float(sb),3),MAE_gt7=round(float(np.abs(obs[m]-p[m]).mean()),3),**sg)
def soft(c,s): w=1/(1+np.exp(-(arzt-c)/s)); return (1-w)*short+w*long

cs=[6,7,8,9,10,12]; ss=[0.5,1.0,2.0,3.0]
rows=[]
for c in cs:
    for s in ss:
        rows.append({"c":c,"s":s,**met(los,soft(c,s))})
res=pd.DataFrame(rows); res.to_csv(OUT/"soft_gate_sweep.csv",sep=";",index=False)
phys=met(los,arzt); hard=met(los,np.where(arzt>=7,long,short))
print("=== WEICHE Arzt-Weiche: Sweep c (Zentrum) x s (Steilheit), prospektiv n=%d ==="%len(los))
print(res[["c","s","MAE","R2","MAE_gt7","MAE_2-4","MAE_4-7","MAE_>7"]].to_string(index=False))
best=res.sort_values("MAE").iloc[0]
print(f"\nBestes (Gesamt-MAE): c={best.c}, s={best.s} -> MAE {best.MAE:.3f}, R2 {best.R2:.3f}, MAE>7 {best.MAE_gt7:.2f}")
print(f"Harte Weiche (arzt>=7):  MAE {hard['MAE']:.3f}, R2 {hard['R2']:.3f}, 2-4:{hard['MAE_2-4']} 4-7:{hard['MAE_4-7']} >7:{hard['MAE_>7']}")
print(f"Oberarzt allein:         MAE {phys['MAE']:.3f}, R2 {phys['R2']:.3f}, 2-4:{phys['MAE_2-4']} 4-7:{phys['MAE_4-7']} >7:{phys['MAE_>7']}")

# ===== Figuren =====
fig,ax=plt.subplots(1,2,figsize=(14,5.4))
# (A) Heatmap MAE ueber c x s
M=res.pivot(index="s",columns="c",values="MAE")
im=ax[0].imshow(M.values,aspect="auto",cmap="viridis_r",origin="lower")
ax[0].set_xticks(range(len(cs))); ax[0].set_xticklabels(cs); ax[0].set_yticks(range(len(ss))); ax[0].set_yticklabels(ss)
ax[0].set_xlabel("gate centre c (days)"); ax[0].set_ylabel("steepness s")
for i in range(len(ss)):
    for j in range(len(cs)): ax[0].text(j,i,f"{M.values[i,j]:.2f}",ha="center",va="center",color="white",fontsize=8)
ax[0].set_title("(A) overall MAE over soft-gate parameters",weight="bold",fontsize=11); fig.colorbar(im,ax=ax[0],label="MAE (days)")
ax[0].plot([list(cs).index(best.c)],[list(ss).index(best.s)],"r*",ms=16)
# (B) Subgruppen best soft vs hard vs physician
binsl=["1-2","2-4","4-7",">7"]; xb=np.arange(4); w=0.27
bestp=soft(best.c,best.s)
for i,(lab,p,c) in enumerate([("soft gate (best)",bestp,"#1b7f3b"),("hard gate (≥7)",np.where(arzt>=7,long,short),"#7f8c8d"),("physician",arzt,"#c0392b")]):
    mm=met(los,p); vals=[mm["MAE_"+b] for b in binsl]; bb=ax[1].bar(xb+(i-1)*w,vals,w,label=lab,color=c); ax[1].bar_label(bb,fmt="%.1f",fontsize=6.5,padding=1)
ax[1].set_xticks(xb); ax[1].set_xticklabels([b+" d" for b in binsl]); ax[1].set_ylabel("MAE (days)")
ax[1].set_title(f"(B) subgroup MAE — best soft gate (c={best.c}, s={best.s})",weight="bold",fontsize=11); ax[1].legend(fontsize=8.5)
fig.suptitle("Soft physician gate: blending short/long experts by sigmoid(arzt) (prospective n=286, leak-free)",weight="bold",fontsize=12)
fig.tight_layout(); fig.savefig(str(OUT/"fig_soft_gate.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print(f"\nGespeichert: fig_soft_gate.png + soft_gate_sweep.csv")
