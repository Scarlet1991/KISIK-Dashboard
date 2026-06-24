# -*- coding: utf-8 -*-
"""EXPLORATIV: Routing-Sweep mit ARZT-GATE vs ML-GATE, ueber K in {5,7,10,14} und Tool {ExtraTrees,Tweedie}.
Experten: Kurz (1<LoS<=K) + Lang (LoS>K), je OOF-rekalibriert.
  ML-Gate:   soft  (1-p)*kurz + p*lang ,  p=P(LOS>=K) ExtraTreesClassifier
  Arzt-Gate: hard  arzt>=K -> lang, sonst kurz
Evaluierung prospektiv no_isopen (alle n=286). Baselines: Arzt, Einzelmodell.
Leckfrei. Ausgabe: Eigene Auswertung/exploratory_riley/
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; OUT=AN/"exploratory_riley"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"
bp=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]; bpet=bp["ExtraTrees"]; bptw=bp["Tweedie"]

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
def factory(tool):
    if tool=="ExtraTrees":
        return TransformedTargetRegressor(Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=2))]),func=np.log1p,inverse_func=np.expm1)
    return Pipeline([("pre",pre()),("mdl",XGBRegressor(objective="reg:tweedie",**bptw,random_state=RS,n_jobs=2,tree_method="hist"))])
def fit_recal(tool,Xt,yt,gt):
    m=factory(tool); m.fit(Xt,yt); oof=np.clip(cross_val_predict(factory(tool),Xt,yt,groups=gt,cv=GroupKFold(3),n_jobs=1),0,None)
    b,a=np.polyfit(oof,yt,1); return m,a,b
def ap(m,a,b,Xs): return np.clip(a+b*np.clip(m.predict(Xs),0,None),0,None)
def met(obs,p,gate,tool,K):
    p=np.clip(p,0,None); sb,_=np.polyfit(p,obs,1); m=obs>7
    return dict(gate=gate,tool=tool,K=K,MAE=round(float(mean_absolute_error(obs,p)),3),R2=round(float(r2_score(obs,p)),3),
                slope=round(float(sb),3),bias=round(float((p-obs).mean()),3),MAE_gt7=round(float(np.abs(obs[m]-p[m]).mean()),3))
rows=[]
for tool in ["ExtraTrees","Tweedie"]:
    for K in [5,7,10,14]:
        ms=(ytr>1)&(ytr<=K); ml=ytr>K
        if ml.sum()<200: continue
        Rs,as_,bs=fit_recal(tool,Xtr.iloc[ms.nonzero()[0]],ytr[ms],gtr[ms]); short=ap(Rs,as_,bs,Xp)
        Rl,al,bl=fit_recal(tool,Xtr.iloc[ml.nonzero()[0]],ytr[ml],gtr[ml]); long=ap(Rl,al,bl,Xp)
        C=Pipeline([("pre",pre()),("mdl",ExtraTreesClassifier(n_estimators=500,min_samples_leaf=5,max_features=0.5,random_state=RS,n_jobs=2,class_weight="balanced"))]); C.fit(Xtr,(ytr>=K).astype(int))
        p=C.predict_proba(Xp)[:,1]
        rows.append(met(los,(1-p)*short+p*long,"ML-gate (soft)",tool,K))
        rows.append(met(los,np.where(arzt>=K,long,short),"Physician-gate (hard)",tool,K))
        print(f"  {tool:<10} K={K:<3}  ML {rows[-2]['MAE']:.3f} | Arzt {rows[-1]['MAE']:.3f}")
res=pd.DataFrame(rows); res.to_csv(OUT/"routing_physgate_sweep.csv",sep=";",index=False)
phys=met(los,arzt,"—","Physician","—")
print("\n=== ARZT-GATE vs ML-GATE Routing-Sweep (prospektiv n=%d) ==="%len(los))
print(res.to_string(index=False)); print(f"\n[Oberarzt allein: MAE {phys['MAE']:.3f}, R² {phys['R2']:.3f}, MAE>7 {phys['MAE_gt7']:.3f}]")
best=res.sort_values("MAE").iloc[0]; print(f"Bestes Routing (MAE): {best['gate']} {best['tool']} K={best['K']} -> MAE {best['MAE']:.3f}")

# ===== Figur: MAE und MAE>7 ueber K, ML-Gate vs Arzt-Gate (Tweedie-Experten) =====
fig,ax=plt.subplots(1,2,figsize=(13.5,5.4)); Ks=[5,7,10,14]
for gate,c,mk in [("ML-gate (soft)","#1f5f9e","o"),("Physician-gate (hard)","#1b7f3b","s")]:
    sub=res[(res.gate==gate)&(res.tool=="Tweedie")]
    ax[0].plot(Ks,[sub[sub.K==k]["MAE"].iloc[0] for k in Ks],mk+"-",color=c,lw=2,ms=7,label=gate)
    ax[1].plot(Ks,[sub[sub.K==k]["MAE_gt7"].iloc[0] for k in Ks],mk+"-",color=c,lw=2,ms=7,label=gate)
for a,key,ttl in [(ax[0],"MAE","(A) overall MAE vs threshold K"),(ax[1],"MAE_gt7","(B) MAE for stays >7 d vs K")]:
    a.axhline(phys[key],color="#c0392b",ls="--",lw=1.8,label=f"physician ({phys[key]:.2f})")
    a.set_xlabel("long-stay threshold K (days)"); a.set_ylabel(key.replace("_"," ")+" (days)"); a.set_xticks(Ks)
    a.set_title(ttl,weight="bold",fontsize=11); a.legend(fontsize=8.5)
fig.suptitle("Routing gate comparison: ML-gate vs physician-gate over threshold K (Tweedie experts, prospective n=286)",weight="bold",fontsize=11.5)
fig.tight_layout(); fig.savefig(str(OUT/"fig_routing_physgate_sweep.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print(f"\nGespeichert: fig_routing_physgate_sweep.png + routing_physgate_sweep.csv")
