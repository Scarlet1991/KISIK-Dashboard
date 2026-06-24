# -*- coding: utf-8 -*-
"""EXPLORATIV: ARZT-GESTEUERTER Hybrid. Oberarzt-Schaetzung als Weiche (arzt>=K) statt ML-Klassifikator;
ML-Experten verfeinern innerhalb des Regimes.
  Kurz-Experte: Tweedie+Rekal., retro 1<LoS<=7
  Lang-Experte: Tweedie+Rekal., retro LoS>7
Hybride (prospektiv, alle n=286):
  phys-gate>=7 : arzt>=7 -> Lang-Experte, sonst Kurz-Experte
  phys-gate>=10: arzt>=10-> Lang-Experte, sonst Kurz-Experte
  ML-short+phys-long: arzt<7 -> Kurz-Experte, arzt>=7 -> Arzt-Schaetzung selbst
Vergleich: Arzt allein, Einzelmodell (volle Kohorte), ML-gesteuertes Routing (soft).
Leckfrei. Ausgabe: Eigene Auswertung/exploratory_riley/
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; OUT=AN/"exploratory_riley"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; B=5000; rng=np.random.default_rng(RS)
bptw=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["Tweedie"]

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
    m=tw(); m.fit(Xt,yt); oof=np.clip(cross_val_predict(m,Xt,yt,groups=gt,cv=GroupKFold(4),n_jobs=1),0,None); b,a=np.polyfit(oof,yt,1); return m,a,b
def ap(m,a,b,Xset): return np.clip(a+b*np.clip(m.predict(Xset),0,None),0,None)
ms=(ytr>1)&(ytr<=7); ml=ytr>7
Rs,as_,bs=fit_recal(Xtr.iloc[ms.nonzero()[0]],ytr[ms],gtr[ms])
Rl,al,bl=fit_recal(Xtr.iloc[ml.nonzero()[0]],ytr[ml],gtr[ml])
Rfull,af,bf=fit_recal(Xtr,ytr,gtr)
# ML-Weiche (zum Vergleich, soft)
C=Pipeline([("pre",pre()),("mdl",ExtraTreesClassifier(n_estimators=600,min_samples_leaf=5,max_features=0.5,random_state=RS,n_jobs=2,class_weight="balanced"))]); C.fit(Xtr,(ytr>=7).astype(int))
short=ap(Rs,as_,bs,Xp); long=ap(Rl,al,bl,Xp); full=ap(Rfull,af,bf,Xp); p_ml=C.predict_proba(Xp)[:,1]
print(f"Kurz-Experte recal a={as_:.2f} b={bs:.2f} | Lang-Experte recal a={al:.2f} b={bl:.2f}")
print(f"Arzt-Weiche: {int((arzt>=7).sum())} Faelle arzt>=7, {int((arzt>=10).sum())} arzt>=10  (real >7d: {int((los>7).sum())})")

variants={
 "Senior physician (alone)":arzt,
 "Single Tweedie+recal (full)":full,
 "ML-gated routing soft":(1-p_ml)*short+p_ml*long,
 "Physician-gated >=7 (experts)":np.where(arzt>=7,long,short),
 "Physician-gated >=10 (experts)":np.where(arzt>=10,long,short),
 "ML-short + physician-long (>=7)":np.where(arzt>=7,arzt,short),
}
def met(obs,p):
    p=np.clip(p,0,None); sb,_=np.polyfit(p,obs,1); m=obs>7
    return dict(MAE=round(float(mean_absolute_error(obs,p)),3),RMSE=round(float(np.sqrt(mean_squared_error(obs,p))),3),
                R2=round(float(r2_score(obs,p)),3),calib_slope=round(float(sb),3),bias=round(float((p-obs).mean()),3),
                MAE_gt7=round(float(np.abs(obs[m]-p[m]).mean()),3))
res=pd.DataFrame([{"approach":k,**met(los,v)} for k,v in variants.items()])
res.to_csv(OUT/"physgated_metrics.csv",sep=";",index=False)
print("\n=== PROSPEKTIV alle (n=%d) — arzt-gesteuerter Hybrid vs Alternativen ==="%len(los)); print(res.to_string(index=False))

# Subgruppen
sg=[("1-2 d",1,2),("2-4 d",2,4),("4-7 d",4,7),(">7 d",7,999)]
print("\n=== MAE nach Subgruppe ===")
print("approach".ljust(34)+"".join(f"{l:>9}" for l,_,_ in sg))
subrows=[]
for k,v in variants.items():
    line=k.ljust(34); r={"approach":k}
    for l,lo_,hi_ in sg:
        m=(los>lo_)&(los<=hi_) if hi_<999 else los>7; mae=float(np.abs(los[m]-np.clip(v[m],0,None)).mean()); line+=f"{mae:>9.2f}"; r[l]=round(mae,2)
    print(line); subrows.append(r)
pd.DataFrame(subrows).to_csv(OUT/"physgated_subgroups.csv",sep=";",index=False)

# Superioritaet phys-gated>=7 vs Arzt allein (gepaart, Bootstrap)
def supr(mask,pred):
    yt=los[mask]; ea=np.abs(yt-arzt[mask]); ee=np.abs(yt-np.clip(pred[mask],0,None)); n=len(yt)
    idx=rng.integers(0,n,size=(B,n)); d=ea[idx].mean(1)-ee[idx].mean(1); lo,hi=np.percentile(d,[2.5,97.5])
    return round(float(ea.mean()-ee.mean()),2),round(float(lo),2),round(float(hi),2),("hybrid" if lo>0 else "physician" if hi<0 else "n.s.")
hg=variants["Physician-gated >=7 (experts)"]
print("\n=== Superioritaet Physician-gated>=7 vs Oberarzt allein (dMAE=Arzt-Hybrid) ===")
for l,lo_,hi_ in [("overall",1,999)]+sg:
    m=np.ones(len(los),bool) if l=="overall" else ((los>lo_)&(los<=hi_) if hi_<999 else los>7)
    d,lo,hi,v=supr(m,hg); print(f"  {l:<9} (n={int(m.sum()):>3}): dMAE {d:+.2f} [{lo:+.2f},{hi:+.2f}] -> {v}")

# ===== Figur =====
fig,ax=plt.subplots(1,2,figsize=(14,5.6)); CAP=25
def cv(p,q=8):
    d=pd.DataFrame({"p":np.clip(p,0,None),"o":los}); d["b"]=pd.qcut(d["p"],q,duplicates="drop"); g=d.groupby("b",observed=True)
    return g["p"].mean().to_numpy(),g["o"].mean().to_numpy()
ax[0].plot([0,CAP],[0,CAP],"--",color="#888",label="ideal")
for p,lab,c in [(arzt,"physician alone","#c0392b"),(hg,"physician-gated hybrid","#1b7f3b"),(full,"single model","#7f8c8d"),(variants["ML-gated routing soft"],"ML-gated routing","#1f5f9e")]:
    mp,mo=cv(p); ax[0].plot(mp,mo,"o-",color=c,lw=1.8,ms=5,label=lab)
ax[0].set_xlim(0,CAP); ax[0].set_ylim(0,CAP); ax[0].set_aspect("equal","box")
ax[0].set_xlabel("Predicted LoS (days)"); ax[0].set_ylabel("Observed LoS (days)")
ax[0].set_title("(A) Calibration on ALL cases (prospective)",weight="bold",fontsize=11); ax[0].legend(fontsize=8.5,loc="upper left")
binsl=["1-2 d","2-4 d","4-7 d",">7 d"]; xb=np.arange(4); w=0.2
def smae(p):
    o=[]
    for l,lo_,hi_ in sg:
        m=(los>lo_)&(los<=hi_) if hi_<999 else los>7; o.append(float(np.abs(los[m]-np.clip(p[m],0,None)).mean()))
    return o
sel=[("physician alone",arzt,"#c0392b"),("physician-gated",hg,"#1b7f3b"),("single model",full,"#7f8c8d"),("ML-gated routing",variants["ML-gated routing soft"],"#1f5f9e")]
for i,(lab,p,c) in enumerate(sel):
    bb=ax[1].bar(xb+(i-1.5)*w,smae(p),w,label=lab,color=c); ax[1].bar_label(bb,fmt="%.1f",fontsize=6.3,padding=1)
ax[1].set_xticks(xb); ax[1].set_xticklabels(binsl); ax[1].set_ylabel("MAE (days)")
ax[1].set_title("(B) MAE by subgroup — all cases",weight="bold",fontsize=11); ax[1].legend(fontsize=8.5)
fig.suptitle(f"Physician-gated hybrid (physician decides regime, ML expert predicts) — prospective n={len(los)}, leak-free",weight="bold",fontsize=12)
fig.tight_layout(); fig.savefig(str(OUT/"fig_physician_gated.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print(f"\nGespeichert: fig_physician_gated.png + physgated_metrics.csv + physgated_subgroups.csv")
