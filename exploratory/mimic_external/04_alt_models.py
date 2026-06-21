# -*- coding: utf-8 -*-
"""
MIMIC-IV externe Validierung — Schritt 4: alternative Ansaetze (wie im KISIK-Repo) auf MIMIC.
(A) Tweedie/Gamma/Hazard (analog tweedie_hazard.py):
    log1p-Mean (Referenz), Tweedie (var_power 1.3/1.5/1.7), Gamma, diskretes Hazard-Modell
    (Person-Period -> erwartete & mediane LoS). Fokus Langlieger (>7/>14 d).
(B) Quantilregression (analog quantile_op_prospective.py):
    XGBoost Mean / P50 / P80 auf log1p; P80-Coverage als Kapazitaetsplanungs-Metrik.
Daten: bereits gebaute mimic_features.parquet (First-24h-Features). Kein Oberarzt (MIMIC).
Patientengruppierter 80/20-Split (seed 42), identisch zu 02_model.py.
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
import time
OUT=Path(r"D:\Ausgangsdaten\KISIK Projekt\mimic_external"); RS=42; TMAX=90
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

X=pd.read_parquet(OUT/"mimic_features.parquet")
los=X["los_days"].clip(lower=0.01).to_numpy(float); groups=X["subject_id"].astype(str).to_numpy()
CAT=[c for c in ["first_careunit","gender","admission_type"] if c in X.columns]
feats=[c for c in X.columns if c not in ["los_days","subject_id"]]; num=[c for c in feats if c not in CAT]
parts=[X[num].apply(pd.to_numeric,errors="coerce")]
if CAT: parts.append(pd.get_dummies(X[CAT].astype(str),prefix=CAT).astype(float))
Xm=pd.concat(parts,axis=1); Xm.columns=[str(c) for c in Xm.columns]
Xm=Xm.apply(pd.to_numeric,errors="coerce").astype(np.float64).values   # NaN bleibt -> XGBoost-nativ
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(Xm,los,groups))
Xtr,Xte,los_tr,los_te=Xm[tr],Xm[te],los[tr],los[te]
log(f"Kohorte {len(los):,} | Train {len(tr):,} | Holdout {len(te):,} | {Xm.shape[1]} Modell-Spalten")

COMMON=dict(n_estimators=600,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
            min_child_weight=3,random_state=RS,n_jobs=-1)
def met(yt,yp,label,sub=None):
    if sub is not None: yt,yp=yt[sub],yp[sub]
    ae=np.abs(yt-yp)
    return {"Modell":label,"n":len(yt),"MAE":round(float(ae.mean()),3),"Median_AE":round(float(np.median(ae)),3),
            "RMSE":round(float(np.sqrt(mean_squared_error(yt,yp))),3),"R2":round(float(r2_score(yt,yp)),3),
            "Bias":round(float((yp-yt).mean()),3)}

fitted={}
log("log1p-Mean ..."); m=XGBRegressor(objective="reg:squarederror",**COMMON); m.fit(Xtr,np.log1p(los_tr)); fitted["log1p_Mean"]=("log",m)
for vp in [1.3,1.5,1.7]:
    log(f"Tweedie {vp} ..."); m=XGBRegressor(objective="reg:tweedie",tweedie_variance_power=vp,**COMMON); m.fit(Xtr,los_tr); fitted[f"Tweedie_{vp}"]=("raw",m)
log("Gamma ..."); m=XGBRegressor(objective="reg:gamma",**COMMON); m.fit(Xtr,los_tr); fitted["Gamma"]=("raw",m)

# Quantile P50/P80 (XGB reg:quantileerror; Fallback HistGBR)
def quantile_model(alpha):
    try:
        m=XGBRegressor(objective="reg:quantileerror",quantile_alpha=alpha,**COMMON); m.fit(Xtr,np.log1p(los_tr)); return ("log",m)
    except Exception as e:
        from sklearn.ensemble import HistGradientBoostingRegressor
        log(f"  XGB-Quantil nicht verfuegbar ({e}); Fallback HistGBR");
        m=HistGradientBoostingRegressor(loss="quantile",quantile=alpha,max_iter=500,learning_rate=0.05,random_state=RS); m.fit(Xtr,np.log1p(los_tr)); return ("log",m)
log("Quantil P50 ..."); fitted["Quantile_P50"]=quantile_model(0.5)
log("Quantil P80 ..."); fitted["Quantile_P80"]=quantile_model(0.8)

def predict(tag,Xmat):
    kind,m=fitted[tag]; p=m.predict(Xmat); return np.clip(np.expm1(p) if kind=="log" else p,0,None)

# Diskretes Hazard-Modell
log("Hazard: Person-Period-Aufbau ...")
T_tr=np.clip(np.ceil(los_tr).astype(int),1,TMAX); counts=T_tr
Xrep=np.repeat(Xtr,counts,axis=0)
day_t=np.concatenate([np.arange(1,c+1) for c in counts]).astype(np.float64)
T_rep=np.repeat(T_tr,counts); y_haz=(day_t==T_rep).astype(int)
Xhaz=np.column_stack([Xrep,day_t]).astype(np.float32)
log(f"  Person-Period-Zeilen: {Xhaz.shape[0]:,}")
clf=XGBClassifier(objective="binary:logistic",n_estimators=400,max_depth=6,learning_rate=0.05,
                  subsample=0.8,colsample_bytree=0.8,min_child_weight=5,random_state=RS,n_jobs=-1,eval_metric="logloss")
clf.fit(Xhaz,y_haz); log("  Hazard-Klassifikator trainiert")
def hazard_predict(Xmat):
    n=Xmat.shape[0]; S=np.ones(n); E=np.zeros(n); med=np.full(n,np.nan)
    for t in range(1,TMAX+1):
        h=np.clip(clf.predict_proba(np.column_stack([Xmat,np.full(n,t)]).astype(np.float32))[:,1],1e-6,1-1e-6)
        E+=S; Snew=S*(1-h); newly=(med!=med)&(Snew<0.5); med[newly]=t; S=Snew
    med[med!=med]=TMAX; return E,med

# ---- Holdout-Auswertung nach Subgruppen
log("Holdout-Auswertung ...")
preds={t:predict(t,Xte) for t in fitted}
E_te,med_te=hazard_predict(Xte); preds["Hazard_E"]=E_te; preds["Hazard_Median"]=med_te
subs={"overall":np.ones(len(te),bool),"1-7d":(los_te>1)&(los_te<=7),">7d":los_te>7,">14d":los_te>14}
rows=[]
for sg,mask in subs.items():
    for tag,p in preds.items(): rows.append({**met(los_te,p,tag,mask),"Subgroup":sg})
retro=pd.DataFrame(rows)[["Subgroup","Modell","n","MAE","Median_AE","RMSE","R2","Bias"]]
retro.to_csv(OUT/"mimic_alt_models_retro.csv",sep=";",index=False)
print("\n=== MIMIC Holdout — alternative Ansaetze (Tage) ==="); print(retro.to_string(index=False))

# ---- P80-Coverage (Kapazitaetsplanung)
covrows=[]
for label in ["log1p_Mean","Quantile_P50","Quantile_P80","Hazard_Median"]:
    for sg,mask in subs.items():
        cov=float((los_te[mask]<=preds[label][mask]).mean())
        covrows.append({"Estimator":label,"Subgroup":sg,"Coverage_%":round(100*cov,1),
                        "Mean_pred":round(float(preds[label][mask].mean()),2),"Mean_obs":round(float(los_te[mask].mean()),2)})
cov=pd.DataFrame(covrows); cov.to_csv(OUT/"mimic_p80_coverage.csv",sep=";",index=False)
print("\n=== P80-Coverage (Anteil Stays mit beobachteter LoS <= Vorhersage) ==="); print(cov.to_string(index=False))

# ========================= FIGUREN =========================
# Fig A: MAE je Ansatz nach Subgruppen (Heatmap-artige Balken)
order=["log1p_Mean","Tweedie_1.3","Tweedie_1.5","Tweedie_1.7","Gamma","Quantile_P50","Hazard_E","Hazard_Median"]
order=[o for o in order if o in set(retro["Modell"])]
fig,ax=plt.subplots(figsize=(11,5.5)); sgs=["overall","1-7d",">7d",">14d"]; w=0.2; xb=np.arange(len(order))
cols={"overall":"#1f5f9e","1-7d":"#5b9bd5",">7d":"#e08214","›14d":"#b2182b",">14d":"#b2182b"}
for i,sg in enumerate(sgs):
    vals=[retro[(retro.Modell==o)&(retro.Subgroup==sg)]["MAE"].iloc[0] for o in order]
    ax.bar(xb+(i-1.5)*w,vals,w,label=sg,color=cols[sg])
ax.set_xticks(xb); ax.set_xticklabels(order,rotation=30,ha="right",fontsize=9); ax.set_ylabel("MAE (days)")
ax.set_title("MIMIC-IV: alternative LoS approaches by actual-LoS subgroup (holdout)",weight="bold",fontsize=12); ax.legend(title="observed LoS",fontsize=9)
fig.tight_layout(); fig.savefig(OUT/"fig_mimic_alt_mae.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# Fig B: P80-Coverage
fig,ax=plt.subplots(figsize=(9,4.8)); ests=["log1p_Mean","Quantile_P50","Quantile_P80","Hazard_Median"]; w=0.2; xb=np.arange(len(sgs))
palette={"log1p_Mean":"#1f5f9e","Quantile_P50":"#5b9bd5","Quantile_P80":"#1a9850","Hazard_Median":"#762a83"}
for i,es in enumerate(ests):
    vals=[cov[(cov.Estimator==es)&(cov.Subgroup==sg)]["Coverage_%"].iloc[0] for sg in sgs]
    ax.bar(xb+(i-1.5)*w,vals,w,label=es,color=palette[es])
ax.axhline(80,color="#d6604d",ls="--",lw=1.5,label="80% target")
ax.set_xticks(xb); ax.set_xticklabels(sgs); ax.set_ylabel("Coverage (%)  observed ≤ predicted")
ax.set_title("MIMIC-IV: capacity-planning coverage (P80 vs point estimators)",weight="bold",fontsize=12); ax.legend(fontsize=8.5,ncol=2)
fig.tight_layout(); fig.savefig(OUT/"fig_mimic_p80_coverage.png",dpi=300,bbox_inches="tight"); plt.close(fig)

log("FERTIG. mimic_alt_models_retro.csv, mimic_p80_coverage.csv, fig_mimic_alt_mae.png, fig_mimic_p80_coverage.png")
