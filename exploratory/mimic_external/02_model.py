# -*- coding: utf-8 -*-
"""
MIMIC-IV externe Validierung — Schritt 2: Modelltraining + Auswertung (spiegelt KISIK-Methodik).
- Patientengruppierter 80/20-Split (GroupShuffleSplit, seed 42), log1p-Ziel, expm1+clip(0).
- 4 Modelle (Ridge/RF/ExtraTrees/XGBoost), 4-fold GroupKFold-CV-MAE -> Auswahl des finalen Modells.
- Holdout-Metriken (MAE/MedianAE/RMSE/R²/Bias) + gepaarter Bootstrap-95%-KI.
- Null-Baseline = Trainings-Median (unverändert auf Holdout angewandt).
- Kalibration: Slope (obs~pred)+KI, CITL, nach echten LoS-Gruppen (Modell vs Null).
- Permutations-Importance, Figuren, KISIK-vs-MIMIC-Vergleichstabelle, summary.json.
"""
import sys, io, json, time, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    from xgboost import XGBRegressor; HAS_XGB=True
except Exception: HAS_XGB=False

OUT=Path(r"D:\Ausgangsdaten\KISIK Projekt\mimic_external"); RS=42; CAP=20
KIS=Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung\canonical")
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

X=pd.read_parquet(OUT/"mimic_features.parquet")
y=X["los_days"].to_numpy(float); groups=X["subject_id"].astype(str).to_numpy()
CAT=[c for c in ["first_careunit","gender","admission_type"] if c in X.columns]
drop=["los_days","subject_id"]
feats=[c for c in X.columns if c not in drop]
num=[c for c in feats if c not in CAT]
Xf=X[feats].copy()
for c in num: Xf[c]=pd.to_numeric(Xf[c],errors="coerce")
for c in CAT: Xf[c]=Xf[c].astype(str)
log(f"Kohorte {len(Xf):,} Stays | {len(feats)} Features ({len(num)} numerisch, {len(CAT)} kategorial)")

tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(Xf,y,groups))
Xtr,Xte,ytr,yte,gtr=Xf.iloc[tr],Xf.iloc[te],y[tr],y[te],groups[tr]
log(f"Train {len(tr):,} | Holdout {len(te):,}")

def pre(scale=False):
    ns=[("imp",SimpleImputer(strategy="median"))]+([("sc",StandardScaler())] if scale else [])
    return ColumnTransformer([("num",Pipeline(ns),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                         ("ohe",OneHotEncoder(handle_unknown="ignore"))]),CAT)])
def ttr(mdl,scale=False): return TransformedTargetRegressor(
    regressor=Pipeline([("pre",pre(scale)),("mdl",mdl)]), func=np.log1p, inverse_func=np.expm1)

models={"Ridge":ttr(Ridge(alpha=10.0,random_state=RS),scale=True),
        "RandomForest":ttr(RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=RS)),
        "ExtraTrees":ttr(ExtraTreesRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=RS))}
if HAS_XGB: models["XGBoost"]=ttr(XGBRegressor(n_estimators=400,learning_rate=0.05,max_depth=6,
        subsample=0.9,colsample_bytree=0.8,n_jobs=-1,random_state=RS,verbosity=0,tree_method="hist"))

def clip(p): return np.clip(p,0,None)
def mets(yt,yp):
    yp=clip(yp); e=yp-yt
    return dict(MAE=mean_absolute_error(yt,yp),MedianAE=float(np.median(np.abs(e))),
                RMSE=float(np.sqrt(np.mean(e**2))),R2=r2_score(yt,yp),Bias=float(e.mean()))

# ---- 4-fold GroupKFold CV-MAE auf Train -> Modellauswahl
gkf=GroupKFold(n_splits=4); rows=[]; fitted={}
for nm,m in models.items():
    log(f"CV {nm} ...")
    cvp=clip(cross_val_predict(m,Xtr,ytr,cv=gkf,groups=gtr,n_jobs=1))
    cv_mae=mean_absolute_error(ytr,cvp)
    m.fit(Xtr,ytr); fitted[nm]=m
    hm=mets(yte,m.predict(Xte))
    rows.append({"Modell":nm,"n":len(te),"CV_MAE":round(cv_mae,3),
                 "MAE":round(hm["MAE"],3),"MedianAE":round(hm["MedianAE"],3),
                 "RMSE":round(hm["RMSE"],3),"R2":round(hm["R2"],3),"Bias":round(hm["Bias"],3)})
    log(f"  {nm}: CV-MAE {cv_mae:.3f} | Holdout MAE {hm['MAE']:.3f} R² {hm['R2']:.3f}")
res=pd.DataFrame(rows).sort_values("CV_MAE")
final=res.iloc[0]["Modell"]; res.to_csv(OUT/"mimic_metrics.csv",sep=";",index=False)
log(f"Finales Modell (niedrigste CV-MAE): {final}")
pred=clip(fitted[final].predict(Xte))

# ---- Null-Baseline (Trainings-Median)
train_med=float(np.median(ytr)); null_mae=float(np.mean(np.abs(yte-train_med)))
log(f"Null-Baseline (Train-Median {train_med:.2f}d): Holdout-MAE {null_mae:.2f} | Modell {mets(yte,pred)['MAE']:.2f}")

# ---- Bootstrap-95%-KIs (Holdout, finales Modell)
rng=np.random.default_rng(RS); B=2000; n=len(yte); idx=rng.integers(0,n,(B,n))
bt={k:[] for k in ["MAE","RMSE","R2","Bias"]}
for b in idx:
    mm=mets(yte[b],pred[b])
    for k in bt: bt[k].append(mm[k])
ci={k:(round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)) for k,v in bt.items()}
pt=mets(yte,pred)
boot=pd.DataFrame([{"Metric":k,"Estimate":round(pt[k],3),"CI_low":ci[k][0],"CI_high":ci[k][1]} for k in bt])
boot.to_csv(OUT/"mimic_bootstrap_ci.csv",sep=";",index=False)

# ---- Kalibration: Slope (obs~pred)+KI, CITL
def slope(yt,yp): b,a=np.polyfit(clip(yp),yt,1); return b
sl=slope(yte,pred); sl_bt=[slope(yte[b],pred[b]) for b in idx]
sl_ci=(round(float(np.percentile(sl_bt,2.5)),3),round(float(np.percentile(sl_bt,97.5)),3))
citl=float((clip(pred)-yte).mean())

# ---- nach echten LoS-Gruppen: Modell vs Null
bins=[("1-2 d",(yte>1)&(yte<=2)),("2-4 d",(yte>2)&(yte<=4)),("4-7 d",(yte>4)&(yte<=7)),(">7 d",yte>7)]
grp=[]
for nm,mk in bins:
    if mk.sum()==0: continue
    grp.append({"LoS_bin":nm,"n":int(mk.sum()),"obs_median":round(float(np.median(yte[mk])),2),
                "MAE_model":round(float(np.mean(np.abs(pred[mk]-yte[mk]))),2),
                "Bias_model":round(float(np.mean(pred[mk]-yte[mk])),2),
                "MAE_null":round(float(np.mean(np.abs(train_med-yte[mk]))),2)})
grp=pd.DataFrame(grp); grp.to_csv(OUT/"mimic_by_losgroup.csv",sep=";",index=False)

# ---- Permutations-Importance (finales Modell; Subsample für Tempo)
log("Permutations-Importance ...")
si=rng.choice(len(te),size=min(4000,len(te)),replace=False)
pi=permutation_importance(fitted[final],Xte.iloc[si],yte[si],scoring="neg_mean_absolute_error",
                          n_repeats=5,random_state=RS,n_jobs=2)
imp=pd.DataFrame({"Feature":feats,"MAE_increase_days":pi.importances_mean,"sd":pi.importances_std}
                 ).sort_values("MAE_increase_days",ascending=False)
imp.to_csv(OUT/"mimic_importance.csv",sep=";",index=False)

# ---- summary.json
summary=dict(n_stays=int(len(Xf)),n_patients=int(pd.Series(groups).nunique()),
             n_train=int(len(tr)),n_test=int(len(te)),n_features=len(feats),
             los_median=round(float(np.median(y)),2),los_mean=round(float(y.mean()),2),
             los_std=round(float(y.std()),2),pct_gt7=round(100*float((y>7).mean()),1),
             final_model=final,train_median=round(train_med,2),null_mae_holdout=round(null_mae,2),
             calib_slope=round(sl,3),calib_slope_ci=list(sl_ci),citl=round(citl,3))
json.dump(summary,open(OUT/"mimic_summary.json","w"),indent=2)

# ========================= FIGUREN =========================
PHYC,MODC,NULC="#c0392b","#1f5f9e","#c7ccd1"
# Fig A: Modellvergleich (MAE + R²) MIMIC, mit Null-Linie
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.6)); order=list(res["Modell"]); xr=np.arange(len(order))
a1.bar(xr,res["MAE"],color=MODC); a1.axhline(null_mae,color=NULC,ls="--",lw=1.6)
a1.text(len(order)-1,null_mae+0.03,f"null model (MAE {null_mae:.2f} d)",color="#777",ha="right",fontsize=8.5)
a1.bar_label(a1.containers[0],fmt="%.2f",fontsize=8,padding=2)
a1.set_xticks(xr);a1.set_xticklabels(order,fontsize=9);a1.set_ylabel("MAE (days)");a1.set_title("(A) MIMIC holdout MAE",weight="bold")
a2.bar(xr,res["R2"],color=MODC); a2.bar_label(a2.containers[0],fmt="%.2f",fontsize=8,padding=2)
a2.set_xticks(xr);a2.set_xticklabels(order,fontsize=9);a2.set_ylabel("R²");a2.set_title("(B) MIMIC holdout R²",weight="bold")
fig.suptitle(f"MIMIC-IV external validation: model performance (holdout n={len(te):,})",weight="bold",fontsize=12.5)
fig.tight_layout(); fig.savefig(OUT/"fig_mimic_models.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# Fig B: Kalibration (obs vs pred, binned) + Vorhersageverteilung
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5)); pc=clip(pred)
a1.scatter(pc,yte,s=6,alpha=.15,color=MODC,edgecolor="none"); a1.plot([0,CAP],[0,CAP],"--",color=PHYC,lw=1.5,label="Identity")
qs=np.quantile(pc,np.linspace(0,1,9)); mp=[];mo=[]
for i in range(8):
    mk=(pc>qs[i])&(pc<=qs[i+1])
    if mk.sum()>20: mp.append(pc[mk].mean()); mo.append(yte[mk].mean())
a1.plot(mp,mo,"o-",color="#762a83",lw=1.6,label="binned observed mean")
a1.set_xlim(0,CAP);a1.set_ylim(0,CAP);a1.set_aspect("equal");a1.set_xlabel("Predicted LoS (d)");a1.set_ylabel("Observed LoS (d)")
a1.set_title(f"(A) Calibration ({final})\nslope {sl:.2f} [{sl_ci[0]:.2f},{sl_ci[1]:.2f}], CITL {citl:+.2f} d",weight="bold",fontsize=10.5);a1.legend(fontsize=8.5)
bins2=np.linspace(0,CAP,31)
a2.hist(np.clip(yte,0,CAP),bins=bins2,alpha=.45,color="#7f8c8d",density=True,label="Observed LoS")
a2.hist(np.clip(pc,0,CAP),bins=bins2,histtype="step",lw=2,color=MODC,density=True,label=f"{final} predicted")
a2.set_xlabel("ICU LoS (days)");a2.set_ylabel("Density");a2.set_title("(B) Prediction distribution",weight="bold");a2.legend(fontsize=9)
fig.suptitle(f"MIMIC-IV external validation: calibration (holdout n={len(te):,})",weight="bold",fontsize=12.5)
fig.tight_layout(); fig.savefig(OUT/"fig_mimic_calibration.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# Fig C: nach LoS-Gruppen (Modell vs Null)
fig,ax=plt.subplots(figsize=(8.5,4.8)); xb=np.arange(len(grp)); w=0.38
ax.bar(xb-w/2,grp["MAE_model"],w,label=f"{final}",color=MODC)
ax.bar(xb+w/2,grp["MAE_null"],w,label=f"null (const. {train_med:.1f} d)",color=NULC)
for c in ax.containers: ax.bar_label(c,fmt="%.2f",fontsize=7.5,padding=2)
for i,r in grp.iterrows(): ax.annotate(f"n={int(r['n'])}",(xb[i],max(r['MAE_model'],r['MAE_null'])+0.2),ha="center",fontsize=8,color="#777")
ax.set_xticks(xb);ax.set_xticklabels(grp["LoS_bin"]);ax.set_ylabel("MAE (days)")
ax.set_title("MIMIC-IV: MAE by actual LoS — model vs null baseline",weight="bold",fontsize=12);ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT/"fig_mimic_losgroup.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ========================= KISIK-VS-MIMIC-VERGLEICH =========================
try:
    kret=pd.read_csv(KIS/"metrics_retrospective.csv",sep=";").set_index("Modell")
    knull=pd.read_csv(KIS/"prospective_null_baseline.csv",sep=";")
    ket=kret.loc["ExtraTrees"]
    cmp=pd.DataFrame([
        {"Dataset":"KISIK (retro holdout)","n_holdout":int(ket.get("n",2601)),"final":"ExtraTrees",
         "MAE":round(float(ket["MAE_days"]),2),"R2":round(float(ket["R2"]),2),
         "null_MAE":round(float(knull.iloc[0]["Null_MAE_trainMed"]),2),
         "beats_null":"yes" if float(ket["MAE_days"])<float(knull.iloc[0]["Null_MAE_trainMed"]) else "no"},
        {"Dataset":"MIMIC-IV (holdout)","n_holdout":int(len(te)),"final":final,
         "MAE":round(pt["MAE"],2),"R2":round(pt["R2"],2),"null_MAE":round(null_mae,2),
         "beats_null":"yes" if pt["MAE"]<null_mae else "no"},
    ])
    cmp.to_csv(OUT/"kisik_vs_mimic.csv",sep=";",index=False)
    log("KISIK-vs-MIMIC Vergleich:\n"+cmp.to_string(index=False))
except Exception as e:
    log(f"Vergleich übersprungen: {e}")

log(f"FERTIG. final={final} | Holdout MAE {pt['MAE']:.2f} R² {pt['R2']:.2f} | null {null_mae:.2f} | slope {sl:.2f} {sl_ci}")
print(res.to_string(index=False))
