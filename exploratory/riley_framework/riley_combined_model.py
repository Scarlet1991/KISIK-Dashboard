# -*- coding: utf-8 -*-
"""EXPLORATIV: umgesetzte Riley-Konsequenz fuer KISIK (leckfreies Feature-Set, getrennt vom Manuskript).
A) kalibrierte kontinuierliche Prognose: Tweedie + eingefrorene lineare Rekalibrierung (statt naivem log1p)
B) separate Langlieger-Klassifikatoren P(LOS>=7) und P(LOS>=10) fuer die Kapazitaetsplanung
   (Schwellen-Tuning auf Dev-OOF, Kalibration, AUROC, erwartete vs beobachtete Fallzahl)
Ausgabe: Eigene Auswertung/exploratory_riley/
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
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score, brier_score_loss,
                             roc_curve, confusion_matrix)
from sklearn.calibration import calibration_curve
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"
OUT=AN/"exploratory_riley"; OUT.mkdir(parents=True,exist_ok=True)
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"
bp=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]; bptw=bp["Tweedie"]

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
Xtr,Xte,ytr,yte,gtr=X.iloc[tr],X.iloc[te],y[tr],y[te],groups[tr]
def pre(): return ColumnTransformer([("num",SimpleImputer(strategy="median"),numc),
    ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat)])
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float)
Xp=pd.DataFrame(index=PR.index)
for c in present: Xp[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
Xp=Xp[present]
gkf=GroupKFold(4)

# ===================== A) kalibrierte kontinuierliche Prognose =====================
print("="*72,"\nA) KONTINUIERLICH: Tweedie + eingefrorene lineare Rekalibrierung\n","="*72)
TW=Pipeline([("pre",pre()),("mdl",XGBRegressor(objective="reg:tweedie",**bptw,random_state=RS,n_jobs=-1,tree_method="hist"))])
TW.fit(Xtr,ytr)
# Rekalibrierung aus Dev-OOF schaetzen (obs ~ a + b*pred), einfrieren
oof=np.clip(cross_val_predict(TW,Xtr,ytr,groups=gtr,cv=gkf,n_jobs=-1),0,None)
b,a=np.polyfit(oof,ytr,1); print(f"Dev-OOF Tweedie-Kalibration: slope b={b:.3f}, intercept a={a:.3f} -> recal(p)=a+b*p eingefroren")
def recal(p): return np.clip(a+b*np.clip(p,0,None),0,None)
def cont(obs,pred,lab):
    pred=np.clip(pred,0,None); sb,sa=np.polyfit(pred,obs,1); m=obs>7
    return dict(model=lab,MAE=round(float(mean_absolute_error(obs,pred)),3),R2=round(float(r2_score(obs,pred)),3),
                calib_slope=round(float(sb),3),bias=round(float((pred-obs).mean()),3),bias_gt7=round(float((pred[m]-obs[m]).mean()),3))
controws=[{**cont(los,TW.predict(Xp),"Tweedie raw (prospective)"),"set":"prosp"},
          {**cont(los,recal(TW.predict(Xp)),"Tweedie + recalibration (prospective)"),"set":"prosp"},
          {**cont(yte,TW.predict(Xte),"Tweedie raw (holdout)"),"set":"holdout"},
          {**cont(yte,recal(TW.predict(Xte)),"Tweedie + recalibration (holdout)"),"set":"holdout"}]
contdf=pd.DataFrame(controws)[["set","model","MAE","R2","calib_slope","bias","bias_gt7"]]
contdf.to_csv(OUT/"combined_continuous_metrics.csv",sep=";",index=False); print(contdf.to_string(index=False))

# ===================== B) Langlieger-Klassifikatoren =====================
print("\n"+"="*72,"\nB) KLASSIFIKATOR P(LOS>=7) und P(LOS>=10) — Kapazitaetsplanung\n","="*72)
def make_clf(): return Pipeline([("pre",pre()),("mdl",ExtraTreesClassifier(n_estimators=600,min_samples_leaf=5,max_features=0.5,random_state=RS,n_jobs=-1,class_weight="balanced"))])
thr_table=[]; clf_summary=[]; probs={}
for K in (7,10):
    ytr_k=(ytr>=K).astype(int); yte_k=(yte>=K).astype(int); yp_k=(los>=K).astype(int)
    clf=make_clf(); clf.fit(Xtr,ytr_k)
    # OOF-Wahrscheinlichkeiten auf Dev fuer Schwellenwahl (Ziel-Sensitivitaet ~0.75)
    oofp=cross_val_predict(clf,Xtr,ytr_k,groups=gtr,cv=gkf,method="predict_proba",n_jobs=-1)[:,1]
    fpr,tpr,thr=roc_curve(ytr_k,oofp); want=0.75
    i=np.argmin(np.abs(tpr-want)); thr_star=float(thr[i])
    pp=clf.predict_proba(Xp)[:,1]; probs[K]=pp
    auc=roc_auc_score(yp_k,pp); br=brier_score_loss(yp_k,pp)
    clf_summary.append(dict(target=f"LOS>={K}d",prev_prosp=round(float(yp_k.mean()),3),AUROC_prosp=round(float(auc),3),
                            Brier_prosp=round(float(br),3),thr_star=round(thr_star,3),
                            exp_count=round(float(pp.sum()),1),obs_count=int(yp_k.sum())))
    # Schwellen-Tabelle prospektiv
    for t in [0.2,0.3,0.4,thr_star,0.5]:
        pred=(pp>=t).astype(int); tn,fp,fn,tp=confusion_matrix(yp_k,pred,labels=[0,1]).ravel()
        sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1); ppv=tp/max(tp+fp,1); npv=tn/max(tn+fn,1)
        thr_table.append(dict(target=f"LOS>={K}d",threshold=round(float(t),3),flagged=int((pred==1).sum()),
                              sens=round(sens,3),spec=round(spec,3),PPV=round(ppv,3),NPV=round(npv,3),
                              note=("<-OOF-tuned ~0.75 sens" if abs(t-thr_star)<1e-9 else "")))
cs=pd.DataFrame(clf_summary); tt=pd.DataFrame(thr_table)
cs.to_csv(OUT/"combined_longstay_summary.csv",sep=";",index=False); tt.to_csv(OUT/"combined_longstay_thresholds.csv",sep=";",index=False)
print(cs.to_string(index=False)); print("\nSchwellen-Tabelle (prospektiv):"); print(tt.to_string(index=False))

# ===================== Figuren =====================
fig,ax=plt.subplots(1,3,figsize=(16,5))
# (A) Kalibration kontinuierlich
CAP=25
def cc(pred,q=8):
    d=pd.DataFrame({"p":np.clip(pred,0,None),"o":los}); d["b"]=pd.qcut(d["p"],q,duplicates="drop"); g=d.groupby("b",observed=True)
    return g["p"].mean().to_numpy(),g["o"].mean().to_numpy()
ax[0].plot([0,CAP],[0,CAP],"--",color="#888",label="ideal")
for pred,lab,c in [(TW.predict(Xp),"Tweedie raw","#8e44ad"),(recal(TW.predict(Xp)),"Tweedie + recal","#d9663b")]:
    mp,mo=cc(pred); ax[0].plot(mp,mo,"o-",color=c,lw=1.8,ms=6,label=lab)
ax[0].set_xlim(0,CAP); ax[0].set_ylim(0,CAP); ax[0].set_aspect("equal","box")
ax[0].set_xlabel("Predicted LoS (days)"); ax[0].set_ylabel("Observed LoS (days)")
ax[0].set_title("(A) Continuous calibration (prospective)",weight="bold",fontsize=11); ax[0].legend(fontsize=9,loc="upper left")
# (B) ROC LOS>=7 / >=10
for K,c in [(7,"#1f5f9e"),(10,"#c0392b")]:
    yk=(los>=K).astype(int); fpr,tpr,_=roc_curve(yk,probs[K]); auc=roc_auc_score(yk,probs[K])
    ax[1].plot(fpr,tpr,color=c,lw=2,label=f"LOS≥{K}d (AUROC {auc:.2f})")
ax[1].plot([0,1],[0,1],"--",color="#888"); ax[1].set_xlabel("1 − specificity"); ax[1].set_ylabel("sensitivity")
ax[1].set_title("(B) Long-stay classifier ROC (prospective)",weight="bold",fontsize=11); ax[1].legend(fontsize=9,loc="lower right")
# (C) Wahrscheinlichkeits-Kalibration LOS>=7
yk=(los>=7).astype(int); fp,mp=calibration_curve(yk,probs[7],n_bins=5,strategy="quantile")
ax[2].plot([0,1],[0,1],"--",color="#888",label="ideal")
ax[2].plot(mp,fp,"o-",color="#1f5f9e",lw=1.8,ms=6,label="P(LOS≥7d)")
ax[2].set_xlabel("Predicted probability"); ax[2].set_ylabel("Observed frequency")
ax[2].set_title("(C) Probability calibration LOS≥7d",weight="bold",fontsize=11); ax[2].legend(fontsize=9,loc="upper left")
fig.suptitle("Riley-informed combined model (leak-free): calibrated continuous + long-stay classifier",weight="bold",fontsize=13)
fig.tight_layout(); fig.savefig(str(OUT/"fig_combined_model.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print(f"\nAlle Ausgaben in {OUT} (combined_*.csv, fig_combined_model.png)")
