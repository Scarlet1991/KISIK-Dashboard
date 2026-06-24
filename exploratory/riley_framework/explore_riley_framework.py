# -*- coding: utf-8 -*-
"""EXPLORATIV (gesondert, nicht Teil der Manuskript-Pipeline): Riley/Collins-Rahmen fuer KISIK.
Leckfreies Feature-Set (8-98f entfernt). Bausteine:
 (1) Kalibrationssteigung beta + flexible Kalibrationskurve (retro OOF + prospektiv)
 (2) lineare Rekalibrierung (alpha,beta aus OOF-Dev, eingefroren, prospektiv geprueft)
 (3) log1p-Ruecktransformationsbias -> Duan-Smearing vs naiv vs Tweedie
 (4) Bootstrap-Instabilitaetsanalyse (prediction instability + Instabilitaetsindex)
 (5) Langlieger als Estimand: Klassifikator P(LOS>=7) AUROC/Sensitivitaet/Kalibration
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
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, brier_score_loss
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"
OUT=AN/"exploratory_riley"; OUT.mkdir(parents=True,exist_ok=True)
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; rng=np.random.default_rng(RS)
bp=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]
bpet=bp["ExtraTrees"]; bptw=bp["Tweedie"]

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
def slope(obs,pred): b,a=np.polyfit(np.clip(pred,0,None),obs,1); return b,a

# prospektive Kohorte (no_isopen, rekonstruiert)
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float)
Xp=pd.DataFrame(index=PR.index)
for c in present: Xp[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
Xp=Xp[present]

# ===== Modelle =====
ET=TransformedTargetRegressor(Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=-1))]),func=np.log1p,inverse_func=np.expm1)
ET.fit(Xtr,ytr)
ETlog=Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=-1))])  # arbeitet auf log1p-Skala
ETlog.fit(Xtr,np.log1p(ytr))
TW=Pipeline([("pre",pre()),("mdl",XGBRegressor(objective="reg:tweedie",**bptw,random_state=RS,n_jobs=-1,tree_method="hist"))])
TW.fit(Xtr,ytr)

def report(obs,pred,lab):
    pred=np.clip(pred,0,None); b,a=slope(obs,pred); ae=np.abs(obs-pred); m=obs>7
    return dict(model=lab,MAE=round(float(ae.mean()),3),R2=round(float(r2_score(obs,pred)),3),
                calib_slope=round(float(b),3),calib_int=round(float(a),3),
                bias_all=round(float((pred-obs).mean()),3),bias_gt7=round(float((pred[m]-obs[m]).mean()),3),
                MAE_gt7=round(float(ae[m].mean()),3))

print("="*70,"\n(1) KALIBRATION + (3) Ruecktransformation\n","="*70)
gkf=GroupKFold(4)
# OOF dev predictions (naive ET expm1)
oof=cross_val_predict(ET,Xtr,ytr,groups=gtr,cv=gkf,n_jobs=-1)
b_oof,a_oof=slope(ytr,oof)
print(f"DEV OOF calibration slope beta={b_oof:.3f}, intercept alpha={a_oof:.3f}  (beta>1 => Prognosen zu eng)")
rows=[]
# naive on holdout & prospective
rows.append({**report(yte,ET.predict(Xte),"ET log1p naive (hold-out)"),"set":"holdout"})
rows.append({**report(los,ET.predict(Xp),"ET log1p naive (prospective)"),"set":"prospective"})

# (3) Duan-Smearing aus OOF-Residuen (log-Skala)
zhat_oof=cross_val_predict(ETlog,Xtr,np.log1p(ytr),groups=gtr,cv=gkf,n_jobs=-1)
resid=np.log1p(ytr)-zhat_oof; smear=float(np.mean(np.exp(resid)))
print(f"Duan-Smearing-Faktor (OOF, log-Skala) = {smear:.4f}  (>1 => naive Rueck-Transf. unterschaetzt)")
def smeared(Xset): z=ETlog.predict(Xset); return np.clip(np.exp(z)*smear-1,0,None)
rows.append({**report(yte,smeared(Xte),"ET smearing (hold-out)"),"set":"holdout"})
rows.append({**report(los,smeared(Xp),"ET smearing (prospective)"),"set":"prospective"})
# Tweedie (kein Rueck-Transf.-Bias)
rows.append({**report(yte,TW.predict(Xte),"Tweedie (hold-out)"),"set":"holdout"})
rows.append({**report(los,TW.predict(Xp),"Tweedie (prospective)"),"set":"prospective"})

print("\n"+"="*70,"\n(2) LINEARE REKALIBRIERUNG (alpha,beta aus OOF-Dev, eingefroren)\n","="*70)
# recal(p) = a_oof + b_oof * p  (auf Dev geschaetzt -> auf holdout/prosp angewandt)
def recal(p): return np.clip(a_oof+b_oof*np.clip(p,0,None),0,None)
rows.append({**report(yte,recal(ET.predict(Xte)),"ET recalibrated (hold-out)"),"set":"holdout"})
rows.append({**report(los,recal(ET.predict(Xp)),"ET recalibrated (prospective)"),"set":"prospective"})
res=pd.DataFrame(rows)[["set","model","MAE","R2","calib_slope","calib_int","bias_all","bias_gt7","MAE_gt7"]]
res.to_csv(OUT/"riley_recalibration_smearing.csv",sep=";",index=False)
print(res.to_string(index=False))

# ---- Figur: flexible Kalibrationskurve prospektiv (naiv / smearing / recal / tweedie) ----
def calib_curve(pred,q=8):
    d=pd.DataFrame({"p":np.clip(pred,0,None),"o":los}); d["b"]=pd.qcut(d["p"],q,duplicates="drop"); g=d.groupby("b",observed=True)
    return g["p"].mean().to_numpy(),g["o"].mean().to_numpy()
CAP=25
fig,ax=plt.subplots(figsize=(7.5,7))
ax.plot([0,CAP],[0,CAP],"--",color="#888",lw=1.4,label="ideal")
for pred,lab,c in [(ET.predict(Xp),"naive expm1","#1f5f9e"),(smeared(Xp),f"smearing (×{smear:.2f})","#2e9e4f"),
                   (recal(ET.predict(Xp)),f"linear recal (β={b_oof:.2f})","#d9663b"),(TW.predict(Xp),"Tweedie","#8e44ad")]:
    mp,mo=calib_curve(pred); ax.plot(mp,mo,"o-",color=c,lw=1.8,ms=6,label=lab)
ax.set_xlim(0,CAP); ax.set_ylim(0,CAP); ax.set_aspect("equal","box")
ax.set_xlabel("Predicted ICU LoS (days)"); ax.set_ylabel("Observed ICU LoS (days)")
ax.set_title("Prospective calibration — back-transform / recalibration variants (leak-free)",weight="bold",fontsize=11)
ax.legend(fontsize=9,loc="upper left"); fig.tight_layout(); fig.savefig(str(OUT/"fig_calibration_variants.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

print("\n"+"="*70,"\n(4) BOOTSTRAP-INSTABILITAET (B=200 Refits, prospektive Patienten)\n","="*70)
B=200; n=len(Xtr); P=np.zeros((B,len(Xp)))
for bi in range(B):
    idx=rng.integers(0,n,size=n)
    m=TransformedTargetRegressor(Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=bi,n_jobs=-1))]),func=np.log1p,inverse_func=np.expm1)
    m.fit(Xtr.iloc[idx],ytr[idx]); P[bi]=np.clip(m.predict(Xp),0,None)
    if (bi+1)%50==0: print(f"  ... {bi+1}/{B}")
full=np.clip(ET.predict(Xp),0,None); sd=P.std(0); mean_b=P.mean(0)
instab=pd.DataFrame({"stay":PR["__stay_id__"].astype(str),"obs":los,"pred_full":full,"boot_mean":mean_b,"boot_sd":sd})
instab.to_csv(OUT/"riley_instability_per_stay.csv",sep=";",index=False)
print(f"Instabilitaetsindex (SD der Bootstrap-Prognosen): Median {np.median(sd):.2f} d, p90 {np.percentile(sd,90):.2f} d")
print(f"Bootstrap-Prognosespanne max (p2.5-p97.5) Median: {np.median(np.percentile(P,97.5,0)-np.percentile(P,2.5,0)):.2f} d")
mlong=los>7
print(f"SD bei Langliegern (>7d): Median {np.median(sd[mlong]):.2f} d vs kurz (<=7d): {np.median(sd[~mlong]):.2f} d")
# prediction instability plot
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.4))
samp=rng.choice(len(Xp),min(60,len(Xp)),replace=False)
for j in samp: a1.plot([full[j]]*B,P[:,j],".",color="#1f5f9e",alpha=0.12,ms=3)
a1.plot([0,CAP],[0,CAP],"--",color="#888"); a1.set_xlim(0,CAP); a1.set_ylim(0,CAP); a1.set_aspect("equal","box")
a1.set_xlabel("Prediction from full model (days)"); a1.set_ylabel("Predictions across 200 bootstrap models")
a1.set_title("(A) Prediction instability",weight="bold",fontsize=11)
a2.scatter(los,sd,s=14,color="#762a83",alpha=.6); a2.axvline(7,color="#c0392b",ls="--",lw=1)
a2.set_xlabel("Observed ICU LoS (days)"); a2.set_ylabel("Per-stay bootstrap SD (days)")
a2.set_title("(B) Instability index vs observed LoS",weight="bold",fontsize=11)
fig.suptitle(f"Bootstrap instability (B={B}, leak-free Extra Trees)",weight="bold",fontsize=12)
fig.tight_layout(); fig.savefig(str(OUT/"fig_instability.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

print("\n"+"="*70,"\n(5) LANGLIEGER-ESTIMAND: Klassifikator P(LOS>=7)\n","="*70)
y7tr=(ytr>=7).astype(int); y7te=(yte>=7).astype(int); y7p=(los>=7).astype(int)
clf=Pipeline([("pre",pre()),("mdl",ExtraTreesClassifier(n_estimators=400,min_samples_leaf=5,max_features=0.5,random_state=RS,n_jobs=-1))])
clf.fit(Xtr,y7tr)
def clf_eval(Xset,yt,lab):
    pp=clf.predict_proba(Xset)[:,1]; auc=roc_auc_score(yt,pp); br=brier_score_loss(yt,pp)
    thr=0.5; sens=((pp>=thr)&(yt==1)).sum()/max(yt.sum(),1); spec=((pp<thr)&(yt==0)).sum()/max((yt==0).sum(),1)
    return pp,dict(set=lab,AUROC=round(auc,3),Brier=round(br,3),sens_at_0_5=round(float(sens),3),spec_at_0_5=round(float(spec),3),prev=round(float(yt.mean()),3))
crows=[]
_,r=clf_eval(Xte,y7te,"holdout"); crows.append(r)
ppp,r=clf_eval(Xp,y7p,"prospective"); crows.append(r)
# Regressionsprognose als Score fuer LOS>=7 (Vergleich)
auc_reg_te=roc_auc_score(y7te,ET.predict(Xte)); auc_reg_p=roc_auc_score(y7p,ET.predict(Xp))
print(pd.DataFrame(crows).to_string(index=False))
print(f"AUROC der REGRESSIONSprognose als LOS>=7-Score: holdout {auc_reg_te:.3f} | prospective {auc_reg_p:.3f}")
print(f"(dedizierter Klassifikator prospektiv AUROC {crows[1]['AUROC']:.3f})")
pd.DataFrame(crows).assign(AUROC_regression_score=[round(auc_reg_te,3),round(auc_reg_p,3)]).to_csv(OUT/"riley_longstay_classifier.csv",sep=";",index=False)

print(f"\nAlle Ausgaben in {OUT}")
