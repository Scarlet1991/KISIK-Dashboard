# -*- coding: utf-8 -*-
"""
SIGNIFIKANZTEST: Ist irgendein ML-Ansatz dem Oberarzt UEBERLEGEN (nicht nur verschieden)?

Pro Ansatz, gepaart pro Stay gegen den Oberarzt:
  e_arzt = |LoS - Arzt|,  e_ml = |LoS - Vorhersage|
  diff   = e_arzt - e_ml          (> 0  => ML besser)
Tests (alle gerichtet auf UEBERLEGENHEIT des ML):
  1) Einseitiger Wilcoxon-Vorzeichen-Rang-Test  H1: e_arzt > e_ml  (ML besser)
  2) Paired Bootstrap (B=5000) auf dMAE = MAE_Arzt - MAE_ML, 95%-CI
     -> signifikant ueberlegen, wenn das gesamte CI > 0 liegt
Ausgewertet: overall + je LoS-Subgruppe, fuer beide Kohorten (is_open=0 n=193, no_isopen n=286).

Quellen: alt_matrices/ (retro_train, feature_lists) + prospective_rebuilt_{193,286}.parquet,
canonical/summary.json (beste Hyperparameter der finalen Modelle).
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor, XGBClassifier

AN=Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung"); CAN=AN/"canonical"
ALT=CAN/"alt_matrices"; ALT286=CAN/"alt_matrices_no_isopen"
OUTX=AN/"exploratory_no_isopen"; RS=42; TMAX=90; B=5000
rng=np.random.default_rng(RS)

fl=json.load(open(ALT/"feature_lists.json")); present,numc,cat=fl["present"],fl["numc"],fl["cat"]
tr=pd.read_parquet(ALT/"retro_train.parquet"); ytr=tr["__y__"].to_numpy(float); Xtr_df=tr[present]
bp=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]

# ---------- kanonische Modelle (sklearn-Pipelines, identisch zum Manuskript) ----------
def pre(scale=False):
    ns=[("imp",SimpleImputer(strategy="median"))]+([("sc",StandardScaler())] if scale else [])
    return ColumnTransformer([("num",Pipeline(ns),numc),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)])
def ttr(reg,scale=False): return TransformedTargetRegressor(Pipeline([("pre",pre(scale)),("mdl",reg)]),func=np.log1p,inverse_func=np.expm1)
def plain(reg): return Pipeline([("pre",pre(False)),("mdl",reg)])
canon={
 "Ridge":ttr(Ridge(**bp["Ridge"],random_state=RS),scale=True),
 "RandomForest":ttr(RandomForestRegressor(**bp["RandomForest"],random_state=RS,n_jobs=1)),
 "ExtraTrees":ttr(ExtraTreesRegressor(**bp["ExtraTrees"],random_state=RS,n_jobs=1)),
 "XGBoost":ttr(XGBRegressor(**bp["XGBoost"],random_state=RS,n_jobs=1,tree_method="hist")),
}
if "Tweedie" in bp:
    canon["Tweedie"]=plain(XGBRegressor(objective="reg:tweedie",**bp["Tweedie"],random_state=RS,n_jobs=1,tree_method="hist"))
print("Training kanonische Modelle ..."); [m.fit(Xtr_df,ytr) for m in canon.values()]

# ---------- alternative Zielfunktionen (XGB design matrix) ----------
def design(frame):
    parts=[frame[numc].apply(pd.to_numeric,errors="coerce")]
    if cat: parts.append(pd.get_dummies(frame[cat].astype(str),prefix=cat).astype(float))
    X=pd.concat(parts,axis=1); X.columns=[str(c) for c in X.columns]; return X
Xtr=design(Xtr_df); COLS=Xtr.columns.tolist()
Xtr_v=Xtr.to_numpy(dtype=np.float64,na_value=np.nan)
COMMON=dict(n_estimators=600,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,random_state=RS,n_jobs=-1)
alt={}
print("Training alternative Zielfunktionen ...")
m=XGBRegressor(objective="reg:tweedie",tweedie_variance_power=1.5,**COMMON); m.fit(Xtr_v,ytr); alt["Tweedie_1.5"]=("raw",m)
for a,nm in [(0.5,"Quantile_P50"),(0.8,"Quantile_P80")]:
    m=XGBRegressor(objective="reg:quantileerror",quantile_alpha=a,**COMMON); m.fit(Xtr_v,np.log1p(ytr)); alt[nm]=("log",m)
# Hazard (E[T])
T_tr=np.clip(np.ceil(ytr).astype(int),1,TMAX); Xrep=np.repeat(Xtr_v,T_tr,axis=0)
day_t=np.concatenate([np.arange(1,c+1) for c in T_tr]).astype(np.float64); y_haz=(day_t==np.repeat(T_tr,T_tr)).astype(int)
clf=XGBClassifier(objective="binary:logistic",n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,random_state=RS,n_jobs=-1,eval_metric="logloss")
clf.fit(np.column_stack([Xrep,day_t]).astype(np.float32),y_haz)
def hazard_E(Xv):
    n=Xv.shape[0]; S=np.ones(n); E=np.zeros(n)
    for t in range(1,TMAX+1):
        h=np.clip(clf.predict_proba(np.column_stack([Xv,np.full(n,t)]).astype(np.float32))[:,1],1e-6,1-1e-6)
        E+=S; S=S*(1-h)
    return E

def predict_all(Xdf):
    Xv=design(Xdf)
    for c in COLS:
        if c not in Xv.columns: Xv[c]=0.0
    Xv=Xv[COLS].to_numpy(dtype=np.float64,na_value=np.nan)
    P={n:np.clip(m.predict(Xdf),0,None) for n,m in canon.items()}
    for tag,(kind,m) in alt.items():
        p=m.predict(Xv); P[tag]=np.clip(np.expm1(p) if kind=="log" else p,0,None)
    P["Hazard_E"]=np.clip(hazard_E(Xv),0,None)
    return P

# ---------- Tests ----------
def superiority(los,arzt,pred,mask):
    yt=los[mask]; a=arzt[mask]; p=pred[mask]
    e_a=np.abs(yt-a); e_m=np.abs(yt-p)
    mae_a=e_a.mean(); mae_m=e_m.mean(); dmae=mae_a-mae_m   # >0 => ML besser
    # einseitiger Wilcoxon: H1 e_arzt > e_ml  (ML besser)
    try: pgt=stats.wilcoxon(e_a,e_m,alternative="greater").pvalue
    except Exception: pgt=np.nan
    # paired bootstrap auf dMAE
    n=len(yt); idx=rng.integers(0,n,size=(B,n))
    boot=(e_a[idx].mean(axis=1)-e_m[idx].mean(axis=1))
    lo,hi=np.percentile(boot,[2.5,97.5])
    sup = (lo>0)   # gesamtes CI > 0 -> signifikant ueberlegen
    return dict(n=n,MAE_Arzt=round(mae_a,3),MAE_ML=round(mae_m,3),dMAE=round(dmae,3),
                CI_low=round(lo,3),CI_high=round(hi,3),
                p_one_sided=("<0.001" if pgt<0.001 else f"{pgt:.3f}"),
                ML_besser_pct=round(100*float((e_m<e_a).mean()),1),
                ueberlegen=("JA" if sup else "nein"))

approaches=["ExtraTrees","RandomForest","XGBoost","Ridge","Tweedie","Tweedie_1.5","Quantile_P50","Quantile_P80","Hazard_E"]
cohorts=[("is_open=0",ALT/"prospective_rebuilt_193.parquet"),
         ("no_isopen",ALT286/"prospective_rebuilt_286.parquet")]

allrows=[]
for cname,path in cohorts:
    pp=pd.read_parquet(path); los=pp["__los__"].to_numpy(float); arzt=pp["__arzt__"].to_numpy(float)
    Xp_df=pp[present]; P=predict_all(Xp_df)
    subs={"overall":np.ones(len(los),bool),"2-4 d":(los>2)&(los<=4),
          "4-7 d":(los>4)&(los<=7),">7 d":los>7}
    print(f"\n{'='*78}\nKOHORTE: {cname}\n{'='*78}")
    for sg,mask in subs.items():
        if mask.sum()<8: continue
        block=[]
        for ap in approaches:
            if ap not in P: continue
            r=superiority(los,arzt,P[ap],mask); r.update(Modell=ap,Kohorte=cname,Subgruppe=sg); block.append(r); allrows.append(r)
        bdf=pd.DataFrame(block)[["Modell","n","MAE_Arzt","MAE_ML","dMAE","CI_low","CI_high","p_one_sided","ML_besser_pct","ueberlegen"]]
        print(f"\n--- {sg} (n={int(mask.sum())}) — dMAE>0 & CI>0 => ML signifikant besser als Oberarzt ---")
        print(bdf.to_string(index=False))

res=pd.DataFrame(allrows)[["Kohorte","Subgruppe","Modell","n","MAE_Arzt","MAE_ML","dMAE","CI_low","CI_high","p_one_sided","ML_besser_pct","ueberlegen"]]
res.to_csv(OUTX/"superiority_vs_oberarzt.csv",sep=";",index=False)
sig=res[res["ueberlegen"]=="JA"]
print(f"\n{'='*78}\nFAZIT: signifikante UEBERLEGENHEIT (CI komplett > 0) in {len(sig)} von {len(res)} Faellen")
if len(sig): print(sig[["Kohorte","Subgruppe","Modell","dMAE","CI_low","CI_high","p_one_sided"]].to_string(index=False))
else: print("Kein Ansatz ist dem Oberarzt in irgendeiner Gruppe signifikant ueberlegen.")
print(f"\nGespeichert: {OUTX/'superiority_vs_oberarzt.csv'}")
