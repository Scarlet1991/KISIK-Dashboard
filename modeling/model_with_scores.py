# -*- coding: utf-8 -*-
"""Integriert SAPS II + TISS-28 (24h) ins retrospektive Modell und prueft, ob sie die Vorhersage verbessern.
Prospektiv sind die Scores ~0% -> wird transparent berichtet."""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
SCORES=AN/"canonical"/"scores24_retro.csv"; RS=42
allowed=[("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31")]  # nur AIN-Intensiveinheiten IZ32/IZ21/IZ31
asql=", ".join(f"('{w}','{o}')" for w,o in allowed); con=duckdb.connect()
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
df["los_days"]=df["icu_duration_h"]/24.0
sc=pd.read_csv(SCORES,sep=";")
SCORE_FEATS=["score_saps_first","score_saps_max","score_tiss_first","score_tiss_max"]
df=df.merge(sc[["stay_id"]+[c for c in SCORE_FEATS if c in sc.columns]],on="stay_id",how="left")
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns and not f.startswith(("lab_","vital_","proc_","zugang_"))]
cat=[c for c in present if c=="oebenekurz"]; numc=[c for c in present if c not in cat]
# univariate Korrelation Score vs LoS
print("Spearman-Korrelation Score (first, 24h) vs ICU-LoS:")
for c in ["score_saps_first","score_tiss_first"]:
    v=pd.to_numeric(df[c],errors="coerce"); m=v.notna()
    rho,p=stats.spearmanr(v[m],df["los_days"][m]); print(f"  {c:20} rho={rho:+.3f} (n={m.sum():,}, p={p:.1e})")
y=df["los_days"].values; groups=df["pid"].fillna("unknown").astype(str).values
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(df,y,groups)); los_te=y[te]
def pre(numcols,scale=False):
    ns=[("imp",SimpleImputer(strategy="median"))]+([("sc",StandardScaler())] if scale else [])
    return ColumnTransformer([("num",Pipeline(ns),numcols),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)])
def ttr(reg,numcols,scale=False): return TransformedTargetRegressor(Pipeline([("pre",pre(numcols,scale)),("mdl",reg)]),func=np.log1p,inverse_func=np.expm1)
def models(numcols):
    return {"Ridge":ttr(Ridge(alpha=0.1),numcols,True),
            "RandomForest":ttr(RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=0.5,max_depth=20,random_state=RS,n_jobs=1),numcols),
            "ExtraTrees":ttr(ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=0.5,max_depth=20,random_state=RS,n_jobs=1),numcols),
            "XGBoost":ttr(XGBRegressor(n_estimators=500,max_depth=8,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,min_child_weight=1,reg_lambda=5,random_state=RS,n_jobs=1,tree_method="hist"),numcols)}
def met(yp,label,fs):
    yp=np.clip(yp,0,None); ae=np.abs(los_te-yp)
    return {"Modell":label,"Featureset":fs,"MAE":round(float(ae.mean()),3),"MedianAE":round(float(np.median(ae)),3),
            "RMSE":round(float(np.sqrt(mean_squared_error(los_te,yp))),3),"R2":round(float(r2_score(los_te,yp)),3)}
rows=[]
for fs,numcols in [("base (84)",numc),("base+scores",numc+[c for c in SCORE_FEATS if c in df.columns])]:
    X=df[numcols+cat].copy()
    for c in numcols: X[c]=pd.to_numeric(X[c],errors="coerce")
    for c in cat: X[c]=X[c].astype(str)
    for name,m in models(numcols).items():
        m.fit(X.iloc[tr],np.log1p(np.clip(y[tr],0,None)))
        rows.append(met(np.expm1(m.predict(X.iloc[te])),name,fs))
res=pd.DataFrame(rows).pivot_table(index="Modell",columns="Featureset",values=["MAE","R2"])
print("\n=== Retrospektiver Holdout: ohne vs. mit SAPS II + TISS-28 ===")
print(res.to_string())
pd.DataFrame(rows).to_csv(AN/"canonical"/"model_with_scores_retro.csv",sep=";",index=False)
print(f"\nGespeichert: canonical/model_with_scores_retro.csv")
print("Hinweis: prospektiv sind SAPS II/TISS-28 ~0% -> diese Score-Features transferieren NICHT (Recording-Shift).")
