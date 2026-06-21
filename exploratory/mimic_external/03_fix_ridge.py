# -*- coding: utf-8 -*-
"""
Ridge-Baseline robust nachrechnen: MIMIC-Laborwerte enthalten extreme Ausreisser
(z.B. 999999), die ein lineares Modell auf der log1p-Skala numerisch sprengen
(expm1-Overflow). Loesung: Winsorizing der numerischen Features auf die Trainings-1./99.-Perzentile
(Standard-EHR-Preprocessing fuer lineare Modelle; Baeume sind robust und unveraendert).
Aktualisiert nur die Ridge-Zeile in mimic_metrics.csv und regeneriert fig_mimic_models.png.
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT=Path(r"D:\Ausgangsdaten\KISIK Projekt\mimic_external"); RS=42
class Winsorizer(BaseEstimator,TransformerMixin):
    def __init__(self,lo=0.01,hi=0.99): self.lo=lo; self.hi=hi
    def fit(self,X,y=None):
        A=np.asarray(X,float); self.lo_=np.nanpercentile(A,self.lo*100,axis=0); self.hi_=np.nanpercentile(A,self.hi*100,axis=0); return self
    def transform(self,X): return np.clip(np.asarray(X,float),self.lo_,self.hi_)

X=pd.read_parquet(OUT/"mimic_features.parquet")
y=X["los_days"].to_numpy(float); groups=X["subject_id"].astype(str).to_numpy()
CAT=[c for c in ["first_careunit","gender","admission_type"] if c in X.columns]
feats=[c for c in X.columns if c not in ["los_days","subject_id"]]; num=[c for c in feats if c not in CAT]
Xf=X[feats].copy()
for c in num: Xf[c]=pd.to_numeric(Xf[c],errors="coerce")
for c in CAT: Xf[c]=Xf[c].astype(str)
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(Xf,y,groups))

pre=ColumnTransformer([
    ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("win",Winsorizer()),("sc",StandardScaler())]),num),
    ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),CAT)])
ridge=TransformedTargetRegressor(regressor=Pipeline([("pre",pre),("mdl",Ridge(alpha=10.0,random_state=RS))]),
                                 func=np.log1p, inverse_func=np.expm1)
ridge.fit(Xf.iloc[tr],y[tr]); p=np.clip(ridge.predict(Xf.iloc[te]),0,None); yte=y[te]; e=p-yte
r=dict(MAE=mean_absolute_error(yte,p),MedianAE=float(np.median(np.abs(e))),
       RMSE=float(np.sqrt(np.mean(e**2))),R2=r2_score(yte,p),Bias=float(e.mean()))
print(f"Ridge (winsorized): MAE {r['MAE']:.3f} | RMSE {r['RMSE']:.3f} | R² {r['R2']:.3f} | Bias {r['Bias']:+.3f}")

m=pd.read_csv(OUT/"mimic_metrics.csv",sep=";")
for k,col in [("MAE","MAE"),("MedianAE","MedianAE"),("RMSE","RMSE"),("R2","R2"),("Bias","Bias")]:
    m.loc[m["Modell"]=="Ridge",col]=round(r[k],3)
m=m.sort_values("CV_MAE" if m["CV_MAE"].max()<1e6 else "MAE")  # keep order sensible
m.to_csv(OUT/"mimic_metrics.csv",sep=";",index=False)
print(m.to_string(index=False))

# Figur neu (jetzt mit endlichem Ridge-R²)
s=json.load(open(OUT/"mimic_summary.json")); null_mae=s["null_mae_holdout"]; nte=s["n_test"]
order=["XGBoost","ExtraTrees","RandomForest","Ridge"]; order=[o for o in order if o in set(m["Modell"])]
mm=m.set_index("Modell").loc[order]
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.6)); xr=np.arange(len(order)); MODC="#1f5f9e"; NULC="#c7ccd1"
a1.bar(xr,mm["MAE"],color=MODC); a1.axhline(null_mae,color="#999",ls="--",lw=1.6)
a1.text(len(order)-1,null_mae+0.04,f"null model (MAE {null_mae:.2f} d)",color="#777",ha="right",fontsize=8.5)
a1.bar_label(a1.containers[0],fmt="%.2f",fontsize=8,padding=2)
a1.set_xticks(xr);a1.set_xticklabels(order,fontsize=9);a1.set_ylabel("MAE (days)");a1.set_title("(A) MIMIC holdout MAE — lower is better",weight="bold",fontsize=11)
a2.bar(xr,mm["R2"],color=MODC); a2.axhline(0,color="#888",lw=0.8); a2.bar_label(a2.containers[0],fmt="%.2f",fontsize=8,padding=2)
a2.set_xticks(xr);a2.set_xticklabels(order,fontsize=9);a2.set_ylabel("R²");a2.set_title("(B) MIMIC holdout R² — higher is better",weight="bold",fontsize=11)
fig.suptitle(f"MIMIC-IV external validation: model performance (holdout n={nte:,})",weight="bold",fontsize=12.5)
fig.tight_layout(); fig.savefig(OUT/"fig_mimic_models.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print("fig_mimic_models.png regeneriert.")
