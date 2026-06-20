# -*- coding: utf-8 -*-
"""
Verbesserte Feature-Selektion (LASSO) + SVM (SVR) fuer die LoS-Regression,
methodisch orientiert an Chen et al. (Front Med 2026, ID 1910537):
 - Split zuerst, Preprocessing/Selektion nur auf Train (kein Leakage)
 - Z-Score-Standardisierung, One-Hot, Median-Imputation
 - LASSO zur Feature-Selektion (lambda.min / lambda.1se) + klassische LASSO-Grafiken
 - Modellvergleich Full- vs. LASSO-Features inkl. SVR (RBF) und LinearSVR
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, lasso_path
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"
FEAT=AN/"los_selected_features_ain_24h_compact.csv"
OUT=AN/"canonical"/"lasso_svm"; OUT.mkdir(parents=True,exist_ok=True)
RS=42
allowed=[("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31")]  # nur AIN-Intensiveinheiten IZ32/IZ21/IZ31
asql=", ".join(f"('{w}','{o}')" for w,o in allowed)
con=duckdb.connect()
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
df["los_days"]=df["icu_duration_h"]/24.0
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns and not f.startswith(("lab_","vital_","proc_","zugang_"))]
cat=[c for c in present if c=="oebenekurz"]; numc=[c for c in present if c not in cat]
X=df.reindex(columns=present).copy()
for c in numc: X[c]=pd.to_numeric(X[c],errors="coerce")
for c in cat:  X[c]=X[c].astype(str)
y=df["los_days"].values; ylog=np.log1p(np.clip(y,0,None))
groups=df["pid"].fillna("unknown").astype(str).values
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(X,y,groups))
print(f"Kohorte {len(df):,} | Train {len(tr):,} | Test {len(te):,} | Features (roh) {len(present)}")

# ---- Preprocessing (nur auf Train gefittet): Impute+Z-Score (num), Impute+OneHot (cat) ----
pre=ColumnTransformer([
 ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),numc),
 ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat),
])
Xtr=pre.fit_transform(X.iloc[tr]); Xte=pre.transform(X.iloc[te])
Xtr=np.asarray(Xtr,float); Xte=np.asarray(Xte,float)
fnames=list(pre.get_feature_names_out())
fnames=[f.replace("num__","").replace("cat__","") for f in fnames]
ytr,yte=ylog[tr],ylog[te]; gtr=groups[tr]; los_te=y[te]
print(f"Preprocessed Features: {Xtr.shape[1]}")

# ====================== LASSO Feature-Selektion ======================
gkf=list(GroupKFold(5).split(Xtr,ytr,gtr))
# Alpha-Pfad
alphas, coefs, _ = lasso_path(Xtr, ytr, n_alphas=100, eps=1e-3)
# CV-MSE je Alpha (gruppiert)
from sklearn.linear_model import Lasso as L
mse=np.zeros((len(alphas),len(gkf)))
for j,(trf,vaf) in enumerate(gkf):
    for i,a in enumerate(alphas):
        m=L(alpha=a,max_iter=5000).fit(Xtr[trf],ytr[trf])
        mse[i,j]=mean_squared_error(ytr[vaf],m.predict(Xtr[vaf]))
mse_mean=mse.mean(1); mse_se=mse.std(1)/np.sqrt(mse.shape[1])
imin=int(np.argmin(mse_mean)); a_min=alphas[imin]
thr=mse_mean[imin]+mse_se[imin]
cand=np.where(mse_mean<=thr)[0]; a_1se=alphas[cand.min()]  # groesstes alpha (alphas absteigend) -> kleinster Index
lasso_min=L(alpha=a_min,max_iter=10000).fit(Xtr,ytr)
lasso_1se=L(alpha=a_1se,max_iter=10000).fit(Xtr,ytr)
sel_min=[fnames[i] for i in np.where(np.abs(lasso_min.coef_)>1e-8)[0]]
sel_1se=[fnames[i] for i in np.where(np.abs(lasso_1se.coef_)>1e-8)[0]]
print(f"\nLASSO: lambda.min={a_min:.4f} -> {len(sel_min)} Features | lambda.1se={a_1se:.4f} -> {len(sel_1se)} Features")
coef_df=pd.DataFrame({"Feature":fnames,"coef_lambda_min":lasso_min.coef_}).query("coef_lambda_min!=0").reindex()
coef_df["abs"]=coef_df["coef_lambda_min"].abs(); coef_df=coef_df.sort_values("abs",ascending=False)
coef_df.drop(columns="abs").to_csv(OUT/"lasso_selected_coefficients.csv",sep=";",index=False)
print("Top-12 LASSO-Koeffizienten:"); print(coef_df.head(12).drop(columns="abs").to_string(index=False))
idx_sel=[fnames.index(f) for f in sel_min]

# ====================== LASSO-Grafiken ======================
# Englische, lesbare Achsenbeschriftung (Rohspaltennamen enthalten dt. Labor-Tokens)
_DE_TOK={"natrium":"sodium","chlorid":"chloride","kalium":"potassium","kalzium":"calcium",
 "calcium":"calcium","harnstoff":"urea","gesamt":"total","anzahl":"count","oebenekurz":"care-unit",
 "alter":"age","saeure":"acid"}
_PREF={"lab24_":"Lab: ","vital24_":"Vital: ","proc24_":"Procedure ","zugang24_":"Access: ","diag_main_":"Diagnosis "}
def clean_label(f):
    s=f
    if s.startswith(("num__","cat__")): s=s.split("__",1)[1]
    pre=""
    for k,v in _PREF.items():
        if s.startswith(k): pre=v; s=s[len(k):]; break
    for de,en in _DE_TOK.items(): s=s.replace(de,en)
    return (pre+s.replace("_"," ")).strip()
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
la=-np.log10(alphas)
# Fig 1: Koeffizientenpfade
fig,ax=plt.subplots(figsize=(8,5))
top=coef_df.head(8)["Feature"].tolist()
for k in range(coefs.shape[0]):
    lab=fnames[k]
    ax.plot(la,coefs[k],lw=(1.8 if lab in top else 0.5),alpha=(1 if lab in top else 0.25),
            color=None if lab in top else "#999")
ax.axvline(-np.log10(a_min),color="#2166ac",ls="--",lw=1.3,label=f"λ.min ({len(sel_min)} feat.)")
ax.axvline(-np.log10(a_1se),color="#d6604d",ls="--",lw=1.3,label=f"λ.1se ({len(sel_1se)} feat.)")
ax.set_xlabel("-log10(λ)"); ax.set_ylabel("Coefficient"); ax.set_title("LASSO coefficient paths (log1p ICU-LoS)",weight="bold")
ax.legend(fontsize=9,framealpha=.4)
fig.tight_layout(); fig.savefig(str(OUT/"fig_lasso_paths.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
# Fig 2: CV-Kurve
fig,ax=plt.subplots(figsize=(8,5))
ax.errorbar(la,mse_mean,yerr=mse_se,fmt="o",ms=3,color="#185fa5",ecolor="#b5d4f4",elinewidth=1,capsize=2)
ax.axvline(-np.log10(a_min),color="#2166ac",ls="--",lw=1.3,label="λ.min")
ax.axvline(-np.log10(a_1se),color="#d6604d",ls="--",lw=1.3,label="λ.1se")
ax.set_xlabel("-log10(λ)"); ax.set_ylabel("Cross-validated MSE (log1p scale)")
ax.set_title("LASSO 5-fold (patient-grouped) cross-validation",weight="bold"); ax.legend(fontsize=9,framealpha=.4)
fig.tight_layout(); fig.savefig(str(OUT/"fig_lasso_cv.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
# Fig 3: selektierte Koeffizienten
fig,ax=plt.subplots(figsize=(8,max(4,0.32*min(len(coef_df),20))))
show=coef_df.head(20).iloc[::-1]
ax.barh([clean_label(f) for f in show["Feature"]],show["coef_lambda_min"],color=["#d6604d" if v<0 else "#2166ac" for v in show["coef_lambda_min"]])
ax.axvline(0,color="#444",lw=.8); ax.set_xlabel("LASSO coefficient (λ.min, standardised features)")
ax.set_title(f"Selected predictors (LASSO, {len(sel_min)} non-zero)",weight="bold")
fig.tight_layout(); fig.savefig(str(OUT/"fig_lasso_coefficients.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("LASSO-Grafiken gespeichert.")

# ====================== Modellvergleich: Full vs LASSO-Features (+ SVM) ======================
def metrics(yt_log,yp_log,label,featset):
    yp=np.clip(np.expm1(yp_log),0,None); ae=np.abs(los_te-yp)
    return {"Modell":label,"Features":featset,"k":(Xtr.shape[1] if featset=="full" else len(idx_sel)),
            "MAE":round(float(ae.mean()),3),"MedianAE":round(float(np.median(ae)),3),
            "RMSE":round(float(np.sqrt(mean_squared_error(los_te,yp))),3),"R2":round(float(r2_score(los_te,yp)),3)}

def make(name):
    if name=="Ridge": return Ridge(alpha=1.0,random_state=RS)
    if name=="RandomForest": return RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=0.5,max_depth=20,random_state=RS,n_jobs=1)
    if name=="ExtraTrees": return ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=0.5,max_depth=20,random_state=RS,n_jobs=1)
    if name=="XGBoost": return XGBRegressor(n_estimators=500,max_depth=8,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,min_child_weight=1,reg_lambda=5,random_state=RS,n_jobs=1,tree_method="hist")
    if name=="LinearSVR": return LinearSVR(C=1.0,epsilon=0.1,max_iter=20000,random_state=RS)

rows=[]
# Full-Feature-Modelle (ohne RBF-SVR, da langsam)
for name in ["Ridge","RandomForest","ExtraTrees","XGBoost","LinearSVR"]:
    m=make(name).fit(Xtr,ytr); rows.append(metrics(yte,m.predict(Xte),name,"full"))
# LASSO-Linearmodell selbst
rows.append(metrics(yte,lasso_min.predict(Xte),"LASSO (λ.min)","full"))
# LASSO-selektierte Features
Xtr_s,Xte_s=Xtr[:,idx_sel],Xte[:,idx_sel]
for name in ["Ridge","RandomForest","ExtraTrees","XGBoost","LinearSVR"]:
    m=make(name).fit(Xtr_s,ytr); rows.append(metrics(yte,m.predict(Xte_s),name,"lasso"))

# ---- SVR (RBF): C-Tuning per GroupKFold auf Subsample, dann Full-Train (auf LASSO-Features) ----
print("\nSVR (RBF) Tuning auf LASSO-Features ...")
sub=np.random.RandomState(RS).choice(len(tr),min(5000,len(tr)),replace=False)
gs=GridSearchCV(SVR(kernel="rbf",gamma="scale",epsilon=0.1),{"C":[1,10,30]},
                scoring="neg_mean_absolute_error",cv=GroupKFold(3),n_jobs=3)
gs.fit(Xtr_s[sub],ytr[sub],groups=gtr[sub])
svr=SVR(kernel="rbf",gamma="scale",epsilon=0.1,C=gs.best_params_["C"]).fit(Xtr_s,ytr)
rows.append(metrics(yte,svr.predict(Xte_s),f"SVR-RBF (C={gs.best_params_['C']})","lasso"))
print(f"  bestes C={gs.best_params_['C']}")

res=pd.DataFrame(rows)[["Modell","Features","k","MAE","MedianAE","RMSE","R2"]]
res.to_csv(OUT/"model_comparison_lasso_svm.csv",sep=";",index=False)
print("\n=== MODELLVERGLEICH (Holdout, Tage) ===")
print(res.to_string(index=False))
print(f"\nAlles gespeichert in: {OUT}")
