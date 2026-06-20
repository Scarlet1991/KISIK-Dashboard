# -*- coding: utf-8 -*-
"""
KANONISCHE, reproduzierbare LoS-Analyse (Source of Truth fuer das Manuskript).
- leckage-frei: NUR direkt vorhandene first-24h-Features (kein Full-Stay-Fallback)
- konsistenter Modellsatz (Ridge, RandomForest, ExtraTrees, XGBoost), alle mit log1p-Ziel
- patienten-gruppierte 4-fold CV-Hyperparametersuche (RandomizedSearchCV) auf dem Trainingsset
- MAE explizit in Tagen: mean(|y_obs_days - y_pred_days|)
- Permutation-Importance fuer das finale Modell
- prospektive Evaluierung gegen Oberarzt (best_senior_estimate_days)
- Figuren bis 20 Tage, gut lesbar
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"
PROS =BASE/"kisik2"/"kisik2_prospektiv_ml_dataset.parquet"
SENIOR=AN/"los_senior_estimates_tagesausleitung_stay_level.csv"
FEAT=AN/"los_selected_features_ain_24h_compact.csv"
OUT=AN/"canonical"; OUT.mkdir(exist_ok=True)
RS=42; CAP=20  # Figuren-Achsenlimit (Tage)

allowed=[("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31"),("AIN","IZ01"),("AUG","IZ01"),("AVT","IZ01"),("GCH","IZ01"),("GYN","IZ01"),("HNO","IZ01"),("HTC","IZ01"),("IZPV","IZ01"),("MKG","IZ01"),("NCH","IZ01"),("NUK","IZ01"),("STR","IZ01"),("UCH","IZ01"),("URO","IZ01")]
asql=", ".join(f"('{w}','{o}')" for w,o in allowed)
con=duckdb.connect()

# ---------------------------------------------------------------- Kohorte
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
df["los_days"]=df["icu_duration_h"]/24.0          # icu_duration_h ist in STUNDEN
y=df["los_days"].values
groups=df["pid"].fillna("unknown").astype(str).values
summary=dict(n_stays=int(len(df)), n_patients=int(df["pid"].nunique()),
             n_fallid=int(df["fallid"].nunique()),
             patients_gt1_stay=int((df.groupby("pid").size()>1).sum()),
             los_days_median=round(float(df["los_days"].median()),2),
             los_days_p90=round(float(df["los_days"].quantile(0.9)),2),
             los_days_max=round(float(df["los_days"].max()),1))

# ---------------------------------------------------------------- leckage-freie Features
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns]               # NUR direkt vorhandene 24h-Features
# Sicherheits-Ausschluss: keinerlei Full-Stay-Spalten (lab_/vital_/proc_/zugang_ ohne 24)
present=[f for f in present if not (f.startswith(("lab_","vital_","proc_","zugang_")))]
missing=[f for f in feat if f not in present]
cat=[c for c in present if c=="oebenekurz"]; numc=[c for c in present if c not in cat]
def domain(f):
    if f.startswith("lab24_"): return "lab_24h"
    if f.startswith("vital24_"): return "vital_24h"
    if f.startswith("proc24_"): return "procedure_24h"
    if f.startswith("zugang24_"): return "access_24h"
    if f.startswith("diag_main_"): return "diagnosis"
    return "demographics_admission"
dom_counts=pd.Series([domain(f) for f in present]).value_counts().to_dict()
summary["n_features_selected"]=len(feat)
summary["n_features_used_leakagefree"]=len(present)
summary["n_features_excluded"]=len(missing)
summary["feature_domains"]={k:int(v) for k,v in dom_counts.items()}

def Xframe(frame):
    X=frame.reindex(columns=present).copy()
    for c in numc: X[c]=pd.to_numeric(X[c],errors="coerce")
    for c in cat:  X[c]=X[c].astype(str)
    return X
X=Xframe(df)

# ---------------------------------------------------------------- Split (patienten-gruppiert)
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(X,y,groups))
Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y[tr],y[te]; gtr=groups[tr]
summary["n_train"]=int(len(tr)); summary["n_test"]=int(len(te))
summary["units"]="icu_duration_h are HOURS; target = hours/24 = DAYS; senior estimate in DAYS"

def pre(scale=False):
    num_steps=[("imp",SimpleImputer(strategy="median"))]+([("sc",StandardScaler())] if scale else [])
    return ColumnTransformer([("num",Pipeline(num_steps),numc),
                              ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                                               ("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)])
def ttr(reg,scale=False):
    return TransformedTargetRegressor(Pipeline([("pre",pre(scale)),("mdl",reg)]),func=np.log1p,inverse_func=np.expm1)

# ---------------------------------------------------------------- Hyperparametersuche (GroupKFold=4)
gkf=GroupKFold(n_splits=4)
SCORER="neg_mean_absolute_error"   # MAE in TAGEN (TTR liefert Tage zurueck)
searches={
 "Ridge": (ttr(Ridge(random_state=RS),scale=True),
           {"regressor__mdl__alpha":[0.1,0.3,1,3,10,30,100]}, "grid"),
 "RandomForest": (ttr(RandomForestRegressor(random_state=RS,n_jobs=1)),
           {"regressor__mdl__n_estimators":[300,500],"regressor__mdl__max_depth":[None,12,20],
            "regressor__mdl__min_samples_leaf":[2,5,10],"regressor__mdl__max_features":["sqrt",0.5]}, "rand"),
 "ExtraTrees": (ttr(ExtraTreesRegressor(random_state=RS,n_jobs=1)),
           {"regressor__mdl__n_estimators":[300,500],"regressor__mdl__max_depth":[None,12,20],
            "regressor__mdl__min_samples_leaf":[2,5,10],"regressor__mdl__max_features":["sqrt",0.5]}, "rand"),
 "XGBoost": (ttr(XGBRegressor(random_state=RS,n_jobs=1,verbosity=0,tree_method="hist")),
           {"regressor__mdl__n_estimators":[300,500,800],"regressor__mdl__max_depth":[4,6,8],
            "regressor__mdl__learning_rate":[0.03,0.05,0.1],"regressor__mdl__subsample":[0.7,0.9],
            "regressor__mdl__colsample_bytree":[0.7,0.9],"regressor__mdl__min_child_weight":[1,3,5],
            "regressor__mdl__reg_lambda":[1,2,5]}, "rand"),
}
print("Hyperparametersuche (4-fold GroupKFold auf pid) ...")
fitted={}; cv_rows=[]; best_params={}
for name,(est,space,kind) in searches.items():
    if kind=="grid":
        s=GridSearchCV(est,space,scoring=SCORER,cv=gkf,n_jobs=4)
    else:
        s=RandomizedSearchCV(est,space,n_iter=12,scoring=SCORER,cv=gkf,random_state=RS,n_jobs=4)
    s.fit(Xtr,ytr,groups=gtr)
    fitted[name]=s.best_estimator_
    best_params[name]={k.replace("regressor__mdl__",""):v for k,v in s.best_params_.items()}
    cv_rows.append({"Modell":name,"CV_MAE_days":round(-s.best_score_,3)})
    print(f"  {name}: CV-MAE={-s.best_score_:.3f} d | {best_params[name]}")

cv_df=pd.DataFrame(cv_rows).sort_values("CV_MAE_days")
final_name=cv_df.iloc[0]["Modell"]
summary["final_model"]=final_name
summary["best_params"]=best_params
print(f"\n==> Finales Modell (beste CV-MAE): {final_name}")

# ---------------------------------------------------------------- Metriken
def metrics(yt,yp,label,n_groups=None):
    yt=np.asarray(yt,float); yp=np.clip(np.asarray(yp,float),0,None)
    ae=np.abs(yt-yp)
    return {"Modell":label,"n":int(len(yt)),
            "MAE_days":round(float(ae.mean()),3),"MedianAE_days":round(float(np.median(ae)),3),
            "RMSE_days":round(float(np.sqrt(mean_squared_error(yt,yp))),3),
            "R2":round(float(r2_score(yt,yp)),3),"Bias_days":round(float((yp-yt).mean()),3)}

retro_rows=[]; preds_te={}
for name,m in fitted.items():
    p=m.predict(Xte); preds_te[name]=p; retro_rows.append(metrics(yte,p,name))
retro_df=pd.DataFrame(retro_rows).merge(cv_df,on="Modell").sort_values("MAE_days")
print("\n--- Retrospektiver Holdout (Tage) ---"); print(retro_df.to_string(index=False))

# ---------------------------------------------------------------- Prospektiv + Oberarzt
dp=con.execute(f"SELECT * FROM read_parquet('{PROS.as_posix()}')").df()
dp["los_days"]=dp["icu_duration_h"]/24.0
sen=pd.read_csv(SENIOR,sep=";")
dp["stay_id"]=dp["stay_id"].astype(str); sen["tages_stay_id"]=sen["tages_stay_id"].astype(str)
mg=dp.merge(sen,left_on="stay_id",right_on="tages_stay_id",how="inner")
mg["arzt"]=pd.to_numeric(mg["best_senior_estimate_days"],errors="coerce")
mg=mg.dropna(subset=["los_days","arzt"]).reset_index(drop=True)
Xp=Xframe(mg)   # fehlende 24h-Features -> NaN -> Imputer-Median (Deployment-Realitaet)
summary["n_prospective_matched"]=int(len(mg))
pros_rows=[metrics(mg["los_days"].values,mg["arzt"].values,"Oberarzt")]
preds_pp={"Oberarzt":mg["arzt"].values}
for name,m in fitted.items():
    p=m.predict(Xp); preds_pp[name]=p; pros_rows.append(metrics(mg["los_days"].values,p,name))
pros_df=pd.DataFrame(pros_rows)
print("\n--- Prospektiv vs. Oberarzt (Tage) ---"); print(pros_df.to_string(index=False))

# ---------------------------------------------------------------- Permutation-Importance (finales Modell)
print(f"\nPermutation-Importance ({final_name}) ...")
pi=permutation_importance(fitted[final_name],Xte,yte,scoring=SCORER,n_repeats=10,random_state=RS,n_jobs=2)
imp=pd.DataFrame({"Feature":Xte.columns,"MAE_increase_days":pi.importances_mean,"sd":pi.importances_std})
imp=imp.sort_values("MAE_increase_days",ascending=False)
imp.to_csv(OUT/"feature_importance.csv",sep=";",index=False)
print(imp.head(15).to_string(index=False))

# ---------------------------------------------------------------- Exporte
retro_df.to_csv(OUT/"metrics_retrospective.csv",sep=";",index=False)
pros_df.to_csv(OUT/"metrics_prospective.csv",sep=";",index=False)
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

# ---------------------------------------------------------------- Figuren (bis 20 Tage, gross/lesbar)
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False})
def hexbin(obs,pred,title,path):
    obs=np.asarray(obs,float); pred=np.clip(np.asarray(pred,float),0,None)
    m=metrics(obs,pred,title)
    fig,ax=plt.subplots(figsize=(6,5.2))
    hb=ax.hexbin(obs,pred,gridsize=24,cmap="viridis",bins="log",mincnt=1,extent=[0,CAP,0,CAP])
    ax.plot([0,CAP],[0,CAP],"--",color="#d6604d",lw=1.6,label="Identity (perfect prediction)")
    cb=fig.colorbar(hb,ax=ax); cb.set_label("Stays per bin (log scale)",fontsize=11)
    ax.set_xlim(0,CAP); ax.set_ylim(0,CAP)
    ax.set_xlabel("Observed ICU length of stay (days)"); ax.set_ylabel("Predicted ICU LoS (days)")
    ax.set_title(title,weight="bold",fontsize=13); ax.legend(loc="upper left",fontsize=10,framealpha=.4)
    box=f"n = {m['n']}\nMAE = {m['MAE_days']:.2f} d\nRMSE = {m['RMSE_days']:.2f} d\nR² = {m['R2']:.2f}\nMedian AE = {m['MedianAE_days']:.2f} d"
    ax.text(0.97,0.03,box,transform=ax.transAxes,ha="right",va="bottom",fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#bbb",alpha=.9))
    fig.tight_layout(); fig.savefig(str(path),dpi=300,bbox_inches="tight"); plt.close(fig)

hexbin(yte,preds_te[final_name],f"Retrospective hold-out — {final_name} (log1p)",OUT/f"fig_hexbin_retro_{final_name}.png")
hexbin(mg["los_days"].values,preds_pp[final_name],f"Prospective — {final_name} (log1p)",OUT/f"fig_hexbin_pros_{final_name}.png")
hexbin(mg["los_days"].values,preds_pp["Oberarzt"],"Prospective — senior physician",OUT/"fig_hexbin_pros_oberarzt.png")

# Modellvergleich (retro + prospektiv MAE)
fig,ax=plt.subplots(figsize=(8,4.6)); order=["Ridge","RandomForest","ExtraTrees","XGBoost"]
xr=np.arange(len(order)); w=0.38
rt=[retro_df.set_index("Modell").loc[m,"MAE_days"] for m in order]
pp=[pros_df.set_index("Modell").loc[m,"MAE_days"] for m in order]
ax.bar(xr-w/2,rt,w,label="retrospective hold-out",color="#b5d4f4")
ax.bar(xr+w/2,pp,w,label="prospective",color="#185fa5")
ax.axhline(pros_df.set_index("Modell").loc["Oberarzt","MAE_days"],color="#d6604d",ls="--",lw=1.5)
ax.text(len(order)-1,pros_df.set_index("Modell").loc["Oberarzt","MAE_days"]+0.05,"Senior physician (prosp.)",color="#d6604d",ha="right",fontsize=9)
ax.set_xticks(xr); ax.set_xticklabels(order); ax.set_ylabel("MAE (days)")
ax.set_title("Model comparison — MAE (retrospective vs prospective)",weight="bold"); ax.legend()
fig.tight_layout(); fig.savefig(str(OUT/"fig_model_comparison.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

# Permutation-Importance Top-15 (englische, lesbare Labels statt Rohspaltennamen)
IMP_LABELS={
 "proc24_8_98f_0":"ICU complex treatment, base (8-98f.0)","proc24_8_98f_10":"ICU complex treatment, extended (8-98f.10)",
 "oebenekurz":"ICU care-unit type","proc24_8_931_0":"Extended haemodynamic monitoring (8-931)",
 "proc24_8_924":"Cardiac monitoring (8-924)","proc24_anzahl_gesamt":"Total procedure count (24 h)",
 "proc24_8_98f_11":"ICU complex treatment, prolonged (8-98f.11)","diag_main_z99_1":"Ventilator dependence (Z99.1)",
 "proc24_8_930":"Basic haemodynamic monitoring (8-930)","stay_nr":"ICU stay number","alter":"Age",
 "diag_main_j12_8":"Viral pneumonia (J12.8)","diag_main_g91_0":"Communicating hydrocephalus (G91.0)",
 "diag_main_g91_8":"Hydrocephalus, other (G91.8)","proc24_3_200":"Native cranial CT (3-200)"}
def _implabel(f):
    if f in IMP_LABELS: return IMP_LABELS[f]
    return (f.replace("lab24_","Lab: ").replace("vital24_","Vital: ").replace("proc24_","Procedure ")
             .replace("zugang24_","Access: ").replace("diag_main_","Diagnosis ").replace("_"," "))
fig,ax=plt.subplots(figsize=(8.4,6)); top=imp.head(15).iloc[::-1]
ax.barh([_implabel(f) for f in top["Feature"]],top["MAE_increase_days"],xerr=top["sd"],color="#762a83")
ax.set_xlabel("Increase in MAE when permuted (days)"); ax.margins(y=0.01)
ax.set_title(f"Permutation feature importance — {final_name}",weight="bold")
fig.tight_layout(); fig.savefig(str(OUT/"fig_importance.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

print(f"\nAlle Exporte + Figuren in: {OUT}")
