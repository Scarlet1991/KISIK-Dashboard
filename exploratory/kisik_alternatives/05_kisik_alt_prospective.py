# -*- coding: utf-8 -*-
"""
Manuskriptfaehiger prospektiver Vergleich der ALTERNATIVEN Ansaetze (Tweedie/Gamma/Hazard/Quantil)
auf den REKONSTRUIERTEN 193-Features (86% Abdeckung) — identische Features wie das finale ExtraTrees
im Manuskript, nur andere Zielfunktionen. Matrizen aus prospective_24h_rebuild.py (alt_matrices/).
Vergleich gegen Oberarzt (n=193) + ExtraTrees (aus metrics_prospective_fair24h_predictions.csv).
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, XGBClassifier
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

AN=Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung"); CAN=AN/"canonical"; ALT=CAN/"alt_matrices"
RS=42; TMAX=90; OUT=AN/"los_alt_prospective_193"
fl=json.load(open(ALT/"feature_lists.json")); present,numc,cat=fl["present"],fl["numc"],fl["cat"]
tr=pd.read_parquet(ALT/"retro_train.parquet"); ytr=tr["__y__"].to_numpy(float); Xtr_df=tr[present]
pp=pd.read_parquet(ALT/"prospective_rebuilt_193.parquet")
los=pp["__los__"].to_numpy(float); arzt=pp["__arzt__"].to_numpy(float); Xp_df=pp[present]
print(f"Retro-Train {Xtr_df.shape} | prospektiv rekonstruiert {Xp_df.shape} (n={len(los)})")

def design(frame):
    parts=[frame[numc].apply(pd.to_numeric,errors="coerce")]
    if cat: parts.append(pd.get_dummies(frame[cat].astype(str),prefix=cat).astype(float))
    X=pd.concat(parts,axis=1); X.columns=[str(c) for c in X.columns]
    return X
Xtr=design(Xtr_df); COLS=Xtr.columns.tolist(); Xp=design(Xp_df)
for c in COLS:
    if c not in Xp.columns: Xp[c]=0.0
Xp=Xp[COLS]
Xtr_v=Xtr.to_numpy(dtype=np.float64,na_value=np.nan); Xp_v=Xp.to_numpy(dtype=np.float64,na_value=np.nan)  # pd.NA -> np.nan, XGBoost-nativ

COMMON=dict(n_estimators=600,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
            min_child_weight=3,random_state=RS,n_jobs=-1)
fitted={}
print("Training auf rekonstruierten Features ...")
m=XGBRegressor(objective="reg:squarederror",**COMMON); m.fit(Xtr_v,np.log1p(ytr)); fitted["log1p_Mean"]=("log",m)
for vp in [1.3,1.5,1.7]:
    m=XGBRegressor(objective="reg:tweedie",tweedie_variance_power=vp,**COMMON); m.fit(Xtr_v,ytr); fitted[f"Tweedie_{vp}"]=("raw",m)
m=XGBRegressor(objective="reg:gamma",**COMMON); m.fit(Xtr_v,np.clip(ytr,0.01,None)); fitted["Gamma"]=("raw",m)
for a,nm in [(0.5,"Quantile_P50"),(0.8,"Quantile_P80")]:
    m=XGBRegressor(objective="reg:quantileerror",quantile_alpha=a,**COMMON); m.fit(Xtr_v,np.log1p(ytr)); fitted[nm]=("log",m)
def predict(tag,Xv):
    kind,m=fitted[tag]; p=m.predict(Xv); return np.clip(np.expm1(p) if kind=="log" else p,0,None)

# Hazard
T_tr=np.clip(np.ceil(ytr).astype(int),1,TMAX)
Xrep=np.repeat(Xtr_v,T_tr,axis=0)
day_t=np.concatenate([np.arange(1,c+1) for c in T_tr]).astype(np.float64)
y_haz=(day_t==np.repeat(T_tr,T_tr)).astype(int)
clf=XGBClassifier(objective="binary:logistic",n_estimators=400,max_depth=6,learning_rate=0.05,
                  subsample=0.8,colsample_bytree=0.8,min_child_weight=5,random_state=RS,n_jobs=-1,eval_metric="logloss")
clf.fit(np.column_stack([Xrep,day_t]).astype(np.float32),y_haz)
print(f"Hazard trainiert (Person-Period {len(y_haz):,})")
def hazard_predict(Xv):
    n=Xv.shape[0]; S=np.ones(n); E=np.zeros(n); med=np.full(n,np.nan)
    for t in range(1,TMAX+1):
        h=np.clip(clf.predict_proba(np.column_stack([Xv,np.full(n,t)]).astype(np.float32))[:,1],1e-6,1-1e-6)
        E+=S; Snew=S*(1-h); nw=(med!=med)&(Snew<0.5); med[nw]=t; S=Snew
    med[med!=med]=TMAX; return E,med

preds={t:predict(t,Xp_v) for t in fitted}
E_p,med_p=hazard_predict(Xp_v); preds["Hazard_E"]=E_p; preds["Hazard_Median"]=med_p
preds["Oberarzt"]=arzt
# ExtraTrees aus Manuskript-Vorhersagen (gleiche rekonstruierte Features)
try:
    et=pd.read_csv(CAN/"metrics_prospective_fair24h_predictions.csv",sep=";")
    et_map=dict(zip(et["stay_id"].astype(str),et["pred_ExtraTrees"]))
    preds["ExtraTrees"]=np.array([et_map.get(s,np.nan) for s in pp["__stay_id__"].astype(str)])
except Exception as e: print("ExtraTrees-Merge übersprungen:",e)
preds["Null"]=np.full(len(los),float(np.median(ytr)))  # Null-Baseline = Trainings-Median

def met(yt,yp,label,sub=None):
    if sub is not None: yt,yp=yt[sub],yp[sub]
    yp=np.clip(yp,0,None); ae=np.abs(yt-yp)
    return {"Modell":label,"n":int(len(yt)),"MAE":round(float(ae.mean()),3),"MedianAE":round(float(np.median(ae)),3),
            "RMSE":round(float(np.sqrt(mean_squared_error(yt,yp))),3),"R2":round(float(r2_score(yt,yp)),3),"Bias":round(float((yp-yt).mean()),3)}
order=["Oberarzt","ExtraTrees","log1p_Mean","Tweedie_1.3","Tweedie_1.5","Tweedie_1.7","Gamma","Hazard_E","Hazard_Median","Quantile_P50","Quantile_P80","Null"]
order=[o for o in order if o in preds and np.isfinite(preds[o]).all()]
subs={"1-2 d":(los>1)&(los<=2),"2-4 d":(los>2)&(los<=4),"4-7 d":(los>4)&(los<=7),">7 d":los>7}
rows=[]
for sg,mask in subs.items():
    for o in order: rows.append({**met(los,preds[o],o,mask),"Subgroup":sg})
res=pd.DataFrame(rows)[["Subgroup","Modell","n","MAE","MedianAE","RMSE","R2","Bias"]]
res.to_csv(f"{OUT}.csv",sep=";",index=False)
print("\n=== PROSPEKTIV auf rekonstruierten 193-Features (Tage) ==="); print(res.to_string(index=False))

# P80-Coverage + Wilcoxon
cov=[]
for o in ["Oberarzt","ExtraTrees","Quantile_P50","Quantile_P80","Hazard_Median"]:
    if o not in preds: continue
    for sg,mask in subs.items():
        cov.append({"Estimator":o,"Subgroup":sg,"Coverage_%":round(100*float((los[mask]<=np.clip(preds[o],0,None)[mask]).mean()),1),
                    "Mean_pred":round(float(np.clip(preds[o],0,None)[mask].mean()),2),"Mean_obs":round(float(los[mask].mean()),2)})
pd.DataFrame(cov).to_csv(f"{OUT}_coverage.csv",sep=";",index=False)
wr=[]
for o in [x for x in order if x not in ("Oberarzt",)]:
    ae_ml=np.abs(los-np.clip(preds[o],0,None)); ae_a=np.abs(los-arzt)
    try: p=stats.wilcoxon(ae_ml,ae_a).pvalue
    except Exception: p=np.nan
    wr.append({"vs_Oberarzt":o,"MedianAE_ML":round(float(np.median(ae_ml)),2),"MedianAE_Arzt":round(float(np.median(ae_a)),2),
               "dMAE":round(float(ae_ml.mean()-ae_a.mean()),3),"p":("<0.001" if p<0.001 else f"{p:.3f}"),"ML_better_%":round(100*float((ae_ml<ae_a).mean()),1)})
pd.DataFrame(wr).to_csv(f"{OUT}_wilcoxon.csv",sep=";",index=False)
print("\n=== P80-Coverage ==="); print(pd.DataFrame(cov).to_string(index=False))
print("\n=== Wilcoxon (|Fehler| ML vs Oberarzt, overall) ==="); print(pd.DataFrame(wr).to_string(index=False))

# Figur: MAE nach Subgruppe (Oberarzt / ExtraTrees / Hazard_E / Quantile_P80)
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
sel=[o for o in ["Oberarzt","ExtraTrees","Tweedie_1.5","Hazard_E","Quantile_P80","Null"] if o in order]
sgs=["1-2 d","2-4 d","4-7 d",">7 d"]; w=0.145; xb=np.arange(len(sgs))
pal={"Oberarzt":"#c0392b","ExtraTrees":"#1f5f9e","Tweedie_1.5":"#5b9bd5","Hazard_E":"#1a9850","Quantile_P80":"#762a83","Null":"#c7ccd1"}
fig,ax=plt.subplots(figsize=(12,5.6))
for i,o in enumerate(sel):
    vals=[res[(res.Modell==o)&(res.Subgroup==sg)]["MAE"].iloc[0] for sg in sgs]
    ax.bar(xb+(i-(len(sel)-1)/2)*w,vals,w,label=o,color=pal.get(o,"#888"),edgecolor="white",linewidth=0.4)
    for j,v in enumerate(vals): ax.annotate(f"{v:.1f}",(xb[j]+(i-(len(sel)-1)/2)*w,v),ha="center",va="bottom",fontsize=6.6)
ax.set_xticks(xb); ax.set_xticklabels([f"{s}\n(n={int(subs[s].sum())})" for s in sgs]); ax.set_ylabel("MAE (days) — lower is better")
ax.set_title("KISIK prospective (n=193, reconstructed features): physician vs ML approaches by actual ICU LoS",weight="bold",fontsize=11.5)
ax.legend(fontsize=9,ncol=6,loc="upper center",bbox_to_anchor=(0.5,-0.08))
fig.tight_layout(); fig.savefig(f"{OUT}_mae.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print(f"\nGespeichert: {OUT}.csv / _coverage.csv / _wilcoxon.csv / _mae.png")
