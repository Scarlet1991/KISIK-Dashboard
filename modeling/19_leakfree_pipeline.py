# -*- coding: utf-8 -*-
"""LEAKAGE-CORRECTED pipeline: identical to the primary analysis but with the OPS 8-98f
complex-treatment family (a target leak) removed from the feature set.
Re-trains the 5 models, re-selects the final model, re-evaluates retrospectively and
prospectively (no_isopen n=200), and writes all CSVs + figures into  Eigene Auswertung/leakfree/."""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"
LF=AN/"leakfree"; LF.mkdir(parents=True,exist_ok=True)
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"
FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"
bp=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]

con=duckdb.connect()
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
df["los_days"]=df["icu_duration_h"]/24.0
y=df["los_days"].values; groups=df["pid"].fillna("unknown").astype(str).values
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns and not f.startswith(("lab_","vital_","proc_","zugang_"))]
LEAK=[c for c in present if c.startswith("proc24_8_98f")]
present=[c for c in present if c not in LEAK]     # <-- leak removed
cat=[c for c in present if c=="oebenekurz"]; numc=[c for c in present if c not in cat]
print(f"n={len(df)} | removed {len(LEAK)} leak features {LEAK} | {len(present)} features remain")

def Xframe(frame):
    X=frame.reindex(columns=present).copy()
    for c in numc: X[c]=pd.to_numeric(X[c],errors="coerce")
    for c in cat:  X[c]=X[c].astype(str)
    return X
X=Xframe(df)
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(X,y,groups))
def pre(scale=False):
    ns=[("imp",SimpleImputer(strategy="median"))]+([("sc",StandardScaler())] if scale else [])
    return ColumnTransformer([("num",Pipeline(ns),numc),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)])
def ttr(reg,scale=False): return TransformedTargetRegressor(Pipeline([("pre",pre(scale)),("mdl",reg)]),func=np.log1p,inverse_func=np.expm1)
def plain(reg): return Pipeline([("pre",pre(False)),("mdl",reg)])
models={
 "Ridge":ttr(Ridge(**bp["Ridge"],random_state=RS),scale=True),
 "RandomForest":ttr(RandomForestRegressor(**bp["RandomForest"],random_state=RS,n_jobs=1)),
 "ExtraTrees":ttr(ExtraTreesRegressor(**bp["ExtraTrees"],random_state=RS,n_jobs=1)),
 "XGBoost":ttr(XGBRegressor(**bp["XGBoost"],random_state=RS,n_jobs=1,tree_method="hist")),
 "Tweedie":plain(XGBRegressor(objective="reg:tweedie",**bp["Tweedie"],random_state=RS,n_jobs=1,tree_method="hist")),
}
# ---- CV-MAE (GroupKFold) for model selection ----
gkf=GroupKFold(4)
cvmae={}
for n,m in models.items():
    sc=cross_val_score(m,X.iloc[tr],y[tr],groups=groups[tr],cv=gkf,scoring="neg_mean_absolute_error",n_jobs=1)
    cvmae[n]=float(-sc.mean())
final=min(cvmae,key=cvmae.get)
print("CV-MAE:",{k:round(v,3) for k,v in cvmae.items()},"-> final:",final)

# ---- fit on train, eval on holdout ----
def metr(yt,yp):
    yp=np.clip(yp,0,None); ae=np.abs(yt-yp)
    return dict(MAE_days=round(float(ae.mean()),3),MedianAE_days=round(float(np.median(ae)),3),
                RMSE_days=round(float(np.sqrt(mean_squared_error(yt,yp))),3),R2=round(float(r2_score(yt,yp)),3),
                Bias_days=round(float((yp-yt).mean()),3))
rows=[]
for n,m in models.items():
    m.fit(X.iloc[tr],y[tr]); d=metr(y[te],m.predict(X.iloc[te]))
    rows.append({"Modell":n,"n":len(te),**d,"CV_MAE_days":round(cvmae[n],3)})
retro=pd.DataFrame(rows).sort_values("CV_MAE_days")
retro.to_csv(LF/"metrics_retrospective_lf.csv",sep=";",index=False)
print("\n--- Retrospektiv leakfree (Holdout) ---"); print(retro.to_string(index=False))

# ---- permutation importance (final model) ----
fm=models[final]
pi=permutation_importance(fm,X.iloc[te],y[te],n_repeats=10,random_state=RS,scoring="neg_mean_absolute_error",n_jobs=1)
imp=pd.DataFrame({"Feature":present,"MAE_increase_days":pi.importances_mean,"sd":pi.importances_std}).sort_values("MAE_increase_days",ascending=False)
imp.to_csv(LF/"feature_importance_lf.csv",sep=";",index=False)

# ---- summary ----
summ={"n_stays":int(len(df)),"n_patients":int(df["pid"].nunique()),"n_train":int(len(tr)),"n_test":int(len(te)),
      "los_days_median":round(float(np.median(y)),2),"los_days_p90":round(float(np.percentile(y,90)),2),
      "los_days_max":round(float(y.max()),1),"n_features_leakfree":len(present),"removed_leak":LEAK,
      "final_model":final,"cv_mae":{k:round(v,3) for k,v in cvmae.items()},"best_params":bp}
json.dump(summ,open(LF/"summary_lf.json","w"),indent=2)

# ============ PROSPECTIVE (no_isopen n=200), reuse reconstructed matrix minus leak ============
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float); iso=PR["__is_open__"].to_numpy(int)
def Xpros(frame):
    Xp=pd.DataFrame(index=frame.index)
    for c in present:
        if c in frame.columns: Xp[c]=(frame[c].astype(str) if c in cat else pd.to_numeric(frame[c],errors="coerce"))
        else: Xp[c]=("nan" if c in cat else np.nan)
    for c in cat: Xp[c]=Xp[c].astype(str)
    return Xp[present]
Xp=Xpros(PR)
def mp(yt,yp,sub=None):
    if sub is not None: yt,yp=yt[sub],yp[sub]
    yp=np.clip(np.asarray(yp,float),0,None); ae=np.abs(yt-yp)
    return dict(n=int(len(yt)),MAE=round(float(ae.mean()),3),MedianAE=round(float(np.median(ae)),3),
               RMSE=round(float(np.sqrt(mean_squared_error(yt,yp))),3),R2=round(float(r2_score(yt,yp)),3),Bias=round(float((yp-yt).mean()),3))
preds={n:np.clip(m.predict(Xp),0,None) for n,m in models.items()}
order=["Ridge","RandomForest","ExtraTrees","XGBoost","Tweedie"]
prows=[{"Modell":"Oberarzt",**mp(los,arzt)}]+[{"Modell":n,**mp(los,preds[n])} for n in order]
pros=pd.DataFrame(prows).set_index("Modell")
pros.reset_index().to_csv(LF/"prospektiv_overall_lf.csv",sep=";",index=False)
print(f"\n--- Prospektiv leakfree (n={len(los)}) ---"); print(pros.to_string())

# subgroups (2-4/4-7/>7) + Null
null=float(np.median(y[tr]))
sg=[("1–2 d",(los>=1)&(los<=2)),("2–4 d",(los>2)&(los<=4)),("4–7 d",(los>4)&(los<=7)),(">7 d",los>7)]
allp={"Oberarzt":arzt,**preds,"Null":np.full(len(los),null)}
sgrows=[]
for lab,mask in sg:
    for mn,pv in allp.items():
        ae=np.abs(los[mask]-np.asarray(pv,float)[mask])
        sgrows.append({"Subgroup":lab,"n":int(mask.sum()),"Modell":mn,"MAE":round(float(ae.mean()),3)})
sgdf=pd.DataFrame(sgrows); sgdf.to_csv(LF/"metrics_subgroups_lf.csv",sep=";",index=False)

# superiority final-model vs physician (paired bootstrap)
rng=np.random.default_rng(RS); B=5000
def superiority(mask):
    yt=los[mask]; a=arzt[mask]; p=np.clip(preds[final],0,None)[mask]
    ea=np.abs(yt-a); ee=np.abs(yt-p); n=len(yt); idx=rng.integers(0,n,size=(B,n))
    boot=ea[idx].mean(1)-ee[idx].mean(1); lo,hi=np.percentile(boot,[2.5,97.5])
    return round(float(ea.mean()-ee.mean()),3),round(float(lo),3),round(float(hi),3)
suprows=[]
for lab,mask in [("overall",np.ones(len(los),bool))]+sg:
    d,lo,hi=superiority(mask); suprows.append({"Subgruppe":lab,"Modell":final,"n":int(mask.sum()),"dMAE":d,"CI_low":lo,"CI_high":hi,
        "verdict":("model" if lo>0 else "physician" if hi<0 else "n.s.")})
supdf=pd.DataFrame(suprows); supdf.to_csv(LF/"superiority_lf.csv",sep=";",index=False)
print("\n--- Superiority (final vs physician) ---"); print(supdf.to_string(index=False))

# ============ FIGURES ============
RC="#185fa5"; PC="#c0392b"
# Fig: model comparison (retro vs prospective, MAE + R2)
fo=[m for m in order if m in retro.set_index("Modell").index]
labs=[{"RandomForest":"Random forest","ExtraTrees":"Extra Trees"}.get(m,m) for m in fo]
rmae=[retro.set_index("Modell").loc[m,"MAE_days"] for m in fo]; pmae=[pros.loc[m,"MAE"] for m in fo]
rr2=[retro.set_index("Modell").loc[m,"R2"] for m in fo]; pr2=[pros.loc[m,"R2"] for m in fo]
xi=np.arange(len(fo)); w=0.38
fig,(axA,axB)=plt.subplots(1,2,figsize=(13,4.8))
axA.bar(xi-w/2,rmae,w,label=f"retrospective hold-out (n={len(te):,})",color="#b5d4f4")
axA.bar(xi+w/2,pmae,w,label=f"prospective (n={len(los)})",color="#185fa5")
axA.axhline(pros.loc["Oberarzt","MAE"],color=PC,ls="--",lw=1.6)
axA.text(len(fo)-1,pros.loc["Oberarzt","MAE"]+0.05,f"Senior physician (MAE {pros.loc['Oberarzt','MAE']:.2f} d)",color=PC,ha="right",fontsize=8.5,weight="bold")
axA.set_xticks(xi); axA.set_xticklabels(labs,fontsize=9.5); axA.set_ylabel("MAE (days) — lower is better")
axA.set_title("(A) MAE — lower is better",weight="bold",fontsize=11); axA.legend(fontsize=8.5)
R2F=-0.35; pr2c=[max(v,R2F) for v in pr2]
axB.bar(xi-w/2,rr2,w,label="retrospective hold-out",color="#b5d4f4"); axB.bar(xi+w/2,pr2c,w,label="prospective",color="#185fa5")
axB.axhline(0,color="#888",lw=.8); axB.axhline(pros.loc["Oberarzt","R2"],color=PC,ls="--",lw=1.6)
axB.text(0,pros.loc["Oberarzt","R2"]+0.02,f"Senior physician (R² {pros.loc['Oberarzt','R2']:.2f})",color=PC,fontsize=8.5,weight="bold")
for i,v in enumerate(pr2): axB.text(xi[i]+w/2,pr2c[i]+0.01,f"{v:.2f}",ha="center",va="bottom",fontsize=7.2,color="#1f5f9e")
axB.set_ylim(R2F,0.62); axB.set_xticks(xi); axB.set_xticklabels(labs,fontsize=9.5); axB.set_ylabel("R² — higher is better")
axB.set_title("(B) R² — higher is better",weight="bold",fontsize=11); axB.legend(fontsize=8.5,loc="upper right")
fig.suptitle(f"Leakage-corrected model (8-98f excluded): retrospective (n={len(te):,}) vs prospective (n={len(los)})",weight="bold",fontsize=12)
fig.tight_layout(); fig.savefig(str(LF/"fig_model_comparison_lf.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

# Fig: importance
top=imp.head(12).iloc[::-1]
fig,ax=plt.subplots(figsize=(8.4,6))
ax.barh(range(len(top)),top["MAE_increase_days"],xerr=top["sd"],color="#762a83")
ax.set_yticks(range(len(top))); ax.set_yticklabels([f.replace("proc24_","Procedure ").replace("diag_main_","Diagnosis ").replace("vital24_","Vital ").replace("lab24_","Lab ").replace("_"," ") for f in top["Feature"]],fontsize=8.5)
ax.set_xlabel("Increase in MAE when permuted (days)"); ax.set_title("Permutation importance — leakage-corrected final model ("+final+")",weight="bold",fontsize=11)
fig.tight_layout(); fig.savefig(str(LF/"fig_importance_lf.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

# Fig: subgroup MAE with significance (final vs physician)
mlist=["Oberarzt","Ridge","RandomForest","ExtraTrees","XGBoost","Tweedie","Null"]
disp={"Oberarzt":"Senior physician","RandomForest":"Random Forest","ExtraTrees":"Extra Trees","Null":"Null (mean)"}
col={"Oberarzt":"#c0392b","Ridge":"#7f8c8d","RandomForest":"#3498db","ExtraTrees":"#1a6ea3","XGBoost":"#27ae60","Tweedie":"#8e44ad","Null":"#bdc3c7"}
binl=[s[0] for s in sg]; xi=np.arange(len(binl)); nm=len(mlist); bw=0.82/nm; off=np.linspace(-(0.82-bw)/2,(0.82-bw)/2,nm)
def smae(mn,b): r=sgdf[(sgdf.Modell==mn)&(sgdf.Subgroup==b)]; return float(r.MAE.iloc[0]) if len(r) else 0
fig,ax=plt.subplots(figsize=(12,5.6))
for i,mn in enumerate(mlist):
    bars=ax.bar(xi+off[i],[smae(mn,b) for b in binl],bw,label=disp.get(mn,mn),color=col[mn]); ax.bar_label(bars,fmt="%.1f",fontsize=6.6,padding=1)
sup_by={r["Subgruppe"]:r for _,r in supdf.iterrows()}
ymax=max(sgdf.MAE)*1.25; ax.set_ylim(0,ymax)
for k,b in enumerate(binl):
    r=sup_by[b]; c={"model":"#1b7f3b","physician":"#b03030","n.s.":"#777"}[r["verdict"]]
    h=max(smae("Oberarzt",b),smae(final,b));
    txt={"model":f"{final} superior\nΔ{r['dMAE']:+.2f} [{r['CI_low']:.2f},{r['CI_high']:.2f}]",
         "physician":f"Physician superior\nΔ{r['dMAE']:+.2f} [{r['CI_low']:.2f},{r['CI_high']:.2f}]",
         "n.s.":f"n.s.\nΔ{r['dMAE']:+.2f} [{r['CI_low']:.2f},{r['CI_high']:.2f}]"}[r["verdict"]]
    ax.text(xi[k],h+ymax*0.07,txt,ha="center",va="bottom",fontsize=8,weight="bold",color=c)
ax.set_xticks(xi); ax.set_xticklabels([f"{b}\n(n={dict((s[0],int(s[1].sum())) for s in sg)[b]})" for b in binl],fontsize=10)
ax.set_ylabel("MAE (days) — lower is better"); ax.legend(fontsize=8.5,ncol=7,loc="upper center",bbox_to_anchor=(0.5,-0.07))
ax.set_title(f"Leakage-corrected: prospective MAE by LoS subgroup (n={len(los)}) with {final}-vs-physician superiority test",weight="bold",fontsize=11.5)
fig.tight_layout(); fig.savefig(str(LF/"fig_subgroup_mae_lf.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

# Fig: calibration (final + physician)
def calib(pred,q=10):
    d=pd.DataFrame({"p":np.clip(pred,0,None),"o":los}); d["b"]=pd.qcut(d["p"],q,duplicates="drop"); g=d.groupby("b",observed=True)
    return g["p"].mean().to_numpy(),g["o"].mean().to_numpy(),1.96*g["o"].std(ddof=1).to_numpy()/np.sqrt(g["o"].size().to_numpy())
CAP=20
fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharex=True,sharey=True)
for ax,(pv,nm,c) in zip(axes,[(preds[final],f"{final} (leak-free final)",RC),(arzt,"Senior physician",PC)]):
    mp_,mo_,ci=calib(pv); b,a=np.polyfit(np.clip(pv,0,None),los,1); xs=np.linspace(0,CAP,50)
    ax.plot([0,CAP],[0,CAP],"--",color="#888",lw=1.4,label="ideal"); ax.errorbar(mp_,mo_,yerr=ci,fmt="o",color=c,ms=7,capsize=3,label="observed per decile (95% CI)")
    ax.plot(xs,a+b*xs,"-",color=c,lw=2,alpha=.8,label=f"slope {b:.2f}, intercept {a:+.2f}")
    ax.set_xlim(0,CAP); ax.set_ylim(0,CAP); ax.set_aspect("equal","box"); ax.set_xlabel("Predicted ICU LoS (days)"); ax.set_title(nm,weight="bold",fontsize=12); ax.legend(fontsize=8.5,loc="upper left")
axes[0].set_ylabel("Observed ICU LoS (days)")
fig.suptitle(f"Leakage-corrected calibration — prospective (n={len(los)})",weight="bold",fontsize=12)
fig.tight_layout(); fig.savefig(str(LF/"fig_calibration_lf.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print(f"\nAlle leakfree-Ausgaben in {LF}")
