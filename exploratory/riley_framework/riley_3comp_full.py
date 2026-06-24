# -*- coding: utf-8 -*-
"""EXPLORATIV (Voll-Auswertung): 3-Komponenten-Blend, MID = Arzt-Zahl.
  pred = p_short*SHORT + p_mid*ARZT + p_long*LONG
  p_long=sigma((arzt-c_hi)/s), p_short=sigma((c_lo-arzt)/s), p_mid=max(0,1-p_long-p_short); renorm.
SHORT/LONG = log1p-ExtraTrees+recal (retro 1<LoS<=7 bzw. >7). (c_lo,s) EHRLICH per nested-5-fold-CV
auf der prospektiven Kohorte (Arzt nur prospektiv) -> OOF-Vorhersage = ausgewertetes Modell.
Outputs: Metriken, Kalibrierung, Subgruppen+Signifikanz, Hexbins (retro ML-Kern + prospektiv Hybrid/Arzt),
Forest-Plot der Ueberlegenheit. Signifikanz vs Oberarzt: gepaarter Bootstrap-CI (B=5000) auf
dMAE=MAE_Arzt-MAE_Modell + Wilcoxon. Leckfrei, no_isopen n=286. Ausgabe: exploratory_riley/
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
from collections import Counter
import duckdb, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; OUT=AN/"exploratory_riley"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; B=5000; rng=np.random.default_rng(RS)
bpet=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["ExtraTrees"]

con=duckdb.connect()
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
allcols=list(con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') LIMIT 0").df().columns)
meta=["icu_duration_h","wardshort","oebenekurz","pid"]
present=[f for f in feat if f in allcols and not f.startswith(("lab_","vital_","proc_","zugang_")) and not f.startswith("proc24_8_98f")]
selc=meta+[f for f in present if f not in meta]; colstr=", ".join('"'+c+'"' for c in selc)
df=con.execute(f"SELECT {colstr} FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
y=(df["icu_duration_h"]/24.0).values; groups=df["pid"].fillna("unknown").astype(str).values
cat=[c for c in present if c=="oebenekurz"]; numc=[c for c in present if c not in cat]
def Xf(frame):
    X=frame.reindex(columns=present).copy()
    for c in present: X[c]=(X[c].astype(str) if c=="oebenekurz" else pd.to_numeric(X[c],errors="coerce"))
    return X
X=Xf(df); tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(X,y,groups))
Xtr,ytr,gtr=X.iloc[tr],y[tr],groups[tr]; Xte,yte=X.iloc[te],y[te]
def pre(): return ColumnTransformer([("num",SimpleImputer(strategy="median"),numc),
    ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat)])
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float)
Xp=pd.DataFrame(index=PR.index)
for c in present: Xp[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
Xp=Xp[present]

def et(): return TransformedTargetRegressor(Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=2))]),func=np.log1p,inverse_func=np.expm1)
def fit_recal(Xt,yt,gt):
    m=et(); m.fit(Xt,yt); oof=np.clip(cross_val_predict(et(),Xt,yt,groups=gt,cv=GroupKFold(3),n_jobs=1),0,None); b,a=np.polyfit(oof,yt,1); return m,a,b
def ap(m,a,b,Xs): return np.clip(a+b*np.clip(m.predict(Xs),0,None),0,None)

ms=(ytr>1)&(ytr<=7); ml=ytr>7
Ms,as_,bs=fit_recal(Xtr.iloc[ms.nonzero()[0]],ytr[ms],gtr[ms]); SHORT=ap(Ms,as_,bs,Xp)
Ml,al,bl=fit_recal(Xtr.iloc[ml.nonzero()[0]],ytr[ml],gtr[ml]); LONG=ap(Ml,al,bl,Xp)
# Voll-Kohorte-Kontinuierlich (retro ML-Kern fuer Hexbin retro; Arzt-Gate existiert retro nicht)
Mf,af,bf=fit_recal(Xtr,ytr,gtr); RETRO_PRED=ap(Mf,af,bf,Xte)

def soft3(idx,c_lo,s,c_hi=7.0):
    a=arzt[idx]; pl=1/(1+np.exp(-(a-c_hi)/s)); psh=1/(1+np.exp(-(c_lo-a)/s)); pm=np.clip(1-pl-psh,0,None)
    tot=pl+psh+pm; return (psh*SHORT[idx]+pm*a+pl*LONG[idx])/tot

CLO=[3,4,5]; SS=[1.0,1.5,2.0]; kf=KFold(5,shuffle=True,random_state=RS); N=len(los)
oof=np.full(N,np.nan); picks=[]
for trn,tst in kf.split(np.arange(N)):
    best=None
    for c_lo in CLO:
        for s in SS:
            v=mean_absolute_error(los[trn],np.clip(soft3(trn,c_lo,s),0,None))
            if best is None or v<best[0]: best=(v,c_lo,s)
    _,c_lo,s=best; picks.append((c_lo,s)); oof[tst]=np.clip(soft3(tst,c_lo,s),0,None)
HYB=oof  # ehrliche OOF-Vorhersage
(c_lo_f,s_f),_=Counter(picks).most_common(1)[0]
FROZEN=np.clip(soft3(np.arange(N),c_lo_f,s_f),0,None)

SG=[("1-2 d",1,2),("2-4 d",2,4),("4-7 d",4,7),(">7 d",7,999)]
def met(p):
    p=np.clip(p,0,None); m=los>7; sb,_=np.polyfit(p,los,1); d={}
    for l,lo_,hi_ in SG:
        mm=(los>lo_)&(los<=hi_) if hi_<999 else los>7; d["MAE_"+l]=round(float(np.abs(los[mm]-p[mm]).mean()),2)
    return dict(MAE=round(float(mean_absolute_error(los,p)),3),RMSE=round(float(np.sqrt(mean_squared_error(los,p))),3),
                R2=round(float(r2_score(los,p)),3),slope=round(float(sb),3),bias=round(float((p-los).mean()),3),
                MAE_gt7=round(float(np.abs(los[m]-p[m]).mean()),2),**d)
mh,mf2,mp=met(HYB),met(FROZEN),met(arzt)
pd.DataFrame([{"model":"3-comp hybrid (nested-CV OOF)",**mh},
             {"model":f"3-comp hybrid (frozen c_lo={c_lo_f},s={s_f})",**mf2},
             {"model":"Senior physician",**mp}]).to_csv(OUT/"3comp_metrics.csv",sep=";",index=False)

# ===== Signifikanz vs Oberarzt: gepaarter Bootstrap + Wilcoxon =====
def supr(mask):
    yt=los[mask]; ea=np.abs(yt-arzt[mask]); ee=np.abs(yt-HYB[mask]); n=len(yt)
    idx=rng.integers(0,n,size=(B,n)); d=ea[idx].mean(1)-ee[idx].mean(1); lo,hi=np.percentile(d,[2.5,97.5])
    try: w=wilcoxon(ea,ee,alternative="two-sided"); p=float(w.pvalue)
    except Exception: p=float("nan")
    v=("model better" if lo>0 else "physician better" if hi<0 else "n.s.")
    return dict(n=int(n),MAE_phys=round(float(ea.mean()),2),MAE_model=round(float(ee.mean()),2),
                dMAE=round(float(ea.mean()-ee.mean()),3),CI_low=round(float(lo),3),CI_high=round(float(hi),3),
                wilcoxon_p=round(p,4),verdict=v)
rows=[{"subgroup":"overall",**supr(np.ones(N,bool))}]
for l,lo_,hi_ in SG:
    m=(los>lo_)&(los<=hi_) if hi_<999 else los>7; rows.append({"subgroup":l,**supr(m)})
sup=pd.DataFrame(rows); sup.to_csv(OUT/"3comp_superiority.csv",sep=";",index=False)

print("=== 3-Komponenten-Hybrid (MID=Arzt), prospektiv n=%d ==="%N)
print(pd.DataFrame([{"model":"hybrid (OOF)",**mh},{"model":"physician",**mp}])[
    ["model","MAE","RMSE","R2","slope","MAE_1-2 d","MAE_2-4 d","MAE_4-7 d","MAE_>7 d"]].to_string(index=False))
print("\n=== Signifikanz vs Oberarzt (dMAE = MAE_Arzt - MAE_Modell; >0 = Modell besser) ===")
print(sup[["subgroup","n","MAE_phys","MAE_model","dMAE","CI_low","CI_high","wilcoxon_p","verdict"]].to_string(index=False))
print(f"\nnested-CV (c_lo,s)-Wahl je Fold: {picks}  -> frozen ({c_lo_f},{s_f})")

# ===== FIGUR 1: Hexbins (retro ML-Kern | prospektiv Hybrid | prospektiv Arzt) =====
def hexpanel(ax,pred,obs,title,cap):
    p=np.clip(pred,0,cap); o=np.clip(obs,0,cap)
    hb=ax.hexbin(p,o,gridsize=28,cmap="viridis",mincnt=1,extent=(0,cap,0,cap))
    ax.plot([0,cap],[0,cap],"--",color="#e74c3c",lw=1.4)
    mae=mean_absolute_error(obs,np.clip(pred,0,None)); r2=r2_score(obs,np.clip(pred,0,None))
    sb,_=np.polyfit(np.clip(pred,0,None),obs,1)
    ax.set_xlim(0,cap); ax.set_ylim(0,cap); ax.set_aspect("equal","box")
    ax.set_xlabel("Predicted LoS (days)"); ax.set_ylabel("Observed LoS (days)")
    ax.set_title(title,weight="bold",fontsize=10.5)
    ax.text(0.04,0.96,f"MAE {mae:.2f}\nR² {r2:.2f}\nslope {sb:.2f}\nn={len(obs)}",transform=ax.transAxes,
            va="top",ha="left",fontsize=8.5,bbox=dict(boxstyle="round",fc="white",ec="#ccc",alpha=.85))
    return hb
fig,ax=plt.subplots(1,3,figsize=(16.5,5.4))
hexpanel(ax[0],RETRO_PRED,yte,"(A) Retrospective ML core — hold-out\nlog1p-ExtraTrees + recalibration",30)
hexpanel(ax[1],HYB,los,"(B) Prospective — 3-component hybrid\n(MID = physician, nested-CV)",30)
hb=hexpanel(ax[2],arzt,los,"(C) Prospective — senior physician",30)
fig.colorbar(hb,ax=ax,label="cases per bin",fraction=0.025,pad=0.02)
fig.suptitle("Predicted vs observed ICU LoS — retrospective ML core and prospective hybrid vs physician (leak-free)",weight="bold",fontsize=12.5)
fig.savefig(str(OUT/"fig_3comp_hexbins.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

# ===== FIGUR 2: Subgruppen-MAE mit Signifikanz-Markern =====
fig,ax=plt.subplots(figsize=(9,5.4)); labels=[l for l,_,_ in SG]; xb=np.arange(len(labels)); w=0.36
vh=[mh["MAE_"+l] for l in labels]; vp=[mp["MAE_"+l] for l in labels]
b1=ax.bar(xb-w/2,vh,w,label="3-component hybrid",color="#1b7f3b"); b2=ax.bar(xb+w/2,vp,w,label="senior physician",color="#c0392b")
ax.bar_label(b1,fmt="%.2f",fontsize=8); ax.bar_label(b2,fmt="%.2f",fontsize=8)
for i,l in enumerate(labels):
    r=sup[sup.subgroup==l].iloc[0]; mark={"model better":"*","physician better":"†","n.s.":"n.s."}[r.verdict]
    yy=max(vh[i],vp[i]); ax.plot([xb[i]-w/2,xb[i]+w/2],[yy+0.25,yy+0.25],color="#444",lw=1.1)
    ax.text(xb[i],yy+0.32,mark,ha="center",fontsize=11 if mark in("*","†") else 8,weight="bold")
ax.set_xticks(xb); ax.set_xticklabels(labels); ax.set_ylabel("MAE (days)")
ax.set_title("Subgroup MAE: 3-component hybrid vs senior physician\n(* hybrid sig. better, † physician sig. better, n.s. = not significant; paired bootstrap 95% CI)",
             weight="bold",fontsize=10.5); ax.legend(fontsize=9.5); ax.set_ylim(0,max(max(vh),max(vp))+1.2)
fig.tight_layout(); fig.savefig(str(OUT/"fig_3comp_subgroup_sig.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

# ===== FIGUR 3: Forest-Plot dMAE (Arzt - Modell) =====
fig,ax=plt.subplots(figsize=(8.6,4.8)); order=["overall"]+labels; yy=np.arange(len(order))[::-1]
for i,sgl in enumerate(order):
    r=sup[sup.subgroup==sgl].iloc[0]; c="#1b7f3b" if r.dMAE>0 else "#c0392b"
    ax.plot([r.CI_low,r.CI_high],[yy[i],yy[i]],color=c,lw=2.2); ax.plot(r.dMAE,yy[i],"o",color=c,ms=8)
    ax.text(r.CI_high+0.05,yy[i],f"{r.dMAE:+.2f} [{r.CI_low:+.2f},{r.CI_high:+.2f}]  {r.verdict}",va="center",fontsize=8.3)
ax.axvline(0,color="#444",ls="--",lw=1); ax.set_yticks(yy); ax.set_yticklabels([f"{o} (n={sup[sup.subgroup==o].n.iloc[0]})" for o in order])
ax.set_xlabel("ΔMAE = MAE(physician) − MAE(hybrid)   →   right = hybrid better")
ax.set_title("Superiority of the 3-component hybrid over the senior physician\n(paired bootstrap, B=5000, 95% CI)",weight="bold",fontsize=10.5)
ax.set_xlim(min(sup.CI_low)-0.3,max(sup.CI_high)+1.8)
fig.tight_layout(); fig.savefig(str(OUT/"fig_3comp_forest.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

# ===== FIGUR 4: Kalibrierung (binned) =====
fig,ax=plt.subplots(figsize=(6.4,6.0)); CAP=20
def cv(p,q=8):
    d=pd.DataFrame({"p":np.clip(p,0,None),"o":los}); d["b"]=pd.qcut(d["p"],q,duplicates="drop"); g=d.groupby("b",observed=True)
    return g["p"].mean().to_numpy(),g["o"].mean().to_numpy()
ax.plot([0,CAP],[0,CAP],"--",color="#888",label="ideal")
for p,lab,c in [(HYB,"3-component hybrid","#1b7f3b"),(arzt,"senior physician","#c0392b")]:
    mpx,mox=cv(p); ax.plot(mpx,mox,"o-",color=c,lw=1.9,ms=6,label=lab)
ax.set_xlim(0,CAP); ax.set_ylim(0,CAP); ax.set_aspect("equal","box")
ax.set_xlabel("Predicted LoS (days)"); ax.set_ylabel("Observed LoS (days)")
ax.set_title("Calibration (prospective, n=286)",weight="bold",fontsize=11); ax.legend(fontsize=9.5,loc="upper left")
fig.tight_layout(); fig.savefig(str(OUT/"fig_3comp_calibration.png"),dpi=300,bbox_inches="tight"); plt.close(fig)

print("\nGespeichert: fig_3comp_hexbins.png, fig_3comp_subgroup_sig.png, fig_3comp_forest.png, fig_3comp_calibration.png")
print("           + 3comp_metrics.csv, 3comp_superiority.csv")
