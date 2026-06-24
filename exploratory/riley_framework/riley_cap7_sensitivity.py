# -*- coding: utf-8 -*-
"""SENSITIVITAET: kontinuierliche Prognose NUR fuer 1 < LoS <= 7 d (Cap), Training UND Evaluierung.
Leckfrei. Tweedie + eingefrorene Rekalibrierung vs Oberarzt, prospektiv (Beobachtung <=7 d).
Hinweis: Evaluations-Subset ist OUTCOME-bedingt (obs<=7) -> nur deskriptive Sensitivitaet, nicht deploybar.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; OUT=AN/"exploratory_riley"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; B=5000; rng=np.random.default_rng(RS)
bp=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]; bptw=bp["Tweedie"]; bpet=bp["ExtraTrees"]
CAPLO,CAPHI=1.0,7.0

con=duckdb.connect()
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>{CAPLO} AND icu_duration_h/24.0<={CAPHI}").df()
y=(df["icu_duration_h"]/24.0).values; groups=df["pid"].fillna("unknown").astype(str).values
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns and not f.startswith(("lab_","vital_","proc_","zugang_")) and not f.startswith("proc24_8_98f")]
cat=[c for c in present if c=="oebenekurz"]; numc=[c for c in present if c not in cat]
def Xf(frame):
    X=frame.reindex(columns=present).copy()
    for c in present: X[c]=(X[c].astype(str) if c=="oebenekurz" else pd.to_numeric(X[c],errors="coerce"))
    return X
X=Xf(df); tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(X,y,groups))
Xtr,ytr,gtr=X.iloc[tr],y[tr],groups[tr]
def pre(): return ColumnTransformer([("num",SimpleImputer(strategy="median"),numc),
    ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat)])
print(f"CAP 1<LoS<=7: Training-Kohorte n={len(df)} (median {np.median(y):.1f} d, max {y.max():.1f} d)")

# prospektiv: nur Beobachtung 1<LoS<=7
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
losA=PR["__los__"].to_numpy(float); arztA=PR["__arzt__"].to_numpy(float)
msk=(losA>CAPLO)&(losA<=CAPHI)
los=losA[msk]; arzt=arztA[msk]
Xp=pd.DataFrame(index=PR.index)
for c in present: Xp[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
Xp=Xp[present].iloc[msk.nonzero()[0]]
print(f"Prospektive Evaluierung (obs 1<LoS<=7): n={len(los)} von {len(losA)} (median {np.median(los):.1f} d)")

gkf=GroupKFold(4)
TW=Pipeline([("pre",pre()),("mdl",XGBRegressor(objective="reg:tweedie",**bptw,random_state=RS,n_jobs=-1,tree_method="hist"))]); TW.fit(Xtr,ytr)
oof=np.clip(cross_val_predict(TW,Xtr,ytr,groups=gtr,cv=gkf,n_jobs=-1),0,None); b,a=np.polyfit(oof,ytr,1)
def recal(p): return np.clip(a+b*np.clip(p,0,None),0,None)
ET=TransformedTargetRegressor(Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=-1))]),func=np.log1p,inverse_func=np.expm1); ET.fit(Xtr,ytr)
print(f"Tweedie-Rekalibrierung (cap): slope b={b:.3f}, intercept a={a:.3f}")

def met(obs,p):
    p=np.clip(p,0,None); sb,_=np.polyfit(p,obs,1)
    return dict(MAE=round(float(mean_absolute_error(obs,p)),3),RMSE=round(float(np.sqrt(mean_squared_error(obs,p))),3),
                R2=round(float(r2_score(obs,p)),3),calib_slope=round(float(sb),3),bias=round(float((p-obs).mean()),3))
preds={"Combined model (Tweedie+recal)":recal(TW.predict(Xp)),"ExtraTrees log1p naive":ET.predict(Xp),"Senior physician":arzt}
rows=[{"estimator":k,**met(los,v)} for k,v in preds.items()]
res=pd.DataFrame(rows); res.to_csv(OUT/"cap7_continuous_metrics.csv",sep=";",index=False)
print("\n=== KONTINUIERLICH, CAP 1<LoS<=7 (prospektiv n=%d) ==="%len(los)); print(res.to_string(index=False))

# Superioritaet Modell vs Arzt (gepaart, Bootstrap) overall + Subgruppen
def supr(mask,pred):
    yt=los[mask]; ar=arzt[mask]; pr=np.clip(pred[mask],0,None); ea=np.abs(yt-ar); ee=np.abs(yt-pr); n=len(yt)
    idx=rng.integers(0,n,size=(B,n)); d=ea[idx].mean(1)-ee[idx].mean(1); lo,hi=np.percentile(d,[2.5,97.5])
    v="model" if lo>0 else ("physician" if hi<0 else "n.s.")
    return n,round(float(ea.mean()),2),round(float(ee.mean()),2),round(float(ea.mean()-ee.mean()),2),round(float(lo),2),round(float(hi),2),v
mc=recal(TW.predict(Xp))
print("\n=== Superioritaet Combined model vs Oberarzt (cap, ΔMAE=Arzt-Modell) ===")
print(f"{'bin':<9}{'n':>5}{'MAE_Arzt':>10}{'MAE_Mod':>9}{'dMAE[CI]':>22}  verdict")
srows=[]
for lab,m in [("overall",np.ones(len(los),bool)),("1-2 d",(los>1)&(los<=2)),("2-4 d",(los>2)&(los<=4)),("4-7 d",(los>4)&(los<=7))]:
    if m.sum()<5: continue
    n,ma,mm,d,lo,hi,v=supr(m,mc); srows.append({"bin":lab,"n":n,"MAE_Arzt":ma,"MAE_model":mm,"dMAE":d,"CI_low":lo,"CI_high":hi,"verdict":v})
    print(f"{lab:<9}{n:>5}{ma:>10.2f}{mm:>9.2f}{f'  {d:+.2f}[{lo:+.2f},{hi:+.2f}]':>22}  {v}")
pd.DataFrame(srows).to_csv(OUT/"cap7_superiority.csv",sep=";",index=False)

# ===== Figur =====
fig,ax=plt.subplots(1,2,figsize=(13,5.3))
CAP=8
def cv(p,q=6):
    d=pd.DataFrame({"p":np.clip(p,0,None),"o":los}); d["b"]=pd.qcut(d["p"],q,duplicates="drop"); g=d.groupby("b",observed=True)
    return g["p"].mean().to_numpy(),g["o"].mean().to_numpy()
ax[0].plot([0,CAP],[0,CAP],"--",color="#888",label="ideal")
for p,lab,c in [(mc,"Combined model","#1f5f9e"),(arzt,"Senior physician","#c0392b")]:
    mp,mo=cv(p); ax[0].plot(mp,mo,"o-",color=c,lw=1.9,ms=6,label=lab)
ax[0].set_xlim(0,CAP); ax[0].set_ylim(0,CAP); ax[0].set_aspect("equal","box")
ax[0].set_xlabel("Predicted LoS (days)"); ax[0].set_ylabel("Observed LoS (days)")
ax[0].set_title("(A) Calibration, capped 1–7 d (prospective)",weight="bold",fontsize=11); ax[0].legend(fontsize=9,loc="upper left")
bins=["1-2 d","2-4 d","4-7 d"]; xb=np.arange(3); w=0.38
def smae(p,lab):
    out=[]
    for lo_,hi_ in [(1,2),(2,4),(4,7)]:
        m=(los>lo_)&(los<=hi_); out.append(np.abs(los[m]-np.clip(p[m],0,None)).mean())
    return out
b1=ax[1].bar(xb-w/2,smae(mc,""),w,label="Combined model",color="#1f5f9e"); b2=ax[1].bar(xb+w/2,smae(arzt,""),w,label="Senior physician",color="#c0392b")
ax[1].bar_label(b1,fmt="%.2f",fontsize=8.5); ax[1].bar_label(b2,fmt="%.2f",fontsize=8.5)
ns={"1-2 d":int(((los>1)&(los<=2)).sum()),"2-4 d":int(((los>2)&(los<=4)).sum()),"4-7 d":int(((los>4)&(los<=7)).sum())}
ax[1].set_xticks(xb); ax[1].set_xticklabels([f"{b}\n(n={ns[b]})" for b in bins]); ax[1].set_ylabel("MAE (days)")
ax[1].set_title("(B) MAE by subgroup within 1–7 d",weight="bold",fontsize=11); ax[1].legend(fontsize=9)
fig.suptitle(f"Sensitivity: continuous model capped to 1–7 d — model vs physician (prospective n={len(los)}, leak-free)",weight="bold",fontsize=12.5)
fig.tight_layout(); fig.savefig(str(OUT/"fig_cap7_vs_physician.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print(f"\nGespeichert: fig_cap7_vs_physician.png + cap7_continuous_metrics.csv + cap7_superiority.csv")
