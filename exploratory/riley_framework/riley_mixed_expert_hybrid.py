# -*- coding: utf-8 -*-
"""EXPLORATIV: GEMISCHTER Experten-Hybrid mit Arzt-Gate.
Idee: log1p(ExtraTrees) als KURZ-Experte (ideal fuer Kurzlieger), Tweedie als LANG-Experte
(kein Ruecktransformations-Bias), Oberarzt-Schaetzung (arzt>=K) als Weiche.
Vergleich der Kurz-Experten-Wahl {Tweedie | ExtraTrees-log1p(+recal) | ExtraTrees-log1p(naiv)}
fuer K in {5,7}. Evaluierung prospektiv no_isopen (alle n=286). Leckfrei.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; OUT=AN/"exploratory_riley"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"
bp=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]; bpet=bp["ExtraTrees"]; bptw=bp["Tweedie"]

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
Xtr,ytr,gtr=X.iloc[tr],y[tr],groups[tr]
def pre(): return ColumnTransformer([("num",SimpleImputer(strategy="median"),numc),
    ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat)])
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float)
Xp=pd.DataFrame(index=PR.index)
for c in present: Xp[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
Xp=Xp[present]

def factory(tool):
    if tool=="ETlog":
        return TransformedTargetRegressor(Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=2))]),func=np.log1p,inverse_func=np.expm1)
    return Pipeline([("pre",pre()),("mdl",XGBRegressor(objective="reg:tweedie",**bptw,random_state=RS,n_jobs=2,tree_method="hist"))])
def fit(tool,Xt,yt): m=factory(tool); m.fit(Xt,yt); return m
def recal_ab(tool,Xt,yt,gt):
    oof=np.clip(cross_val_predict(factory(tool),Xt,yt,groups=gt,cv=GroupKFold(3),n_jobs=1),0,None); b,a=np.polyfit(oof,yt,1); return a,b
def pr_(m,Xs,a=0.0,b=1.0): return np.clip(a+b*np.clip(m.predict(Xs),0,None),0,None)

def met(obs,p):
    p=np.clip(p,0,None); sb,_=np.polyfit(p,obs,1); m=obs>7
    sg={}
    for l,lo_,hi_ in [("1-2",1,2),("2-4",2,4),("4-7",4,7),(">7",7,999)]:
        mm=(obs>lo_)&(obs<=hi_) if hi_<999 else obs>7; sg["MAE_"+l]=round(float(np.abs(obs[mm]-p[mm]).mean()),2)
    return dict(MAE=round(float(mean_absolute_error(obs,p)),3),R2=round(float(r2_score(obs,p)),3),
                slope=round(float(sb),3),MAE_gt7=round(float(np.abs(obs[m]-p[m]).mean()),3),**sg)

rows=[]
for K in [5,7]:
    ms=(ytr>1)&(ytr<=K); ml=ytr>K
    # Lang-Experte = Tweedie+recal (fix)
    Ml=fit("Tweedie",Xtr.iloc[ml.nonzero()[0]],ytr[ml]); al,bl=recal_ab("Tweedie",Xtr.iloc[ml.nonzero()[0]],ytr[ml],gtr[ml]); long=pr_(Ml,Xp,al,bl)
    # Kurz-Experten-Varianten
    Mt=fit("Tweedie",Xtr.iloc[ms.nonzero()[0]],ytr[ms]); at,bt=recal_ab("Tweedie",Xtr.iloc[ms.nonzero()[0]],ytr[ms],gtr[ms])
    Me=fit("ETlog",Xtr.iloc[ms.nonzero()[0]],ytr[ms]); ae,be=recal_ab("ETlog",Xtr.iloc[ms.nonzero()[0]],ytr[ms],gtr[ms])
    shorts={"short=Tweedie+recal":pr_(Mt,Xp,at,bt),
            "short=log1p-ExtraTrees+recal":pr_(Me,Xp,ae,be),
            "short=log1p-ExtraTrees naive":pr_(Me,Xp)}
    gate=arzt>=K
    for lab,short in shorts.items():
        hyb=np.where(gate,long,short); rows.append({"K":K,"variant":lab,**met(los,hyb)})
res=pd.DataFrame(rows); res.to_csv(OUT/"mixed_expert_hybrid.csv",sep=";",index=False)
phys=met(los,arzt)
print("=== GEMISCHTER Hybrid (Arzt-Gate), Kurz-Experten-Wahl, prospektiv n=%d ==="%len(los))
cols=["K","variant","MAE","R2","slope","MAE_gt7","MAE_1-2","MAE_2-4","MAE_4-7","MAE_>7"]
print(res[cols].to_string(index=False))
print("\n[Oberarzt allein]  MAE %.3f R2 %.3f | 1-2:%.2f 2-4:%.2f 4-7:%.2f >7:%.2f"%(phys["MAE"],phys["R2"],phys["MAE_1-2"],phys["MAE_2-4"],phys["MAE_4-7"],phys["MAE_>7"]))

# ===== Figur: Subgruppen-MAE der Kurz-Experten-Varianten (K=5) vs Arzt =====
fig,ax=plt.subplots(1,2,figsize=(14,5.4))
for ai,K in enumerate([5,7]):
    sub=res[res.K==K]; binsl=["1-2","2-4","4-7",">7"]; xb=np.arange(4); w=0.2
    series=[("short=Tweedie+recal","#8e44ad"),("short=log1p-ExtraTrees+recal","#1f5f9e"),("short=log1p-ExtraTrees naive","#5b9bd5")]
    for i,(v,c) in enumerate(series):
        r=sub[sub.variant==v].iloc[0]; vals=[r["MAE_"+b] for b in binsl]
        bb=ax[ai].bar(xb+(i-1.5)*w,vals,w,label=v.replace("short=",""),color=c); ax[ai].bar_label(bb,fmt="%.1f",fontsize=6.2,padding=1)
    physvals=[phys["MAE_"+b] for b in binsl]; bb=ax[ai].bar(xb+1.5*w,physvals,w,label="physician",color="#c0392b"); ax[ai].bar_label(bb,fmt="%.1f",fontsize=6.2,padding=1)
    ax[ai].set_xticks(xb); ax[ai].set_xticklabels([b+" d" for b in binsl]); ax[ai].set_ylabel("MAE (days)")
    ax[ai].set_title(f"Physician-gated hybrid, K={K} d — subgroup MAE",weight="bold",fontsize=11); ax[ai].legend(fontsize=8)
fig.suptitle("Mixed-expert hybrid: short expert log1p-ExtraTrees vs Tweedie (long=Tweedie, physician gate), prospective n=286",weight="bold",fontsize=12)
fig.tight_layout(); fig.savefig(str(OUT/"fig_mixed_expert_hybrid.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print(f"\nGespeichert: fig_mixed_expert_hybrid.png + mixed_expert_hybrid.csv")
