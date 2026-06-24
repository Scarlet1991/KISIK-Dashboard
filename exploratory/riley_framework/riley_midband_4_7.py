# -*- coding: utf-8 -*-
"""EXPLORATIV: Kann die 4-7-Tage-Bande verbessert werden?
Aktuelle Soft-Arzt-Weiche (2 Experten) verliert bei 4-7 d gegen den Arzt (2.74 vs 1.95),
weil dort w klein ist -> fast nur Kurz-Experte, der 4-7 nach unten komprimiert.
Idee: 3-Komponenten-Blend mit MITTEL-Komponente, die in der Mittelbande zieht:
  pred = p_short*SHORT + p_mid*MID + p_long*LONG
  p_long  = sigma((arzt-7)/s);  p_short = sigma((c_lo-arzt)/s);  p_mid = max(0,1-p_long-p_short); renorm.
MID-Varianten: (V1) MID = Arzt-Zahl selbst, (V2) MID = Mittel-Experte (log1p-ET+recal, retro 3<LoS<=7).
(c_lo, s) ehrlich per nested 5-fold CV auf der prospektiven Kohorte (Arzt nur prospektiv).
Vergleich: 2-Komp-Soft-Gate (aktuell) vs 3-Komp (V1/V2) vs Arzt. Leckfrei, no_isopen n=286.
Ausgabe: Eigene Auswertung/exploratory_riley/
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
from collections import Counter
import duckdb, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; OUT=AN/"exploratory_riley"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"
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
Xtr,ytr,gtr=X.iloc[tr],y[tr],groups[tr]
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

ms=(ytr>1)&(ytr<=7); ml=ytr>7; mm_=(ytr>3)&(ytr<=7)
Ms,as_,bs=fit_recal(Xtr.iloc[ms.nonzero()[0]],ytr[ms],gtr[ms]); SHORT=ap(Ms,as_,bs,Xp)
Ml,al,bl=fit_recal(Xtr.iloc[ml.nonzero()[0]],ytr[ml],gtr[ml]); LONG=ap(Ml,al,bl,Xp)
Mm,am,bm=fit_recal(Xtr.iloc[mm_.nonzero()[0]],ytr[mm_],gtr[mm_]); MIDEXP=ap(Mm,am,bm,Xp)  # Mittel-Experte (3<LoS<=7)

def met(p):
    p=np.clip(p,0,None); m=los>7; sb,_=np.polyfit(p,los,1); sg={}
    for l,lo_,hi_ in [("1-2",1,2),("2-4",2,4),("4-7",4,7),(">7",7,999)]:
        mmk=(los>lo_)&(los<=hi_) if hi_<999 else los>7; sg["MAE_"+l]=round(float(np.abs(los[mmk]-p[mmk]).mean()),2)
    return dict(MAE=round(float(mean_absolute_error(los,p)),3),R2=round(float(r2_score(los,p)),3),
                slope=round(float(sb),3),MAE_gt7=round(float(np.abs(los[m]-p[m]).mean()),2),**sg)

def soft2(idx,c=7.0,s=1.0): w=1/(1+np.exp(-(arzt[idx]-c)/s)); return (1-w)*SHORT[idx]+w*LONG[idx]
def soft3(idx,MID,c_lo,s,c_hi=7.0):
    a=arzt[idx]; pl=1/(1+np.exp(-(a-c_hi)/s)); psh=1/(1+np.exp(-(c_lo-a)/s)); pm=np.clip(1-pl-psh,0,None)
    tot=pl+psh+pm; return (psh*SHORT[idx]+pm*MID[idx]+pl*LONG[idx])/tot

CLO=[3,4,5]; SS=[1.0,1.5,2.0]
def nested_cv(MID,objective="MAE"):
    kf=KFold(5,shuffle=True,random_state=RS); oof=np.full(len(los),np.nan); picks=[]
    def score(idx,c_lo,s):
        p=np.clip(soft3(idx,MID,c_lo,s),0,None); o=los[idx]
        if objective=="MAE": return mean_absolute_error(o,p)
        mk=(o>4)&(o<=7); return float(np.abs(o[mk]-p[mk]).mean()) if mk.any() else mean_absolute_error(o,p)
    for trn,tst in kf.split(np.arange(len(los))):
        best=None
        for c_lo in CLO:
            for s in SS:
                v=score(trn,c_lo,s)
                if best is None or v<best[0]: best=(v,c_lo,s)
        _,c_lo,s=best; picks.append((c_lo,s)); oof[tst]=np.clip(soft3(tst,MID,c_lo,s),0,None)
    return oof,Counter(picks).most_common(1)[0]

# 2-Komp Baseline (aktuell)
base2=met(soft2(np.arange(len(los))))
# 3-Komp, MID=Arzt, getunt auf Gesamt-MAE
oofA,pickA=nested_cv(arzt,"MAE"); v1=met(oofA)
# 3-Komp, MID=Mittel-Experte, getunt auf Gesamt-MAE
oofE,pickE=nested_cv(MIDEXP,"MAE"); v2=met(oofE)
# 3-Komp, MID=Arzt, getunt gezielt auf 4-7-MAE (zeigt erreichbares Maximum dort + Kosten)
oofA47,pickA47=nested_cv(arzt,"47"); v1b=met(oofA47)
phys=met(arzt)

rows=[{"approach":"Soft gate 2-comp (current)","tuned_for":"MAE","pick":"c=7,s=1",**base2},
      {"approach":"Soft 3-comp, MID=physician","tuned_for":"MAE","pick":f"c_lo={pickA[0][0]},s={pickA[0][1]}",**v1},
      {"approach":"Soft 3-comp, MID=mid-expert(3-7)","tuned_for":"MAE","pick":f"c_lo={pickE[0][0]},s={pickE[0][1]}",**v2},
      {"approach":"Soft 3-comp, MID=physician","tuned_for":"4-7 MAE","pick":f"c_lo={pickA47[0][0]},s={pickA47[0][1]}",**v1b},
      {"approach":"Senior physician","tuned_for":"-","pick":"-",**phys}]
res=pd.DataFrame(rows); res.to_csv(OUT/"midband_4_7.csv",sep=";",index=False)
cols=["approach","tuned_for","pick","MAE","R2","MAE_1-2","MAE_2-4","MAE_4-7","MAE_>7"]
print("=== 4-7-Tage-Bande verbessern? 3-Komponenten-Blend (nested-CV), prospektiv n=%d ==="%len(los))
print(res[cols].to_string(index=False))
print("\nGewaehlte (c_lo,s): MID=Arzt/MAE %s | MID=Experte/MAE %s | MID=Arzt/4-7 %s"%(pickA,pickE,pickA47))

# ===== Figur =====
fig,ax=plt.subplots(1,2,figsize=(14,5.6))
binsl=["1-2","2-4","4-7",">7"]; xb=np.arange(4); w=0.2
sel=[("2-comp (current)",base2,"#7f8c8d"),("3-comp MID=physician",v1,"#1b7f3b"),("3-comp MID=mid-expert",v2,"#2e86c1"),("physician",phys,"#c0392b")]
for i,(lab,mm,c) in enumerate(sel):
    vals=[mm["MAE_"+b] for b in binsl]; bb=ax[0].bar(xb+(i-1.5)*w,vals,w,label=lab,color=c); ax[0].bar_label(bb,fmt="%.1f",fontsize=6.2,padding=1)
ax[0].axvspan(2-0.5,2+0.5,color="#f1c40f",alpha=0.12)  # highlight 4-7 column
ax[0].set_xticks(xb); ax[0].set_xticklabels([b+" d" for b in binsl]); ax[0].set_ylabel("MAE (days)")
ax[0].set_title("(A) Subgroup MAE — does the mid-band (4-7 d) improve?",weight="bold",fontsize=11); ax[0].legend(fontsize=8)
# (B) overall MAE vs 4-7 MAE scatter (trade-off)
pts=[("2-comp",base2,"#7f8c8d"),("3c MID=phys (MAE)",v1,"#1b7f3b"),("3c MID=expert",v2,"#2e86c1"),("3c MID=phys (4-7)",v1b,"#27ae60"),("physician",phys,"#c0392b")]
for lab,mm,c in pts:
    ax[1].scatter(mm["MAE"],mm["MAE_4-7"],s=90,color=c,zorder=3); ax[1].annotate(lab,(mm["MAE"],mm["MAE_4-7"]),fontsize=8,xytext=(5,4),textcoords="offset points")
ax[1].set_xlabel("overall MAE (days)  →  worse"); ax[1].set_ylabel("4-7 d MAE (days)  →  worse")
ax[1].set_title("(B) Trade-off: mid-band accuracy vs overall",weight="bold",fontsize=11); ax[1].grid(alpha=.25)
fig.suptitle("Improving the 4-7 day band: 3-component physician-blended hybrid (leak-free, prospective n=286)",weight="bold",fontsize=12)
fig.tight_layout(); fig.savefig(str(OUT/"fig_midband_4_7.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("\nGespeichert: fig_midband_4_7.png + midband_4_7.csv")
