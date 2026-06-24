# -*- coding: utf-8 -*-
"""EXPLORATIV: Bringt ein Quantil-Ansatz etwas fuer die LANGLIEGER?
Zwei Fragen, leckfrei, prospektiv no_isopen (n=286):
 (1) Standalone-Quantilregression (P50/P70/P80/P90, volle Kohorte) — Trade-off Kurz vs Lang.
 (2) Quantil als LANG-Experte innerhalb der Arzt-Weiche (hart arzt>=7 / weich c=7,s=1.0),
     Kurz-Experte bleibt log1p-ExtraTrees+recal. Vergleich gegen aktuellen Lang-Experten
     (log1p-ExtraTrees+recal) und gegen den Oberarzt.
Quantil = GradientBoostingRegressor(loss="quantile") (sklearn-nativ, stabil).
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
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
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

# ---- Experten ----
def et(): return TransformedTargetRegressor(Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=2))]),func=np.log1p,inverse_func=np.expm1)
def fit_recal(Xt,yt,gt):
    m=et(); m.fit(Xt,yt); oof=np.clip(cross_val_predict(et(),Xt,yt,groups=gt,cv=GroupKFold(3),n_jobs=1),0,None); b,a=np.polyfit(oof,yt,1); return m,a,b
def ap(m,a,b,Xs): return np.clip(a+b*np.clip(m.predict(Xs),0,None),0,None)
def qgbr(q): return Pipeline([("pre",pre()),("mdl",GradientBoostingRegressor(loss="quantile",alpha=q,n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=RS))])
def qfit(q,Xt,yt): m=qgbr(q); m.fit(Xt,yt); return lambda Xs: np.clip(m.predict(Xs),0,None)

ms=(ytr>1)&(ytr<=7); ml=ytr>7
Ms,as_,bs=fit_recal(Xtr.iloc[ms.nonzero()[0]],ytr[ms],gtr[ms]); SHORT=ap(Ms,as_,bs,Xp)
Ml,al,bl=fit_recal(Xtr.iloc[ml.nonzero()[0]],ytr[ml],gtr[ml]); LONG_et=ap(Ml,al,bl,Xp)   # aktueller Lang-Experte

QS=[0.5,0.7,0.8,0.9]
full_q={q:qfit(q,Xtr,ytr)(Xp) for q in QS}                          # standalone, volle Kohorte
long_q={q:qfit(q,Xtr.iloc[ml.nonzero()[0]],ytr[ml])(Xp) for q in QS} # Quantil als Lang-Experte

def cov_gt7(p): m=los>7; return round(float((np.clip(p,0,None)[m]>=los[m]).mean()),3)
def met(p):
    p=np.clip(p,0,None); m=los>7; sb,_=np.polyfit(p,los,1); sg={}
    for l,lo_,hi_ in [("1-2",1,2),("2-4",2,4),("4-7",4,7),(">7",7,999)]:
        mm=(los>lo_)&(los<=hi_) if hi_<999 else los>7; sg["MAE_"+l]=round(float(np.abs(los[mm]-p[mm]).mean()),2)
    return dict(MAE=round(float(mean_absolute_error(los,p)),3),R2=round(float(r2_score(los,p)),3),slope=round(float(sb),3),
                bias_gt7=round(float((p[m]-los[m]).mean()),2),MAE_gt7=round(float(np.abs(los[m]-p[m]).mean()),2),cov_gt7=cov_gt7(p),**sg)

# (1) Standalone-Quantil-Sweep
rows=[{"approach":"Single log1p-ExtraTrees (long expert form)","q":"-",**met(LONG_et)}]  # ref (mean-target)
for q in QS: rows.append({"approach":"Standalone quantile (full cohort)","q":q,**met(full_q[q])})
rows.append({"approach":"Senior physician","q":"-",**met(arzt)})
t1=pd.DataFrame(rows); t1.to_csv(OUT/"quantile_longstay_standalone.csv",sep=";",index=False)

# (2) Quantil als Lang-Experte in der Arzt-Weiche
def hard(long): return np.where(arzt>=7,long,SHORT)
def soft(long,c=7.0,s=1.0): w=1/(1+np.exp(-(arzt-c)/s)); return (1-w)*SHORT+w*long
rows2=[{"gate":"hard (arzt>=7)","long_expert":"log1p-ExtraTrees+recal","q":"-",**met(hard(LONG_et))},
       {"gate":"soft (c=7,s=1)","long_expert":"log1p-ExtraTrees+recal","q":"-",**met(soft(LONG_et))}]
for q in QS:
    rows2.append({"gate":"hard (arzt>=7)","long_expert":"quantile-GBR","q":q,**met(hard(long_q[q]))})
    rows2.append({"gate":"soft (c=7,s=1)","long_expert":"quantile-GBR","q":q,**met(soft(long_q[q]))})
rows2.append({"gate":"-","long_expert":"Senior physician","q":"-",**met(arzt)})
t2=pd.DataFrame(rows2); t2.to_csv(OUT/"quantile_longstay_hybrid.csv",sep=";",index=False)

cols1=["approach","q","MAE","R2","MAE_gt7","bias_gt7","cov_gt7","MAE_2-4","MAE_4-7"]
cols2=["gate","long_expert","q","MAE","R2","MAE_gt7","bias_gt7","cov_gt7","MAE_4-7"]
print("=== (1) STANDALONE Quantil — volle Kohorte, prospektiv n=%d ==="%len(los))
print(t1[cols1].to_string(index=False))
print("\n  [Coverage>7 = Anteil Langlieger, deren Prognose >= beobachtete LoS (Kapazitaetsplanung)]")
print("\n=== (2) Quantil als LANG-Experte in der Arzt-Weiche ===")
print(t2[cols2].to_string(index=False))

# ===== Figur =====
fig,ax=plt.subplots(1,2,figsize=(14,5.6))
# (A) Standalone: MAE>7 + overall MAE + coverage>7 ueber Quantile
qx=[0.5,0.7,0.8,0.9]
ax[0].plot(qx,[t1[t1.q==q].MAE_gt7.iloc[0] for q in qx],"o-",color="#c0392b",lw=2,label="MAE long-stay (>7 d)")
ax[0].plot(qx,[t1[t1.q==q].MAE.iloc[0] for q in qx],"s-",color="#185fa5",lw=2,label="MAE overall")
ax[0].axhline(met(LONG_et)["MAE_gt7"],ls="--",color="#c0392b",alpha=.5,label="mean-target ET, >7 d")
ax[0].axhline(met(arzt)["MAE_gt7"],ls=":",color="#222",alpha=.7,label="physician, >7 d (7.74)")
ax[0].set_xlabel("quantile level α"); ax[0].set_ylabel("MAE (days)"); ax[0].set_xticks(qx)
ax[0].set_title("(A) Standalone quantile — long-stay vs overall trade-off",weight="bold",fontsize=11); ax[0].legend(fontsize=8.2)
axb=ax[0].twinx(); axb.plot(qx,[t1[t1.q==q].cov_gt7.iloc[0] for q in qx],"^:",color="#1b7f3b",lw=1.6)
axb.set_ylabel("coverage >7 d (pred ≥ obs)",color="#1b7f3b"); axb.set_ylim(0,1); axb.tick_params(axis="y",colors="#1b7f3b")
# (B) Subgruppen-MAE: soft gate mit long=ET vs long=P80 vs physician
binsl=["1-2","2-4","4-7",">7"]; xb=np.arange(4); w=0.27
sel=[("soft, long=ET+recal",soft(LONG_et),"#7f8c8d"),("soft, long=quantile P80",soft(long_q[0.8]),"#1b7f3b"),("physician",arzt,"#c0392b")]
for i,(lab,p,c) in enumerate(sel):
    mm=met(p); vals=[mm["MAE_"+b] for b in binsl]; bb=ax[1].bar(xb+(i-1)*w,vals,w,label=lab,color=c); ax[1].bar_label(bb,fmt="%.1f",fontsize=6.5,padding=1)
ax[1].set_xticks(xb); ax[1].set_xticklabels([b+" d" for b in binsl]); ax[1].set_ylabel("MAE (days)")
ax[1].set_title("(B) Soft physician gate — quantile vs mean long expert",weight="bold",fontsize=11); ax[1].legend(fontsize=8.5)
fig.suptitle("Does a quantile approach help long-stayers? (leak-free, prospective n=286)",weight="bold",fontsize=12.5)
fig.tight_layout(); fig.savefig(str(OUT/"fig_quantile_longstay.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("\nGespeichert: fig_quantile_longstay.png + quantile_longstay_standalone.csv + quantile_longstay_hybrid.csv")
