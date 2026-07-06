# -*- coding: utf-8 -*-
"""24h-Scores: (1) Hexbin prospektiv (Oberarzt / Score-ML / Hybrid).
(2) Hybrid mit Scores: wird er GESAMT signifikant besser als der Oberarzt allein?
   pred = (1-p)*arzt + p*L,  p = sigmoid((arzt-c)/w). L = score-augmentiertes ML (general und long).
   Gepaarter Bootstrap dMAE-CI (B=5000) + Wilcoxon, gesamt + >7d + je Bin.
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from matplotlib.colors import LogNorm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, roc_auc_score
BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; K=BASE/"kisik2"; OUT=AN/"exploratory_riley"
RETRO=K/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
SCALL=Path(r"D:\Ausgangsdaten\Live-Daten\OLD\Entlassdaten\score_all.csv"); RS=42; B=5000; rng=np.random.default_rng(RS)
asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; bpet=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["ExtraTrees"]
con=duckdb.connect(); feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
allcols=list(con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') LIMIT 0").df().columns)
present=[f for f in feat if f in allcols and not f.startswith(("lab_","vital_","proc_","zugang_")) and not f.startswith("proc24_8_98f")]
base=[f for f in present if not f.startswith("score_")]
STATS=["first","max","mean","min"]; TYPES=["saps_ii_14","tiss_28_10","sofa"]; SCCOLS=[f"score_{t}_{s}" for t in TYPES for s in STATS]
meta=["icu_duration_h","oebenekurz","wardshort","pid","stay_id","fallid","planbegin"]
df=con.execute(f"SELECT {', '.join(chr(34)+c+chr(34) for c in dict.fromkeys(meta+base))} FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
con.register("stays",df[["stay_id","fallid","planbegin"]])
def clean(k): return k.strip().lower().replace(" ","_").replace("(","").replace(")","")
rsc=con.execute(f"""WITH j AS (SELECT s.stay_id, trim(x.kurzbez) kb, try_cast(x.scoreergebnis AS DOUBLE) v, try_cast(x.von AS TIMESTAMP) von,
  date_diff('minute', s.planbegin, try_cast(x.von AS TIMESTAMP))/60.0 hrs
  FROM read_csv('{(K/'score.csv').as_posix()}', delim=';', header=true, all_varchar=true) x JOIN stays s ON x.fallid=s.fallid)
  SELECT stay_id, kb, arg_min(v,von) AS "first", max(v) AS "max", avg(v) AS "mean", min(v) AS "min"
  FROM j WHERE v IS NOT NULL AND hrs>=0 AND hrs<24 GROUP BY stay_id, kb""").df()
rsc["typ"]=rsc["kb"].map(clean); rw=rsc.pivot_table(index="stay_id",columns="typ",values=STATS); rw.columns=[f"score_{t}_{st}" for st,t in rw.columns]
df=df.merge(rw.reset_index(),on="stay_id",how="left")
for c in SCCOLS:
    if c not in df.columns: df[c]=np.nan
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float); N=len(los)
fallnr=PR["__stay_id__"].astype(str).str.split("_stay").str[0]
raw=con.execute(f"""SELECT FALLNR fallnr, KURZBEZ kb, try_cast(SCOREERGEBNIS AS DOUBLE) v, strptime(VON,'%d.%m.%Y %H:%M:%S') von
FROM read_csv('{SCALL.as_posix()}', delim=';', header=true, all_varchar=true)
WHERE try_cast(SCOREERGEBNIS AS DOUBLE) IS NOT NULL AND strptime(VON,'%d.%m.%Y %H:%M:%S') IS NOT NULL""").df()
raw["typ"]=raw["kb"].map(clean); raw=raw.sort_values("von")
fv=raw.groupby(["fallnr","typ"])["von"].transform("min"); w=raw[raw["von"]<fv+pd.Timedelta(hours=24)]; gb=w.groupby(["fallnr","typ"])["v"]
pa=pd.DataFrame({"first":gb.first(),"max":gb.max(),"mean":gb.mean(),"min":gb.min()}).reset_index()
pw=pa.pivot_table(index="fallnr",columns="typ",values=STATS); pw.columns=[f"score_{t}_{st}" for st,t in pw.columns]
prosc=pd.DataFrame({"fallnr":fallnr}).merge(pw.reset_index(),on="fallnr",how="left")
cols=base+SCCOLS
def Xr(frame):
    X=frame.reindex(columns=cols).copy()
    for c in cols: X[c]=(X[c].astype(str) if c=="oebenekurz" else pd.to_numeric(X[c],errors="coerce"))
    return X
def Xp():
    X=pd.DataFrame(index=PR.index)
    for c in cols:
        if c=="oebenekurz": X[c]=PR[c].astype(str) if c in PR.columns else "NA"
        elif c in SCCOLS: X[c]=pd.to_numeric(prosc[c],errors="coerce") if c in prosc.columns else np.nan
        else: X[c]=pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan
    return X[cols]
y=(df["icu_duration_h"]/24.0).values; groups=df["pid"].fillna("unknown").astype(str).values
def pre():
    c=["oebenekurz"] if "oebenekurz" in cols else []; n=[x for x in cols if x!="oebenekurz"]
    parts=[("num",SimpleImputer(strategy="median"),n)]
    if c: parts.append(("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),c))
    return ColumnTransformer(parts)
def et(): return TransformedTargetRegressor(Pipeline([("pre",pre()),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=4))]),func=np.log1p,inverse_func=np.expm1)
Xpros=Xp()
# --- L_gen: general ML (LoS>1) ---
trg,_=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(df,y,groups))
Xtr=Xr(df).iloc[trg]; m=et(); m.fit(Xtr,y[trg])
oof=np.clip(cross_val_predict(et(),Xtr,y[trg],groups=groups[trg],cv=GroupKFold(3),n_jobs=1),0,None)
b0,a0=np.polyfit(oof,y[trg],1); Lgen=np.clip(a0+b0*np.clip(m.predict(Xpros),0,None),0,None)
# --- L_long: long model (retro LoS>7) ---
lm=y>7; Xl=Xr(df)[lm]; yl=y[lm]; gl=groups[lm]
ml=et(); ml.fit(Xl,yl)
oofl=np.clip(cross_val_predict(et(),Xl,yl,groups=gl,cv=GroupKFold(3),n_jobs=1),0,None)
b1,a1=np.polyfit(oofl,yl,1); Llong=np.clip(a1+b1*np.clip(ml.predict(Xpros),0,None),0,None)
# --- Hybrids ---
def gate(a,c,wd): return 1/(1+np.exp(-(a-c)/wd))
def hybrid(L,c=7.0,wd=1.0): p=gate(arzt,c,wd); return (1-p)*arzt+p*L
H_gen=hybrid(Lgen); H_long=hybrid(Llong)
# --- Signifikanz-Helfer ---
def sig(ref,alt,mask=None):  # dMAE=MAE_ref-MAE_alt (>0 => alt besser)
    m=np.ones(N,bool) if mask is None else mask; n=int(m.sum())
    ea=np.abs(los[m]-ref[m]); eb=np.abs(los[m]-alt[m]); idx=rng.integers(0,n,size=(B,n))
    dd=ea[idx].mean(1)-eb[idx].mean(1); lo,hi=np.percentile(dd,[2.5,97.5])
    try: pw=float(wilcoxon(ea,eb).pvalue)
    except: pw=1.0
    return round(float(ea.mean()-eb.mean()),3),round(float(lo),3),round(float(hi),3),round(pw,4),round(float(eb.mean()),2)
def cindex(pred):
    comp=los[:,None]<los[None,:]; pi=pred[:,None]; pj=pred[None,:]; conc=(pi<pj)&comp; tie=(pi==pj)&comp; d=comp.sum()
    return float((conc.sum()+0.5*tie.sum())/d)
print("=== Prospektiv n=%d — MAE / C-index ==="%N)
for nm,pr in [("Oberarzt",arzt),("Score-ML (general)",Lgen),("Score-Long",Llong),("Hybrid-gen (c7,w1)",H_gen),("Hybrid-long (c7,w1)",H_long)]:
    print(f"  {nm:22s} MAE {mean_absolute_error(los,pr):.2f} | C-index {cindex(pr):.3f}")
print("\n=== GESAMT: Hybrid vs Oberarzt (dMAE=MAE_Arzt-MAE_Hybrid; >0 => Hybrid besser) ===")
rows=[]
for nm,H in [("Hybrid-gen",H_gen),("Hybrid-long",H_long),("Score-ML allein",Lgen)]:
    d,lo,hi,pw,mh=sig(arzt,H); verdict="Hybrid sig. besser" if (lo>0 and pw<0.05) else ("Arzt sig. besser" if hi<0 and pw<0.05 else "n.s.")
    rows.append(dict(Modell=nm,MAE=mh,MAE_Arzt=2.94,dMAE=d,CI=f"[{lo:+.2f},{hi:+.2f}]",wilcox_p=pw,Urteil=verdict));
R=pd.DataFrame(rows); pd.set_option("display.width",170); print(R.to_string(index=False))
R.to_csv(OUT/"hybrid_scores_signif.csv",sep=";",index=False)
print("\n=== >7 d (Langlieger): Hybrid vs Oberarzt ===")
lmk=los>7
for nm,H in [("Hybrid-gen",H_gen),("Hybrid-long",H_long)]:
    d,lo,hi,pw,mh=sig(arzt,H,lmk); print(f"  {nm:12s} n={int(lmk.sum())} MAE {mh:.2f} vs Arzt {mean_absolute_error(los[lmk],arzt[lmk]):.2f} | dMAE {d:+.2f} CI[{lo:+.2f},{hi:+.2f}] p={pw}")
print("\n=== je Bin: Hybrid-long vs Oberarzt ===")
for lab,lo_,hi_ in [("1-2",1,2),("2-4",2,4),("4-7",4,7),(">7",7,1e9)]:
    mk=(los>lo_)&(los<=hi_) if hi_<1e8 else los>7; d,lo,hi,pw,mh=sig(arzt,H_long,mk)
    print(f"  {lab:4s} n={int(mk.sum()):3d}  MAE Hyb {mh:.2f} vs Arzt {mean_absolute_error(los[mk],arzt[mk]):.2f}  dMAE {d:+.2f} CI[{lo:+.2f},{hi:+.2f}] p={pw}")
print("\n=== Gate-Sweep (Hybrid-long): ist c=7,w=1 die beste Konfiguration? (Sensitivitaet, nicht optimiert) ===")
sweep=[]
for c in (5,6,7,8,9):
    for wd in (0.5,1.0,2.0):
        H=hybrid(Llong,c,wd); dO,_,_,pO,_=sig(arzt,H); dL,_,_,pL,_=sig(arzt,H,los>7)
        sweep.append(dict(c=c,w=wd,MAE=round(float(mean_absolute_error(los,H)),3),Cidx=round(cindex(H),3),
                          dMAE_gesamt=dO,p_gesamt=pO,dMAE_7d=dL,p_7d=pL))
SW=pd.DataFrame(sweep); print(SW.to_string(index=False)); SW.to_csv(OUT/"hybrid_gate_sweep.csv",sep=";",index=False)
# Praediktionen sichern
pd.DataFrame({"los":los,"arzt":arzt,"Lgen":Lgen,"Llong":Llong,"H_gen":H_gen,"H_long":H_long}).to_parquet(OUT/"hybrid_scores_preds.parquet")
# --- Hexbin ---
ACC="#103A5C"; ORA="#C44E1E"; GRY="#888"
fig,axes=plt.subplots(1,3,figsize=(14.4,4.8),sharex=True,sharey=True)
for ax,(pr,nm,col) in zip(axes,[(arzt,"Oberarzt",ORA),(Lgen,"Standalone-ML + 24h-Scores",ACC),(H_long,"Hybrid (Arzt + Score-Long-Modell)","#2E7D5B")]):
    hb=ax.hexbin(los,pr,gridsize=24,extent=(0,25,0,25),cmap="viridis",mincnt=1,norm=LogNorm())
    ax.plot([0,25],[0,25],"--",color=GRY,lw=1.3); ax.set_xlim(0,25); ax.set_ylim(0,25)
    ax.set_xlabel("Beobachtete LoS (d)"); ax.set_title(nm,color=col,fontweight="bold")
    ax.text(0.04,0.96,"MAE %.2f\nC-index %.3f"%(mean_absolute_error(los,pr),cindex(pr)),transform=ax.transAxes,
            va="top",fontsize=9.5,bbox=dict(boxstyle="round,pad=0.35",fc="white",ec="0.7"))
axes[0].set_ylabel("Vorhergesagte LoS (d)")
fig.suptitle("Prospektiv (n=286): Oberarzt vs. Standalone-ML mit 24h-Scores vs. Hybrid",fontsize=13,color=ACC,weight="bold",y=1.02)
fig.tight_layout(); fig.savefig(OUT/"fig_hexbin_scores_hybrid.png",dpi=300,bbox_inches="tight")
print("\nGespeichert: fig_hexbin_scores_hybrid.png, hybrid_scores_signif.csv")
