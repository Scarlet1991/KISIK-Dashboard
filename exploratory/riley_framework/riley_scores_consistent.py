# -*- coding: utf-8 -*-
"""Train/serve-KONSISTENTER Score-Test: beide Seiten ECHTE 24h-Scores (kein Leak-Fenster).
Retro-Scores aus kisik2/score.csv im Fenster [planbegin, planbegin+24h). Prospektiv aus score_all.csv.
Base = deployment-aware (ohne Scores). Vergleich Ranking/MAE: base vs base+Scores vs Oberarzt.
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, roc_auc_score
BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; K=BASE/"kisik2"; OUT=AN/"exploratory_riley"
RETRO=K/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
SCALL=Path(r"D:\Ausgangsdaten\Live-Daten\OLD\Entlassdaten\score_all.csv"); RS=42
asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; bpet=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["ExtraTrees"]
con=duckdb.connect(); feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
allcols=list(con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') LIMIT 0").df().columns)
present=[f for f in feat if f in allcols and not f.startswith(("lab_","vital_","proc_","zugang_")) and not f.startswith("proc24_8_98f")]
base=[f for f in present if not f.startswith("score_")]
STATS=["first","max","mean","min"]; TYPES=["saps_ii_14","tiss_28_10","sofa"]
SCCOLS=[f"score_{t}_{s}" for t in TYPES for s in STATS]
# --- retro base + keys ---
meta=["icu_duration_h","oebenekurz","wardshort","pid","stay_id","fallid","planbegin"]
selc=list(dict.fromkeys(meta+base)); colstr=", ".join('"'+c+'"' for c in selc)
df=con.execute(f"SELECT {colstr} FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
con.register("stays",df[["stay_id","fallid","planbegin"]])
# --- saubere 24h retro-Scores aus score.csv ---
rsc=con.execute(f"""
WITH j AS (
  SELECT s.stay_id, trim(x.kurzbez) kb, try_cast(x.scoreergebnis AS DOUBLE) v, try_cast(x.von AS TIMESTAMP) von,
         date_diff('minute', s.planbegin, try_cast(x.von AS TIMESTAMP))/60.0 hrs
  FROM read_csv('{(K/'score.csv').as_posix()}', delim=';', header=true, all_varchar=true) x JOIN stays s ON x.fallid=s.fallid )
SELECT stay_id, kb, arg_min(v,von) AS "first", max(v) AS "max", avg(v) AS "mean", min(v) AS "min"
FROM j WHERE v IS NOT NULL AND hrs>=0 AND hrs<24 GROUP BY stay_id, kb
""").df()
def clean(k): return k.strip().lower().replace(" ","_").replace("(","").replace(")","")
rsc["typ"]=rsc["kb"].map(clean)
rw=rsc.pivot_table(index="stay_id",columns="typ",values=STATS)
rw.columns=[f"score_{t}_{st}" for st,t in rw.columns]; rw=rw.reset_index()
df=df.merge(rw,on="stay_id",how="left")
for c in SCCOLS:
    if c not in df.columns: df[c]=np.nan
print("retro Score-Befuellung:",{t:f"{100*df[f'score_{t}_max'].notna().mean():.0f}%" for t in TYPES})
# --- prospektiv ---
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float); N=len(los)
fallnr=PR["__stay_id__"].astype(str).str.split("_stay").str[0]
raw=con.execute(f"""SELECT FALLNR fallnr, KURZBEZ kb, try_cast(SCOREERGEBNIS AS DOUBLE) v, strptime(VON,'%d.%m.%Y %H:%M:%S') von
FROM read_csv('{SCALL.as_posix()}', delim=';', header=true, all_varchar=true)
WHERE try_cast(SCOREERGEBNIS AS DOUBLE) IS NOT NULL AND strptime(VON,'%d.%m.%Y %H:%M:%S') IS NOT NULL""").df()
raw["typ"]=raw["kb"].map(clean); raw=raw.sort_values("von")
fv=raw.groupby(["fallnr","typ"])["von"].transform("min"); w=raw[raw["von"]<fv+pd.Timedelta(hours=24)]; gb=w.groupby(["fallnr","typ"])["v"]
pa=pd.DataFrame({"first":gb.first(),"max":gb.max(),"mean":gb.mean(),"min":gb.min()}).reset_index()
pw=pa.pivot_table(index="fallnr",columns="typ",values=STATS); pw.columns=[f"score_{t}_{st}" for st,t in pw.columns]; pw=pw.reset_index()
prosc=pd.DataFrame({"fallnr":fallnr}).merge(pw,on="fallnr",how="left")
# --- Feature-Frames ---
def Xretro(cols):
    X=df.reindex(columns=cols).copy()
    for c in cols: X[c]=(X[c].astype(str) if c=="oebenekurz" else pd.to_numeric(X[c],errors="coerce"))
    return X
def Xpros(cols):
    X=pd.DataFrame(index=PR.index)
    for c in cols:
        if c=="oebenekurz": X[c]=PR[c].astype(str) if c in PR.columns else "NA"
        elif c in SCCOLS: X[c]=pd.to_numeric(prosc[c],errors="coerce") if c in prosc.columns else np.nan
        else: X[c]=pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan
    return X[cols]
y=(df["icu_duration_h"]/24.0).values; groups=df["pid"].fillna("unknown").astype(str).values
tr,_=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(df,y,groups))
def pre(cols):
    c=[x for x in cols if x=="oebenekurz"]; n=[x for x in cols if x!="oebenekurz"]
    parts=[("num",SimpleImputer(strategy="median"),n)]
    if c: parts.append(("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),c))
    return ColumnTransformer(parts)
def et(cols): return TransformedTargetRegressor(Pipeline([("pre",pre(cols)),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=4))]),func=np.log1p,inverse_func=np.expm1)
def cindex(pred,yy):
    yy=np.asarray(yy,float); pred=np.asarray(pred,float)
    comp=yy[:,None]<yy[None,:]; pi=pred[:,None]; pj=pred[None,:]
    conc=(pi<pj)&comp; tie=(pi==pj)&comp; d=comp.sum()
    return float((conc.sum()+0.5*tie.sum())/d) if d>0 else float("nan")
def run(cols,label):
    Xtr=Xretro(cols).iloc[tr]; ytr=y[tr]; gtr=groups[tr]; Xp=Xpros(cols)
    m=et(cols); m.fit(Xtr,ytr)
    oof=np.clip(cross_val_predict(et(cols),Xtr,ytr,groups=gtr,cv=GroupKFold(3),n_jobs=1),0,None)
    b0,a0=np.polyfit(oof,ytr,1); pred=np.clip(a0+b0*np.clip(m.predict(Xp),0,None),0,None)
    return dict(Modell=label,n_feat=len(cols),C_index=round(cindex(pred,los),3),
                Spearman=round(float(spearmanr(pred,los).correlation),3),MAE=round(float(mean_absolute_error(los,pred)),2)),pred
rows=[]; r1,p1=run(base,"ML ohne Scores"); rows.append(r1)
r2,p2=run(base+SCCOLS,"ML + saubere 24h-Scores (konsistent)"); rows.append(r2)
rows.append(dict(Modell="Oberarzt",n_feat=np.nan,C_index=round(cindex(arzt,los),3),Spearman=round(float(spearmanr(arzt,los).correlation),3),MAE=round(float(mean_absolute_error(los,arzt)),2)))
R=pd.DataFrame(rows); pd.set_option("display.width",160)
print("\n=== PROSPEKTIV (n=%d) train/serve-konsistent ==="%N); print(R.to_string(index=False))
sh=los<=3; vl=los>10; mk=sh|vl; yb=(los[mk]>10).astype(int)
print("\nDiskrimination Extreme AUROC: ohne %.3f | mit Scores %.3f | Arzt %.3f"%(roc_auc_score(yb,p1[mk]),roc_auc_score(yb,p2[mk]),roc_auc_score(yb,arzt[mk])))
R.to_csv(OUT/"scores_consistent_effekt.csv",sep=";",index=False); print("\nGespeichert: scores_consistent_effekt.csv")
