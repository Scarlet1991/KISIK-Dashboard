# -*- coding: utf-8 -*-
"""Bestes LEAK-FREIES retrospektives Modell (unabhaengig von prospektiver Verfuegbarkeit).
Alle legitim-24h-Features: lab24/vital24/proc24/zugang24/diag + Demografie + ECHTE 24h-Scores.
Ausgeschlossen: NUR echte Leaks (OPS 8-98f/8-98-Familie = Intensiv-Komplexbehandlung ~ LoS-Proxy),
Ganzstay-Aggregate (lab_/vital_/...), und die leaky canonical score_-Spalten (durch clean-24h ersetzt).
retro-CV GroupKFold C-index/Spearman/MAE. Referenz: Arzt prospektiv 0.766, deployment-aware 81 = 0.668.
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
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error
BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; K=BASE/"kisik2"; OUT=AN/"exploratory_riley"
RETRO=K/"kisik2_icu_ml_dataset_24h.parquet"; RS=42
asql="('AIN','IZ32'),('AIN','IZ21'),('AIN','IZ31')"
con=duckdb.connect()
allc=list(con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') LIMIT 0").df().columns)
def is_leak(c): return ("8_98" in c)   # Intensiv-Komplexbehandlung OPS = LoS-Proxy
rich=[c for c in allc if c.startswith(("lab24_","vital24_","proc24_","zugang24_","diag_")) and not is_leak(c) and not c.startswith("score_")]
rich=[c for c in rich if not c.startswith(("lab_","vital_","proc_","zugang_"))]  # sicherheitshalber keine Ganzstay
extra=[c for c in ["alter"] if c in allc]
STATS=["first","max","mean","min"]; TYPES=["saps_ii_14","tiss_28_10","sofa"]; SCCOLS=[f"score_{t}_{s}" for t in TYPES for s in STATS]
meta=["icu_duration_h","oebenekurz","wardshort","pid","stay_id","fallid","planbegin"]
sel=list(dict.fromkeys(meta+rich+extra))
print("Rich-Features (leak-frei, ohne Scores):",len(rich),"| + Demografie:",len(extra))
df=con.execute(f"SELECT {', '.join(chr(34)+c+chr(34) for c in sel)} FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
con.register("stays",df[["stay_id","fallid","planbegin"]])
# echte 24h-Scores aus score.csv
def clean(k): return k.strip().lower().replace(" ","_").replace("(","").replace(")","")
rsc=con.execute(f"""WITH j AS (SELECT s.stay_id, trim(x.kurzbez) kb, try_cast(x.scoreergebnis AS DOUBLE) v, try_cast(x.von AS TIMESTAMP) von,
  date_diff('minute', s.planbegin, try_cast(x.von AS TIMESTAMP))/60.0 hrs
  FROM read_csv('{(K/'score.csv').as_posix()}', delim=';', header=true, all_varchar=true) x JOIN stays s ON x.fallid=s.fallid)
  SELECT stay_id, kb, arg_min(v,von) AS "first", max(v) AS "max", avg(v) AS "mean", min(v) AS "min"
  FROM j WHERE v IS NOT NULL AND hrs>=0 AND hrs<24 GROUP BY stay_id, kb""").df()
rsc["typ"]=rsc["kb"].map(clean); rw=rsc.pivot_table(index="stay_id",columns="typ",values=STATS); rw.columns=[f"score_{t}_{st}" for st,t in rw.columns]
df=df.merge(rw.reset_index(),on="stay_id",how="left")
scin=[c for c in SCCOLS if c in df.columns]
y=(df["icu_duration_h"]/24.0).values; g=df["pid"].fillna("unknown").astype(str).values
def cindex(pred,yy):
    yy=np.asarray(yy,float); pred=np.asarray(pred,float)
    comp=yy[:,None]<yy[None,:]; pi=pred[:,None]; pj=pred[None,:]
    conc=(pi<pj)&comp; tie=(pi==pj)&comp; d=comp.sum()
    return float((conc.sum()+0.5*tie.sum())/d) if d>0 else float("nan")
def ev(cols,label,losmin=1.0):
    m=df["icu_duration_h"]/24.0>losmin
    X=pd.DataFrame(index=df.index[m])
    for c in cols: X[c]=(df.loc[m,c].astype(str) if c=="oebenekurz" else pd.to_numeric(df.loc[m,c],errors="coerce"))
    keep=[c for c in cols if c=="oebenekurz" or X[c].notna().any()]; X=X[keep]
    cat=[c for c in keep if c=="oebenekurz"]; num=[c for c in keep if c!="oebenekurz"]
    parts=[("num",SimpleImputer(strategy="median"),num)]
    if cat: parts.append(("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat))
    pipe=TransformedTargetRegressor(Pipeline([("pre",ColumnTransformer(parts)),("mdl",ExtraTreesRegressor(n_estimators=250,min_samples_leaf=2,max_features="sqrt",random_state=RS,n_jobs=4))]),func=np.log1p,inverse_func=np.expm1)
    yy=y[m]; gg=g[m]
    oof=np.clip(cross_val_predict(pipe,X,yy,groups=gg,cv=GroupKFold(4),n_jobs=1),0,None)
    return dict(Variante=label,n=int(m.sum()),n_feat=len(keep),C_index=round(cindex(oof,yy),3),Spearman=round(float(spearmanr(oof,yy).correlation),3),MAE=round(float(mean_absolute_error(yy,oof)),2))
rows=[
 ev(rich+extra+["oebenekurz"],"Rich leak-frei, OHNE Scores | LoS>1"),
 ev(rich+extra+scin+["oebenekurz"],"Rich leak-frei, MIT 24h-Scores | LoS>1"),
 ev(rich+extra+scin+["oebenekurz"],"Rich leak-frei, MIT 24h-Scores | LoS>2 (sauber)",losmin=2.0),
]
R=pd.DataFrame(rows); pd.set_option("display.width",180)
print("\n=== Bestes leak-freies retrospektives Modell (retro-CV) ==="); print(R.to_string(index=False))
print("\nReferenz: Oberarzt prospektiv C-index 0.766 | deployment-aware 81 retro 0.668 | clean-24h(labs+vital+proc+score) 0.711")
R.to_csv(OUT/"clean_ceiling_retro.csv",sep=";",index=False); print("Gespeichert: clean_ceiling_retro.csv")
