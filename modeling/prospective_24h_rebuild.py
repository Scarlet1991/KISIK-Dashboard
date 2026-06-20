# -*- coding: utf-8 -*-
"""
Option A: echte first-24h-Features fuer die PROSPEKTIVE Kohorte aus den OLD-Rohdaten bauen
(namensgleich zum Training), Abdeckung messen (real vs. Median-Ersatz) und Modelle fair neu auswerten.
"""
import sys, io, os, re, glob, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"
PROS =BASE/"kisik2"/"kisik2_prospektiv_ml_dataset.parquet"
SENIOR=AN/"los_senior_estimates_tagesausleitung_stay_level.csv"
FEAT=AN/"los_selected_features_ain_24h_compact.csv"
OLD="D:/Ausgangsdaten/Live-Daten/OLD"
RS=42
con=duckdb.connect()
def rc(p): return f"read_csv_auto('{p}', delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true)"
def rcl(name):  # nur Dateien mit gueltiger FALLNR-Kopfzeile (umgeht Sniffing-Fehler bei leeren/kaputten Dateien)
    fs=[]
    for p in glob.glob(f"{OLD}/*/{name}"):
        try:
            if os.path.getsize(p)<30: continue
            with open(p,"r",encoding="utf-8",errors="ignore") as fh: head=fh.readline()
            if "FALLNR" in head.upper(): fs.append(p.replace("\\","/"))
        except Exception: pass
    lit="["+",".join("'"+p+"'" for p in fs)+"]"
    return f"read_csv_auto({lit}, delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true)"
TS=lambda c: f"COALESCE(TRY_CAST({c} AS TIMESTAMP), TRY_STRPTIME({c}, '%d.%m.%Y %H:%M:%S'))"

def sanitize(v):
    t=re.sub(r"[^a-z0-9]+","_",str(v).strip().lower()); t=re.sub(r"_+","_",t).strip("_")
    return t[:60] or "unknown"

# ---------------- Training (kanonisch, beste Hyperparameter) ----------------
allowed=[("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31"),("AIN","IZ01"),("AUG","IZ01"),("AVT","IZ01"),("GCH","IZ01"),("GYN","IZ01"),("HNO","IZ01"),("HTC","IZ01"),("IZPV","IZ01"),("MKG","IZ01"),("NCH","IZ01"),("NUK","IZ01"),("STR","IZ01"),("UCH","IZ01"),("URO","IZ01")]
asql=", ".join(f"('{w}','{o}')" for w,o in allowed)
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
df["los_days"]=df["icu_duration_h"]/24.0
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns and not f.startswith(("lab_","vital_","proc_","zugang_"))]
cat=[c for c in present if c=="oebenekurz"]; numc=[c for c in present if c not in cat]
def Xframe(frame):
    X=frame.reindex(columns=present).copy()
    for c in numc: X[c]=pd.to_numeric(X[c],errors="coerce")
    for c in cat:  X[c]=X[c].astype(str)
    return X
X=Xframe(df); y=df["los_days"].values; groups=df["pid"].fillna("unknown").astype(str).values
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(X,y,groups))
def pre(scale=False):
    ns=[("imp",SimpleImputer(strategy="median"))]+([("sc",StandardScaler())] if scale else [])
    return ColumnTransformer([("num",Pipeline(ns),numc),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)])
def ttr(reg,scale=False): return TransformedTargetRegressor(Pipeline([("pre",pre(scale)),("mdl",reg)]),func=np.log1p,inverse_func=np.expm1)
models={
 "Ridge":ttr(Ridge(alpha=0.1,random_state=RS),scale=True),
 "RandomForest":ttr(RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=0.5,max_depth=20,random_state=RS,n_jobs=1)),
 "ExtraTrees":ttr(ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=0.5,max_depth=20,random_state=RS,n_jobs=1)),
 "XGBoost":ttr(XGBRegressor(n_estimators=500,max_depth=8,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,min_child_weight=1,reg_lambda=5,random_state=RS,n_jobs=1,tree_method="hist")),
}
print("Training (beste Hyperparameter) ..."); [m.fit(X.iloc[tr],y[tr]) for m in models.values()]

# ---------------- prospektive Kohorte + Senior-Match ----------------
# nur abgeschlossene Aufenthalte (is_open=0) mit tatsaechlicher LoS > 1 Tag, konsistent zur retrospektiven Kohorte
dp=con.execute(f"SELECT * FROM read_parquet('{PROS.as_posix()}') WHERE is_open=0 AND icu_duration_h/24.0>1").df()
dp["los_days"]=dp["icu_duration_h"]/24.0
sen=pd.read_csv(SENIOR,sep=";"); dp["stay_id"]=dp["stay_id"].astype(str); sen["tages_stay_id"]=sen["tages_stay_id"].astype(str)
mg=dp.merge(sen,left_on="stay_id",right_on="tages_stay_id",how="inner")
mg["arzt"]=pd.to_numeric(mg["best_senior_estimate_days"],errors="coerce")
mg=mg.dropna(subset=["los_days","arzt"]).reset_index(drop=True)
mg["planbegin"]=pd.to_datetime(mg["planbegin"],errors="coerce")
print(f"Matched prospektiv: {len(mg)} Stays")

stays=mg[["stay_id","fallnr","planbegin"]].copy(); stays["fallnr"]=stays["fallnr"].astype(str)
con.register("stays_df",stays); con.execute("CREATE OR REPLACE TEMP TABLE s AS SELECT stay_id, CAST(fallnr AS VARCHAR) idc, CAST(planbegin AS TIMESTAMP) pb FROM stays_df")

def long_to_wide(long_df,prefix):
    if long_df.empty: return pd.DataFrame(columns=["stay_id"])
    long_df["fk"]=long_df["feature_name"].map(sanitize)
    parts=[]
    for col,suf in [("mean_value","mean"),("median_value","median"),("first_value","first"),("last_value","last"),("min_value","min"),("max_value","max"),("count_value","count")]:
        if col not in long_df.columns: continue
        p=long_df.pivot_table(index="stay_id",columns="fk",values=col,aggfunc="first")
        if p.empty: continue
        p.columns=[f"{prefix}_{c}_{suf}" for c in p.columns]; parts.append(p)
    return pd.concat(parts,axis=1).reset_index() if parts else pd.DataFrame(columns=["stay_id"])
def presence(long_df,prefix):
    if long_df.empty: return pd.DataFrame(columns=["stay_id"])
    loc=long_df[["stay_id","feature_name"]].dropna().copy(); loc["fk"]=loc["feature_name"].map(sanitize)
    w=(pd.crosstab(loc["stay_id"],loc["fk"])>0).astype("uint8"); w.columns=[f"{prefix}_{c}" for c in w.columns]
    return w.reset_index()

G=lambda f: f"{OLD}/*/{f}"
# Labor
lab=con.execute(f"""WITH l AS (SELECT s.stay_id, COALESCE(NULLIF(TRIM(BESCHREIBUNG),''),NULLIF(TRIM(CODE),''),NULLIF(TRIM(ANALYTX),'')) fn,
  COALESCE(TRY_CAST(REPLACE(ERGEBNISF,',','.') AS DOUBLE),TRY_CAST(REPLACE(ERGEBNIST,',','.') AS DOUBLE)) v, {TS('ERFASSDAT')} ts
  FROM {rcl('lab.csv')} lab JOIN s ON CAST(lab.FALLNR AS VARCHAR)=s.idc
  WHERE {TS('ERFASSDAT')} BETWEEN s.pb AND s.pb+INTERVAL 24 HOURS)
  SELECT stay_id,fn feature_name,AVG(v) mean_value,MEDIAN(v) median_value,ARG_MIN(v,ts) first_value,ARG_MAX(v,ts) last_value,MIN(v) min_value,MAX(v) max_value,COUNT(v) count_value
  FROM l WHERE fn IS NOT NULL AND v IS NOT NULL AND ts IS NOT NULL GROUP BY 1,2""").df()
# Vitals
vit=con.execute(f"""WITH v AS (SELECT s.stay_id, BEFUNDARTKURZBEZ fn, TRY_CAST(REPLACE(WERT,',','.') AS DOUBLE) val, {TS('ZEITPUNKT')} ts
  FROM {rcl('vitalzeichen.csv')} vt JOIN s ON CAST(vt.FALLNR AS VARCHAR)=s.idc
  WHERE {TS('ZEITPUNKT')} BETWEEN s.pb AND s.pb+INTERVAL 24 HOURS)
  SELECT stay_id,fn feature_name,AVG(val) mean_value,MEDIAN(val) median_value,ARG_MIN(val,ts) first_value,ARG_MAX(val,ts) last_value,MIN(val) min_value,MAX(val) max_value,COUNT(val) count_value
  FROM v WHERE fn IS NOT NULL AND val IS NOT NULL AND ts IS NOT NULL GROUP BY 1,2""").df()
# Prozeduren
proc=con.execute(f"""SELECT DISTINCT s.stay_id, OPS feature_name FROM {rcl('prozeduren.csv')} p JOIN s ON CAST(p.FALLNR AS VARCHAR)=s.idc
  WHERE OPS IS NOT NULL AND {TS('DURCHF_DATUM')} BETWEEN s.pb AND s.pb+INTERVAL 24 HOURS""").df()
# Zugaenge (Datum aus ANLEGEDATUM + Zeit aus ANLEGEZEIT)
acc=con.execute(f"""SELECT s.stay_id, COALESCE(NULLIF(TRIM(SUBKLASSIFIKATION1),''),NULLIF(TRIM(TEXT),'')) feature_name
  FROM {rcl('zugaenge.csv')} a JOIN s ON CAST(a.FALLNR AS VARCHAR)=s.idc
  WHERE COALESCE(TRY_CAST(strftime({TS('ANLEGEDATUM')},'%Y-%m-%d')||' '||strftime({TS('ANLEGEZEIT')},'%H:%M:%S') AS TIMESTAMP), {TS('ANLEGEDATUM')})
        BETWEEN s.pb - INTERVAL 1 DAY AND s.pb+INTERVAL 24 HOURS""").df().dropna(subset=["feature_name"])
# Diagnosen (Hauptdiagnose, <= 24h)
diag=con.execute(f"""SELECT DISTINCT s.stay_id, DIAGNR feature_name FROM {rcl('diagnose.csv')} d JOIN s ON CAST(d.FALLNR AS VARCHAR)=s.idc
  WHERE UPPER(TRIM(HAUPTNEBEN))='H' AND DIAGNR IS NOT NULL AND {TS('FESTSTDATUM')} <= s.pb+INTERVAL 24 HOURS""").df()

lab_w=long_to_wide(lab,"lab24"); vit_w=long_to_wide(vit,"vital24")
proc_w=presence(proc,"proc24"); acc_w=presence(acc,"zugang24")
diag_w=presence(diag,"diag_main") if not diag.empty else pd.DataFrame(columns=["stay_id"])
# diag-Praefix: retro nutzt 'diag_main_<icd>' -> presence() liefert genau das
built=mg[["stay_id"]].copy()
for part in [lab_w,vit_w,proc_w,acc_w,diag_w]:
    if len(part.columns)>1: built=built.merge(part,on="stay_id",how="left")
print(f"Neu gebaute prospektive 24h-Spalten: {built.shape[1]-1}")

# ---------------- Prospektive Feature-Matrix mit den 84 Namen ----------------
mg2=mg.merge(built,on="stay_id",how="left")
def Xframe_pros(frame):
    X=pd.DataFrame(index=frame.index)
    for c in present:
        if c in built.columns: X[c]=pd.to_numeric(frame[c],errors="coerce")
        elif c in frame.columns: X[c]=(frame[c].astype(str) if c in cat else pd.to_numeric(frame[c],errors="coerce"))
        else: X[c]=np.nan
    for c in cat: X[c]=X[c].astype(str)
    return X[present]
Xp=Xframe_pros(mg2)

# ---------------- Abdeckung messen ----------------
def coverage(frame):
    real=[c for c in present if c!="oebenekurz" and pd.to_numeric(frame.get(c),errors="coerce").notna().sum()>0]
    return real
real_after=coverage(mg2 if False else Xp.assign(**{c:pd.to_numeric(Xp[c],errors='coerce') for c in numc}))
# robust: pruefe auf den gebauten/vorhandenen Werten
real_after=[c for c in present if c!="oebenekurz" and pd.to_numeric(Xp[c],errors="coerce").notna().sum()>0]
import collections
def dom(f):
    for p in ["lab24_","vital24_","proc24_","zugang24_","diag_main_"]:
        if f.startswith(p): return p
    return "demo"
print("\n=== ABDECKUNG (von 84 leckage-freien Features) ===")
print(f"vorher (nur Prospektiv-Parquet): 4 numerische + oebenekurz")
print(f"nachher (mit echten 24h-Features): {len(real_after)} mit >=1 echtem Wert")
print("nachher nach Domain:", dict(collections.Counter(dom(f) for f in real_after)))
miss=[f for f in present if f!="oebenekurz" and f not in real_after]
print(f"weiterhin per Median ersetzt: {len(miss)} ->", dict(collections.Counter(dom(f) for f in miss)))
# Befuellungsgrad pro Stay (Anteil der 84 mit echtem Wert)
fillrate=pd.to_numeric(Xp[[c for c in present if c!='oebenekurz']].apply(lambda s:pd.to_numeric(s,errors='coerce')).notna().sum(axis=1))/ (len(present)-1)
print(f"Median-Befuellungsgrad pro Stay: {fillrate.median()*100:.0f}% (min {fillrate.min()*100:.0f}%, max {fillrate.max()*100:.0f}%)")

# ---------------- Faire Neu-Auswertung ----------------
def metrics(yt,yp,label,sub=None):
    if sub is not None: yt,yp=yt[sub],yp[sub]
    yt=np.asarray(yt,float); yp=np.clip(np.asarray(yp,float),0,None); ae=np.abs(yt-yp)
    return {"Modell":label,"n":int(len(yt)),"MAE":round(float(ae.mean()),3),"MedianAE":round(float(np.median(ae)),3),
            "RMSE":round(float(np.sqrt(mean_squared_error(yt,yp))),3),"R2":round(float(r2_score(yt,yp)),3),"Bias":round(float((yp-yt).mean()),3)}
yt=mg2["los_days"].values
rows=[metrics(yt,mg2["arzt"].values,"Oberarzt")]
for n,m in models.items(): rows.append(metrics(yt,m.predict(Xp),n))
res=pd.DataFrame(rows)
print("\n=== PROSPEKTIV (faire 24h-Features) — Tage ===")
print(res.to_string(index=False))
res.to_csv(AN/"canonical"/"metrics_prospective_fair24h.csv",sep=";",index=False)
# Subgruppen
print("\nnach LoS-Subgruppe (MAE):")
for sg,mask in [("1-7d",(yt>1)&(yt<=7)),(">7d",yt>7)]:
    if mask.sum()<5: continue
    r=[f"Oberarzt {metrics(yt,mg2['arzt'].values,'',mask)['MAE']}"]+[f"{n} {metrics(yt,m.predict(Xp),'',mask)['MAE']}" for n,m in models.items()]
    print(f"  {sg} (n={int(mask.sum())}): "+" | ".join(r))
print("\nGespeichert: canonical/metrics_prospective_fair24h.csv")

# ---- faire prospektive Hexbin-Figur (finales Modell ExtraTrees, bis 20 Tage) + Vorhersagen ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CAP=20
predf=pd.DataFrame({"stay_id":mg2["stay_id"].values,"los_obs":yt,"arzt":mg2["arzt"].values})
for n,m in models.items(): predf[f"pred_{n}"]=np.clip(m.predict(Xp),0,None)
predf.to_csv(AN/"canonical"/"metrics_prospective_fair24h_predictions.csv",sep=";",index=False)
def hexbin(obs,pred,title,path):
    obs=np.asarray(obs,float); pred=np.clip(np.asarray(pred,float),0,None); ae=np.abs(obs-pred)
    n=len(obs); mae=ae.mean(); rmse=np.sqrt(mean_squared_error(obs,pred)); r2=r2_score(obs,pred); med=np.median(ae)
    plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(6,5.2))
    hb=ax.hexbin(obs,pred,gridsize=24,cmap="viridis",bins="log",mincnt=1,extent=[0,CAP,0,CAP])
    ax.plot([0,CAP],[0,CAP],"--",color="#d6604d",lw=1.6,label="Identity (perfect prediction)")
    cb=fig.colorbar(hb,ax=ax); cb.set_label("Stays per bin (log scale)",fontsize=11)
    ax.set_xlim(0,CAP); ax.set_ylim(0,CAP); ax.set_xlabel("Observed ICU length of stay (days)"); ax.set_ylabel("Predicted ICU LoS (days)")
    ax.set_title(title,weight="bold",fontsize=13); ax.legend(loc="upper left",fontsize=10,framealpha=.4)
    box=f"n = {n}\nMAE = {mae:.2f} d\nRMSE = {rmse:.2f} d\nR² = {r2:.2f}\nMedian AE = {med:.2f} d"
    ax.text(0.97,0.03,box,transform=ax.transAxes,ha="right",va="bottom",fontsize=10,bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#bbb",alpha=.9))
    fig.tight_layout(); fig.savefig(str(path),dpi=300,bbox_inches="tight"); plt.close(fig)
hexbin(yt,predf["pred_ExtraTrees"].values,"Prospective — ExtraTrees (fair 24 h features)",AN/"canonical"/"fig_hexbin_pros_ExtraTrees.png")
print("Figur aktualisiert: canonical/fig_hexbin_pros_ExtraTrees.png")
