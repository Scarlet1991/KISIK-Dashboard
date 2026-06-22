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
allowed=[("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31")]  # nur AIN-Intensiveinheiten IZ32/IZ21/IZ31
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
def plain(reg): return Pipeline([("pre",pre(False)),("mdl",reg)])  # Tweedie: eigener Loss/Link, kein log1p
# beste Hyperparameter aus der kanonischen Analyse (summary.json) laden -> konsistent zum finalen Modell
import json
bp=json.loads((AN/"canonical"/"summary.json").read_text(encoding="utf-8"))["best_params"]
models={
 "Ridge":ttr(Ridge(**bp["Ridge"],random_state=RS),scale=True),
 "RandomForest":ttr(RandomForestRegressor(**bp["RandomForest"],random_state=RS,n_jobs=1)),
 "ExtraTrees":ttr(ExtraTreesRegressor(**bp["ExtraTrees"],random_state=RS,n_jobs=1)),
 "XGBoost":ttr(XGBRegressor(**bp["XGBoost"],random_state=RS,n_jobs=1,tree_method="hist")),
}
if "Tweedie" in bp:
    models["Tweedie"]=plain(XGBRegressor(objective="reg:tweedie",**bp["Tweedie"],random_state=RS,n_jobs=1,tree_method="hist"))
print(f"Training (beste Hyperparameter aus summary.json) ... ExtraTrees={bp['ExtraTrees']}"); [m.fit(X.iloc[tr],y[tr]) for m in models.values()]

# ---------------- prospektive Kohorte + Senior-Match ----------------
# gleiche Stations-/Einheiten-Filter wie retrospektiv (AIN IZ32/21/31), nur abgeschlossene
# Aufenthalte (is_open=0) mit tatsaechlicher LoS > 1 Tag
dp=con.execute(f"SELECT * FROM read_parquet('{PROS.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()  # ALTER ANSATZ: KEIN is_open-Filter (offene/zensierte Stays einbezogen)
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

# ============ AUSWERTUNG OHNE is_open-Filter (5 Modelle + Tweedie) ============
# ACHTUNG: bei is_open=1 ist los_days ZENSIERT (bisher vergangene Zeit, nicht endgueltige LoS)
# -> Fehler bei is_open=1-Stays sind durch zensierte Zielwerte verzerrt. Nur als Sensitivitaet.
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
OUTX=AN/"exploratory_no_isopen"; OUTX.mkdir(parents=True,exist_ok=True)

yt=mg2["los_days"].to_numpy(float)
isopen=pd.to_numeric(mg2["is_open"],errors="coerce").fillna(0).astype(int).to_numpy()
arzt=mg2["arzt"].to_numpy(float)

def met(y,p,label,sub=None):
    if sub is not None: y,p=y[sub],p[sub]
    p=np.clip(np.asarray(p,float),0,None); ae=np.abs(y-p)
    return {"Modell":label,"n":int(len(y)),"MAE":round(float(ae.mean()),3),"MedianAE":round(float(np.median(ae)),3),
            "RMSE":round(float(np.sqrt(mean_squared_error(y,p))),3),"R2":round(float(r2_score(y,p)),3),"Bias":round(float((p-y).mean()),3)}

preds={"Oberarzt":arzt}
for nm,m in models.items(): preds[nm]=np.clip(m.predict(Xp),0,None)
ml_order=[m for m in ["Ridge","RandomForest","ExtraTrees","XGBoost","Tweedie"] if m in models]
order=["Oberarzt"]+ml_order
n0=int((isopen==0).sum()); n1=int((isopen==1).sum())
print(f"\nKohorte OHNE is_open-Filter: n={len(yt)} (abgeschlossen {n0}, offen/zensiert {n1})")

# ---- Gesamt-Metriken ----
allres=pd.DataFrame([met(yt,preds[o],o) for o in order])
allres.to_csv(OUTX/"prospektiv_no_isopen_overall.csv",sep=";",index=False)
print("\n=== PROSPEKTIV OHNE is_open-Filter (gegen erfasste/zensierte LoS) ==="); print(allres.to_string(index=False))
A=allres.set_index("Modell")

# ---- Stratifiziert nach is_open ----
strat=[]
for st,lab in [(isopen==0,"is_open=0 abgeschlossen"),(isopen==1,"is_open=1 offen/zensiert")]:
    for o in order: strat.append({**met(yt,preds[o],o,st),"Gruppe":lab})
stratdf=pd.DataFrame(strat)[["Gruppe","Modell","n","MAE","MedianAE","RMSE","R2","Bias"]]
stratdf.to_csv(OUTX/"prospektiv_no_isopen_by_isopen.csv",sep=";",index=False)
print("\n=== Stratifiziert nach is_open ==="); print(stratdf.to_string(index=False))

# ---- Vergleich is_open=0 vs ohne Filter ----
comp=[]
for o in order:
    c=met(yt,preds[o],o,isopen==0); a=met(yt,preds[o],o)
    comp.append({"Modell":o,"MAE_isopen0":c["MAE"],"R2_isopen0":c["R2"],
                 "MAE_no_filter":a["MAE"],"R2_no_filter":a["R2"],"dMAE":round(a["MAE"]-c["MAE"],3)})
compdf=pd.DataFrame(comp); compdf.to_csv(OUTX/"prospektiv_vergleich_isopen.csv",sep=";",index=False)
print("\n=== Vergleich is_open=0 vs ohne Filter ==="); print(compdf.to_string(index=False))

# ---- Figur 1: MAE + R² zwei Panels (retro vs prospektiv ohne Filter) ----
retro_csv=pd.read_csv(AN/"canonical"/"metrics_retrospective.csv",sep=";").set_index("Modell")
fig_order=[m for m in ["Ridge","RandomForest","ExtraTrees","XGBoost","Tweedie"] if m in retro_csv.index and m in A.index]
fig_labels=[{"RandomForest":"Random forest","ExtraTrees":"Extra Trees"}.get(m,m) for m in fig_order]
rt =[float(retro_csv.loc[m,"MAE_days"]) for m in fig_order]
pp =[float(A.loc[m,"MAE"]) for m in fig_order]
rt2=[float(retro_csv.loc[m,"R2"])       for m in fig_order]
pp2=[float(A.loc[m,"R2"])               for m in fig_order]
sen_mae=float(A.loc["Oberarzt","MAE"]); sen_r2=float(A.loc["Oberarzt","R2"])
n_retro=int(retro_csv["n"].iloc[0]); n_pros=len(yt)
xi=np.arange(len(fig_order)); w=0.38
fig,(axA,axB)=plt.subplots(1,2,figsize=(13,4.8))
ba1=axA.bar(xi-w/2,rt,w,label=f"retrospective hold-out (n={n_retro:,})",color="#b5d4f4")
ba2=axA.bar(xi+w/2,pp,w,label=f"prospective, no is_open filter (n={n_pros})",color="#185fa5")
axA.bar_label(ba1,fmt="%.2f",fontsize=7.2,padding=2,color="#555")
axA.bar_label(ba2,fmt="%.2f",fontsize=7.2,padding=2,color="#1f5f9e")
axA.axhline(sen_mae,color="#c0392b",ls="--",lw=1.6)
axA.text(len(fig_order)-1,sen_mae+0.12,f"Senior physician (MAE {sen_mae:.2f} d)",color="#c0392b",ha="right",fontsize=8.5,weight="bold")
axA.set_xticks(xi); axA.set_xticklabels(fig_labels,fontsize=9.5); axA.set_ylabel("MAE (days) — lower is better")
axA.set_ylim(0,max(max(rt),max(pp))+1.0); axA.set_title("(A) MAE — lower is better",weight="bold",fontsize=11); axA.legend(fontsize=8.5)
R2FLOOR=-0.35; pp2c=[max(v,R2FLOOR) for v in pp2]
bb1=axB.bar(xi-w/2,rt2,w,label="retrospective hold-out",color="#b5d4f4")
axB.bar(xi+w/2,pp2c,w,label="prospective (no filter)",color="#185fa5")
axB.bar_label(bb1,fmt="%.2f",fontsize=7.2,padding=2,color="#555")
axB.axhline(0,color="#888",lw=0.8); axB.axhline(sen_r2,color="#c0392b",ls="--",lw=1.6)
axB.text(0,sen_r2+0.03,f"Senior physician (R² {sen_r2:.2f})",color="#c0392b",ha="left",fontsize=8.5,weight="bold")
for i,v in enumerate(pp2):
    if v<R2FLOOR: axB.text(xi[i]+w/2,R2FLOOR+0.02,f"{v:.1f}↓",color="white",ha="center",va="bottom",fontsize=8,weight="bold")
    else: axB.text(xi[i]+w/2,pp2c[i]+0.01,f"{v:.2f}",color="#1f5f9e",ha="center",va="bottom",fontsize=7.2)
axB.set_ylim(R2FLOOR,0.62); axB.set_xticks(xi); axB.set_xticklabels(fig_labels,fontsize=9.5); axB.set_ylabel("R² — higher is better")
axB.set_title("(B) R² — higher is better",weight="bold",fontsize=11); axB.legend(fontsize=8.5,loc="upper right")
fig.suptitle(f"Model performance: retrospective hold-out (n={n_retro:,}) vs prospective NO is_open filter (n={n_pros})\n"
             f"[is_open=1: LoS is censored elapsed time, not final LOS — MAE artificially inflated for open stays]",weight="bold",fontsize=11)
fig.tight_layout(); fig.savefig(str(OUTX/"fig_model_comparison_no_isopen.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Figur gespeichert: fig_model_comparison_no_isopen.png")

# ---- Figur 2: MAE nach is_open-Status (3 Balken pro Modell) ----
g0=[met(yt,preds[m],m,isopen==0)["MAE"] for m in ml_order]
ga=[float(A.loc[m,"MAE"]) for m in ml_order]
g1=[met(yt,preds[m],m,isopen==1)["MAE"] for m in ml_order]
xr4=np.arange(len(ml_order)); w3=0.27
fig,ax=plt.subplots(figsize=(10,5))
c0=ax.bar(xr4-w3,g0,w3,label=f"is_open=0 completed (n={n0})",color="#b5d4f4")
ca=ax.bar(xr4,ga,w3,label=f"all / no filter (n={n_pros})",color="#185fa5")
c1=ax.bar(xr4+w3,g1,w3,label=f"is_open=1 open/censored (n={n1}) [ZENSIERT]",color="#c0392b")
for c in (c0,ca,c1): ax.bar_label(c,fmt="%.1f",fontsize=6.8,padding=2)
ax.set_xticks(xr4); ax.set_xticklabels([{"RandomForest":"Random Forest","ExtraTrees":"Extra Trees"}.get(m,m) for m in ml_order])
ax.set_ylabel("MAE (days)"); ax.set_title("Prospective MAE by is_open status — open stays drive apparent degradation",weight="bold",fontsize=10.5)
ax.legend(fontsize=8.5); fig.tight_layout(); fig.savefig(str(OUTX/"fig_mae_by_isopen.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Figur gespeichert: fig_mae_by_isopen.png")

# ---- Figur 3: Subgruppen-MAE nach LoS-Kategorie (4 Bins) ----
null_pred_val=float(y[tr].mean())
sg_bins=[("1–2 d",(yt>=1)&(yt<=2)),("2–4 d",(yt>2)&(yt<=4)),("4–7 d",(yt>4)&(yt<=7)),(">7 d",yt>7)]
all_sg_preds={"Oberarzt":arzt,**{n:np.clip(m.predict(Xp),0,None) for n,m in models.items()},"Null":np.full(len(yt),null_pred_val)}
sg_rows=[]
for sg_label,mask in sg_bins:
    if mask.sum()<5: continue
    for mod_name,pred in all_sg_preds.items():
        ae=np.abs(yt[mask]-np.asarray(pred,float)[mask])
        sg_rows.append({"Subgroup":sg_label,"n":int(mask.sum()),"Modell":mod_name,
                        "MAE":round(float(ae.mean()),3),"MedianAE":round(float(np.median(ae)),3)})
sg_df=pd.DataFrame(sg_rows); sg_df.to_csv(OUTX/"metrics_subgroups_no_isopen.csv",sep=";",index=False)
print("\n=== MAE NACH LoS-SUBGRUPPE (ohne is_open-Filter, n=%d) ===" % len(yt))
print(sg_df.pivot_table(index="Modell",columns="Subgroup",values="MAE").to_string())

sg_order=["Oberarzt"]+[m for m in ["Ridge","RandomForest","ExtraTrees","XGBoost","Tweedie"] if m in models]+["Null"]
sg_label_map={"Oberarzt":"Oberarzt","RandomForest":"Random Forest","ExtraTrees":"Extra Trees","Null":"Null (mean)"}
sg_colors={"Oberarzt":"#c0392b","Ridge":"#7f8c8d","RandomForest":"#3498db","ExtraTrees":"#1a6ea3",
           "XGBoost":"#27ae60","Tweedie":"#8e44ad","Null":"#bdc3c7"}
bins_order=[r[0] for r in sg_bins if r[1].sum()>=5]
ns_sg={r[0]:int(r[1].sum()) for r in sg_bins}
xi2=np.arange(len(bins_order)); n_mods=len(sg_order); total_w=0.82; bw=total_w/n_mods
offsets=np.linspace(-(total_w-bw)/2,(total_w-bw)/2,n_mods)
fig,ax=plt.subplots(figsize=(13,5.2))
for i,mod_name in enumerate(sg_order):
    mae_vals=[float(sg_df[(sg_df["Modell"]==mod_name)&(sg_df["Subgroup"]==b)]["MAE"].iloc[0]) if len(sg_df[(sg_df["Modell"]==mod_name)&(sg_df["Subgroup"]==b)])>0 else 0 for b in bins_order]
    bars=ax.bar(xi2+offsets[i],mae_vals,bw,label=sg_label_map.get(mod_name,mod_name),color=sg_colors.get(mod_name,"#888"))
    ax.bar_label(bars,fmt="%.1f",fontsize=6.8,padding=2)
ax.set_xticks(xi2); ax.set_xticklabels([f"{b}\n(n={ns_sg[b]})" for b in bins_order],fontsize=10)
ax.set_ylabel("MAE (days) — lower is better"); ax.set_ylim(0,max(sg_df["MAE"])*1.22)
ax.legend(fontsize=8.5,ncol=4,loc="upper left"); ax.spines[["top","right"]].set_visible(False)
ax.set_title(f"MAE by LoS subgroup — prospective cohort WITHOUT is_open filter (n={len(yt)}, incl. {n1} censored)\n"
             "[is_open=1 LoS = elapsed time only — MAE for open stays scored against lower bound]",weight="bold",fontsize=11)
fig.tight_layout(); fig.savefig(str(OUTX/"fig_subgroup_mae_no_isopen.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Figur gespeichert: fig_subgroup_mae_no_isopen.png")

print(f"\nAlle Ausgaben in {OUTX}: 3 CSVs + 3 Figuren")
con.close()
