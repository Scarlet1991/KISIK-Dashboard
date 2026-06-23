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
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>2").df()
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
dp=con.execute(f"SELECT * FROM read_parquet('{PROS.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND is_open=0 AND icu_duration_h/24.0>2").df()
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

# ---- Matrizen fuer alternative Modelle (Tweedie/Hazard/Quantile) auf rekonstruierten Features speichern ----
import json as _json
ALT=AN/"canonical"/"alt_matrices"; ALT.mkdir(parents=True,exist_ok=True)
_xtr=X.iloc[tr].copy(); _xtr["__y__"]=y[tr]; _xtr.to_parquet(ALT/"retro_train.parquet")
_xp=Xp.copy().reset_index(drop=True)
_xp["__los__"]=mg2["los_days"].values; _xp["__arzt__"]=mg2["arzt"].values; _xp["__stay_id__"]=mg2["stay_id"].astype(str).values
_xp.to_parquet(ALT/"prospective_rebuilt_193.parquet")
_json.dump({"present":present,"numc":numc,"cat":cat}, open(ALT/"feature_lists.json","w"))
print(f"Alt-Matrizen gespeichert: retro_train {_xtr.shape} | prospective_rebuilt {_xp.shape} -> {ALT}")

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
# Subgruppen (print)
print("\nnach LoS-Subgruppe (MAE):")
for sg,mask in [("2-7d",(yt>2)&(yt<=7)),(">7d",yt>7)]:
    if mask.sum()<5: continue
    r=[f"Oberarzt {metrics(yt,mg2['arzt'].values,'',mask)['MAE']}"]+[f"{n} {metrics(yt,m.predict(Xp),'',mask)['MAE']}" for n,m in models.items()]
    print(f"  {sg} (n={int(mask.sum())}): "+" | ".join(r))
print("\nGespeichert: canonical/metrics_prospective_fair24h.csv")

# ---- Null-Modell (Trainings-Mittelwert) ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
null_pred_val = float(y[tr].mean())
print(f"\nNull-Modell: Trainings-Mittelwert = {null_pred_val:.2f} d")

# ---- Subgruppen-Analyse: MAE nach LoS-Kategorie (LoS>2 -> 3 Bins) + Null ----
sg_bins=[("2–4 d",(yt>2)&(yt<=4)),
         ("4–7 d",(yt>4)&(yt<=7)),(">7 d",yt>7)]
all_preds={"Oberarzt":mg2["arzt"].values}
for n,m in models.items(): all_preds[n]=np.clip(m.predict(Xp),0,None)
all_preds["Null"]=np.full(len(yt),null_pred_val)

sg_rows=[]
for sg_label,mask in sg_bins:
    if mask.sum()<5: continue
    for mod_name,pred in all_preds.items():
        ae=np.abs(yt[mask]-np.asarray(pred,float)[mask])
        sg_rows.append({"Subgroup":sg_label,"n":int(mask.sum()),"Modell":mod_name,
                        "MAE":round(float(ae.mean()),3),"MedianAE":round(float(np.median(ae)),3)})
sg_df=pd.DataFrame(sg_rows)
sg_df.to_csv(AN/"canonical"/"metrics_subgroups.csv",sep=";",index=False)
print("\n=== MAE NACH LoS-SUBGRUPPE (is_open=0, n=193) ===")
pivot=sg_df.pivot_table(index="Modell",columns="Subgroup",values="MAE")
print(pivot.to_string())

# ---- Subgruppen-Figur ----
sg_order=["Oberarzt"]+[m for m in ["Ridge","RandomForest","ExtraTrees","XGBoost","Tweedie"] if m in models]+["Null"]
sg_label_map={"Oberarzt":"Oberarzt","RandomForest":"Random Forest","ExtraTrees":"Extra Trees","Null":"Null (mean)"}
sg_colors={"Oberarzt":"#c0392b","Ridge":"#7f8c8d","RandomForest":"#3498db","ExtraTrees":"#1a6ea3",
           "XGBoost":"#27ae60","Tweedie":"#8e44ad","Null":"#bdc3c7"}
bins_order=[r[0] for r in sg_bins if r[1].sum()>=5]
ns_sg={r[0]:int(r[1].sum()) for r in sg_bins}
xi=np.arange(len(bins_order)); n_mods=len(sg_order); total_w=0.82; bw=total_w/n_mods
offsets=np.linspace(-(total_w-bw)/2,(total_w-bw)/2,n_mods)
fig,ax=plt.subplots(figsize=(13,5.2))
for i,mod_name in enumerate(sg_order):
    mae_vals=[float(sg_df[(sg_df["Modell"]==mod_name)&(sg_df["Subgroup"]==b)]["MAE"].iloc[0]) if len(sg_df[(sg_df["Modell"]==mod_name)&(sg_df["Subgroup"]==b)])>0 else 0 for b in bins_order]
    bars=ax.bar(xi+offsets[i],mae_vals,bw,label=sg_label_map.get(mod_name,mod_name),color=sg_colors.get(mod_name,"#888"))
    ax.bar_label(bars,fmt="%.1f",fontsize=6.8,padding=2)
ax.set_xticks(xi); ax.set_xticklabels([f"{b}\n(n={ns_sg[b]})" for b in bins_order],fontsize=10)
ax.set_ylabel("MAE (days) — lower is better"); ax.set_ylim(0,max(sg_df["MAE"])*1.22)
ax.legend(fontsize=8.5,ncol=4,loc="upper left"); ax.spines[["top","right"]].set_visible(False)
ax.set_title("MAE by LoS subgroup — prospective cohort (is_open=0, n=193, completed stays)",weight="bold",fontsize=12)
fig.tight_layout(); fig.savefig(str(AN/"canonical"/"fig_subgroup_mae.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Figur gespeichert: canonical/fig_subgroup_mae.png")

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
# Senior-physician hexbin (English title, fair n=193) — overwrites the stale n=360 version from canonical_analysis.py
hexbin(yt,predf["arzt"].values,"Prospective — senior physician",AN/"canonical"/"fig_hexbin_pros_oberarzt.png")
print("Figuren aktualisiert: canonical/fig_hexbin_pros_ExtraTrees.png, fig_hexbin_pros_oberarzt.png")

# ---- Modellvergleich (retro Holdout vs. faire prospektive n=193): MAE + R^2 Panels, English ----
# Zwei Panels, weil der MAE allein irrefuehrt: ExtraTrees prospektiv ~ retrospektiv, aber R^2 kollabiert.
retro_csv=pd.read_csv(AN/"canonical"/"metrics_retrospective.csv",sep=";").set_index("Modell")
resi=res.set_index("Modell")
order=[m for m in ["Ridge","RandomForest","ExtraTrees","XGBoost","Tweedie"] if m in retro_csv.index and m in resi.index]
labels=[{"RandomForest":"Random forest","ExtraTrees":"Extra Trees"}.get(m,m) for m in order]
rt =[float(retro_csv.loc[m,"MAE_days"]) for m in order]; pp =[float(resi.loc[m,"MAE"]) for m in order]
rt2=[float(retro_csv.loc[m,"R2"])       for m in order]; pp2=[float(resi.loc[m,"R2"])  for m in order]
sen_mae=float(resi.loc["Oberarzt","MAE"]); sen_r2=float(resi.loc["Oberarzt","R2"])
n_retro=int(retro_csv["n"].iloc[0]); n_pros=int(resi.loc["Oberarzt","n"])
xr=np.arange(len(order)); w=0.38
fig,(axA,axB)=plt.subplots(1,2,figsize=(13,4.8))
# Panel A: MAE (mit Wertelabels)
ba1=axA.bar(xr-w/2,rt,w,label=f"retrospective hold-out (n={n_retro:,})",color="#b5d4f4")
ba2=axA.bar(xr+w/2,pp,w,label=f"prospective (n={n_pros})",color="#185fa5")
axA.bar_label(ba1,fmt="%.2f",fontsize=7.2,padding=2,color="#555")
axA.bar_label(ba2,fmt="%.2f",fontsize=7.2,padding=2,color="#1f5f9e")
axA.axhline(sen_mae,color="#c0392b",ls="--",lw=1.6)
axA.text(len(order)-1,sen_mae+0.12,f"Senior physician (MAE {sen_mae:.2f} d)",color="#c0392b",ha="right",fontsize=8.5,weight="bold")
axA.set_xticks(xr); axA.set_xticklabels(labels,fontsize=9.5); axA.set_ylabel("MAE (days) — lower is better")
axA.set_ylim(0,max(max(rt),max(pp))+1.0); axA.set_title("(A) MAE — lower is better",weight="bold",fontsize=11); axA.legend(fontsize=8.5)
# Panel B: R^2 (Ridge prospektiv ~ -22 -> abgeschnitten und annotiert) mit Wertelabels
R2FLOOR=-0.35; pp2c=[max(v,R2FLOOR) for v in pp2]
bb1=axB.bar(xr-w/2,rt2,w,label="retrospective hold-out",color="#b5d4f4")
axB.bar(xr+w/2,pp2c,w,label="prospective",color="#185fa5")
axB.bar_label(bb1,fmt="%.2f",fontsize=7.2,padding=2,color="#555")
axB.axhline(0,color="#888",lw=0.8)
axB.axhline(sen_r2,color="#c0392b",ls="--",lw=1.6)
axB.text(0,sen_r2+0.03,f"Senior physician (R² {sen_r2:.2f})",color="#c0392b",ha="left",fontsize=8.5,weight="bold")
for i,v in enumerate(pp2):
    if v<R2FLOOR: axB.text(xr[i]+w/2,R2FLOOR+0.02,f"{v:.0f}↓",color="white",ha="center",va="bottom",fontsize=8,weight="bold")
    else: axB.text(xr[i]+w/2,pp2c[i]+0.01,f"{v:.2f}",color="#1f5f9e",ha="center",va="bottom",fontsize=7.2)
axB.set_ylim(R2FLOOR,0.62); axB.set_xticks(xr); axB.set_xticklabels(labels,fontsize=9.5); axB.set_ylabel("R² — higher is better")
axB.set_title("(B) R² — higher is better",weight="bold",fontsize=11); axB.legend(fontsize=8.5,loc="upper right")
fig.suptitle(f"Model performance: retrospective hold-out (n={n_retro:,}) vs prospective (n={n_pros} completed stays)",weight="bold",fontsize=12.5)
fig.tight_layout(); fig.savefig(str(AN/"canonical"/"fig_model_comparison.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Figur aktualisiert: canonical/fig_model_comparison.png (MAE + R^2 Panels, faire prospektive n=193)")

# ---- Permutation-Importance-Figur mit englischen Labels (aus feature_importance.csv) ----
LABELS={
 "proc24_8_98f_0":"ICU complex treatment, base (8-98f.0)",
 "proc24_8_98f_10":"ICU complex treatment, extended (8-98f.10)",
 "oebenekurz":"ICU care-unit type",
 "proc24_8_931_0":"Extended haemodynamic monitoring (8-931)",
 "proc24_8_924":"Cardiac monitoring (8-924)",
 "proc24_anzahl_gesamt":"Total procedure count (24 h)",
 "proc24_8_98f_11":"ICU complex treatment, prolonged (8-98f.11)",
 "diag_main_z99_1":"Ventilator dependence (Z99.1)",
 "proc24_8_930":"Basic haemodynamic monitoring (8-930)",
 "stay_nr":"ICU stay number",
 "diag_main_j12_8":"Viral pneumonia (J12.8)",
 "diag_main_g91_0":"Communicating hydrocephalus (G91.0)",
 "alter":"Age",
 "diag_main_g91_8":"Hydrocephalus, other (G91.8)",
 "proc24_3_200":"Native cranial CT (3-200)",
}
def eng_label(f):
    if f in LABELS: return LABELS[f]
    t=f.replace("lab24_","Lab: ").replace("vital24_","Vital: ").replace("proc24_","Procedure ").replace("zugang24_","Access: ").replace("diag_main_","Diagnosis ").replace("_"," ")
    return t
imp=pd.read_csv(AN/"canonical"/"feature_importance.csv",sep=";").head(15).iloc[::-1]
fig,ax=plt.subplots(figsize=(8.4,6));
ax.barh([eng_label(f) for f in imp["Feature"]],imp["MAE_increase_days"],xerr=imp["sd"],color="#762a83")
ax.set_xlabel("Increase in MAE when permuted (days)"); ax.margins(y=0.01)
ax.set_title("Permutation feature importance — Extra Trees (final model)",weight="bold")
fig.tight_layout(); fig.savefig(str(AN/"canonical"/"fig_importance.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Figur aktualisiert: canonical/fig_importance.png (englische Labels)")
