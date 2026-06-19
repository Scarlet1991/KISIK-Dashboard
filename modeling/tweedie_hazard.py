# -*- coding: utf-8 -*-
"""
Tweedie/Gamma-Objective + diskrete Hazard-Modellierung fuer ICU-LoS.
Vergleich gegen log1p-Baseline, retrospektiv + prospektiver Oberarzt-Vergleich.
Fokus: Langlieger (>7 / >14 Tage).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, XGBClassifier

BASE   = Path(r"D:\Ausgangsdaten\KISIK Projekt")
AN     = BASE / "Eigene Auswertung"
RETRO  = BASE / "kisik2" / "kisik2_icu_ml_dataset_24h.parquet"
PROS   = BASE / "kisik2" / "kisik2_prospektiv_ml_dataset.parquet"
OPAN_R = BASE / "kisik2" / "op_an.csv";            OPZ_R = BASE / "kisik2" / "op_zeitintervalle.csv"
OPAN_P = AN / "oldlive_kisik2_core_old_op_an_filtered_dedup.csv"
OPZ_P  = "D:/Ausgangsdaten/Live-Daten/OLD/*/op_zeitintervalle.csv"
SENIOR = AN / "los_senior_estimates_tagesausleitung_stay_level.csv"
FEAT   = AN / "los_selected_features_ain_24h_compact.csv"
OUT    = AN / "los_tweedie_hazard"
RS = 42; TMAX = 90

allowed = [("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31"),("AIN","IZ01"),
           ("AUG","IZ01"),("AVT","IZ01"),("GCH","IZ01"),("GYN","IZ01"),
           ("HNO","IZ01"),("HTC","IZ01"),("IZPV","IZ01"),("MKG","IZ01"),
           ("NCH","IZ01"),("NUK","IZ01"),("STR","IZ01"),("UCH","IZ01"),("URO","IZ01")]
allowed_sql = ", ".join(f"('{w}','{o}')" for w,o in allowed)
con = duckdb.connect()
def rc(p): return f"read_csv_auto('{p}', delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true)"
MIN = lambda c: f"COALESCE(EXTRACT(EPOCH FROM TRY_CAST({c} AS INTERVAL))/60.0, 0)"
TS  = lambda c: f"COALESCE(TRY_CAST({c} AS TIMESTAMP), TRY_STRPTIME({c}, '%d.%m.%Y %H:%M:%S'))"
OP_COLS = ["op_schnittnaht_min","op_anaesth_min","op_anaesth_praesenz_min","op_hlm_min",
           "op_hlm_flag","op_n_intervalle","op_plandauer_sum","op_asa_max","op_n_eingriffe","op_any"]

def build_op(stays_df, idcol, opan, opz, asa_col, plan_col, opan_date,
             z_dauer, z_zeit, z_beginn, z_fallid, z_fallid_an):
    con.register("st", stays_df[["stay_id", idcol, "planbegin"]].rename(columns={idcol:"idc"}))
    con.execute("CREATE OR REPLACE TEMP TABLE s AS SELECT stay_id, CAST(idc AS VARCHAR) idc, CAST(planbegin AS TIMESTAMP) pb FROM st")
    opz_df = con.execute(f"""
        WITH z AS (SELECT DISTINCT CAST({z_fallid} AS VARCHAR) idc, {z_zeit} zb, {z_beginn} bg, {z_dauer} du FROM {rc(opz)}),
        zw AS (SELECT s.stay_id, z.zb, {MIN('z.du')} m FROM s JOIN z ON z.idc=s.idc
               WHERE {TS('z.bg')} BETWEEN s.pb - INTERVAL 1 DAY AND s.pb + INTERVAL 24 HOURS)
        SELECT stay_id,
          SUM(CASE WHEN zb='Schnitt-Naht' THEN m ELSE 0 END) op_schnittnaht_min,
          SUM(CASE WHEN zb='Reine Anästhesiezeit' THEN m ELSE 0 END) op_anaesth_min,
          SUM(CASE WHEN zb='Anästhesiepräsenz' THEN m ELSE 0 END) op_anaesth_praesenz_min,
          SUM(CASE WHEN zb='HLM' THEN m ELSE 0 END) op_hlm_min,
          MAX(CASE WHEN zb='HLM' THEN 1 ELSE 0 END) op_hlm_flag,
          COUNT(*) op_n_intervalle FROM zw GROUP BY stay_id""").df()
    opan_df = con.execute(f"""
        WITH a AS (SELECT DISTINCT CAST({z_fallid_an} AS VARCHAR) idc, TRY_CAST({asa_col} AS INT) asa,
                          TRY_CAST({plan_col} AS DOUBLE) plan, {opan_date} dt FROM {rc(opan)}),
        aw AS (SELECT s.stay_id, a.asa, a.plan FROM s JOIN a ON a.idc=s.idc
               WHERE {TS('a.dt')} BETWEEN s.pb - INTERVAL 1 DAY AND s.pb + INTERVAL 24 HOURS)
        SELECT stay_id, MAX(asa) op_asa_max, SUM(plan) op_plandauer_sum, COUNT(*) op_n_eingriffe FROM aw GROUP BY stay_id""").df()
    op = pd.merge(opz_df, opan_df, on="stay_id", how="outer")
    op["op_any"] = (op["op_n_intervalle"].notna() | op["op_n_eingriffe"].notna()).astype(int)
    for c in OP_COLS:
        if c not in op.columns: op[c]=0.0
        op[c]=pd.to_numeric(op[c],errors="coerce").fillna(0.0)
    return op[["stay_id"]+OP_COLS]

# ------------------------------------------------------------------ retro
print("=== Daten laden + OP-Features ===")
df = con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({allowed_sql}) AND icu_duration_h/24.0 > 1").df()
df["planbegin"]=pd.to_datetime(df["planbegin"],errors="coerce"); df["los_days"]=df["icu_duration_h"]/24.0
df = df.merge(build_op(df,"fallid",OPAN_R.as_posix(),OPZ_R.as_posix(),"asaid","indivdauer","opplandatum","dauer","zeitintbez","beginn","fallid","fallid"),on="stay_id",how="left")
for c in OP_COLS: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
print(f"  retro {len(df):,} Stays")

feat = pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns]; cat=[c for c in present if c=="oebenekurz"]; num=[c for c in present if c not in cat]
def build_X(frame):
    parts=[frame.reindex(columns=num).apply(pd.to_numeric,errors="coerce")]
    if cat and all(c in frame.columns for c in cat):
        parts.append(pd.get_dummies(frame[cat].astype(str),prefix=cat).astype(float))
    parts.append(frame.reindex(columns=OP_COLS).apply(pd.to_numeric,errors="coerce"))
    X=pd.concat(parts,axis=1); X.columns=[str(c) for c in X.columns]
    return X.apply(pd.to_numeric,errors="coerce").astype(np.float64)
X=build_X(df); TRAIN_COLS=X.columns.tolist()
los=df["los_days"].clip(lower=0.01).values
groups=df["pid"].fillna("unknown").astype(str).values
tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(df,los,groups))
Xtr,Xte=X.values[tr],X.values[te]; los_tr,los_te=los[tr],los[te]
print(f"  Train {len(tr):,} | Test {len(te):,}")

COMMON=dict(n_estimators=600,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,random_state=RS,n_jobs=-1)

def metrics(yt,yp,label,sub=None):
    if sub is not None: yt,yp=yt[sub],yp[sub]
    ae=np.abs(yt-yp)
    return {"Modell":label,"n":len(yt),"MAE":round(float(ae.mean()),3),"Median_AE":round(float(np.median(ae)),3),
            "RMSE":round(float(np.sqrt(mean_squared_error(yt,yp))),3),"R2":round(float(r2_score(yt,yp)),3),"Bias":round(float((yp-yt).mean()),3)}

# ------------------------------------------------------------------ Modelle
print("=== Training ===")
fitted={}
# log1p Referenz
m=XGBRegressor(objective="reg:squarederror",**COMMON); m.fit(Xtr,np.log1p(los_tr)); fitted["log1p_Mean"]=("log",m)
# Tweedie
for vp in [1.3,1.5,1.7]:
    m=XGBRegressor(objective="reg:tweedie",tweedie_variance_power=vp,**COMMON); m.fit(Xtr,los_tr)
    fitted[f"Tweedie_{vp}"]=("raw",m)
# Gamma
m=XGBRegressor(objective="reg:gamma",**COMMON); m.fit(Xtr,los_tr); fitted["Gamma"]=("raw",m)
print("  log1p/Tweedie/Gamma trainiert")

# Diskrete Hazard
T_tr=np.clip(np.ceil(los_tr).astype(int),1,TMAX)
counts=T_tr
Xrep=np.repeat(Xtr,counts,axis=0)
day_t=np.concatenate([np.arange(1,c+1) for c in counts]).astype(np.float64)
T_rep=np.repeat(T_tr,counts)
y_haz=(day_t==T_rep).astype(int)
Xhaz=np.column_stack([Xrep,day_t]).astype(np.float32)
print(f"  Hazard Person-Period-Zeilen: {Xhaz.shape[0]:,}")
clf=XGBClassifier(objective="binary:logistic",n_estimators=400,max_depth=6,learning_rate=0.05,
                  subsample=0.8,colsample_bytree=0.8,min_child_weight=5,random_state=RS,n_jobs=-1,
                  eval_metric="logloss")
clf.fit(Xhaz,y_haz)
print("  Hazard-Klassifikator trainiert")

def hazard_predict(Xmat):
    """erwartete + mediane LoS aus diskreter Hazard-/Survival-Kurve."""
    n=Xmat.shape[0]; S=np.ones(n); E=np.zeros(n); med=np.full(n,np.nan)
    for t in range(1,TMAX+1):
        feat_t=np.column_stack([Xmat,np.full(n,t)]).astype(np.float32)
        h=np.clip(clf.predict_proba(feat_t)[:,1],1e-6,1-1e-6)
        E += S                         # P(LoS>=t)=S_{t-1}
        Snew=S*(1-h)
        newly=(med!=med)&(Snew<0.5)    # erstes t mit S(t)<0.5
        med[newly]=t
        S=Snew
    med[med!=med]=TMAX
    return E, med

def predict(tag,Xmat):
    kind,m=fitted[tag]
    p=m.predict(Xmat)
    return np.clip(np.expm1(p) if kind=="log" else p,0,None)

# ------------------------------------------------------------------ Retro-Holdout
print("\n=== Retrospektiver Holdout ===")
subs_r={"gesamt":np.ones(len(te),bool),"1-7d":(los_te>1)&(los_te<=7),">7d":los_te>7,">14d":los_te>14}
rows=[]
preds_te={t:predict(t,Xte) for t in fitted}
E_te,med_te=hazard_predict(Xte)
preds_te["Hazard_E"]=E_te; preds_te["Hazard_Median"]=med_te
for sg,mask in subs_r.items():
    for tag,p in preds_te.items():
        rows.append({**metrics(los_te,p,tag,mask),"Subgruppe":sg})
retro=pd.DataFrame(rows)[["Subgruppe","Modell","n","MAE","Median_AE","RMSE","R2","Bias"]]
print(retro.to_string(index=False))
retro.to_csv(f"{OUT}_retro.csv",sep=";",index=False)

# ------------------------------------------------------------------ Prospektiv
print("\n=== Prospektiv (Oberarzt-Vergleich) ===")
dp=con.execute(f"SELECT * FROM read_parquet('{PROS.as_posix()}')").df()
dp["planbegin"]=pd.to_datetime(dp["planbegin"],errors="coerce"); dp["los_days"]=dp["icu_duration_h"]/24.0
dp=dp.merge(build_op(dp,"fallnr",OPAN_P.as_posix(),OPZ_P,"ASAID","indivdauer_min","opplandatum_dt","DAUER","ZEITINTBEZ","BEGINN","FALLNR","FALLNR"),on="stay_id",how="left")
for c in OP_COLS: dp[c]=pd.to_numeric(dp[c],errors="coerce").fillna(0.0)
Xp=build_X(dp)
for c in TRAIN_COLS:
    if c not in Xp.columns: Xp[c]=0.0
Xp=Xp[TRAIN_COLS].astype(np.float64).values
sen=pd.read_csv(SENIOR,sep=";"); dp["stay_id"]=dp["stay_id"].astype(str); sen["tages_stay_id"]=sen["tages_stay_id"].astype(str)
mg=dp.merge(sen,left_on="stay_id",right_on="tages_stay_id",how="inner")
mg["los_obs"]=pd.to_numeric(mg["los_days"],errors="coerce"); mg["arzt"]=pd.to_numeric(mg["best_senior_estimate_days"],errors="coerce")
keep=mg.dropna(subset=["los_obs","arzt"]).index
mg=mg.loc[keep].reset_index(drop=True)
Xpm=Xp[mg.index.values] if len(mg)==len(Xp) else None
# Index-Mapping: baue Xp fuer gematchte Zeilen erneut sauber
mask_match=dp["stay_id"].isin(mg["stay_id"]).values
# robust: Vorhersagen fuer ganze prospektive Kohorte, dann mergen
predP={t:predict(t,Xp) for t in fitted}
E_p,med_p=hazard_predict(Xp)
predP["Hazard_E"]=E_p; predP["Hazard_Median"]=med_p
for t,p in predP.items(): dp[f"pp_{t}"]=p
mg=dp.merge(sen,left_on="stay_id",right_on="tages_stay_id",how="inner")
mg["los_obs"]=pd.to_numeric(mg["los_days"],errors="coerce"); mg["arzt"]=pd.to_numeric(mg["best_senior_estimate_days"],errors="coerce")
mg=mg.dropna(subset=["los_obs","arzt"]).reset_index(drop=True)
print(f"  Match {len(mg)} Stays")
yt=mg["los_obs"].values
subs_p={"gesamt":np.ones(len(mg),bool),"1-7d":(yt>1)&(yt<=7),">7d":yt>7,">14d":yt>14}
rows=[]
for sg,mask in subs_p.items():
    if mask.sum()<5: continue
    rows.append({**metrics(yt,mg["arzt"].values,"Oberarzt",mask),"Subgruppe":sg})
    for t in list(fitted)+["Hazard_E","Hazard_Median"]:
        rows.append({**metrics(yt,mg[f"pp_{t}"].values,t,mask),"Subgruppe":sg})
pros=pd.DataFrame(rows)[["Subgruppe","Modell","n","MAE","Median_AE","RMSE","R2","Bias"]]
print(pros.to_string(index=False))
pros.to_csv(f"{OUT}_prospektiv.csv",sep=";",index=False)
print(f"\nGespeichert: {OUT}_retro.csv / {OUT}_prospektiv.csv")
