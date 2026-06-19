# -*- coding: utf-8 -*-
"""
Quantilregression (P50/P80) + OP-Features  ->  prospektiver Oberarzt-Vergleich.

- Training: retrospektive 24h-Kohorte + OP-Features (op_an, op_zeitintervalle via fallid)
- 3 XGBoost-Modelle: Mean (reg:squarederror), P50 und P80 (reg:quantileerror)
- Anwendung auf prospektive Kohorte + OP-Features (via fallnr; OLD-Snapshots)
- Head-to-Head gegen best_senior_estimate_days (359 Stays), Subgruppen, Wilcoxon
- P80-Coverage als Kapazitaetsplanungs-Metrik
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE   = Path(r"D:\Ausgangsdaten\KISIK Projekt")
AN     = BASE / "Eigene Auswertung"
RETRO  = BASE / "kisik2" / "kisik2_icu_ml_dataset_24h.parquet"
PROS   = BASE / "kisik2" / "kisik2_prospektiv_ml_dataset.parquet"
OPAN_R = BASE / "kisik2" / "op_an.csv"
OPZ_R  = BASE / "kisik2" / "op_zeitintervalle.csv"
OPAN_P = AN / "oldlive_kisik2_core_old_op_an_filtered_dedup.csv"
OPZ_P  = "D:/Ausgangsdaten/Live-Daten/OLD/*/op_zeitintervalle.csv"
SENIOR = AN / "los_senior_estimates_tagesausleitung_stay_level.csv"
FEAT   = AN / "los_selected_features_ain_24h_compact.csv"
OUT    = AN / "los_quantile_op_prospective"
RS = 42

allowed = [("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31"),("AIN","IZ01"),
           ("AUG","IZ01"),("AVT","IZ01"),("GCH","IZ01"),("GYN","IZ01"),
           ("HNO","IZ01"),("HTC","IZ01"),("IZPV","IZ01"),("MKG","IZ01"),
           ("NCH","IZ01"),("NUK","IZ01"),("STR","IZ01"),("UCH","IZ01"),("URO","IZ01")]
allowed_sql = ", ".join(f"('{w}','{o}')" for w,o in allowed)

con = duckdb.connect()
def rc(p): return f"read_csv_auto('{p}', delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true)"
MIN = lambda col: f"COALESCE(EXTRACT(EPOCH FROM TRY_CAST({col} AS INTERVAL))/60.0, 0)"
# formatunabhaengiges Timestamp-Parsing (ISO retro + deutsch DD.MM.YYYY prospektiv)
TS  = lambda col: f"COALESCE(TRY_CAST({col} AS TIMESTAMP), TRY_STRPTIME({col}, '%d.%m.%Y %H:%M:%S'))"

OP_COLS = ["op_schnittnaht_min","op_anaesth_min","op_anaesth_praesenz_min","op_hlm_min",
           "op_hlm_flag","op_n_intervalle","op_plandauer_sum","op_asa_max","op_n_eingriffe","op_any"]

# ----------------------------------------------------------------------
def build_op(stays_df, idcol, opan, opz, asa_col, plan_col, opan_date,
             z_dauer, z_zeit, z_beginn, z_fallid):
    """OP-Features fuer eine Kohorte; perioperatives Fenster [pb-1d, pb+24h]."""
    con.register("st", stays_df[["stay_id", idcol, "planbegin"]].rename(columns={idcol:"idc"}))
    con.execute("CREATE OR REPLACE TEMP TABLE s AS SELECT stay_id, CAST(idc AS VARCHAR) idc, "
                "CAST(planbegin AS TIMESTAMP) pb FROM st")
    # op_zeitintervalle
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
          COUNT(*) op_n_intervalle
        FROM zw GROUP BY stay_id""").df()
    # op_an
    opan_df = con.execute(f"""
        WITH a AS (SELECT DISTINCT CAST({z_fallid_an} AS VARCHAR) idc, TRY_CAST({asa_col} AS INT) asa,
                          TRY_CAST({plan_col} AS DOUBLE) plan, {opan_date} dt FROM {rc(opan)}),
        aw AS (SELECT s.stay_id, a.asa, a.plan FROM s JOIN a ON a.idc=s.idc
               WHERE {TS('a.dt')} BETWEEN s.pb - INTERVAL 1 DAY AND s.pb + INTERVAL 24 HOURS)
        SELECT stay_id, MAX(asa) op_asa_max, SUM(plan) op_plandauer_sum, COUNT(*) op_n_eingriffe
        FROM aw GROUP BY stay_id""").df()
    op = pd.merge(opz_df, opan_df, on="stay_id", how="outer")
    op["op_any"] = (op["op_n_intervalle"].notna() | op["op_n_eingriffe"].notna()).astype(int)
    for c in OP_COLS:
        if c not in op.columns: op[c] = 0.0
        op[c] = pd.to_numeric(op[c], errors="coerce").fillna(0.0)
    return op[["stay_id"]+OP_COLS]

# ----------------------------------------------------------------------
print("=== Retrospektives Training ===")
df = con.execute(f"""SELECT * FROM read_parquet('{RETRO.as_posix()}')
    WHERE (wardshort,oebenekurz) IN ({allowed_sql}) AND icu_duration_h/24.0 > 1""").df()
df["planbegin"] = pd.to_datetime(df["planbegin"], errors="coerce")
df["los_days"]  = df["icu_duration_h"]/24.0
print(f"  {len(df):,} Stays")

z_fallid_an = "fallid"
op_r = build_op(df, "fallid", OPAN_R.as_posix(), OPZ_R.as_posix(),
                "asaid","indivdauer","opplandatum","dauer","zeitintbez","beginn","fallid")
df = df.merge(op_r, on="stay_id", how="left")
for c in OP_COLS: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
print(f"  OP retro: {int((df['op_any']==1).sum()):,} Stays mit OP, {int(df['op_hlm_flag'].sum()):,} HLM")

# Feature-Matrix
feat = pd.read_csv(FEAT, sep=";")["Feature"].tolist()
present = [f for f in feat if f in df.columns]
cat = [c for c in present if c == "oebenekurz"]
num = [c for c in present if c not in cat]
TRAIN_COLS = None
def build_X(frame, use_op=True):
    global TRAIN_COLS
    parts = [frame.reindex(columns=num).apply(pd.to_numeric, errors="coerce")]
    if cat and all(c in frame.columns for c in cat):
        parts.append(pd.get_dummies(frame[cat].astype(str), prefix=cat).astype(float))
    if use_op: parts.append(frame.reindex(columns=OP_COLS).apply(pd.to_numeric, errors="coerce"))
    X = pd.concat(parts, axis=1); X.columns=[str(c) for c in X.columns]
    X = X.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    return X

X = build_X(df); TRAIN_COLS = X.columns.tolist()
y = np.log1p(df["los_days"].clip(lower=0).values)
groups = df["pid"].fillna("unknown").astype(str).values
tr,te = next(GroupShuffleSplit(1, test_size=0.2, random_state=RS).split(df,y,groups))
print(f"  Train {len(tr):,} | Test {len(te):,}")

COMMON = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=3, random_state=RS, n_jobs=-1)
models = {
    "Mean": XGBRegressor(objective="reg:squarederror", **COMMON),
    "P50":  XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **COMMON),
    "P80":  XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.8, **COMMON),
}
for name,m in models.items():
    m.fit(X.values[tr], y[tr]); print(f"  trainiert: {name}")

def metrics(yt, yp, label, sub=None):
    if sub is not None: yt,yp = yt[sub],yp[sub]
    ae=np.abs(yt-yp)
    return {"Modell":label,"n":len(yt),"MAE":round(float(ae.mean()),3),
            "Median_AE":round(float(np.median(ae)),3),
            "RMSE":round(float(np.sqrt(mean_squared_error(yt,yp))),3),
            "R2":round(float(r2_score(yt,yp)),3),"Bias":round(float((yp-yt).mean()),3)}

# Retro-Holdout
los_te = df["los_days"].values[te]
print("\n--- Retrospektiver Holdout ---")
retro_rows=[]
for name,m in models.items():
    pred=np.clip(np.expm1(m.predict(X.values[te])),0,None)
    retro_rows.append(metrics(los_te,pred,name))
print(pd.DataFrame(retro_rows).to_string(index=False))

# ----------------------------------------------------------------------
print("\n=== Prospektive Anwendung ===")
dp = con.execute(f"SELECT * FROM read_parquet('{PROS.as_posix()}')").df()
dp["planbegin"] = pd.to_datetime(dp["planbegin"], errors="coerce")
dp["los_days"]  = dp["icu_duration_h"]/24.0
print(f"  {len(dp):,} prospektive Stays")

z_fallid_an = "FALLNR"
op_p = build_op(dp, "fallnr", OPAN_P.as_posix(), OPZ_P,
                "ASAID","indivdauer_min","opplandatum_dt","DAUER","ZEITINTBEZ","BEGINN","FALLNR")
dp = dp.merge(op_p, on="stay_id", how="left")
for c in OP_COLS: dp[c] = pd.to_numeric(dp[c], errors="coerce").fillna(0.0)
print(f"  OP prospektiv: {int((dp['op_any']==1).sum()):,} Stays mit OP, "
      f"{int((dp['op_n_intervalle']>0).sum()):,} mit op_zeitintervalle, "
      f"{int(dp['op_hlm_flag'].sum()):,} HLM")

# Spalten-Alignment
Xp = build_X(dp)
for c in TRAIN_COLS:
    if c not in Xp.columns: Xp[c]=0.0
Xp = Xp[TRAIN_COLS].astype(np.float64)
for name,m in models.items():
    dp[f"pred_{name}"] = np.clip(np.expm1(m.predict(Xp.values)),0,None)

# Senior-Match
sen = pd.read_csv(SENIOR, sep=";")
dp["stay_id"]=dp["stay_id"].astype(str); sen["tages_stay_id"]=sen["tages_stay_id"].astype(str)
mg = dp.merge(sen, left_on="stay_id", right_on="tages_stay_id", how="inner")
mg["los_obs"]=pd.to_numeric(mg["los_days"],errors="coerce")
mg["arzt"]=pd.to_numeric(mg["best_senior_estimate_days"],errors="coerce")
mg=mg.dropna(subset=["los_obs","arzt"])
print(f"  Match: {len(mg)} Stays | mit OP-Info: {int((mg['op_any']==1).sum())}")

# ---------------- Head-to-Head (Punktschaetzer P50 & Mean vs Oberarzt) -------------
print("\n--- Prospektiver Head-to-Head (Tage) ---")
yt=mg["los_obs"].values
subs={"gesamt":np.ones(len(mg),bool),"1-7d":(yt>1)&(yt<=7),">7d":yt>7,">14d":yt>14}
rows=[]
for sgn,mask in subs.items():
    if mask.sum()<5: continue
    for label,col in [("Oberarzt","arzt"),("ML Mean","pred_Mean"),("ML P50","pred_P50")]:
        rows.append({**metrics(yt,mg[col].values,label,mask),"Subgruppe":sgn})
bench=pd.DataFrame(rows)[["Subgruppe","Modell","n","MAE","Median_AE","RMSE","R2","Bias"]]
print(bench.to_string(index=False))

# ---------------- Wilcoxon (P50/Mean vs Oberarzt) ----------------
print("\n--- Wilcoxon (|Fehler| ML vs Oberarzt) ---")
wrows=[]
for label,col in [("ML Mean","pred_Mean"),("ML P50","pred_P50")]:
    ae_ml=np.abs(yt-mg[col].values); ae_a=np.abs(yt-mg["arzt"].values)
    w,p=stats.wilcoxon(ae_ml,ae_a)
    better=int((ae_ml<ae_a).sum())
    wrows.append({"Vergleich":f"{label} vs Oberarzt","Median_AE_ML":round(float(np.median(ae_ml)),2),
                  "Median_AE_Arzt":round(float(np.median(ae_a)),2),"p":f"{p:.4f}",
                  "ML_besser_%":round(100*better/len(mg),1)})
print(pd.DataFrame(wrows).to_string(index=False))

# ---------------- P80-Coverage (Kapazitaetsplanung) ----------------
print("\n--- P80-Coverage: Anteil Stays mit beobachteter LoS <= Vorhersage ---")
covrows=[]
for label,col in [("Oberarzt","arzt"),("ML Mean","pred_Mean"),("ML P50","pred_P50"),("ML P80","pred_P80")]:
    for sgn,mask in subs.items():
        if mask.sum()<5: continue
        cov=float((yt[mask]<=mg[col].values[mask]).mean())
        covrows.append({"Schaetzer":label,"Subgruppe":sgn,"Coverage_%":round(100*cov,1),
                        "Mittel_Vorhersage":round(float(mg[col].values[mask].mean()),2),
                        "Mittel_beob":round(float(yt[mask].mean()),2)})
cov=pd.DataFrame(covrows)
print(cov.to_string(index=False))

# Exporte
bench.to_csv(f"{OUT}_headtohead.csv",sep=";",index=False)
pd.DataFrame(wrows).to_csv(f"{OUT}_wilcoxon.csv",sep=";",index=False)
cov.to_csv(f"{OUT}_p80_coverage.csv",sep=";",index=False)
pd.DataFrame(retro_rows).to_csv(f"{OUT}_retro_holdout.csv",sep=";",index=False)
mg[["stay_id","los_obs","arzt","pred_Mean","pred_P50","pred_P80","op_any","op_hlm_flag"]].to_csv(f"{OUT}_predictions.csv",sep=";",index=False)
print(f"\nGespeichert unter Prefix: {OUT}")
