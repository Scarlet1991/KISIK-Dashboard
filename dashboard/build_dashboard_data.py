# -*- coding: utf-8 -*-
"""Erzeugt JSON fuer das interaktive Stations-Dashboard (1 Tag, AIN/IZ32).
Pro Patient: vorhergesagte LoS + per-Patient-Feature-Importance (SHAP via XGBoost pred_contribs)."""
import sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import duckdb, numpy as np, pandas as pd, xgboost as xgb
from xgboost import XGBRegressor

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"
PQ=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"
FEAT=AN/"los_selected_features_ain_24h_compact.csv"
OUT=AN/"dashboard_station_data.json"
DAY=pd.Timestamp("2019-09-30"); WARD=("AIN","IZ32"); RS=42

con=duckdb.connect()
allowed=[("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31")]  # nur AIN-Intensiveinheiten IZ32/IZ21/IZ31
asql=", ".join(f"('{w}','{o}')" for w,o in allowed)
df=con.execute(f"SELECT * FROM read_parquet('{PQ.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
df["pb"]=pd.to_datetime(df["planbegin"],errors="coerce")
df["los_days"]=df["icu_duration_h"]/24.0
df["end"]=df["pb"]+pd.to_timedelta(df["icu_duration_h"],unit="h")

feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns]; cat=[c for c in present if c=="oebenekurz"]; num=[c for c in present if c not in cat]
def build_X(frame):
    parts=[frame.reindex(columns=num).apply(pd.to_numeric,errors="coerce")]
    if cat: parts.append(pd.get_dummies(frame[cat].astype(str),prefix=cat).astype(float))
    X=pd.concat(parts,axis=1); X.columns=[str(c) for c in X.columns]
    return X.apply(pd.to_numeric,errors="coerce").astype(np.float64)
X=build_X(df); y=np.log1p(df["los_days"].clip(lower=0).values)
model=XGBRegressor(objective="reg:squarederror",n_estimators=500,max_depth=6,learning_rate=0.05,
                   subsample=0.8,colsample_bytree=0.8,min_child_weight=3,random_state=RS,n_jobs=-1)
model.fit(X.values,y)
booster=model.get_booster(); FEATURES=X.columns.tolist()

# --- Stationsbelegung am DAY ---
ain=df[(df["wardshort"]==WARD[0])&(df["oebenekurz"]==WARD[1])].copy()
day_end=DAY+pd.Timedelta(days=1)
present_mask=(ain["pb"]<=day_end)&(ain["end"]>=DAY)
cohort=ain[present_mask].copy().reset_index(drop=True)
print(f"Patienten am {DAY.date()} auf {WARD}: {len(cohort)}")

Xc=build_X(cohort).reindex(columns=FEATURES).fillna(0.0)
dm=xgb.DMatrix(Xc.values, feature_names=FEATURES)
pred_log=booster.predict(dm)
pred_los=np.expm1(pred_log)
contribs=booster.predict(dm, pred_contribs=True)  # (n, F+1); letzte Spalte=bias

# --- globale Feature-Importance (gain) ---
imp=pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

# --- Humanisierung der Feature-Namen ---
ICD={"i25_13":"KHK mit Myokardinfarkt","i21_4":"Akuter NSTEMI","i35_0":"Aortenklappenstenose",
 "i20_8":"Angina pectoris","t81_8":"OP-Komplikation","s06_5":"Subdurale Blutung",
 "j12_8":"Virale Pneumonie","t81_0":"Nachblutung (post-OP)","t81_4":"Wundinfektion (post-OP)",
 "i34_0":"Mitralklappeninsuffizienz","s06_6":"Subarachnoidalblutung (traum.)","i71_01":"Aortendissektion Typ A",
 "i61_0":"Intrazerebrale Blutung","a41_9":"Sepsis","t81_3":"Wunddehiszenz","k55_0":"Darmischämie",
 "g91_8":"Hydrozephalus","k63_1":"Darmperforation","s06_21":"Diffuse Hirnverletzung",
 "g91_0":"Hydrozephalus","z99_1":"Beatmungsabhängigkeit","i71_4":"Bauchaortenaneurysma",
 "i33_0":"Endokarditis","i25_12":"KHK mit Bypass"}
OPS={"8_98f_0":"Intensivmed. Komplexbehandlung","8_931_0":"Monitoring (erweitert)","8_930":"Monitoring",
 "8_831_0":"Zentralvenöser Katheter","3_200":"CT Schädel","8_98f_10":"Intensivmed. Komplexbeh. (lang)",
 "8_800_c0":"Transfusion","3_990":"Bildgebung","8_701":"Intubation/Beatmung","8_706":"Maschinelle Beatmung",
 "8_98f_11":"Intensivmed. Komplexbeh. (sehr lang)","8_924":"Monitoring kardial"}
STAT={"first":"erster","mean":"Mittel","min":"Min","max":"Max","last":"letzter","median":"Median","count":"Anzahl"}
def human(f):
    if f.startswith("lab24_"):
        body=f[len("lab24_"):]; *name,st=body.split("_")
        return f"Labor: {' '.join(name).replace('_',' ')} ({STAT.get(st,st)})"
    if f.startswith("vital24_spo2"):
        st=f.split("_")[-1]; return f"SpO₂ ({STAT.get(st,st)}, 24h)"
    if f.startswith("diag_main_"):
        code=f[len("diag_main_"):]; return f"Diagnose: {ICD.get(code,code.upper().replace('_','.'))}"
    if f.startswith("proc24_"):
        code=f[len("proc24_"):]; return f"Prozedur: {OPS.get(code,code.replace('_','-'))}"
    if f.startswith("zugang24_"):
        return "Zugang: "+f[len("zugang24_"):].replace("_"," ")
    base={"alter":"Alter","stay_nr":"ICU-Aufenthalt-Nr.","admission_hour":"Aufnahme-Stunde",
          "admission_weekday":"Aufnahme-Wochentag","admission_month":"Aufnahme-Monat",
          "proc24_anzahl_gesamt":"Prozeduren (Anzahl 24h)","zugang24_anzahl_gesamt":"Zugänge (Anzahl 24h)"}
    return base.get(f, f)

# --- pro Patient JSON ---
patients=[]
for i,(_,row) in enumerate(cohort.iterrows()):
    c=contribs[i][:-1]  # ohne bias
    order=np.argsort(-np.abs(c))[:7]
    top=[{"feature":human(FEATURES[j]),
          "raw":FEATURES[j],
          "value":(None if pd.isna(Xc.iloc[i,j]) else round(float(Xc.iloc[i,j]),2)),
          "contrib":round(float(c[j]),3)} for j in order]
    dos=int((DAY-row["pb"].normalize()).days)+1
    age=row.get("alter")
    patients.append({
        "bed": f"Bett {i+1:02d}",
        "age": None if pd.isna(age) else int(float(age)),
        "day_of_stay": max(dos,1),
        "pred_los": round(float(pred_los[i]),1),
        "obs_los": round(float(row["los_days"]),1),
        "long_stayer": bool(pred_los[i] > 7),
        "top_features": top,
    })
# nach vorhergesagter LoS sortieren (laengste zuerst)
patients=sorted(patients, key=lambda p:-p["pred_los"])
for k,p in enumerate(patients,1): p["bed"]=f"Bett {k:02d}"

out={
  "date": str(DAY.date()),
  "ward": f"{WARD[0]} / {WARD[1]}",
  "n_patients": len(patients),
  "model": "XGBoost (log1p), 24h-Features",
  "global_importance":[{"feature":human(f),"gain":round(float(g),4)} for f,g in imp.head(12).items()],
  "patients": patients,
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Gespeichert: {OUT}  ({len(patients)} Patienten)")
print("Beispiel-Patient:", json.dumps(patients[0], ensure_ascii=False)[:300])
