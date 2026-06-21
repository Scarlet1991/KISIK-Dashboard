# -*- coding: utf-8 -*-
"""
MIMIC-IV 3.1 — externe Validierung der KISIK-Methodik.
Schritt 1: First-24h-Feature-Matrix bauen (analog zu KISIK, MIMIC-native Features).
- Kohorte: ICU-Stays mit LOS > 1 Tag (24h-Landmark), Outcome = icustays.los (Tage).
- Features NUR aus den ersten 24 h ab ICU-intime:
    * Labore (labevents): Top-30 Itemids, je first/last/min/max/mean/count
    * Vitalparameter (chartevents): kuratierte Vital-Itemids, je first/last/min/max/mean/count
    * Prozeduren (procedureevents): Top-15 Itemids als Binär-Präsenz + Gesamtzahl
    * Demografie/Kontext: Alter, Geschlecht, Aufnahmetyp, ICU-Care-Unit, Stay-Nr
- KEINE Entlass-kodierten Diagnosen (diagnoses_icd wird erst bei Entlassung kodiert -> Leakage).
Schreibt: mimic_features.parquet + mimic_feature_dict.csv
"""
import sys, io, time, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, pandas as pd, numpy as np

M   = Path(r"F:\Mimic\mimic-iv-3.1\mimic-iv-3.1")
OUT = Path(r"D:\Ausgangsdaten\KISIK Projekt\mimic_external"); OUT.mkdir(parents=True, exist_ok=True)
ICU, HOSP = M/"icu", M/"hosp"
def src(p): return f"read_csv_auto('{p.as_posix()}')"
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

con = duckdb.connect(database=str(OUT/"mimic_build.duckdb"))
con.execute("PRAGMA threads=4; PRAGMA memory_limit='6GB';")
con.execute(f"PRAGMA temp_directory='{(OUT/'duck_tmp').as_posix()}';")

# ---------------------------------------------------------------- Kohorte (LOS>1d) + Stay-Nr
log("Kohorte ...")
con.execute(f"""CREATE OR REPLACE TABLE stays AS
  SELECT subject_id, hadm_id, stay_id, first_careunit,
         CAST(intime AS TIMESTAMP) intime, CAST(outtime AS TIMESTAMP) outtime, los,
         row_number() OVER (PARTITION BY subject_id ORDER BY intime) AS stay_nr
  FROM {src(ICU/'icustays.csv.gz')} WHERE los > 1""")
n=con.execute("SELECT count(*) FROM stays").fetchone()[0]
log(f"  Stays (LOS>1d): {n:,}")

# ---------------------------------------------------------------- Demografie/Kontext
log("Demografie ...")
con.execute(f"""CREATE OR REPLACE TABLE demo AS
  SELECT s.stay_id, s.subject_id, s.los AS los_days, s.first_careunit, s.stay_nr,
         p.anchor_age AS age, p.gender, a.admission_type
  FROM stays s
  LEFT JOIN {src(HOSP/'patients.csv.gz')}   p ON p.subject_id=s.subject_id
  LEFT JOIN {src(HOSP/'admissions.csv.gz')} a ON a.hadm_id=s.hadm_id""")

# ---------------------------------------------------------------- Vitalparameter (chartevents, 3.3GB) — ein Scan
VITALS={220045:"heart_rate",220210:"resp_rate",220277:"spo2",223761:"temp_f",223762:"temp_c",
        220179:"nbp_sys",220180:"nbp_dia",220181:"nbp_mean",220050:"abp_sys",220051:"abp_dia",
        220052:"abp_mean",220074:"cvp"}
vid=",".join(map(str,VITALS))
log("Vitalparameter (chartevents-Scan, kann einige Minuten dauern) ...")
con.execute(f"""CREATE OR REPLACE TABLE vit AS
  SELECT c.stay_id, c.itemid,
         arg_min(c.valuenum,c.charttime) AS first, arg_max(c.valuenum,c.charttime) AS last,
         min(c.valuenum) AS min, max(c.valuenum) AS max, avg(c.valuenum) AS mean, count(c.valuenum) AS cnt
  FROM {src(ICU/'chartevents.csv.gz')} c JOIN stays s ON s.stay_id=c.stay_id
  WHERE c.itemid IN ({vid}) AND c.valuenum IS NOT NULL
    AND c.charttime BETWEEN s.intime AND s.intime + INTERVAL 24 HOUR
  GROUP BY 1,2""")
log(f"  Vital-Zeilen (stay×item): {con.execute('SELECT count(*) FROM vit').fetchone()[0]:,}")

# ---------------------------------------------------------------- Labore (labevents, 2.5GB) — ein Scan -> gefiltertes Long
log("Labore (labevents-Scan) ...")
con.execute(f"""CREATE OR REPLACE TABLE lab_long AS
  SELECT s.stay_id, l.itemid, l.valuenum, CAST(l.charttime AS TIMESTAMP) charttime
  FROM {src(HOSP/'labevents.csv.gz')} l JOIN stays s ON s.hadm_id=l.hadm_id
  WHERE l.valuenum IS NOT NULL
    AND CAST(l.charttime AS TIMESTAMP) BETWEEN s.intime AND s.intime + INTERVAL 24 HOUR""")
top_lab=con.execute("""SELECT itemid, count(DISTINCT stay_id) ns FROM lab_long
                       GROUP BY 1 ORDER BY ns DESC LIMIT 30""").df()
lab_ids=top_lab.itemid.tolist()
log(f"  Top-30 Labor-Itemids gewählt (häufigstes deckt {int(top_lab.ns.iloc[0]):,} Stays)")
con.execute(f"""CREATE OR REPLACE TABLE lab AS
  SELECT stay_id, itemid,
         arg_min(valuenum,charttime) AS first, arg_max(valuenum,charttime) AS last,
         min(valuenum) AS min, max(valuenum) AS max, avg(valuenum) AS mean, count(valuenum) AS cnt
  FROM lab_long WHERE itemid IN ({','.join(map(str,lab_ids))}) GROUP BY 1,2""")

# ---------------------------------------------------------------- Prozeduren (procedureevents) — Top-15 binär + Gesamtzahl
log("Prozeduren ...")
con.execute(f"""CREATE OR REPLACE TABLE proc_long AS
  SELECT s.stay_id, p.itemid
  FROM {src(ICU/'procedureevents.csv.gz')} p JOIN stays s ON s.stay_id=p.stay_id
  WHERE CAST(p.starttime AS TIMESTAMP) BETWEEN s.intime AND s.intime + INTERVAL 24 HOUR""")
top_proc=con.execute("""SELECT itemid, count(DISTINCT stay_id) ns FROM proc_long
                        GROUP BY 1 ORDER BY ns DESC LIMIT 15""").df()
proc_ids=top_proc.itemid.tolist()

# ---------------------------------------------------------------- Label-Wörterbücher
ditems =con.execute(f"SELECT itemid,label FROM {src(ICU/'d_items.csv.gz')}").df().set_index("itemid")["label"].to_dict()
dlab   =con.execute(f"SELECT itemid,label FROM {src(HOSP/'d_labitems.csv.gz')}").df().set_index("itemid")["label"].to_dict()

# ---------------------------------------------------------------- Pivot in pandas und zusammenführen
log("Pivot + Assemble ...")
demo=con.execute("SELECT * FROM demo").df()
def pivot(tbl, prefix, labmap):
    df=con.execute(f"SELECT * FROM {tbl}").df()
    if df.empty: return pd.DataFrame(index=demo["stay_id"])
    wide=df.pivot_table(index="stay_id", columns="itemid",
                        values=["first","last","min","max","mean","cnt"], aggfunc="first")
    wide.columns=[f"{prefix}24_{itemid}_{agg}" for agg,itemid in wide.columns]
    return wide
vit_w=pivot("vit","vital",ditems)
lab_w=pivot("lab","lab",dlab)
# Prozeduren binär + Gesamtzahl
pl=con.execute("SELECT stay_id,itemid FROM proc_long").df()
proc_w=pd.DataFrame(index=demo["stay_id"].values)
for iid in proc_ids:
    present=set(pl.loc[pl.itemid==iid,"stay_id"])
    proc_w[f"proc24_{iid}"]=proc_w.index.to_series().isin(present).astype("uint8")
proc_cnt=pl.groupby("stay_id").size().rename("proc24_total_count")
proc_w=proc_w.join(proc_cnt).fillna({"proc24_total_count":0})

X=demo.set_index("stay_id").join([vit_w,lab_w,proc_w])
X.to_parquet(OUT/"mimic_features.parquet")
log(f"  Feature-Matrix: {X.shape[0]:,} Stays × {X.shape[1]} Spalten -> mimic_features.parquet")

# ---------------------------------------------------------------- Feature-Wörterbuch
rows=[]
for iid in VITALS: rows.append({"prefix":"vital","itemid":iid,"label":ditems.get(iid,VITALS[iid])})
for iid in lab_ids: rows.append({"prefix":"lab","itemid":iid,"label":dlab.get(iid,str(iid))})
for iid in proc_ids: rows.append({"prefix":"proc","itemid":iid,"label":ditems.get(iid,str(iid))})
pd.DataFrame(rows).to_csv(OUT/"mimic_feature_dict.csv",sep=";",index=False)
log("Fertig. mimic_features.parquet + mimic_feature_dict.csv geschrieben.")
con.close()
