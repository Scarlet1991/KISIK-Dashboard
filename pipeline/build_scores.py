# -*- coding: utf-8 -*-
"""Baut 24h-gefensterte Schwere-Score-Features (SAPS II, TISS-28, SOFA) fuer
retrospektive (fallid) UND prospektive (fallnr, OLD) Kohorte aus score.csv.
Misst Abdeckung und speichert die Features zur Wiederverwendung."""
import sys, io, os, glob, warnings; warnings.filterwarnings("ignore")
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
from pathlib import Path
import duckdb, pandas as pd, numpy as np
BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"
PROS =BASE/"kisik2"/"kisik2_prospektiv_ml_dataset.parquet"
SCORE_R=BASE/"kisik2"/"score.csv"
OLD="D:/Ausgangsdaten/Live-Daten/OLD"
OUT=AN/"canonical"; OUT.mkdir(exist_ok=True)
allowed=[("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31"),("AIN","IZ01"),("AUG","IZ01"),("AVT","IZ01"),("GCH","IZ01"),("GYN","IZ01"),("HNO","IZ01"),("HTC","IZ01"),("IZPV","IZ01"),("MKG","IZ01"),("NCH","IZ01"),("NUK","IZ01"),("STR","IZ01"),("UCH","IZ01"),("URO","IZ01")]
asql=", ".join(f"('{w}','{o}')" for w,o in allowed)
con=duckdb.connect()
TS=lambda c: f"COALESCE(TRY_CAST({c} AS TIMESTAMP), TRY_STRPTIME({c}, '%d.%m.%Y %H:%M:%S'))"
def rc(p): return f"read_csv_auto('{p}', delim=';', header=true, all_varchar=true, ignore_errors=true)"
def rcl(name):
    fs=[]
    for p in glob.glob(f"{OLD}/*/{name}"):
        try:
            if os.path.getsize(p)<30: continue
            with open(p,"r",encoding="utf-8",errors="ignore") as fh: h=fh.readline()
            if "FALLNR" in h.upper(): fs.append(p.replace("\\","/"))
        except: pass
    return f"read_csv_auto([{','.join(chr(39)+p+chr(39) for p in fs)}], delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true)"

def build(stays_df, idcol, score_sql, von_col, kurz_col, erg_col, fall_col):
    con.register("st",stays_df[["stay_id",idcol,"planbegin"]].rename(columns={idcol:"idc"}))
    con.execute("CREATE OR REPLACE TEMP TABLE s AS SELECT stay_id, CAST(idc AS VARCHAR) idc, CAST(planbegin AS TIMESTAMP) pb FROM st")
    q=f"""
    WITH sc AS (
      SELECT s.stay_id,
        CASE WHEN UPPER({kurz_col}) LIKE 'SAPS%' THEN 'saps'
             WHEN UPPER({kurz_col}) LIKE 'TISS%' THEN 'tiss'
             WHEN UPPER({kurz_col}) LIKE 'SOFA%' THEN 'sofa' END typ,
        TRY_CAST(REPLACE({erg_col},',','.') AS DOUBLE) val,
        {TS(von_col)} ts
      FROM {score_sql} z JOIN s ON CAST(z.{fall_col} AS VARCHAR)=s.idc
      WHERE {TS(von_col)} BETWEEN s.pb AND s.pb + INTERVAL 24 HOURS )
    SELECT stay_id,
      MAX(CASE WHEN typ='saps' THEN val END) AS score_saps_max,
      MAX(CASE WHEN typ='tiss' THEN val END) AS score_tiss_max,
      MAX(CASE WHEN typ='sofa' THEN val END) AS score_sofa_max,
      ARG_MIN(CASE WHEN typ='saps' THEN val END, CASE WHEN typ='saps' THEN ts END) AS score_saps_first,
      ARG_MIN(CASE WHEN typ='tiss' THEN val END, CASE WHEN typ='tiss' THEN ts END) AS score_tiss_first,
      ARG_MIN(CASE WHEN typ='sofa' THEN val END, CASE WHEN typ='sofa' THEN ts END) AS score_sofa_first
    FROM sc WHERE typ IS NOT NULL AND val IS NOT NULL AND ts IS NOT NULL GROUP BY stay_id"""
    return con.execute(q).df()

def cohort(p):
    d=con.execute(f"SELECT * FROM read_parquet('{p}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
    d["planbegin"]=pd.to_datetime(d["planbegin"],errors="coerce"); return d

print("Retro ...")
R=cohort(RETRO.as_posix())
Rs=build(R[["stay_id","fallid","planbegin"]],"fallid",rc(SCORE_R.as_posix()),"von","kurzbez","scoreergebnis","fallid")
print("Prospektiv ...")
P=cohort(PROS.as_posix())
Ps=build(P[["stay_id","fallnr","planbegin"]],"fallnr",rcl("score.csv"),"VON","KURZBEZ","SCOREERGEBNIS","FALLNR")

Rs.to_csv(OUT/"scores24_retro.csv",sep=";",index=False)
Ps.to_csv(OUT/"scores24_prospektiv.csv",sep=";",index=False)

print("\n=== ABDECKUNG (Anteil Stays mit Score in den ersten 24h) ===")
print(f"{'Score':14}{'Retro (n='+str(len(R))+')':>22}{'Prosp (n='+str(len(P))+')':>22}")
def cov(d,full,c):
    if c not in d.columns: return 0.0
    return 100*pd.to_numeric(d[c],errors="coerce").notna().sum()/len(full)
for s in ["saps","tiss","sofa"]:
    c=f"score_{s}_first"
    print(f"{s.upper():14}{cov(Rs,R,c):>20.1f}%{cov(Ps,P,c):>20.1f}%")
print("\nMediane (first, vorhandene):")
for s in ["saps","tiss","sofa"]:
    rv=pd.to_numeric(Rs.get(f'score_{s}_first'),errors='coerce').dropna()
    pv=pd.to_numeric(Ps.get(f'score_{s}_first'),errors='coerce').dropna()
    print(f"  {s.upper():6} retro {('%.1f'%rv.median()) if len(rv) else '--':>6} (n={len(rv)})  | prosp {('%.1f'%pv.median()) if len(pv) else '--':>6} (n={len(pv)})")
print(f"\nGespeichert: {OUT/'scores24_retro.csv'} / scores24_prospektiv.csv")
