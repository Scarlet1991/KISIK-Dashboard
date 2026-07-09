# -*- coding: utf-8 -*-
"""
Perioperative / admission-context features — RETROSPECTIVE and PROSPECTIVE in parallel.

Sources that were previously unused but are available at 24 h prospectively (audit result):
  * op_zeitintervalle.csv  -> surgery duration (Schnitt-Naht), anaesthesia time, bypass (HLM)
  * op_an.csv              -> planned surgery duration (INDIVDAUER)     [ASA is NOT usable —
                              populated retrospectively but empty in the live snapshots]
  * fall_daten.csv         -> admission type (Notfall vs elektiv)
  * aufenthalte_vorher_nachher.csv -> referring specialty (ward before the ICU stay, VOR)

Prospective 24 h coverage (n=276 cohort): OP present ~51 %, planned duration ~51 %,
admission type ~100 %, referring specialty ~51 %. Surgery times are keyed off the peri-
admission window ``[planbegin - 72 h, planbegin + 24 h]`` because the operation precedes
the ICU admission for post-operative cases.

Retro key ``fallid`` / ISO dates (single CSVs); prospective key ``fallnr`` / German dates
(daily OLD snapshots). Column names are identical on both sides.

NOTE: hard-wired local paths -> adapt CONFIG. No patient data included.
"""
from __future__ import annotations
from pathlib import Path
import duckdb
import pandas as pd

# ------------------------------------------------------------------ CONFIG ----
RETRO_DIR     = Path(r"D:\Ausgangsdaten\KISIK Projekt\kisik2")   # op_*/fall_daten/aufenthalte_*
OLD_SNAPSHOTS = Path(r"D:\Ausgangsdaten\Live-Daten\OLD")
# Referring specialties encoded as one-hot ``vorward_<code>`` (top pre-ICU wards on the cohort).
REFERRING_WARDS = ["HTC", "UCH", "AVT", "NCH", "GCH", "NRO", "HNO", "URO", "GYN", "MKG"]
PERI_BEFORE_H, PERI_AFTER_H = 72, 24     # surgery window around planbegin

_TS   = ("COALESCE(try_cast({c} AS TIMESTAMP), "
         "try_strptime({c},'%d.%m.%Y %H:%M:%S'), try_strptime({c},'%d.%m.%Y'))")
_DMIN = "epoch(try_cast({c} AS TIME)) / 60.0"     # 'HH:MM:SS' duration -> minutes


def _snapshot_files(source: str) -> str:
    def ok(p: Path) -> bool:
        try:
            with open(p, "r", encoding="latin1", errors="replace") as fh:
                return "FALLNR" in fh.readline().upper()
        except Exception:
            return False
    files = [p for p in OLD_SNAPSHOTS.glob(f"*/{source}") if p.stat().st_size > 10 and ok(p)]
    return "[" + ",".join("'" + p.as_posix() + "'" for p in files) + "]"


def _ward_cases(ward_prefix: str) -> str:
    """SQL fragment: one max(CASE ...) per referring ward -> vorward_<code> columns."""
    return "".join(
        f", max(CASE WHEN v.w = '{w}' THEN 1 ELSE 0 END) vorward_{w.lower()}"
        for w in REFERRING_WARDS)


# ----------------------------------------------------------- RETROSPECTIVE ----
def build_retro_perioperative(con: duckdb.DuckDBPyConnection, stays: pd.DataFrame) -> pd.DataFrame:
    """``stays`` must have columns (stay_id, fallid, pb=planbegin). Returns features per stay_id."""
    con.register("stays_in", stays[["stay_id", "fallid", "pb"]])
    con.execute("CREATE OR REPLACE TEMP TABLE stays AS SELECT * FROM stays_in")
    r = RETRO_DIR
    o = con.execute(f"""
        WITH z AS (SELECT trim(fallid) fallid, trim(zeitintbez) zi, {_DMIN.format(c='dauer')} dmin,
                          try_cast(op_datum AS TIMESTAMP) ts
                   FROM read_csv('{(r/'op_zeitintervalle.csv').as_posix()}', delim=';', header=true,
                        all_varchar=true, quote='"'))
        SELECT s.stay_id,
               max(dmin) FILTER (WHERE zi = 'Schnitt-Naht')        op_surgery_min,
               max(dmin) FILTER (WHERE zi LIKE 'An%sthesiezeit')   op_anaesthesia_min,
               max(dmin) FILTER (WHERE zi = 'HLM')                 op_bypass_min,
               count(*) op_n_intervals, 1 op_present
        FROM stays s JOIN z ON z.fallid = s.fallid
             AND z.ts >= s.pb - INTERVAL {PERI_BEFORE_H} HOUR AND z.ts < s.pb + INTERVAL {PERI_AFTER_H} HOUR
        GROUP BY s.stay_id""").df()
    a = con.execute(f"""
        WITH a AS (SELECT trim(fallid) fallid, try_cast(indivdauer AS DOUBLE) dur,
                          try_cast(opplandatum AS TIMESTAMP) ts
                   FROM read_csv('{(r/'op_an.csv').as_posix()}', delim=';', header=true,
                        all_varchar=true, quote='"'))
        SELECT s.stay_id, max(dur) op_planned_min FROM stays s JOIN a ON a.fallid = s.fallid
             AND a.ts >= s.pb - INTERVAL {PERI_BEFORE_H} HOUR AND a.ts < s.pb + INTERVAL {PERI_AFTER_H} HOUR
        GROUP BY s.stay_id""").df()
    f = con.execute(f"""
        WITH f AS (SELECT trim(fallid) fallid, trim(aufnahmeart) art
                   FROM read_csv('{(r/'fall_daten.csv').as_posix()}', delim=';', header=true,
                        all_varchar=true, quote='"'))
        SELECT s.stay_id, max(CASE WHEN lower(f.art) LIKE '%notfall%' THEN 1 ELSE 0 END) admission_emergency
        FROM stays s JOIN f ON f.fallid = s.fallid GROUP BY s.stay_id""").df()
    v = con.execute(f"""
        WITH v AS (SELECT DISTINCT trim(fallid) fallid, trim(wardshort) w
                   FROM read_csv('{(r/'aufenthalte_vorher_nachher.csv').as_posix()}', delim=';',
                        header=true, all_varchar=true, quote='"') WHERE trim(vor_nach) = 'V')
        SELECT s.stay_id {_ward_cases('')} FROM stays s JOIN v ON v.fallid = s.fallid GROUP BY s.stay_id""").df()
    out = (o.merge(a, on="stay_id", how="outer").merge(f, on="stay_id", how="outer")
             .merge(v, on="stay_id", how="outer"))
    out["op_present"] = out["op_present"].fillna(0)
    return out


# ------------------------------------------------------------- PROSPECTIVE ----
def build_prospective_perioperative(con: duckdb.DuckDBPyConnection, fallnrs: list[str]) -> pd.DataFrame:
    """Requires a ``ppb`` temp table (fallnr, pb) — created by build_prospective_measurements —
    or pass the cohort and let this create it. Returns features per fallnr."""
    con.register("cohort", pd.DataFrame({"fallnr": sorted(set(fallnrs))}))
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS ppb AS
        SELECT c.fallnr, min({_TS.format(c='fa.PLANBEGIN')}) pb
        FROM cohort c JOIN read_csv('{(OLD_SNAPSHOTS/'*'/'fall_aufenthalt.csv').as_posix()}',
             delim=';', header=true, all_varchar=true, union_by_name=true, ignore_errors=true) fa
          ON c.fallnr = trim(fa.FALLNR) GROUP BY c.fallnr""")
    w = PERI_BEFORE_H, PERI_AFTER_H
    o = con.execute(f"""
        WITH z AS (SELECT trim(FALLNR) fallnr, trim(ZEITINTBEZ) zi, {_DMIN.format(c='DAUER')} dmin,
                          {_TS.format(c='OP_DATUM')} ts
                   FROM read_csv({_snapshot_files('op_zeitintervalle.csv')}, delim=';', header=true,
                        all_varchar=true, union_by_name=true, ignore_errors=true))
        SELECT p.fallnr,
               max(dmin) FILTER (WHERE zi = 'Schnitt-Naht')        op_surgery_min,
               max(dmin) FILTER (WHERE zi LIKE 'An%sthesiezeit')   op_anaesthesia_min,
               max(dmin) FILTER (WHERE zi = 'HLM')                 op_bypass_min,
               count(*) op_n_intervals, 1 op_present
        FROM ppb b JOIN cohort p ON p.fallnr = b.fallnr JOIN z ON z.fallnr = p.fallnr
             AND z.ts >= b.pb - INTERVAL {w[0]} HOUR AND z.ts < b.pb + INTERVAL {w[1]} HOUR
        GROUP BY p.fallnr""").df()
    a = con.execute(f"""
        WITH a AS (SELECT trim(FALLNR) fallnr, try_cast(INDIVDAUER AS DOUBLE) dur, {_TS.format(c='OPPLANDATUM')} ts
                   FROM read_csv({_snapshot_files('op_an.csv')}, delim=';', header=true,
                        all_varchar=true, union_by_name=true, ignore_errors=true))
        SELECT p.fallnr, max(dur) op_planned_min FROM ppb b JOIN cohort p ON p.fallnr = b.fallnr
             JOIN a ON a.fallnr = p.fallnr AND a.ts >= b.pb - INTERVAL {w[0]} HOUR AND a.ts < b.pb + INTERVAL {w[1]} HOUR
        GROUP BY p.fallnr""").df()
    f = con.execute(f"""
        WITH f AS (SELECT trim(FALLNR) fallnr, trim(AUFNAHMEART) art
                   FROM read_csv({_snapshot_files('fall_daten.csv')}, delim=';', header=true,
                        all_varchar=true, union_by_name=true, ignore_errors=true))
        SELECT p.fallnr, max(CASE WHEN lower(f.art) LIKE '%notfall%' THEN 1 ELSE 0 END) admission_emergency
        FROM cohort p JOIN f ON f.fallnr = p.fallnr GROUP BY p.fallnr""").df()
    v = con.execute(f"""
        WITH v AS (SELECT DISTINCT trim(FALLNR) fallnr, trim(WARDSHORT) w
                   FROM read_csv({_snapshot_files('aufenthalte_vorher_nachher.csv')}, delim=';',
                        header=true, all_varchar=true, union_by_name=true, ignore_errors=true)
                   WHERE trim(VOR_NACH) = 'V')
        SELECT p.fallnr {_ward_cases('')} FROM cohort p JOIN v ON v.fallnr = p.fallnr GROUP BY p.fallnr""").df()
    out = (o.merge(a, on="fallnr", how="outer").merge(f, on="fallnr", how="outer")
             .merge(v, on="fallnr", how="outer"))
    out["op_present"] = out["op_present"].fillna(0)
    return out
