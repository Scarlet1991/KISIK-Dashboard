# -*- coding: utf-8 -*-
"""
First-24h measurement features (vitals + labs) — RETROSPECTIVE and PROSPECTIVE in parallel.

Motivation
----------
The prospective feature matrix historically captured only a small slice of the labs/vitals
that are in fact *already measured* within the first 24 h of the live snapshots (e.g. only
SpO2 among vitals, ~6 lab analytes). An input-capture audit (see ``audit_input_capture.py``)
showed that this is a *pipeline capture gap*, not a data-availability gap: at 24 h the live
snapshots contain ~40 labs at >=50 % coverage (incl. haemoglobin ~90 %) and ~18 vitals
(HF/AF/GCS/TEMP/ABP, 80-91 %).

This module rebuilds the first-24h vitals and labs from the raw sources with **identical
column naming** on both sides, so a model trained retrospectively can consume the
prospective columns 1:1:

    vital24_<sanitised-kurzbez>_<stat>      lab24_<sanitised-beschreibung>_<stat>
    stat in {first, mean, min, max, last, count}

Retrospective side : single CSVs, key ``fallid``, ISO timestamps.
Prospective side   : daily OLD snapshots (union), key ``fallnr``, German timestamps
                     (DD.MM.YYYY HH:MM:SS), empty/《header-less》snapshot files skipped.

NOTE: hard-wired local paths -> adapt CONFIG for your environment. No patient data included.
"""
from __future__ import annotations
import re
from pathlib import Path
import duckdb
import pandas as pd

# ------------------------------------------------------------------ CONFIG ----
RETRO_PARQUET = Path(r"D:\Ausgangsdaten\KISIK Projekt\kisik2\kisik2_icu_ml_dataset_24h.parquet")
RETRO_DIR     = Path(r"D:\Ausgangsdaten\KISIK Projekt\kisik2")          # vitalzeichen.csv
RETRO_LAB     = Path(r"D:\Ausgangsdaten\KISIK Projekt\KISIK_Updated_16062025\lab.csv")
OLD_SNAPSHOTS = Path(r"D:\Ausgangsdaten\Live-Daten\OLD")                # daily */*.csv
WARD_FILTER   = "('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"       # (wardshort, oebenekurz)

# Core, clinically-relevant vitals available at 24 h on both sides (BEFUNDARTKURZBEZ).
VITAL_KURZBEZ = ["HF", "AF", "TEMP", "SPO2_", "SPO2", "GCS",
                 "ABPS", "ABPM", "ABPD", "NBPS", "NBPM", "NBPD", "SYS", "DIAS",
                 "FIO2(M)", "PEEP/EPAP(M) GEMESSEN", "AMV GEMESSEN", "PULS_", "PPEAK"]
STATS = ["fst", "mean", "mn", "mx", "lst", "cnt"]                       # -> first/mean/min/max/last/count
LAB_MIN_COVERAGE = 0.02   # keep retro lab analytes measured in >= 2 % of stays

# German-date-tolerant timestamp parse (ISO first, then DD.MM.YYYY [HH:MM:SS]).
_TS = ("COALESCE(try_cast({c} AS TIMESTAMP), "
       "try_strptime({c},'%d.%m.%Y %H:%M:%S'), try_strptime({c},'%d.%m.%Y'))")


def sanitise(name: str) -> str:
    """Lower-case, umlaut/space/punctuation -> underscore, collapse. Deterministic on both sides."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(name).lower())).strip("_")


def _pivot(df: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    """Long (id, k, <stats>) -> wide ``<prefix><sanitised-k>_<stat>`` columns."""
    df = df.copy()
    df["kk"] = prefix + df["k"].map(sanitise)
    out = None
    for st in STATS:
        p = df.pivot_table(index=id_col, columns="kk", values=st)
        p.columns = [f"{c}_{st}" for c in p.columns]
        out = p if out is None else out.join(p, how="outer")
    return out.reset_index()


def _snapshot_files(source: str) -> str:
    """DuckDB list literal of snapshot files that actually carry a FALLNR header (skip empties)."""
    def ok(p: Path) -> bool:
        try:
            with open(p, "r", encoding="latin1", errors="replace") as fh:
                return "FALLNR" in fh.readline().upper()
        except Exception:
            return False
    files = [p for p in OLD_SNAPSHOTS.glob(f"*/{source}") if p.stat().st_size > 10 and ok(p)]
    return "[" + ",".join("'" + p.as_posix() + "'" for p in files) + "]"


# ----------------------------------------------------------- RETROSPECTIVE ----
def build_retro_measurements(con: duckdb.DuckDBPyConnection):
    """Return (vitals_df, labs_df) keyed by ``stay_id`` for the retro cohort (24h window)."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE stays AS
        SELECT CAST(stay_id AS VARCHAR) stay_id, trim(fallid) fallid,
               try_cast(planbegin AS TIMESTAMP) pb
        FROM read_parquet('{RETRO_PARQUET.as_posix()}')
        WHERE (wardshort, oebenekurz) IN ({WARD_FILTER})
          AND icu_duration_h / 24.0 > 1 AND fallid IS NOT NULL AND planbegin IS NOT NULL""")
    n_stays = con.execute("SELECT COUNT(*) FROM stays").fetchone()[0]

    kurz = "(" + ",".join("'" + k.replace("'", "''") + "'" for k in VITAL_KURZBEZ) + ")"
    vit = con.execute(f"""
        WITH vz AS (
          SELECT trim(fallid) fallid, trim(befundartkurzbez) k, try_cast(wert AS DOUBLE) v,
                 try_cast(zeitpunkt AS TIMESTAMP) ts
          FROM read_csv('{(RETRO_DIR/'vitalzeichen.csv').as_posix()}', delim=';', header=true, all_varchar=true)
          WHERE trim(befundartkurzbez) IN {kurz}
            AND try_cast(wert AS DOUBLE) IS NOT NULL AND try_cast(zeitpunkt AS TIMESTAMP) IS NOT NULL)
        SELECT s.stay_id, vz.k, min(vz.v) mn, max(vz.v) mx, avg(vz.v) mean, count(*) cnt,
               arg_min(vz.v, vz.ts) fst, arg_max(vz.v, vz.ts) lst
        FROM stays s JOIN vz ON vz.fallid = s.fallid
             AND vz.ts >= s.pb AND vz.ts < s.pb + INTERVAL 24 HOUR
        GROUP BY s.stay_id, vz.k""").df()

    lab = con.execute(f"""
        WITH lb AS (
          SELECT trim(fallid) fallid, trim(beschreibung) k, try_cast(ergebnisf AS DOUBLE) v,
                 try_cast(erfassdat AS TIMESTAMP) ts
          FROM read_csv('{RETRO_LAB.as_posix()}', delim=';', header=true, all_varchar=true)
          WHERE try_cast(ergebnisf AS DOUBLE) IS NOT NULL
            AND try_cast(erfassdat AS TIMESTAMP) IS NOT NULL AND trim(beschreibung) <> '')
        SELECT s.stay_id, lb.k, min(lb.v) mn, max(lb.v) mx, avg(lb.v) mean, count(*) cnt,
               arg_min(lb.v, lb.ts) fst, arg_max(lb.v, lb.ts) lst
        FROM stays s JOIN lb ON lb.fallid = s.fallid
             AND lb.ts >= s.pb AND lb.ts < s.pb + INTERVAL 24 HOUR
        GROUP BY s.stay_id, lb.k""").df()
    keep = lab.groupby("k")["stay_id"].nunique()
    lab = lab[lab["k"].isin(set(keep[keep >= LAB_MIN_COVERAGE * n_stays].index))]

    return _pivot(vit, "stay_id", "vital24_"), _pivot(lab, "stay_id", "lab24_")


# ------------------------------------------------------------- PROSPECTIVE ----
def build_prospective_measurements(con: duckdb.DuckDBPyConnection, fallnrs: list[str]):
    """Return (vitals_df, labs_df) keyed by ``fallnr`` for the given prospective cases (24h window).

    ``planbegin`` per case is taken from the snapshot ``fall_aufenthalt.csv`` (min over snapshots)."""
    con.register("cohort", pd.DataFrame({"fallnr": sorted(set(fallnrs))}))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ppb AS
        SELECT c.fallnr, min({_TS.format(c='fa.PLANBEGIN')}) pb
        FROM cohort c JOIN read_csv('{(OLD_SNAPSHOTS/'*'/'fall_aufenthalt.csv').as_posix()}',
             delim=';', header=true, all_varchar=true, union_by_name=true, ignore_errors=true) fa
          ON c.fallnr = trim(fa.FALLNR)
        GROUP BY c.fallnr""")

    kurz = "(" + ",".join("'" + k.replace("'", "''") + "'" for k in VITAL_KURZBEZ) + ")"
    pv = con.execute(f"""
        WITH vz AS (
          SELECT trim(FALLNR) fallnr, trim(BEFUNDARTKURZBEZ) k, try_cast(WERT AS DOUBLE) v,
                 {_TS.format(c='ZEITPUNKT')} ts
          FROM read_csv({_snapshot_files('vitalzeichen.csv')}, delim=';', header=true,
               all_varchar=true, union_by_name=true, ignore_errors=true)
          WHERE trim(BEFUNDARTKURZBEZ) IN {kurz} AND try_cast(WERT AS DOUBLE) IS NOT NULL)
        SELECT DISTINCT p.fallnr, vz.k, vz.v, vz.ts, b.pb
        FROM ppb b JOIN cohort p ON p.fallnr = b.fallnr
             JOIN vz ON vz.fallnr = p.fallnr
        WHERE vz.ts >= b.pb AND vz.ts < b.pb + INTERVAL 24 HOUR""").df()
    pv = (pv.groupby(["fallnr", "k"])
            .agg(mn=("v", "min"), mx=("v", "max"), mean=("v", "mean"), cnt=("v", "count"),
                 fst=("v", lambda s: s.iloc[0]), lst=("v", lambda s: s.iloc[-1]))
            .reset_index())

    pl = con.execute(f"""
        WITH lb AS (
          SELECT trim(FALLNR) fallnr, trim(BESCHREIBUNG) k, try_cast(ERGEBNISF AS DOUBLE) v,
                 {_TS.format(c='ERFASSDAT')} ts
          FROM read_csv({_snapshot_files('lab.csv')}, delim=';', header=true,
               all_varchar=true, union_by_name=true, ignore_errors=true)
          WHERE try_cast(ERGEBNISF AS DOUBLE) IS NOT NULL AND trim(BESCHREIBUNG) <> '')
        SELECT DISTINCT p.fallnr, lb.k, lb.v, lb.ts, b.pb
        FROM ppb b JOIN cohort p ON p.fallnr = b.fallnr
             JOIN lb ON lb.fallnr = p.fallnr
        WHERE lb.ts >= b.pb AND lb.ts < b.pb + INTERVAL 24 HOUR""").df()
    pl = (pl.groupby(["fallnr", "k"])
            .agg(mn=("v", "min"), mx=("v", "max"), mean=("v", "mean"), cnt=("v", "count"),
                 fst=("v", lambda s: s.iloc[0]), lst=("v", lambda s: s.iloc[-1]))
            .reset_index())

    return _pivot(pv, "fallnr", "vital24_"), _pivot(pl, "fallnr", "lab24_")


if __name__ == "__main__":
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    rv, rl = build_retro_measurements(con)
    print(f"retro : {rv.shape[1]-1} vital cols, {rl.shape[1]-1} lab cols for {len(rv)} stays")
    # Prospective example: pass the fallnr list of your prospective cohort here.
    # pv, pl = build_prospective_measurements(con, my_fallnr_list)
    # print(f"prosp : {pv.shape[1]-1} vital cols, {pl.shape[1]-1} lab cols for {len(pv)} cases")
