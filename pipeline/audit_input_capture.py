# -*- coding: utf-8 -*-
"""
Input-capture audit: for every input group, how much of the raw signal is (a) captured as a
retrospective 24h feature and (b) actually available in the prospective live snapshots at 24 h.

Why this exists
---------------
The prospective feature matrix under-captures data that *is* present at 24 h. This audit
separates two very different causes of the prospective drop:

  * CAPTURE GAP (fixable in the pipeline) — labs & vitals: at 24 h the snapshots hold ~40 labs
    at >=50 % coverage (haemoglobin ~90 %) and ~18 vitals (HF/AF/GCS/TEMP/ABP, 80-91 %), yet the
    old prospective matrix wired in only ~6 labs + SpO2. Rebuild them with
    ``build_24h_measurement_features.py``.
  * CODING GAP (intrinsic, not fixable) — procedures & main diagnoses are coded later: at 24 h
    only ~1 case carries the specific predictive OPS/ICD codes; severity scores (SOFA ~3 %,
    SAPS ~3 %) and ASA (0 %) are essentially undocumented prospectively.

Perioperative sources (OP duration/type, admission type, referring specialty) are ~50-100 %
available at 24 h and currently unused (see ``build_perioperative_features.py``).

Output: one coverage table per group (CSV). NOTE: hard-wired paths -> adapt CONFIG.
No patient data included.
"""
from __future__ import annotations
from pathlib import Path
import duckdb
import pandas as pd

# ------------------------------------------------------------------ CONFIG ----
OLD_SNAPSHOTS   = Path(r"D:\Ausgangsdaten\Live-Daten\OLD")
PROSPECTIVE_IDS = None   # optional: list[str] of prospective fallnr; None -> all in snapshots
OUT_DIR         = Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung")

_TS = ("COALESCE(try_cast({c} AS TIMESTAMP), "
       "try_strptime({c},'%d.%m.%Y %H:%M:%S'), try_strptime({c},'%d.%m.%Y'))")

# (source file, value column, timestamp column, optional WHERE) per input group.
GROUPS = {
    "labs":       ("lab.csv",         "BESCHREIBUNG",      "ERFASSDAT",  ""),
    "vitals":     ("vitalzeichen.csv","BEFUNDARTKURZBEZ",  "ZEITPUNKT",  ""),
    "procedures": ("prozeduren.csv",  "OPS",               "DURCHF_DATUM",""),
    "diagnoses":  ("diagnose.csv",    "DIAGNR",            "FESTSTDATUM","WHERE upper(trim(HAUPTNEBEN))='H'"),
    "access":     ("zugaenge.csv",    "TEXT",              "ANLEGEDATUM",""),
    "scores":     ("score.csv",       "KURZBEZ",           "VON",        ""),
}


def _snapshot_files(source: str) -> str:
    def ok(p: Path) -> bool:
        try:
            with open(p, "r", encoding="latin1", errors="replace") as fh:
                return "FALLNR" in fh.readline().upper()
        except Exception:
            return False
    files = [p for p in OLD_SNAPSHOTS.glob(f"*/{source}") if p.stat().st_size > 10 and ok(p)]
    return "[" + ",".join("'" + p.as_posix() + "'" for p in files) + "]"


def prospective_availability(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Per group: distinct variables measured within [planbegin, +24 h] and their case coverage."""
    if PROSPECTIVE_IDS is not None:
        con.register("cohort", pd.DataFrame({"fallnr": sorted(set(PROSPECTIVE_IDS))}))
        cohort_sql = "cohort"
    else:
        con.execute(f"""CREATE OR REPLACE TEMP TABLE cohort AS
            SELECT DISTINCT trim(FALLNR) fallnr FROM read_csv('{(OLD_SNAPSHOTS/'*'/'fall_aufenthalt.csv').as_posix()}',
            delim=';', header=true, all_varchar=true, union_by_name=true, ignore_errors=true)""")
        cohort_sql = "cohort"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE ppb AS
        SELECT c.fallnr, min({_TS.format(c='fa.PLANBEGIN')}) pb
        FROM {cohort_sql} c JOIN read_csv('{(OLD_SNAPSHOTS/'*'/'fall_aufenthalt.csv').as_posix()}',
             delim=';', header=true, all_varchar=true, union_by_name=true, ignore_errors=true) fa
          ON c.fallnr = trim(fa.FALLNR) GROUP BY c.fallnr""")
    n = con.execute("SELECT COUNT(*) FROM ppb").fetchone()[0]

    tables = {}
    for grp, (src, keycol, tscol, where) in GROUPS.items():
        df = con.execute(f"""
            WITH ev AS (SELECT trim(FALLNR) fallnr, trim({keycol}) k, {_TS.format(c=tscol)} ts
                        FROM read_csv({_snapshot_files(src)}, delim=';', header=true,
                             all_varchar=true, union_by_name=true, ignore_errors=true) {where})
            SELECT ev.k AS variable, COUNT(DISTINCT ev.fallnr) n_cases
            FROM ppb b JOIN ev ON ev.fallnr = b.fallnr
                 AND ev.ts >= b.pb AND ev.ts < b.pb + INTERVAL 24 HOUR
            WHERE ev.k IS NOT NULL AND trim(ev.k) <> ''
            GROUP BY ev.k ORDER BY n_cases DESC""").df()
        df["coverage_pct"] = (100 * df["n_cases"] / n).round(1)
        tables[grp] = df
    return tables


if __name__ == "__main__":
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='6GB'")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for grp, df in prospective_availability(con).items():
        n_avail = len(df)
        n50 = int((df["coverage_pct"] >= 50).sum())
        print(f"{grp:<11} {n_avail:>4} variables at 24h | {n50} with >=50% coverage")
        df.to_csv(OUT_DIR / f"capture_audit_{grp}_prospective.csv", sep=";", index=False, encoding="utf-8-sig")
