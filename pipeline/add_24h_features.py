"""
Berechnet echte 24h-Zeitfenster-Features und reichert das bestehende Parquet an.
Zeitfenster: planbegin bis planbegin + 24 Stunden (Aufnahme-Snapshot).

Neue Spalten im Parquet:
  vital24_{name}_{stat}    Vitalzeichen in den ersten 24h
  lab24_{name}_{stat}      Laborwerte in den ersten 24h
  proc24_{ops}             Prozedur in den ersten 24h vorhanden (0/1)
  proc24_anzahl_gesamt     Anzahl Prozeduren in den ersten 24h
  zugang24_{text}          Zugang in den ersten 24h angelegt (0/1)
  zugang24_anzahl_gesamt   Anzahl Zugaenge in den ersten 24h
"""
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---- Pfade ----------------------------------------------------------------
BASE_DIR    = Path(r"D:\Ausgangsdaten\KISIK Projekt\kisik2")
LAB_FILE    = Path(r"D:\Ausgangsdaten\KISIK Projekt\KISIK_Updated_16062025\lab.csv")
PARQUET_IN  = BASE_DIR / "kisik2_icu_ml_dataset.parquet"
PARQUET_OUT = BASE_DIR / "kisik2_icu_ml_dataset_24h.parquet"

FILES = {
    "vitals":     BASE_DIR / "vitalzeichen.csv",
    "labs":       LAB_FILE,
    "procedures": BASE_DIR / "prozeduren.csv",
    "access":     BASE_DIR / "zugaenge.csv",
}

TOP_N_FEATURES = {"vital": 100, "lab": 200, "proc": 500, "zugang": 300}

# ---- Hilfsfunktionen ------------------------------------------------------
def sanitize(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:60] or "unknown"


def long_to_wide_24h(long_df: pd.DataFrame, prefix: str, top_n: int) -> pd.DataFrame:
    """Aggregierte numerische Features -> Wide-Format mit prefix."""
    if long_df.empty:
        return pd.DataFrame(columns=["stay_id"])

    # Auf haeufigste Features begrenzen
    counts = long_df.groupby("feature_name")["stay_id"].nunique().sort_values(ascending=False)
    keep   = set(counts.head(top_n).index)
    long_df = long_df[long_df["feature_name"].isin(keep)].copy()

    long_df["fk"] = long_df["feature_name"].map(sanitize)
    stats = {"mean_value": "mean", "median_value": "median",
             "first_value": "first", "last_value": "last",
             "min_value": "min", "max_value": "max",
             "count_value": "count"}
    parts = []
    for col, suffix in stats.items():
        if col not in long_df.columns:
            continue
        p = long_df.pivot_table(index="stay_id", columns="fk", values=col, aggfunc="first")
        if p.empty:
            continue
        if suffix == "count":
            p = p.fillna(0).astype("float32")
        else:
            p = p.astype("float32")
        p.columns = [f"{prefix}_{c}_{suffix}" for c in p.columns]
        parts.append(p)

    if not parts:
        return pd.DataFrame(columns=["stay_id"])
    return pd.concat(parts, axis=1).reset_index()


def presence_to_wide_24h(long_df: pd.DataFrame, prefix: str, top_n: int) -> pd.DataFrame:
    """Binaere Anwesenheits-Features -> Wide-Format."""
    if long_df.empty:
        return pd.DataFrame(columns=["stay_id"])

    counts = long_df.groupby("feature_name")["stay_id"].nunique().sort_values(ascending=False)
    keep   = set(counts.head(top_n).index)
    local  = long_df[long_df["feature_name"].isin(keep)][["stay_id", "feature_name"]].dropna().copy()
    local["fk"] = local["feature_name"].map(sanitize)
    wide   = pd.crosstab(local["stay_id"], local["fk"])
    wide   = (wide > 0).astype("uint8")
    wide.columns = [f"{prefix}_{c}" for c in wide.columns]
    return wide.reset_index()


# ---- DuckDB Verbindung + stays registrieren -------------------------------
print("Lade bestehendes Parquet ...")
con = duckdb.connect()

stays = con.execute(f"""
    SELECT stay_id, fallid, planbegin, planend
    FROM read_parquet('{PARQUET_IN.as_posix()}')
""").df()
stays["planbegin"] = pd.to_datetime(stays["planbegin"], errors="coerce")
stays["planend"]   = pd.to_datetime(stays["planend"],   errors="coerce")
stays["window_end"] = stays["planbegin"] + pd.Timedelta(hours=24)
print(f"  {len(stays):,} Stays geladen")

con.register("stays_df", stays[["stay_id", "fallid", "planbegin", "window_end"]])
con.execute("CREATE OR REPLACE TEMP TABLE stays24 AS SELECT * FROM stays_df")


# ---- Hilfsmakros fuer SQL -------------------------------------------------
def _path(p: Path) -> str:
    return p.as_posix()


# ---- Vitalzeichen 24h -----------------------------------------------------
print("\nBerechne Vitalzeichen 24h ...")
vitals_sql = f"""
    WITH v AS (
        SELECT
            s.stay_id,
            vit.befundartkurzbez AS feature_name,
            TRY_CAST(REPLACE(vit.wert, ',', '.') AS DOUBLE) AS value_num,
            TRY_CAST(vit.zeitpunkt AS TIMESTAMP) AS ts
        FROM read_csv_auto('{_path(FILES["vitals"])}',
             delim=';', header=true, all_varchar=true, ignore_errors=true) vit
        JOIN stays24 s ON vit.fallid = s.fallid
        WHERE TRY_CAST(vit.zeitpunkt AS TIMESTAMP) BETWEEN s.planbegin AND s.window_end
    )
    SELECT
        stay_id, feature_name,
        AVG(value_num) AS mean_value,
        MEDIAN(value_num) AS median_value,
        ARG_MIN(value_num, ts) AS first_value,
        ARG_MAX(value_num, ts) AS last_value,
        MIN(value_num) AS min_value,
        MAX(value_num) AS max_value,
        COUNT(value_num) AS count_value
    FROM v
    WHERE feature_name IS NOT NULL AND value_num IS NOT NULL AND ts IS NOT NULL
    GROUP BY 1, 2
"""
vitals24_long = con.execute(vitals_sql).df()
vitals24_wide = long_to_wide_24h(vitals24_long, "vital24", TOP_N_FEATURES["vital"])
print(f"  {len(vitals24_long):,} Messwerte -> {max(len(vitals24_wide.columns)-1, 0)} Spalten")


# ---- Labor 24h ------------------------------------------------------------
print("\nBerechne Labor 24h ...")
labs_sql = f"""
    WITH l AS (
        SELECT
            s.stay_id,
            COALESCE(
                NULLIF(TRIM(lab.beschreibung), ''),
                NULLIF(TRIM(lab.code), ''),
                NULLIF(TRIM(lab.analytx), '')
            ) AS feature_name,
            COALESCE(
                TRY_CAST(REPLACE(lab.ergebnisf, ',', '.') AS DOUBLE),
                TRY_CAST(REPLACE(lab.ergebnist, ',', '.') AS DOUBLE)
            ) AS value_num,
            TRY_CAST(lab.erfassdat AS TIMESTAMP) AS ts
        FROM read_csv_auto('{_path(FILES["labs"])}',
             delim=';', header=true, all_varchar=true, ignore_errors=true) lab
        JOIN stays24 s ON lab.fallid = s.fallid
        WHERE TRY_CAST(lab.erfassdat AS TIMESTAMP) BETWEEN s.planbegin AND s.window_end
    )
    SELECT
        stay_id, feature_name,
        AVG(value_num) AS mean_value,
        MEDIAN(value_num) AS median_value,
        ARG_MIN(value_num, ts) AS first_value,
        ARG_MAX(value_num, ts) AS last_value,
        MIN(value_num) AS min_value,
        MAX(value_num) AS max_value,
        COUNT(value_num) AS count_value
    FROM l
    WHERE feature_name IS NOT NULL AND value_num IS NOT NULL AND ts IS NOT NULL
    GROUP BY 1, 2
"""
labs24_long = con.execute(labs_sql).df()
labs24_wide = long_to_wide_24h(labs24_long, "lab24", TOP_N_FEATURES["lab"])
print(f"  {len(labs24_long):,} Messwerte -> {max(len(labs24_wide.columns)-1, 0)} Spalten")


# ---- Prozeduren 24h (binaer + Anzahl) ------------------------------------
print("\nBerechne Prozeduren 24h ...")
proc_sql = f"""
    SELECT DISTINCT
        s.stay_id,
        p.ops AS feature_name
    FROM read_csv_auto('{_path(FILES["procedures"])}',
         delim=';', header=true, all_varchar=true, ignore_errors=true) p
    JOIN stays24 s ON p.fallid = s.fallid
    WHERE p.ops IS NOT NULL
      AND TRY_CAST(p.durchf_datum AS TIMESTAMP) BETWEEN s.planbegin AND s.window_end
"""
proc24_long = con.execute(proc_sql).df()
proc24_wide = presence_to_wide_24h(proc24_long, "proc24", TOP_N_FEATURES["proc"])

# Gesamtzahl Prozeduren in ersten 24h
proc24_count = (proc24_long.groupby("stay_id")["feature_name"].count()
                .reset_index(name="proc24_anzahl_gesamt"))
print(f"  {len(proc24_long):,} Prozedur-Eintraege -> {max(len(proc24_wide.columns)-1, 0)} binaere Spalten")


# ---- Zugaenge 24h (binaer + Anzahl) --------------------------------------
print("\nBerechne Zugaenge 24h ...")
access_sql = f"""
    SELECT
        s.stay_id,
        COALESCE(
            NULLIF(TRIM(a.subklassifikation1), ''),
            NULLIF(TRIM(a.longtext), '')
        ) AS feature_name
    FROM read_csv_auto('{_path(FILES["access"])}',
         delim=';', header=true, all_varchar=true, ignore_errors=true) a
    JOIN stays24 s ON a.fallid = s.fallid
    WHERE TRY_CAST(
        a.anlegedatum || ' ' || COALESCE(a.anlegezeit, '00:00:00')
        AS TIMESTAMP
    ) BETWEEN s.planbegin AND s.window_end
"""
access24_long = con.execute(access_sql).df().dropna(subset=["feature_name"])
access24_wide = presence_to_wide_24h(access24_long, "zugang24", TOP_N_FEATURES["zugang"])

access24_count = (access24_long.groupby("stay_id")["feature_name"].count()
                  .reset_index(name="zugang24_anzahl_gesamt"))
print(f"  {len(access24_long):,} Zugangs-Eintraege -> {max(len(access24_wide.columns)-1, 0)} binaere Spalten")

con.close()


# ---- Zusammenfuehren und Parquet speichern --------------------------------
print("\nFuehre zusammen und speichere Parquet ...")

con2 = duckdb.connect()
con2.execute(f"""
    CREATE OR REPLACE TEMP TABLE base AS
    SELECT * FROM read_parquet('{PARQUET_IN.as_posix()}')
""")

def join_part(con, table_name: str, df: pd.DataFrame):
    if df.empty or len(df.columns) <= 1:
        return
    con.register(table_name, df)
    extra_cols = ", ".join(
        f'"{c}"' for c in df.columns if c != "stay_id"
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT base.*, {extra_cols}
        FROM base
        LEFT JOIN {table_name} USING (stay_id)
    """)

parts = [
    ("vitals24_df",   vitals24_wide),
    ("labs24_df",     labs24_wide),
    ("proc24_df",     proc24_wide),
    ("proc24_cnt",    proc24_count),
    ("zugang24_df",   access24_wide),
    ("zugang24_cnt",  access24_count),
]

for tname, df in parts:
    join_part(con2, tname, df)
    print(f"  + {tname}: {max(len(df.columns)-1, 0)} Spalten gejoint")

# Ergebnis schreiben
tmp = PARQUET_OUT.parent / "duckdb_tmp"
tmp.mkdir(exist_ok=True)
con2.execute(f"""
    COPY base TO '{PARQUET_OUT.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

n_rows = con2.execute("SELECT COUNT(*) FROM base").fetchone()[0]
n_cols = con2.execute("SELECT COUNT(*) FROM pragma_table_info('base')").fetchone()[0]
con2.close()

print(f"\nParquet gespeichert: {PARQUET_OUT}")
print(f"  {n_rows:,} Zeilen x {n_cols:,} Spalten")
print(f"  Groesse: {round(PARQUET_OUT.stat().st_size / 1e6, 1)} MB")
