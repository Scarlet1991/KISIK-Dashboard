# -*- coding: utf-8 -*-
"""
Retrospektive KISIK ICU-Daten-Pipeline.
Auto-extrahiert (nur Quellcode, KEINE Notebook-Outputs) aus KISIK-Daten-Pipeline.ipynb.

Baut den retrospektiven ML-Datensatz (kisik2_icu_ml_dataset.parquet) aus den
Roh-KISIK-CSVs: Aufenthalte/Episoden, Diagnosen, Labor, Vitalzeichen, Prozeduren, Zugaenge.
Die Zellen teilen sich Zustand und sind fuer eine Top-to-Bottom-Ausfuehrung gedacht
(urspruenglich ein Jupyter-Notebook).

HINWEIS: enthaelt fest verdrahtete lokale Pfade -> fuer die eigene Umgebung anpassen.
Keine Patientendaten enthalten.
"""

# %% ---- notebook cell [0] ----------------------------------------
"""
ICU ML Preprocessing Pipeline fuer KISIK2
========================================
Verarbeitet die CSV-Dateien aus
D:\Ausgangsdaten\KISIK Projekt\kisik2
zu einem ML-faehigen Dataset pro Aufenthalt.

Verwendete Dateien:
  - fall_aufenthalt.csv
  - vitalzeichen.csv
  - score.csv
  - diagnose.csv
  - prozeduren.csv
  - zugaenge.csv

Ausgaben:
  - kisik2_icu_ml_dataset.parquet
  - kisik2_icu_ml_dataset_summary.xlsx

Hinweis:
Als ICU-Zeitfenster werden planbegin und planend aus fall_aufenthalt.csv verwendet.
Sehr hochkardinale kategoriale Features werden auf die haeufigsten Kategorien begrenzt,
damit das resultierende Wide-Dataset fuer das Notebook handhabbar bleibt.
"""

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE_DIR = Path(r"D:\Ausgangsdaten\KISIK Projekt\kisik2")

FILES = {
    "stays": BASE_DIR / "fall_aufenthalt.csv",
    "vitals": BASE_DIR / "vitalzeichen.csv",
    "scores": BASE_DIR / "score.csv",
    "diagnoses": BASE_DIR / "diagnose.csv",
    "procedures": BASE_DIR / "prozeduren.csv",
    "access": BASE_DIR / "zugaenge.csv",
}

DATASET_FILE = BASE_DIR / "kisik2_icu_ml_dataset.parquet"
SUMMARY_FILE = BASE_DIR / "kisik2_icu_ml_dataset_summary.xlsx"
TOP_FEATURE_LIMITS = {
    "diag_main": 300,
    "proc": 500,
    "zugang": 300,
}

COLS = {
    "case_id": "fallid",
    "patient_id": "pid",
    "age": "alter",
    "icu_start": "planbegin",
    "icu_end": "planend",
    "hospital_start": "aufndat",
    "hospital_end": "entldat",
    "ward": "wardshort",
    "department": "oebenekurz",
    "vital_name": "befundartkurzbez",
    "vital_value": "wert",
    "vital_time": "zeitpunkt",
    "score_name": "kurzbez",
    "score_value": "scoreergebnis",
    "score_start": "von",
    "diag_code": "diagnr",
    "diag_type": "hauptneben",
    "diag_main_value": "H",
    "proc_code": "ops",
    "proc_time": "durchf_datum",
    "access_group": "subklassifikation1",
    "access_text": "longtext",
    "access_date": "anlegedatum",
    "access_time": "anlegezeit",
}


def sanitize_name(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80] if text else "unknown"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  [WARNUNG] Datei nicht gefunden: {path}")
        return pd.DataFrame()

    last_error = None
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(path, sep=";", encoding=encoding)
            print(f"  Geladen: {path.name} -> {len(df):,} Zeilen, {df.shape[1]} Spalten")
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Konnte {path} nicht lesen: {last_error}")


def to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def to_numeric_safe(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def optimize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    for col in result.columns:
        if col == "stay_id":
            continue
        if pd.api.types.is_bool_dtype(result[col]):
            result[col] = result[col].astype("int8")
        elif pd.api.types.is_integer_dtype(result[col]):
            result[col] = pd.to_numeric(result[col], downcast="integer")
        elif pd.api.types.is_float_dtype(result[col]):
            result[col] = pd.to_numeric(result[col], downcast="float")
    return result


def build_stays(stays_raw: pd.DataFrame) -> pd.DataFrame:
    required = [
        COLS["case_id"],
        COLS["icu_start"],
        COLS["icu_end"],
        COLS["hospital_start"],
        COLS["hospital_end"],
    ]
    missing = [col for col in required if col not in stays_raw.columns]
    if missing:
        raise KeyError(f"Fehlende Spalten in fall_aufenthalt.csv: {missing}")

    stays = stays_raw.copy()
    for col in [COLS["icu_start"], COLS["icu_end"], COLS["hospital_start"], COLS["hospital_end"]]:
        stays[col] = to_datetime_safe(stays[col])

    stays[COLS["icu_start"]] = stays[COLS["icu_start"]].fillna(stays[COLS["hospital_start"]])
    stays[COLS["icu_end"]] = stays[COLS["icu_end"]].fillna(stays[COLS["hospital_end"]])
    stays = stays.dropna(subset=[COLS["case_id"], COLS["icu_start"], COLS["icu_end"]]).copy()

    if COLS["age"] in stays.columns:
        stays[COLS["age"]] = to_numeric_safe(stays[COLS["age"]])

    stays = stays.sort_values([COLS["case_id"], COLS["icu_start"], COLS["icu_end"]]).copy()
    stays["stay_nr"] = stays.groupby(COLS["case_id"]).cumcount() + 1
    stays["stay_id"] = stays[COLS["case_id"]].astype(str) + "_stay" + stays["stay_nr"].astype(str)
    stays["icu_duration_h"] = ((stays[COLS["icu_end"]] - stays[COLS["icu_start"]]).dt.total_seconds() / 3600).astype("float32")
    stays["hospital_duration_h"] = ((stays[COLS["hospital_end"]] - stays[COLS["hospital_start"]]).dt.total_seconds() / 3600).astype("float32")
    return stays


def register_stays(con: duckdb.DuckDBPyConnection, stays: pd.DataFrame) -> None:
    con.register("stays_df", stays[["stay_id", COLS["case_id"], COLS["icu_start"], COLS["icu_end"]]].copy())
    con.execute("create or replace temp table stays as select * from stays_df")


def long_metrics_to_wide(long_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["stay_id"])

    local = long_df.copy()
    local["feature_key"] = local["feature_name"].map(sanitize_name)
    metric_map = {
        "mean_value": "mean",
        "median_value": "median",
        "first_value": "first",
        "last_value": "last",
        "min_value": "min",
        "max_value": "max",
    }

    parts = []
    for metric_col, suffix in metric_map.items():
        part = local.pivot_table(index="stay_id", columns="feature_key", values=metric_col, aggfunc="first")
        if part.empty:
            continue
        part = part.astype("float32")
        part.columns = [f"{prefix}_{col}_{suffix}" for col in part.columns]
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=["stay_id"])

    return pd.concat(parts, axis=1).reset_index()


def limit_to_top_features(long_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    top_n = TOP_FEATURE_LIMITS.get(prefix)
    if top_n is None:
        return long_df

    counts = long_df.groupby("feature_name", dropna=False)["stay_id"].nunique().sort_values(ascending=False)
    keep = set(counts.head(top_n).index)
    filtered = long_df[long_df["feature_name"].isin(keep)].copy()
    print(f"  -> {prefix}: {len(counts)} Kategorien gesamt, {len(keep)} beibehalten")
    return filtered


def presence_to_wide(long_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["stay_id"])

    local = limit_to_top_features(long_df[["stay_id", "feature_name"]].dropna().copy(), prefix)
    if local.empty:
        return pd.DataFrame(columns=["stay_id"])

    local["feature_key"] = local["feature_name"].map(sanitize_name)
    wide = pd.crosstab(local["stay_id"], local["feature_key"])
    wide = (wide > 0).astype("uint8")
    wide.columns = [f"{prefix}_{col}" for col in wide.columns]
    return wide.reset_index()


def query_vitals(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    sql = f"""
        with vitals as (
            select
                s.stay_id,
                v.{COLS['vital_name']} as feature_name,
                try_cast(replace(v.{COLS['vital_value']}, ',', '.') as double) as value_num,
                try_cast(v.{COLS['vital_time']} as timestamp) as ts
            from read_csv_auto('{path.as_posix()}', delim=';', header=true, all_varchar=true, ignore_errors=true) v
            join stays s
              on v.{COLS['case_id']} = s.{COLS['case_id']}
             and try_cast(v.{COLS['vital_time']} as timestamp) between s.{COLS['icu_start']} and s.{COLS['icu_end']}
        )
        select
            stay_id,
            feature_name,
            avg(value_num) as mean_value,
            median(value_num) as median_value,
            arg_min(value_num, ts) as first_value,
            arg_max(value_num, ts) as last_value,
            min(value_num) as min_value,
            max(value_num) as max_value
        from vitals
        where feature_name is not null and value_num is not null and ts is not null
        group by 1, 2
    """
    return con.execute(sql).fetch_df()


def query_scores(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    sql = f"""
        with scores as (
            select
                s.stay_id,
                sc.{COLS['score_name']} as feature_name,
                try_cast(replace(sc.{COLS['score_value']}, ',', '.') as double) as value_num,
                try_cast(sc.{COLS['score_start']} as timestamp) as ts
            from read_csv_auto('{path.as_posix()}', delim=';', header=true, all_varchar=true, ignore_errors=true) sc
            join stays s
              on sc.{COLS['case_id']} = s.{COLS['case_id']}
             and try_cast(sc.{COLS['score_start']} as timestamp) between s.{COLS['icu_start']} and s.{COLS['icu_end']}
        )
        select
            stay_id,
            feature_name,
            avg(value_num) as mean_value,
            median(value_num) as median_value,
            arg_min(value_num, ts) as first_value,
            arg_max(value_num, ts) as last_value,
            min(value_num) as min_value,
            max(value_num) as max_value
        from scores
        where feature_name is not null and value_num is not null and ts is not null
        group by 1, 2
    """
    return con.execute(sql).fetch_df()


def query_diagnoses(con: duckdb.DuckDBPyConnection, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_sql = f"""
        select distinct
            s.stay_id,
            d.{COLS['diag_code']} as feature_name
        from read_csv_auto('{path.as_posix()}', delim=';', header=true, all_varchar=true, ignore_errors=true) d
        join stays s
          on d.{COLS['case_id']} = s.{COLS['case_id']}
        where d.{COLS['diag_code']} is not null
          and upper(coalesce(d.{COLS['diag_type']}, '')) = '{COLS['diag_main_value']}'
    """
    side_sql = f"""
        select
            s.stay_id,
            count(*) as neben_diag_anzahl
        from read_csv_auto('{path.as_posix()}', delim=';', header=true, all_varchar=true, ignore_errors=true) d
        join stays s
          on d.{COLS['case_id']} = s.{COLS['case_id']}
        where d.{COLS['diag_code']} is not null
          and upper(coalesce(d.{COLS['diag_type']}, '')) <> '{COLS['diag_main_value']}'
        group by 1
    """
    return con.execute(main_sql).fetch_df(), con.execute(side_sql).fetch_df()


def query_procedures(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    sql = f"""
        select distinct
            s.stay_id,
            p.{COLS['proc_code']} as feature_name
        from read_csv_auto('{path.as_posix()}', delim=';', header=true, all_varchar=true, ignore_errors=true) p
        join stays s
          on p.{COLS['case_id']} = s.{COLS['case_id']}
         and try_cast(p.{COLS['proc_time']} as timestamp) between s.{COLS['icu_start']} and s.{COLS['icu_end']}
        where p.{COLS['proc_code']} is not null
    """
    return con.execute(sql).fetch_df()


def query_access(con: duckdb.DuckDBPyConnection, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sql = f"""
        with access_data as (
            select
                s.stay_id,
                coalesce(a.{COLS['access_group']}, a.{COLS['access_text']}) as feature_name
            from read_csv_auto('{path.as_posix()}', delim=';', header=true, all_varchar=true, ignore_errors=true) a
            join stays s
              on a.{COLS['case_id']} = s.{COLS['case_id']}
             and try_cast(a.{COLS['access_date']} || ' ' || coalesce(a.{COLS['access_time']}, '00:00:00') as timestamp)
                 between s.{COLS['icu_start']} and s.{COLS['icu_end']}
        )
        select stay_id, feature_name
        from access_data
        where feature_name is not null
    """
    features = con.execute(sql).fetch_df()
    if features.empty:
        counts = pd.DataFrame(columns=["stay_id", "zugang_anzahl_gesamt"])
    else:
        counts = features.groupby("stay_id", dropna=False).size().reset_index(name="zugang_anzahl_gesamt")
    return features, counts


def build_feature_summary(base: pd.DataFrame, parts: list[tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    rows = [{"Feature_Gruppe": "Aufenthalt Meta", "Anzahl_Features": max(len(base.columns) - 1, 0)}]
    for part_df, label in parts:
        rows.append({"Feature_Gruppe": label, "Anzahl_Features": max(len(part_df.columns) - 1, 0) if not part_df.empty else 0})
    return pd.DataFrame(rows)


def save_outputs(base: pd.DataFrame, parts: list[tuple[pd.DataFrame, str]], preview_rows: int = 5) -> tuple[int, int, pd.DataFrame]:
    _tmp = DATASET_FILE.parent / "duckdb_tmp"
    _tmp.mkdir(exist_ok=True)
    con = duckdb.connect(config={"temp_directory": str(_tmp)})
    con.register("base_df", optimize_frame(base))

    select_parts = ["base.*"]
    join_parts = []
    for idx, (part_df, _label) in enumerate(parts):
        if part_df.empty:
            continue
        table_name = f"part_{idx}_df"
        alias = f"p{idx}"
        con.register(table_name, optimize_frame(part_df))
        select_parts.append(f"{alias}.* EXCLUDE (stay_id)")
        join_parts.append(f"LEFT JOIN {table_name} {alias} USING (stay_id)")

    final_sql = "\n".join([
        "CREATE OR REPLACE TABLE final_dataset AS",
        "SELECT",
        "    " + ",\n    ".join(select_parts),
        "FROM base_df base",
        *join_parts,
    ])
    con.execute(final_sql)
    con.execute(f"COPY final_dataset TO '{DATASET_FILE.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    n_rows = con.execute("SELECT COUNT(*) FROM final_dataset").fetchone()[0]
    n_cols = con.execute("SELECT COUNT(*) FROM pragma_table_info('final_dataset')").fetchone()[0]
    preview = con.execute(f"SELECT * FROM final_dataset LIMIT {preview_rows}").fetch_df()
    con.close()

    summary = build_feature_summary(base, parts)
    with pd.ExcelWriter(SUMMARY_FILE, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Feature_Uebersicht", index=False)
        preview.to_excel(writer, sheet_name="Preview", index=False)
        pd.DataFrame({
            "Kennzahl": ["Zeilen", "Spalten", "Parquet-Datei", "Summary-Datei"],
            "Wert": [n_rows, n_cols, str(DATASET_FILE), str(SUMMARY_FILE)],
        }).to_excel(writer, sheet_name="Export", index=False)

    return n_rows, n_cols, preview


def run():
    print("\n========================================")
    print(" KISIK2 ICU ML Preprocessing Pipeline")
    print("========================================\n")

    print("-> Schritt 1: Aufenthalte laden")
    stays_raw = load_csv(FILES["stays"])
    if stays_raw.empty:
        raise FileNotFoundError("fall_aufenthalt.csv ist zwingend erforderlich")

    stays = build_stays(stays_raw)
    print(f"  -> {len(stays)} Aufenthalte aus {stays[COLS['case_id']].nunique()} Faellen\n")

    con = duckdb.connect()
    register_stays(con, stays)

    print("-> Schritt 2: Vitalzeichen")
    vitals_long = query_vitals(con, FILES["vitals"])
    vitals_agg = long_metrics_to_wide(vitals_long, "vital")
    print(f"  -> {len(vitals_long):,} Stay-Vital-Kombinationen aggregiert\n")

    print("-> Schritt 3: Scores")
    scores_long = query_scores(con, FILES["scores"])
    scores_agg = long_metrics_to_wide(scores_long, "score")
    print(f"  -> {len(scores_long):,} Stay-Score-Kombinationen aggregiert\n")

    print("-> Schritt 4: Diagnosen")
    main_diag_long, side_diag_count = query_diagnoses(con, FILES["diagnoses"])
    main_diag_ohe = presence_to_wide(main_diag_long, "diag_main")
    print(f"  -> {len(main_diag_long):,} Stay-Hauptdiagnose-Kombinationen")
    print("  -> Nebendiagnosen pro Aufenthalt vorbereitet\n")

    print("-> Schritt 5: Prozeduren")
    proc_long = query_procedures(con, FILES["procedures"])
    proc_ohe = presence_to_wide(proc_long, "proc")
    print(f"  -> {len(proc_long):,} Stay-Prozedur-Kombinationen\n")

    print("-> Schritt 6: Zugaenge")
    access_long, access_count = query_access(con, FILES["access"])
    access_ohe = presence_to_wide(access_long, "zugang")
    print(f"  -> {len(access_long):,} Stay-Zugangs-Kombinationen\n")
    con.close()

    print("-> Schritt 7: Persistente Zusammenfuehrung")
    base_cols = [
        "stay_id",
        COLS["case_id"],
        "stay_nr",
        COLS["patient_id"],
        COLS["age"],
        COLS["icu_start"],
        COLS["icu_end"],
        COLS["hospital_start"],
        COLS["hospital_end"],
        COLS["department"],
        COLS["ward"],
        "icu_duration_h",
        "hospital_duration_h",
    ]
    base_cols = [col for col in base_cols if col in stays.columns]
    base = optimize_frame(stays[base_cols].copy())

    parts = [
        (vitals_agg, "Vitalzeichen"),
        (scores_agg, "Scores"),
        (main_diag_ohe, "Hauptdiagnosen"),
        (optimize_frame(side_diag_count), "Nebendiagnosen"),
        (proc_ohe, "Prozeduren"),
        (access_ohe, "Zugaenge"),
        (optimize_frame(access_count), "Zugang-Anzahl"),
    ]

    for part_df, label in parts:
        print(f"  + {label}: {max(len(part_df.columns) - 1, 0) if not part_df.empty else 0} Spalten")

    n_rows, n_cols, preview = save_outputs(base, parts)
    print(f"\n  Finales Dataset: {n_rows} Zeilen x {n_cols} Spalten")
    print(f"  Parquet: {DATASET_FILE}")
    print(f"  Summary: {SUMMARY_FILE}")
    print("\nPipeline abgeschlossen.\n")
    return preview


df_preview = run()
print(df_preview.head())


# %% ---- notebook cell [3] ----------------------------------------
LAB_SOURCE_FILE = Path(r"D:\Ausgangsdaten\KISIK Projekt\KISIK_Updated_16062025\lab.csv")
LAB_FEATURE_CATALOG_CSV = BASE_DIR / "kisik2_lab_feature_catalog.csv"
LAB_FEATURE_CATALOG_XLSX = BASE_DIR / "kisik2_lab_feature_catalog.xlsx"

FILES['labs'] = LAB_SOURCE_FILE
TOP_FEATURE_LIMITS.setdefault('lab', 200)
LAB_PRIORITY_KEYWORDS = [
    'laktat', 'lactat', 'lactate', 'radiometer', 'poct',
    'ph', 'pco2', 'po2', 'fio2', 'o2',
    'glukose', 'glucose', 'natrium', 'kalium', 'chlorid',
    'hco3', 'bicarbonat', 'base', 'anion',
]
COLS.update({
    'lab_name': 'beschreibung',
    'lab_code': 'code',
    'lab_value_primary': 'ergebnisf',
    'lab_value_secondary': 'ergebnist',
    'lab_time': 'erfassdat',
    'lab_analytic': 'analytx',
})


def matches_lab_priority(name: str) -> bool:
    feature_key = sanitize_name(name)
    return any(keyword in feature_key for keyword in LAB_PRIORITY_KEYWORDS)


def query_labs(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    sql = f"""
        with labs as (
            select
                s.stay_id,
                coalesce(
                    nullif(trim(l.{COLS['lab_name']}), ''),
                    nullif(trim(l.{COLS['lab_code']}), ''),
                    nullif(trim(l.{COLS['lab_analytic']}), '')
                ) as feature_name,
                coalesce(
                    try_cast(replace(l.{COLS['lab_value_primary']}, ',', '.') as double),
                    try_cast(replace(l.{COLS['lab_value_secondary']}, ',', '.') as double)
                ) as value_num,
                try_cast(l.{COLS['lab_time']} as timestamp) as ts
            from read_csv_auto('{path.as_posix()}', delim=';', header=true, all_varchar=true, ignore_errors=true) l
            join stays s
              on l.{COLS['case_id']} = s.{COLS['case_id']}
             and try_cast(l.{COLS['lab_time']} as timestamp) between s.{COLS['icu_start']} and s.{COLS['icu_end']}
        )
        select
            stay_id,
            feature_name,
            avg(value_num) as mean_value,
            median(value_num) as median_value,
            arg_min(value_num, ts) as first_value,
            arg_max(value_num, ts) as last_value,
            min(value_num) as min_value,
            max(value_num) as max_value
        from labs
        where feature_name is not null and value_num is not null and ts is not null
        group by 1, 2
    """
    return con.execute(sql).fetch_df()


def build_lab_feature_catalog(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                'Parameter',
                'Parameter_Key',
                'Anzahl_Aufenthalte',
                'Stay_Labor_Kombinationen',
                'Ist_Prioritaetslabor',
                'Min',
                'Median_ueber_Stays',
                'Max',
            ]
        )

    catalog = (
        long_df.groupby('feature_name', dropna=False)
        .agg(
            Anzahl_Aufenthalte=('stay_id', 'nunique'),
            Stay_Labor_Kombinationen=('stay_id', 'size'),
            Min=('min_value', 'min'),
            Median_ueber_Stays=('median_value', 'median'),
            Max=('max_value', 'max'),
        )
        .reset_index()
        .rename(columns={'feature_name': 'Parameter'})
    )
    catalog['Parameter_Key'] = catalog['Parameter'].map(sanitize_name)
    catalog['Ist_Prioritaetslabor'] = catalog['Parameter'].map(matches_lab_priority)
    catalog = catalog.sort_values(
        ['Ist_Prioritaetslabor', 'Anzahl_Aufenthalte', 'Parameter'],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return catalog


def save_lab_feature_catalog(catalog: pd.DataFrame) -> None:
    if catalog.empty:
        print('  -> Keine Laborparameter fuer Katalog gefunden')
        return

    catalog.to_csv(LAB_FEATURE_CATALOG_CSV, sep=';', index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(LAB_FEATURE_CATALOG_XLSX, engine='openpyxl') as writer:
        catalog.to_excel(writer, sheet_name='Laborparameter', index=False)



def limit_lab_features(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df

    top_n = TOP_FEATURE_LIMITS.get('lab')
    if top_n is None:
        return long_df

    counts = long_df.groupby('feature_name', dropna=False)['stay_id'].nunique().sort_values(ascending=False)
    keep = set(counts.head(top_n).index)
    priority_keep = {name for name in counts.index if matches_lab_priority(name)}
    keep.update(priority_keep)

    filtered = long_df[long_df['feature_name'].isin(keep)].copy()
    print(f"  -> lab: {len(counts)} Kategorien gesamt, {len(keep)} beibehalten ({len(priority_keep)} priorisiert)")
    return filtered



def run_with_labs():
    global lab_feature_catalog

    print("\n==============================================")
    print(" KISIK2 ICU ML Pipeline inklusive Laborwerte")
    print("==============================================\n")

    if not FILES['labs'].exists():
        raise FileNotFoundError(f"Labordatei nicht gefunden: {FILES['labs']}")

    print(f"-> Laborquelle: {FILES['labs']}")
    print("-> Schritt 1: Aufenthalte laden")
    stays_raw = load_csv(FILES['stays'])
    if stays_raw.empty:
        raise FileNotFoundError('fall_aufenthalt.csv ist zwingend erforderlich')

    stays = build_stays(stays_raw)
    print(f"  -> {len(stays)} Aufenthalte aus {stays[COLS['case_id']].nunique()} Faellen\n")

    con = duckdb.connect()
    register_stays(con, stays)

    print("-> Schritt 2: Vitalzeichen")
    vitals_long = query_vitals(con, FILES['vitals'])
    vitals_agg = long_metrics_to_wide(vitals_long, 'vital')
    print(f"  -> {len(vitals_long):,} Stay-Vital-Kombinationen aggregiert\n")

    print("-> Schritt 3: Scores")
    scores_long = query_scores(con, FILES['scores'])
    scores_agg = long_metrics_to_wide(scores_long, 'score')
    print(f"  -> {len(scores_long):,} Stay-Score-Kombinationen aggregiert\n")

    print("-> Schritt 4: Labore")
    labs_long = query_labs(con, FILES['labs'])
    lab_feature_catalog = build_lab_feature_catalog(labs_long)
    save_lab_feature_catalog(lab_feature_catalog)
    labs_long = limit_lab_features(labs_long)
    labs_agg = long_metrics_to_wide(labs_long, 'lab')
    print(f"  -> {len(labs_long):,} Stay-Labor-Kombinationen aggregiert")
    print(f"  -> Laborkatalog: {LAB_FEATURE_CATALOG_CSV.name} / {LAB_FEATURE_CATALOG_XLSX.name}\n")

    print("-> Schritt 5: Diagnosen")
    main_diag_long, side_diag_count = query_diagnoses(con, FILES['diagnoses'])
    main_diag_ohe = presence_to_wide(main_diag_long, 'diag_main')
    print(f"  -> {len(main_diag_long):,} Stay-Hauptdiagnose-Kombinationen")
    print("  -> Nebendiagnosen pro Aufenthalt vorbereitet\n")

    print("-> Schritt 6: Prozeduren")
    proc_long = query_procedures(con, FILES['procedures'])
    proc_ohe = presence_to_wide(proc_long, 'proc')
    print(f"  -> {len(proc_long):,} Stay-Prozedur-Kombinationen\n")

    print("-> Schritt 7: Zugaenge")
    access_long, access_count = query_access(con, FILES['access'])
    access_ohe = presence_to_wide(access_long, 'zugang')
    print(f"  -> {len(access_long):,} Stay-Zugangs-Kombinationen\n")
    con.close()

    print("-> Schritt 8: Persistente Zusammenfuehrung")
    base_cols = [
        'stay_id',
        COLS['case_id'],
        'stay_nr',
        COLS['patient_id'],
        COLS['age'],
        COLS['icu_start'],
        COLS['icu_end'],
        COLS['hospital_start'],
        COLS['hospital_end'],
        COLS['department'],
        COLS['ward'],
        'icu_duration_h',
        'hospital_duration_h',
    ]
    base_cols = [col for col in base_cols if col in stays.columns]
    base = optimize_frame(stays[base_cols].copy())

    parts = [
        (vitals_agg, 'Vitalzeichen'),
        (scores_agg, 'Scores'),
        (labs_agg, 'Labore'),
        (main_diag_ohe, 'Hauptdiagnosen'),
        (optimize_frame(side_diag_count), 'Nebendiagnosen'),
        (proc_ohe, 'Prozeduren'),
        (access_ohe, 'Zugaenge'),
        (optimize_frame(access_count), 'Zugang-Anzahl'),
    ]

    for part_df, label in parts:
        print(f"  + {label}: {max(len(part_df.columns) - 1, 0) if not part_df.empty else 0} Spalten")

    n_rows, n_cols, preview = save_outputs(base, parts)
    print(f"\n  Finales Dataset: {n_rows} Zeilen x {n_cols} Spalten")
    print(f"  Parquet: {DATASET_FILE}")
    print(f"  Summary: {SUMMARY_FILE}")
    print("\nPipeline mit Labordaten abgeschlossen.\n")
    return preview


lab_preview = run_with_labs()
display(lab_preview.head())


# %% ---- notebook cell [4] ----------------------------------------
ANA_WARDS = ["AIN"]
ANA_DEPARTMENTS = []

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"Parquet-Datei nicht gefunden: {DATASET_FILE}. Bitte zuerst Zelle 1 ausfuehren."
    )

ana_query = f"""
    SELECT *
    FROM read_parquet('{DATASET_FILE.as_posix()}')
    WHERE wardshort IN ({', '.join(repr(x) for x in ANA_WARDS)})
"""

if ANA_DEPARTMENTS:
    ana_query = f"""
        SELECT *
        FROM read_parquet('{DATASET_FILE.as_posix()}')
        WHERE wardshort IN ({', '.join(repr(x) for x in ANA_WARDS)})
           OR oebenekurz IN ({', '.join(repr(x) for x in ANA_DEPARTMENTS)})
    """

ana_df = duckdb.sql(ana_query).df()

if ana_df.empty:
    raise ValueError("Keine ANA-Faelle gefunden. Bitte ANA_WARDS bzw. ANA_DEPARTMENTS pruefen.")

num_cols = [col for col in ["alter", "icu_duration_h", "hospital_duration_h"] if col in ana_df.columns]
date_cols = [col for col in ["aufndat", "planbegin", "planend", "entldat"] if col in ana_df.columns]
for col in date_cols:
    ana_df[col] = pd.to_datetime(ana_df[col], errors="coerce")

def describe_series(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }

summary_rows = [
    {"Kennzahl": "ANA-Definition (wardshort)", "Wert": ', '.join(ANA_WARDS)},
    {"Kennzahl": "Anzahl Aufenthalte", "Wert": int(len(ana_df))},
    {"Kennzahl": "Anzahl eindeutige Faelle", "Wert": int(ana_df['fallid'].nunique()) if 'fallid' in ana_df.columns else np.nan},
    {"Kennzahl": "Anzahl eindeutige Patienten", "Wert": int(ana_df['pid'].nunique()) if 'pid' in ana_df.columns else np.nan},
]

for col, label in [("alter", "Alter"), ("icu_duration_h", "ICU-Dauer (h)"), ("hospital_duration_h", "Krankenhausdauer (h)")]:
    if col in ana_df.columns:
        d = describe_series(ana_df[col])
        summary_rows.extend([
            {"Kennzahl": f"{label}: Mittelwert", "Wert": round(d['mean'], 2)},
            {"Kennzahl": f"{label}: SD", "Wert": round(d['std'], 2)},
            {"Kennzahl": f"{label}: Median", "Wert": round(d['median'], 2)},
            {"Kennzahl": f"{label}: Q1-Q3", "Wert": f"{d['q1']:.2f} - {d['q3']:.2f}"},
            {"Kennzahl": f"{label}: Min-Max", "Wert": f"{d['min']:.2f} - {d['max']:.2f}"},
        ])

ana_summary = pd.DataFrame(summary_rows)

zeitraum_rows = []
for col, label in [("aufndat", "Krankenhausaufnahme"), ("planbegin", "ICU-Beginn"), ("planend", "ICU-Ende"), ("entldat", "Krankenhausentlassung")]:
    if col in ana_df.columns:
        non_null = ana_df[col].dropna()
        if not non_null.empty:
            zeitraum_rows.append({
                "Zeitpunkt": label,
                "Von": non_null.min(),
                "Bis": non_null.max(),
            })
ana_zeitraum = pd.DataFrame(zeitraum_rows)

ana_ward_counts = ana_df.groupby("wardshort", dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)
ana_dept_counts = ana_df.groupby("oebenekurz", dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)

feature_prefixes = {
    "Vitalzeichen": "vital_",
    "Scores": "score_",
    "Labore": "lab_",
    "Hauptdiagnosen": "diag_main_",
    "Prozeduren": "proc_",
    "Zugaenge": "zugang_",
}
feature_rows = []
for label, prefix in feature_prefixes.items():
    cols = [c for c in ana_df.columns if c.startswith(prefix)]
    feature_rows.append({"Featureblock": label, "Anzahl Spalten": len(cols)})
ana_feature_summary = pd.DataFrame(feature_rows)

print("ANA-Faelle deskriptiv")
print("=====================")
display(ana_summary)

print("\nZeitraum")
display(ana_zeitraum)

print("\nWard-Verteilung (Top 10)")
display(ana_ward_counts.head(10))

print("\nAbteilungs-Verteilung (Top 10)")
display(ana_dept_counts.head(10))

print("\nFeature-Blocks im gefilterten Datensatz")
display(ana_feature_summary)

print("\nBeispielzeilen")
display(ana_df.head())


# %% ---- notebook cell [5] ----------------------------------------
if 'ana_df' not in globals() or ana_df.empty:
    raise ValueError("ana_df ist nicht vorhanden. Bitte zuerst die ANA-Filterzelle ausfuehren.")

feature_prefixes = {
    'Vitalzeichen': 'vital_',
    'Scores': 'score_',
    'Labore': 'lab_',
    'Hauptdiagnosen': 'diag_main_',
    'Prozeduren': 'proc_',
    'Zugaenge': 'zugang_',
}
feature_rows = []
for label, prefix in feature_prefixes.items():
    cols = [c for c in ana_df.columns if c.startswith(prefix)]
    feature_rows.append({'Featureblock': label, 'Anzahl Spalten': len(cols)})
ana_feature_summary = pd.DataFrame(feature_rows)

print('Feature-Blocks inklusive Labore')
display(ana_feature_summary)


# %% ---- notebook cell [6] ----------------------------------------
if 'ana_df' not in globals() or ana_df.empty:
    raise ValueError("ana_df ist nicht vorhanden. Bitte zuerst Zelle 3 ausfuehren.")

ana_desc_df = ana_df.copy()

numeric_vars = [col for col in ['alter', 'icu_duration_h', 'hospital_duration_h', 'zugang_anzahl_gesamt', 'neben_diag_anzahl'] if col in ana_desc_df.columns]
for col in numeric_vars:
    ana_desc_df[col] = pd.to_numeric(ana_desc_df[col], errors='coerce')

if 'icu_duration_h' in ana_desc_df.columns:
    ana_desc_df['icu_duration_d'] = ana_desc_df['icu_duration_h'] / 24.0
if 'hospital_duration_h' in ana_desc_df.columns:
    ana_desc_df['hospital_duration_d'] = ana_desc_df['hospital_duration_h'] / 24.0

numeric_display = [col for col in ['alter', 'icu_duration_h', 'icu_duration_d', 'hospital_duration_h', 'hospital_duration_d', 'zugang_anzahl_gesamt', 'neben_diag_anzahl'] if col in ana_desc_df.columns]

numeric_summary = (
    ana_desc_df[numeric_display]
    .describe(percentiles=[0.25, 0.5, 0.75])
    .T
    .reset_index()
    .rename(columns={'index': 'Variable', 'count': 'n', '50%': 'median', '25%': 'q1', '75%': 'q3'})
)
if not numeric_summary.empty:
    numeric_summary = numeric_summary[['Variable', 'n', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max']].round(2)

categorical_vars = [col for col in ['wardshort', 'oebenekurz'] if col in ana_desc_df.columns]
categorical_tables = {}
for col in categorical_vars:
    tbl = (
        ana_desc_df[col]
        .fillna('NA')
        .value_counts(dropna=False)
        .rename_axis(col)
        .reset_index(name='n')
    )
    tbl['pct'] = (tbl['n'] / len(ana_desc_df) * 100).round(2)
    categorical_tables[col] = tbl.head(15)

missing_summary = pd.DataFrame({
    'Variable': ana_desc_df.columns,
    'fehlend_n': ana_desc_df.isna().sum().values,
    'fehlend_pct': (ana_desc_df.isna().mean().values * 100).round(2),
})
missing_summary = missing_summary.sort_values(['fehlend_pct', 'fehlend_n'], ascending=False)
missing_summary_top = missing_summary.head(20)

los_day_bands = None
if 'icu_duration_d' in ana_desc_df.columns:
    bins = [-np.inf, 1, 3, 7, 14, np.inf]
    labels = ['<=1 Tag', '>1 bis 3 Tage', '>3 bis 7 Tage', '>7 bis 14 Tage', '>14 Tage']
    los_day_bands = (
        pd.cut(ana_desc_df['icu_duration_d'], bins=bins, labels=labels)
        .value_counts(sort=False, dropna=False)
        .rename_axis('ICU-Dauer-Gruppe')
        .reset_index(name='n')
    )
    los_day_bands['pct'] = (los_day_bands['n'] / len(ana_desc_df) * 100).round(2)

print('Kompakte deskriptive ANA-Darstellung')
print('===================================')
print(f'Anzahl ANA-Aufenthalte: {len(ana_desc_df):,}')
print(f'Eindeutige Faelle: {ana_desc_df["fallid"].nunique():,}' if 'fallid' in ana_desc_df.columns else 'Eindeutige Faelle: n/a')
print(f'Eindeutige Patienten: {ana_desc_df["pid"].nunique():,}' if 'pid' in ana_desc_df.columns else 'Eindeutige Patienten: n/a')

print('\nNumerische Variablen')
display(numeric_summary)

for col, tbl in categorical_tables.items():
    print(f'\nTop-Kategorien: {col}')
    display(tbl)

print('\nMissingness (Top 20)')
display(missing_summary_top)

if los_day_bands is not None:
    print('\nICU-Dauer in Tagen')
    display(los_day_bands)


# %% ---- notebook cell [7] ----------------------------------------
if 'ana_df' not in globals() or ana_df.empty:
    raise ValueError("ana_df ist nicht vorhanden. Bitte zuerst die ANA-Zellen ausfuehren.")

lab_cols = [col for col in ana_df.columns if col.startswith('lab_')]


def split_lab_feature(col: str) -> tuple[str, str]:
    short = col[4:] if col.startswith('lab_') else col
    for suffix in ['_mean', '_median', '_last', '_first', '_max', '_min']:
        if short.endswith(suffix):
            return short[:-len(suffix)], suffix[1:]
    return short, 'value'


lab_rows = []
for col in lab_cols:
    s = pd.to_numeric(ana_df[col], errors='coerce')
    non_null = int(s.notna().sum())
    if non_null == 0:
        continue

    s_non_null = s.dropna()
    parameter_key, stat_name = split_lab_feature(col)
    lab_rows.append({
        'Parameter_Key': parameter_key,
        'Statistik': stat_name,
        'Spalte': col,
        'Nichtleer_n': non_null,
        'Nichtleer_pct': round(non_null / len(ana_df) * 100, 2),
        'Median': round(float(s_non_null.median()), 2),
        'IQR': f"{s_non_null.quantile(0.25):.2f} - {s_non_null.quantile(0.75):.2f}",
    })

lab_analysis_df = pd.DataFrame(lab_rows)
selected_lab_columns = lab_analysis_df['Spalte'].tolist() if not lab_analysis_df.empty else []

if lab_analysis_df.empty:
    ana_lab_compact_summary = pd.DataFrame()
    clinical_lab_selection = pd.DataFrame()
    clinical_lab_matches = pd.DataFrame()
    print('Keine befuellten lab_-Spalten im aktuellen ANA-Datensatz gefunden.')
else:
    ana_lab_compact_summary = (
        lab_analysis_df.groupby('Parameter_Key', dropna=False)
        .agg(
            Statistiken=('Statistik', lambda s: ', '.join(sorted(set(s)))),
            Spalten=('Spalte', 'count'),
            Nichtleer_n_max=('Nichtleer_n', 'max'),
            Nichtleer_pct_max=('Nichtleer_pct', 'max'),
        )
        .reset_index()
    )

    if 'lab_feature_catalog' in globals() and not lab_feature_catalog.empty:
        ana_lab_compact_summary = ana_lab_compact_summary.merge(
            lab_feature_catalog[['Parameter_Key', 'Parameter', 'Anzahl_Aufenthalte', 'Ist_Prioritaetslabor']].drop_duplicates('Parameter_Key'),
            on='Parameter_Key',
            how='left',
        )
    else:
        ana_lab_compact_summary['Parameter'] = ana_lab_compact_summary['Parameter_Key'].str.replace('_', ' ').str.title()
        ana_lab_compact_summary['Anzahl_Aufenthalte'] = np.nan
        ana_lab_compact_summary['Ist_Prioritaetslabor'] = False

    ana_lab_compact_summary = ana_lab_compact_summary.sort_values(
        ['Nichtleer_n_max', 'Anzahl_Aufenthalte', 'Parameter'],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    clinical_lab_selection = lab_analysis_df.sort_values(
        ['Nichtleer_n', 'Parameter_Key', 'Statistik'],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    clinical_lab_matches = ana_lab_compact_summary[
        ana_lab_compact_summary['Parameter_Key'].fillna('').str.contains('lakt|lact', case=False, regex=True)
        | ana_lab_compact_summary['Parameter'].fillna('').str.contains('lakt|lact', case=False, regex=True)
    ].copy()

    print('Laborparameter im ANA-Datensatz')
    print('==============================')
    display(ana_lab_compact_summary.head(40))

    print('\nLaktat-bezogene Parameter')
    if clinical_lab_matches.empty:
        print('Keine Laktat-Parameter unter den finalen lab_-Features gefunden.')
    else:
        display(clinical_lab_matches)

    print('\nBeispielhafte Laborstatistiken (Top 30 Spalten)')
    display(clinical_lab_selection.head(30))


# %% ---- notebook cell [9] ----------------------------------------
if 'ana_desc_df' not in globals() or ana_desc_df.empty:
    raise ValueError("ana_desc_df ist nicht vorhanden. Bitte die vorherige ANA-Zelle ausfuehren.")

import matplotlib.pyplot as plt

plot_df = ana_desc_df.copy()
if 'icu_duration_d' in plot_df.columns:
    icu_duration_days = pd.to_numeric(plot_df['icu_duration_d'], errors='coerce')
    plot_df['icu_duration_d_cap50'] = icu_duration_days.clip(upper=50)
    plot_df['icu_duration_d_cap30'] = icu_duration_days.clip(upper=30)

plot_specs = [
    ('alter', 'Alter (Jahre)'),
    ('icu_duration_d', 'ICU-Dauer (Tage)'),
    ('icu_duration_d_cap50', 'ICU-Dauer (Tage, max. 50)'),
    ('icu_duration_d_cap30', 'ICU-Dauer (Tage, max. 30)'),
    ('hospital_duration_d', 'Krankenhausdauer (Tage)'),
]
plot_specs = [(col, label) for col, label in plot_specs if col in plot_df.columns]

for col, _label in plot_specs:
    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

fig, axes = plt.subplots(len(plot_specs), 2, figsize=(14, 4 * len(plot_specs)))
if len(plot_specs) == 1:
    axes = np.array([axes])

for row_idx, (col, label) in enumerate(plot_specs):
    data = plot_df[col].dropna()

    ax_hist = axes[row_idx, 0]
    ax_box = axes[row_idx, 1]

    ax_hist.hist(data, bins=30, color='#5B7C99', edgecolor='white')
    ax_hist.set_title(f'{label} - Histogramm')
    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel('Hauefigkeit')
    ax_hist.grid(axis='y', alpha=0.2)

    ax_box.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='#D9A441', alpha=0.8))
    ax_box.set_title(f'{label} - Boxplot')
    ax_box.set_xlabel(label)
    ax_box.grid(axis='x', alpha=0.2)

plt.tight_layout()
plt.show()

if 'icu_duration_d' in plot_df.columns:
    icu_scatter = plot_df['icu_duration_d'].dropna().reset_index(drop=True)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.18, 0.18, size=len(icu_scatter))

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.scatter(
        icu_scatter,
        jitter,
        s=10,
        alpha=0.28,
        color='#D1495B',
        edgecolors='none',
    )
    ax.axvline(float(icu_scatter.median()), color='#1F2933', linestyle='--', linewidth=1.5, label='Median')
    ax.set_title('Gesamte ICU-Dauer (Tage) als Punktwolke')
    ax.set_xlabel('ICU-Dauer (Tage)')
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.grid(axis='x', alpha=0.25)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

plot_summary = []
for col, label in plot_specs:
    s = plot_df[col].dropna()
    plot_summary.append({
        'Variable': label,
        'n': int(s.shape[0]),
        'Median': round(float(s.median()), 2),
        'IQR': f"{s.quantile(0.25):.2f} - {s.quantile(0.75):.2f}",
    })

print('Zusammenfassung der visualisierten Variablen')
display(pd.DataFrame(plot_summary))


# %% ---- notebook cell [11] ----------------------------------------
if 'ana_desc_df' not in globals() or ana_desc_df.empty:
    raise ValueError("ana_desc_df ist nicht vorhanden. Bitte die vorherige ANA-Zelle ausfuehren.")

import matplotlib.pyplot as plt

plot_df = ana_desc_df.copy()

if 'icu_duration_d' not in plot_df.columns:
    if 'icu_duration_h' not in plot_df.columns:
        raise ValueError("Weder icu_duration_d noch icu_duration_h vorhanden.")
    plot_df['icu_duration_d'] = pd.to_numeric(plot_df['icu_duration_h'], errors='coerce') / 24.0

plot_df['icu_duration_d'] = pd.to_numeric(plot_df['icu_duration_d'], errors='coerce')
plot_df = plot_df[plot_df['icu_duration_d'] >= 2].copy()

if plot_df.empty:
    raise ValueError("Keine ANA-Faelle mit ICU-Dauer >= 2 Tage gefunden.")

plot_df['icu_duration_d_cap50'] = plot_df['icu_duration_d'].clip(upper=50)
plot_df['icu_duration_d_cap30'] = plot_df['icu_duration_d'].clip(upper=30)

plot_specs = [
    ('alter', 'Alter (Jahre)'),
    ('icu_duration_d', 'ICU-Dauer (Tage)'),
    ('icu_duration_d_cap50', 'ICU-Dauer (Tage, max. 50)'),
    ('icu_duration_d_cap30', 'ICU-Dauer (Tage, max. 30)'),
    ('hospital_duration_d', 'Krankenhausdauer (Tage)'),
]
plot_specs = [(col, label) for col, label in plot_specs if col in plot_df.columns]

for col, _label in plot_specs:
    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

fig, axes = plt.subplots(len(plot_specs), 2, figsize=(14, 4 * len(plot_specs)))
if len(plot_specs) == 1:
    axes = np.array([axes])

for row_idx, (col, label) in enumerate(plot_specs):
    data = plot_df[col].dropna()

    ax_hist = axes[row_idx, 0]
    ax_box = axes[row_idx, 1]

    ax_hist.hist(data, bins=30, color='#5B7C99', edgecolor='white')
    ax_hist.set_title(f'{label} - Histogramm (ICU >= 2 Tage)')
    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel('Hauefigkeit')
    ax_hist.grid(axis='y', alpha=0.2)

    ax_box.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='#D9A441', alpha=0.8))
    ax_box.set_title(f'{label} - Boxplot (ICU >= 2 Tage)')
    ax_box.set_xlabel(label)
    ax_box.grid(axis='x', alpha=0.2)

plt.tight_layout()
plt.show()

icu_scatter = plot_df['icu_duration_d'].dropna().reset_index(drop=True)
rng = np.random.default_rng(42)
jitter = rng.uniform(-0.18, 0.18, size=len(icu_scatter))

fig, ax = plt.subplots(figsize=(14, 4.8))
ax.scatter(
    icu_scatter,
    jitter,
    s=10,
    alpha=0.28,
    color='#D1495B',
    edgecolors='none',
)
ax.axvline(float(icu_scatter.median()), color='#1F2933', linestyle='--', linewidth=1.5, label='Median')
ax.set_title('Gesamte ICU-Dauer (Tage) als Punktwolke bei ICU >= 2 Tage')
ax.set_xlabel('ICU-Dauer (Tage)')
ax.set_yticks([])
ax.set_ylabel('')
ax.grid(axis='x', alpha=0.25)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

plot_summary = []
for col, label in plot_specs:
    s = plot_df[col].dropna()
    plot_summary.append({
        'Variable': label,
        'n': int(s.shape[0]),
        'Median': round(float(s.median()), 2),
        'IQR': f"{s.quantile(0.25):.2f} - {s.quantile(0.75):.2f}",
    })

print('Zusammenfassung der visualisierten Variablen (ICU >= 2 Tage)')
display(pd.DataFrame(plot_summary))


# %% ---- notebook cell [12] ----------------------------------------
if 'ana_desc_df' not in globals() or ana_desc_df.empty:
    raise ValueError("ana_desc_df ist nicht vorhanden. Bitte zuerst die ANA-Zellen ausfuehren.")

import math
import matplotlib.pyplot as plt

icu_ge2_df = ana_desc_df.copy()

if 'icu_duration_d' not in icu_ge2_df.columns:
    if 'icu_duration_h' not in icu_ge2_df.columns:
        raise ValueError("Weder icu_duration_d noch icu_duration_h vorhanden.")
    icu_ge2_df['icu_duration_d'] = pd.to_numeric(icu_ge2_df['icu_duration_h'], errors='coerce') / 24.0

if 'hospital_duration_d' not in icu_ge2_df.columns and 'hospital_duration_h' in icu_ge2_df.columns:
    icu_ge2_df['hospital_duration_d'] = pd.to_numeric(icu_ge2_df['hospital_duration_h'], errors='coerce') / 24.0

for col in ['alter', 'icu_duration_d', 'hospital_duration_d', 'neben_diag_anzahl', 'zugang_anzahl_gesamt']:
    if col in icu_ge2_df.columns:
        icu_ge2_df[col] = pd.to_numeric(icu_ge2_df[col], errors='coerce')

icu_ge2_df = icu_ge2_df[icu_ge2_df['icu_duration_d'] >= 2].copy()

if icu_ge2_df.empty:
    raise ValueError("Keine ANA-Faelle mit ICU-Dauer >= 2 Tage gefunden.")


def describe_metric(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return {'n': 0, 'mean': np.nan, 'std': np.nan, 'median': np.nan, 'q1': np.nan, 'q3': np.nan, 'min': np.nan, 'max': np.nan}
    return {
        'n': int(s.shape[0]),
        'mean': float(s.mean()),
        'std': float(s.std()),
        'median': float(s.median()),
        'q1': float(s.quantile(0.25)),
        'q3': float(s.quantile(0.75)),
        'min': float(s.min()),
        'max': float(s.max()),
    }


def prettify_diag_code(col_name: str) -> str:
    code = col_name.replace('diag_main_', '').replace('_', '').upper()
    if len(code) > 3:
        return f"{code[:3]}.{code[3:]}"
    return code


cohort_overview = pd.DataFrame([
    {'Kennzahl': 'Filter', 'Wert': 'ANA und ICU-Dauer >= 2 Tage'},
    {'Kennzahl': 'Anzahl Aufenthalte', 'Wert': int(len(icu_ge2_df))},
    {'Kennzahl': 'Eindeutige Faelle', 'Wert': int(icu_ge2_df['fallid'].nunique()) if 'fallid' in icu_ge2_df.columns else np.nan},
    {'Kennzahl': 'Eindeutige Patienten', 'Wert': int(icu_ge2_df['pid'].nunique()) if 'pid' in icu_ge2_df.columns else np.nan},
])

age_diag_rows = []
for col, label in [
    ('alter', 'Alter (Jahre)'),
    ('icu_duration_d', 'ICU-Dauer (Tage)'),
    ('hospital_duration_d', 'Krankenhausdauer (Tage)'),
    ('neben_diag_anzahl', 'Anzahl Nebendiagnosen'),
]:
    if col in icu_ge2_df.columns:
        d = describe_metric(icu_ge2_df[col])
        age_diag_rows.append({
            'Variable': label,
            'n': d['n'],
            'Mittelwert': round(d['mean'], 2),
            'SD': round(d['std'], 2),
            'Median': round(d['median'], 2),
            'IQR': f"{d['q1']:.2f} - {d['q3']:.2f}",
            'Min-Max': f"{d['min']:.2f} - {d['max']:.2f}",
        })

icu_ge2_summary = pd.DataFrame(age_diag_rows)

diag_cols = [col for col in icu_ge2_df.columns if col.startswith('diag_main_')]
icu_ge2_top_diag = pd.DataFrame()
if diag_cols:
    diag_rows = []
    for col in diag_cols:
        values = pd.to_numeric(icu_ge2_df[col], errors='coerce').fillna(0)
        present_n = int((values > 0).sum())
        if present_n > 0:
            diag_rows.append({
                'Hauptdiagnose': prettify_diag_code(col),
                'n': present_n,
                'pct': round(present_n / len(icu_ge2_df) * 100, 2),
                'Spalte': col,
            })
    if diag_rows:
        icu_ge2_top_diag = pd.DataFrame(diag_rows).sort_values(['n', 'Hauptdiagnose'], ascending=[False, True]).head(10)

print('Kollektivbeschreibung: ANA mit ICU-Dauer >= 2 Tage')
print('===============================================')
display(cohort_overview)

print('\nAlter, Aufenthaltsdauer und Nebendiagnosen')
display(icu_ge2_summary)

print('\nTop 10 Hauptdiagnosen')
if icu_ge2_top_diag.empty:
    print('Keine Hauptdiagnosen mit positiven Befunden gefunden.')
else:
    display(icu_ge2_top_diag[['Hauptdiagnose', 'n', 'pct']])

metric_specs = [
    ('alter', 'Alter (Jahre)', '#4E79A7'),
    ('icu_duration_d', 'ICU-Dauer (Tage)', '#E15759'),
    ('hospital_duration_d', 'Krankenhausdauer (Tage)', '#76B7B2'),
    ('neben_diag_anzahl', 'Anzahl Nebendiagnosen', '#F28E2B'),
    ('zugang_anzahl_gesamt', 'Anzahl Zugaenge', '#59A14F'),
]
metric_specs = [(col, label, color) for col, label, color in metric_specs if col in icu_ge2_df.columns]

if metric_specs:
    n_plots = len(metric_specs) + (0 if icu_ge2_top_diag.empty else 1)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.3 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    plot_idx = 0
    for col, label, color in metric_specs:
        ax = axes[plot_idx]
        s = pd.to_numeric(icu_ge2_df[col], errors='coerce').dropna().astype(float)
        if s.empty:
            ax.axis('off')
            plot_idx += 1
            continue
        clipped = s
        if s.nunique() > 20:
            lower = float(s.quantile(0.01))
            upper = float(s.quantile(0.99))
            clipped = s.clip(lower=lower, upper=upper)
        ax.hist(clipped, bins=30, color=color, edgecolor='white', alpha=0.9)
        ax.set_title(f'{label} - Histogramm')
        ax.set_xlabel(label)
        ax.set_ylabel('Hauefigkeit')
        ax.grid(axis='y', alpha=0.2)
        plot_idx += 1

    if not icu_ge2_top_diag.empty:
        ax = axes[plot_idx]
        plot_tbl = icu_ge2_top_diag.sort_values('n', ascending=True)
        ax.barh(plot_tbl['Hauptdiagnose'], plot_tbl['n'], color='#B07AA1')
        ax.set_title('Top 10 Hauptdiagnosen')
        ax.set_xlabel('Anzahl Aufenthalte')
        ax.grid(axis='x', alpha=0.2)
        plot_idx += 1

    for ax in axes[plot_idx:]:
        ax.axis('off')

    fig.suptitle('ANA-Kollektiv mit ICU-Dauer >= 2 Tage: Diagnosen und metrische Variablen', fontsize=17, fontweight='bold')
    plt.tight_layout()
    plt.show()


# %% ---- notebook cell [13] ----------------------------------------
if 'icu_ge2_df' not in globals() or icu_ge2_df.empty:
    if 'ana_desc_df' not in globals() or ana_desc_df.empty:
        raise ValueError("Weder icu_ge2_df noch ana_desc_df sind vorhanden. Bitte zuerst die ICU>=2-Kohortenzelle ausfuehren.")
    proc_access_df = ana_desc_df.copy()
    if 'icu_duration_d' not in proc_access_df.columns:
        if 'icu_duration_h' not in proc_access_df.columns:
            raise ValueError("Weder icu_duration_d noch icu_duration_h vorhanden.")
        proc_access_df['icu_duration_d'] = pd.to_numeric(proc_access_df['icu_duration_h'], errors='coerce') / 24.0
    icu_ge2_df = proc_access_df[pd.to_numeric(proc_access_df['icu_duration_d'], errors='coerce') >= 2].copy()

import matplotlib.pyplot as plt

proc_access_df = icu_ge2_df.copy()
proc_cols = [col for col in proc_access_df.columns if col.startswith('proc_')]
zugang_cols = [
    col for col in proc_access_df.columns
    if col.startswith('zugang_') and col != 'zugang_anzahl_gesamt'
]


def prettify_proc_name(col_name: str) -> str:
    return col_name.replace('proc_', '').upper().replace('_', '.')


def prettify_zugang_name(col_name: str) -> str:
    label = col_name.replace('zugang_', '').replace('_', ' ').strip()
    return label.title()


def summarize_binary_features(df: pd.DataFrame, columns: list[str], label_fn, cohort_n: int) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(columns=['Merkmal', 'Spalte', 'n', 'pct'])
    binary_df = df[columns].apply(pd.to_numeric, errors='coerce').fillna(0).gt(0)
    counts = binary_df.sum(axis=0).sort_values(ascending=False)
    result = counts.reset_index()
    result.columns = ['Spalte', 'n']
    result['Merkmal'] = result['Spalte'].map(label_fn)
    result['pct'] = (result['n'] / cohort_n * 100).round(2)
    return result[['Merkmal', 'Spalte', 'n', 'pct']]


icu_ge2_top_proc = summarize_binary_features(proc_access_df, proc_cols, prettify_proc_name, len(proc_access_df))
icu_ge2_top_zugang = summarize_binary_features(proc_access_df, zugang_cols, prettify_zugang_name, len(proc_access_df))

proc_binary = proc_access_df[proc_cols].apply(pd.to_numeric, errors='coerce').fillna(0).gt(0) if proc_cols else pd.DataFrame(index=proc_access_df.index)
zugang_binary = proc_access_df[zugang_cols].apply(pd.to_numeric, errors='coerce').fillna(0).gt(0) if zugang_cols else pd.DataFrame(index=proc_access_df.index)

proc_per_stay = proc_binary.sum(axis=1) if not proc_binary.empty else pd.Series(0, index=proc_access_df.index, dtype='int32')
zugang_types_per_stay = zugang_binary.sum(axis=1) if not zugang_binary.empty else pd.Series(0, index=proc_access_df.index, dtype='int32')

if 'zugang_anzahl_gesamt' in proc_access_df.columns:
    zugang_events_per_stay = pd.to_numeric(proc_access_df['zugang_anzahl_gesamt'], errors='coerce').fillna(0)
else:
    zugang_events_per_stay = zugang_types_per_stay.astype(float)

icu_ge2_proc_access_overview = pd.DataFrame([
    {
        'Bereich': 'Prozeduren',
        'Feature-Spalten': int(len(proc_cols)),
        'Aufenthalte_mit_Befund': int((proc_per_stay > 0).sum()),
        'Anteil_Aufenthalte_pct': round((proc_per_stay > 0).mean() * 100, 2),
        'Median_Merkmale_pro_Aufenthalt': round(float(proc_per_stay.median()), 2),
        'IQR_Merkmale_pro_Aufenthalt': f"{proc_per_stay.quantile(0.25):.2f} - {proc_per_stay.quantile(0.75):.2f}",
        'Max_Merkmale_pro_Aufenthalt': int(proc_per_stay.max()),
    },
    {
        'Bereich': 'Zugaenge (Typen)',
        'Feature-Spalten': int(len(zugang_cols)),
        'Aufenthalte_mit_Befund': int((zugang_types_per_stay > 0).sum()),
        'Anteil_Aufenthalte_pct': round((zugang_types_per_stay > 0).mean() * 100, 2),
        'Median_Merkmale_pro_Aufenthalt': round(float(zugang_types_per_stay.median()), 2),
        'IQR_Merkmale_pro_Aufenthalt': f"{zugang_types_per_stay.quantile(0.25):.2f} - {zugang_types_per_stay.quantile(0.75):.2f}",
        'Max_Merkmale_pro_Aufenthalt': int(zugang_types_per_stay.max()),
    },
    {
        'Bereich': 'Zugaenge (Ereignisse)',
        'Feature-Spalten': np.nan,
        'Aufenthalte_mit_Befund': int((zugang_events_per_stay > 0).sum()),
        'Anteil_Aufenthalte_pct': round((zugang_events_per_stay > 0).mean() * 100, 2),
        'Median_Merkmale_pro_Aufenthalt': round(float(zugang_events_per_stay.median()), 2),
        'IQR_Merkmale_pro_Aufenthalt': f"{zugang_events_per_stay.quantile(0.25):.2f} - {zugang_events_per_stay.quantile(0.75):.2f}",
        'Max_Merkmale_pro_Aufenthalt': int(zugang_events_per_stay.max()),
    },
])

print('Uebersicht ueber Zugaenge und Prozeduren bei ICU-Dauer >= 2 Tage')
print('==============================================================')
display(icu_ge2_proc_access_overview)

print('\nTop 15 Prozeduren nach Anzahl Aufenthalte')
if icu_ge2_top_proc.empty:
    print('Keine proc_-Spalten mit positiven Befunden gefunden.')
else:
    display(icu_ge2_top_proc.head(15)[['Merkmal', 'n', 'pct']])

print('\nTop 15 Zugangstypen nach Anzahl Aufenthalte')
if icu_ge2_top_zugang.empty:
    print('Keine zugang_-Spalten mit positiven Befunden gefunden.')
else:
    display(icu_ge2_top_zugang.head(15)[['Merkmal', 'n', 'pct']])

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('ANA-Kollektiv mit ICU-Dauer >= 2 Tage: Zugaenge und Prozeduren', fontsize=17, fontweight='bold')

ax = axes[0, 0]
proc_plot = proc_per_stay.astype(float)
if proc_plot.sum() == 0:
    ax.axis('off')
else:
    ax.hist(proc_plot.clip(upper=proc_plot.quantile(0.99)), bins=25, color='#4E79A7', edgecolor='white', alpha=0.9)
    ax.set_title('Prozeduren pro Aufenthalt')
    ax.set_xlabel('Anzahl Prozedur-Merkmale')
    ax.set_ylabel('Hauefigkeit')
    ax.grid(axis='y', alpha=0.2)

ax = axes[0, 1]
zugang_plot = zugang_events_per_stay.astype(float)
if zugang_plot.sum() == 0:
    ax.axis('off')
else:
    ax.hist(zugang_plot.clip(upper=zugang_plot.quantile(0.99)), bins=25, color='#59A14F', edgecolor='white', alpha=0.9)
    ax.set_title('Zugaenge pro Aufenthalt')
    ax.set_xlabel('Anzahl Zugangsereignisse')
    ax.set_ylabel('Hauefigkeit')
    ax.grid(axis='y', alpha=0.2)

ax = axes[1, 0]
if icu_ge2_top_proc.empty:
    ax.axis('off')
else:
    proc_bar = icu_ge2_top_proc.head(12).sort_values('n', ascending=True)
    ax.barh(proc_bar['Merkmal'], proc_bar['n'], color='#E15759')
    ax.set_title('Top-Prozeduren')
    ax.set_xlabel('Anzahl Aufenthalte')
    ax.grid(axis='x', alpha=0.2)

ax = axes[1, 1]
if icu_ge2_top_zugang.empty:
    ax.axis('off')
else:
    zugang_bar = icu_ge2_top_zugang.head(12).sort_values('n', ascending=True)
    ax.barh(zugang_bar['Merkmal'], zugang_bar['n'], color='#76B7B2')
    ax.set_title('Top-Zugangstypen')
    ax.set_xlabel('Anzahl Aufenthalte')
    ax.grid(axis='x', alpha=0.2)

plt.tight_layout()
plt.show()


# %% ---- notebook cell [14] ----------------------------------------
if 'icu_ge2_df' not in globals() or icu_ge2_df.empty:
    if 'ana_desc_df' not in globals() or ana_desc_df.empty:
        raise ValueError("Weder icu_ge2_df noch ana_desc_df sind vorhanden. Bitte zuerst die ICU>=2-Kohortenzelle ausfuehren.")
    work_df = ana_desc_df.copy()
    if 'icu_duration_d' not in work_df.columns:
        if 'icu_duration_h' not in work_df.columns:
            raise ValueError("Weder icu_duration_d noch icu_duration_h vorhanden.")
        work_df['icu_duration_d'] = pd.to_numeric(work_df['icu_duration_h'], errors='coerce') / 24.0
    icu_ge2_df = work_df[pd.to_numeric(work_df['icu_duration_d'], errors='coerce') >= 2].copy()

import math
import re
import matplotlib.pyplot as plt

analysis_df = icu_ge2_df.copy()
for col in analysis_df.columns:
    if col.startswith('vital_') or col.startswith('lab_'):
        analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')


def prettify_measure_name(name: str) -> str:
    label = name
    for prefix in ['vital_', 'lab_', 'diag_main_']:
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    for suffix in ['_mean', '_median', '_last', '_first', '_max', '_min', '_std', '_count']:
        if label.endswith(suffix):
            label = label[:-len(suffix)]
            break
    label = label.replace('_', ' ').strip()
    return label.upper() if label.isalpha() and len(label) <= 5 else label.title()


def split_measure_name(name: str) -> tuple[str, str]:
    short = name
    for prefix in ['vital_', 'lab_']:
        if short.startswith(prefix):
            short = short[len(prefix):]
            break
    for suffix in ['_mean', '_median', '_last', '_first', '_max', '_min', '_std', '_count']:
        if short.endswith(suffix):
            return short[:-len(suffix)], suffix[1:]
    return short, 'value'


def matches_keywords(base_name: str, keywords: list[str]) -> bool:
    tokens = [token for token in base_name.split('_') if token]
    for keyword in keywords:
        for token in tokens:
            if len(keyword) <= 3:
                if token == keyword:
                    return True
            elif token == keyword or token.startswith(keyword):
                return True
    return False


def contains_lactate_text(name: str) -> bool:
    lowered = name.lower()
    tokens = [token for token in re.split(r'[^a-z0-9]+', lowered) if token]
    return 'lakt' in lowered or 'lact' in lowered or 'rla' in tokens


def rank_measure_columns(df: pd.DataFrame, columns: list[str], keywords: list[str], min_unique: int = 5) -> pd.DataFrame:
    suffix_priority = {
        'mean': 0,
        'median': 1,
        'last': 2,
        'first': 3,
        'max': 4,
        'min': 5,
        'std': 6,
        'count': 7,
        'value': 8,
    }
    rows = []
    for col in columns:
        base_name, stat_name = split_measure_name(col)
        if not matches_keywords(base_name, keywords):
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna().astype(float)
        if s.empty or s.nunique() < min_unique:
            continue
        rows.append({
            'col': col,
            'base_name': base_name,
            'stat_name': stat_name,
            'n': int(s.shape[0]),
            'unique_n': int(s.nunique()),
            'stat_priority': suffix_priority.get(stat_name, 99),
        })
    if not rows:
        return pd.DataFrame(columns=['col', 'base_name', 'stat_name', 'n', 'unique_n', 'stat_priority'])
    return pd.DataFrame(rows).sort_values(
        ['n', 'stat_priority', 'unique_n', 'base_name'],
        ascending=[False, True, False, True],
    ).drop_duplicates('base_name', keep='first').reset_index(drop=True)


def select_representative_columns(df: pd.DataFrame, columns: list[str], limit: int, keywords: list[str]) -> list[str]:
    ranking_df = rank_measure_columns(df, columns, keywords)
    if ranking_df.empty:
        return []
    return ranking_df.head(limit)['col'].tolist()


def build_lactate_diagnostics(df: pd.DataFrame, columns: list[str], source_label: str) -> pd.DataFrame:
    suffix_priority = {
        'mean': 0,
        'median': 1,
        'last': 2,
        'first': 3,
        'max': 4,
        'min': 5,
        'std': 6,
        'count': 7,
        'value': 8,
    }
    rows = []
    for col in columns:
        if not contains_lactate_text(col):
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna().astype(float)
        base_name, stat_name = split_measure_name(col)
        rows.append({
            'Quelle': source_label,
            'Spalte': col,
            'Parameter': prettify_measure_name(col),
            'Basisname': base_name,
            'Statistik': stat_name,
            'n': int(s.shape[0]),
            'Anteil_pct': round((s.shape[0] / len(df) * 100), 2) if len(df) else 0.0,
            'unique_n': int(s.nunique()) if not s.empty else 0,
            'Median': round(float(s.median()), 2) if not s.empty else np.nan,
            'IQR': f"{s.quantile(0.25):.2f} - {s.quantile(0.75):.2f}" if not s.empty else 'n/a',
            'stat_priority': suffix_priority.get(stat_name, 99),
        })
    if not rows:
        return pd.DataFrame(columns=['Quelle', 'Spalte', 'Parameter', 'Basisname', 'Statistik', 'n', 'Anteil_pct', 'unique_n', 'Median', 'IQR', 'stat_priority'])
    return pd.DataFrame(rows).sort_values(
        ['n', 'stat_priority', 'unique_n', 'Spalte'],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)


def choose_lactate_column(lactate_diag: pd.DataFrame):
    if lactate_diag.empty:
        return None
    usable = lactate_diag[lactate_diag['unique_n'] >= 2].copy()
    if usable.empty:
        usable = lactate_diag.copy()
    return usable.iloc[0]['Spalte']


def find_lactate_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    object_cols = [col for col in df.columns if df[col].dtype == 'object']
    if not object_cols:
        return pd.DataFrame()
    mask = pd.Series(False, index=df.index)
    for col in object_cols:
        mask = mask | df[col].fillna('').astype(str).map(contains_lactate_text)
    return df.loc[mask].copy()


def load_raw_lactate_fallback(stays_df: pd.DataFrame) -> pd.DataFrame:
    if 'FILES' not in globals() or 'labs' not in FILES or not FILES['labs'].exists():
        return pd.DataFrame(columns=['stay_id'])

    required_cols = ['stay_id', COLS['case_id'], COLS['icu_start'], COLS['icu_end']]
    if any(col not in stays_df.columns for col in required_cols):
        return pd.DataFrame(columns=['stay_id'])

    stay_slice = stays_df[required_cols].drop_duplicates('stay_id').copy()
    stay_slice[COLS['icu_start']] = pd.to_datetime(stay_slice[COLS['icu_start']], errors='coerce')
    stay_slice[COLS['icu_end']] = pd.to_datetime(stay_slice[COLS['icu_end']], errors='coerce')
    stay_slice = stay_slice.dropna(subset=[COLS['case_id'], COLS['icu_start'], COLS['icu_end']])
    if stay_slice.empty:
        return pd.DataFrame(columns=['stay_id'])

    con = duckdb.connect()
    con.register('fallback_stays_df', stay_slice)
    con.execute('create or replace temp table fallback_stays as select * from fallback_stays_df')

    sql = f"""
        with lactate_source as (
            select
                s.stay_id,
                coalesce(
                    try_cast(replace(l.{COLS['lab_value_primary']}, ',', '.') as double),
                    try_cast(replace(l.{COLS['lab_value_secondary']}, ',', '.') as double)
                ) as value_num,
                try_cast(l.{COLS['lab_time']} as timestamp) as ts
            from read_csv_auto('{FILES['labs'].as_posix()}', delim=';', header=true, all_varchar=true, ignore_errors=true) l
            join fallback_stays s
              on l.{COLS['case_id']} = s.{COLS['case_id']}
             and try_cast(l.{COLS['lab_time']} as timestamp) between s.{COLS['icu_start']} and s.{COLS['icu_end']}
            where regexp_matches(
                lower(
                    coalesce(l.{COLS['lab_name']}, '') || ' ' ||
                    coalesce(l.{COLS['lab_code']}, '') || ' ' ||
                    coalesce(l.{COLS['lab_analytic']}, '')
                ),
                '(lakt|lact|(^|[^a-z0-9])rla([^a-z0-9]|$))'
            )
        )
        select
            stay_id,
            avg(value_num) as lab_lactat_fallback_mean,
            median(value_num) as lab_lactat_fallback_median,
            min(value_num) as lab_lactat_fallback_min,
            max(value_num) as lab_lactat_fallback_max,
            count(*) as lab_lactat_fallback_count
        from lactate_source
        where value_num is not null and ts is not null
        group by 1
    """

    result = con.execute(sql).fetch_df()
    con.close()
    return result


vital_cols = [col for col in analysis_df.columns if col.startswith('vital_')]
lab_cols = [col for col in analysis_df.columns if col.startswith('lab_')]
wide_lactate_diagnostics = build_lactate_diagnostics(analysis_df, lab_cols, source_label='Wide-Dataset')
raw_lactate_fallback = load_raw_lactate_fallback(analysis_df)

if not raw_lactate_fallback.empty:
    analysis_df = analysis_df.merge(raw_lactate_fallback, on='stay_id', how='left')
    for col in raw_lactate_fallback.columns:
        if col != 'stay_id':
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')

vital_cols = [col for col in analysis_df.columns if col.startswith('vital_')]
lab_cols = [col for col in analysis_df.columns if col.startswith('lab_')]
all_lactate_diagnostics = pd.concat(
    [
        wide_lactate_diagnostics,
        build_lactate_diagnostics(
            analysis_df,
            [col for col in lab_cols if 'fallback' in col.lower()],
            source_label='Raw lab.csv Fallback',
        ),
    ],
    ignore_index=True,
)
all_lactate_diagnostics = all_lactate_diagnostics.sort_values(
    ['n', 'stat_priority', 'unique_n', 'Spalte'],
    ascending=[False, True, False, True],
).reset_index(drop=True) if not all_lactate_diagnostics.empty else all_lactate_diagnostics
lactate_col = choose_lactate_column(all_lactate_diagnostics)

catalog_lactate_rows = pd.DataFrame()
if 'lab_feature_catalog' in globals():
    catalog_lactate_rows = find_lactate_rows(lab_feature_catalog)

presence_rows = [
    {
        'Bereich': 'Wide-Dataset im ICU>=2-Kollektiv',
        'Laktat_Spalten_n': int(len(wide_lactate_diagnostics)),
        'Nichtleer_n_summe': int(wide_lactate_diagnostics['n'].sum()) if not wide_lactate_diagnostics.empty else 0,
        'Kommentar': 'Vorhandene lab_-Spalten im finalen Analyse-DataFrame',
    },
    {
        'Bereich': 'Raw lab.csv Fallback im ICU>=2-Kollektiv',
        'Laktat_Spalten_n': int((len(all_lactate_diagnostics) - len(wide_lactate_diagnostics))) if not all_lactate_diagnostics.empty else 0,
        'Nichtleer_n_summe': int(all_lactate_diagnostics.loc[all_lactate_diagnostics['Quelle'] == 'Raw lab.csv Fallback', 'n'].sum()) if not all_lactate_diagnostics.empty else 0,
        'Kommentar': 'On-the-fly aus lab.csv fuer die aktuelle Kohorte nachgezogen',
    },
]
if not catalog_lactate_rows.empty:
    presence_rows.append({
        'Bereich': 'Upstream-Laborkatalog',
        'Laktat_Spalten_n': int(len(catalog_lactate_rows)),
        'Nichtleer_n_summe': np.nan,
        'Kommentar': 'Laktat ist upstream dokumentiert',
    })
lactate_presence_overview = pd.DataFrame(presence_rows)

vital_keywords = [
    'herz', 'heart', 'puls', 'pulse', 'rr', 'atem', 'resp', 'spo2', 'sao2',
    'temp', 'temperatur', 'fio2', 'blutdruck', 'syst', 'diast'
]
lab_keywords = [
    'hb', 'hgb', 'krea', 'kreat', 'creat', 'crp', 'leuko', 'thrombo', 'plt',
    'natrium', 'sodium', 'kalium', 'potassium', 'bilirubin', 'laktat',
    'lactat', 'lactate', 'glucose', 'glukose', 'ph', 'pco2', 'po2',
    'bicarbonat', 'base', 'anion', 'fallback'
]

selected_vitals = select_representative_columns(analysis_df, vital_cols, limit=6, keywords=vital_keywords)
lab_ranking_df = rank_measure_columns(analysis_df, lab_cols, lab_keywords)
non_lactate_labs = [
    col for col in (lab_ranking_df['col'].tolist() if not lab_ranking_df.empty else [])
    if col != lactate_col
]
selected_labs = non_lactate_labs[:8]
forced_columns = set()
if lactate_col is not None:
    selected_labs = [lactate_col] + selected_labs
    forced_columns.add(lactate_col)
selected_labs = list(dict.fromkeys(selected_labs))

selected_columns = [('Vital', col) for col in selected_vitals] + [('Labor', col) for col in selected_labs]
selected_columns = list(dict.fromkeys(selected_columns))

if not selected_columns:
    raise ValueError("Keine geeigneten Vital- oder Laborparameter fuer ICU>=2 gefunden.")

summary_rows = []
for group_label, col in selected_columns:
    s = pd.to_numeric(analysis_df[col], errors='coerce').dropna().astype(float)
    if s.empty:
        continue
    selection_label = 'Ranking'
    if col in forced_columns:
        selection_label = 'Laktat erzwungen'
    if 'fallback' in col.lower():
        selection_label = 'Raw lab.csv Fallback'
    summary_rows.append({
        'Gruppe': group_label,
        'Parameter': prettify_measure_name(col),
        'Spalte': col,
        'Auswahl': selection_label,
        'n': int(s.shape[0]),
        'Mittelwert': round(float(s.mean()), 2),
        'Median': round(float(s.median()), 2),
        'IQR': f"{s.quantile(0.25):.2f} - {s.quantile(0.75):.2f}",
        'Min-Max': f"{s.min():.2f} - {s.max():.2f}",
    })

icu_ge2_vital_lab_summary = pd.DataFrame(summary_rows)
if not icu_ge2_vital_lab_summary.empty:
    icu_ge2_vital_lab_summary['Ist_Laktat'] = icu_ge2_vital_lab_summary['Spalte'].map(lambda value: contains_lactate_text(value))
    icu_ge2_vital_lab_summary = icu_ge2_vital_lab_summary.sort_values(
        ['Ist_Laktat', 'Gruppe', 'n'],
        ascending=[False, True, False],
    ).drop(columns='Ist_Laktat').reset_index(drop=True)

print('Laktat-Diagnostik im Kollektiv ICU-Dauer >= 2 Tage')
print('=================================================')
display(lactate_presence_overview)
if all_lactate_diagnostics.empty:
    print('Keine Laktat-bezogene Spalte im finalen ICU>=2-DataFrame und auch kein Fallback aus lab.csv gefunden.')
else:
    display(all_lactate_diagnostics.drop(columns='stat_priority'))
    print(f"Verwendete Laktat-Spalte fuer die Darstellung: {lactate_col}")

if not catalog_lactate_rows.empty:
    print('\nLaktat-Hinweis aus dem Upstream-Katalog')
    display(catalog_lactate_rows.head(10))

print('\nAusgewaehlte Vital- und Laborparameter bei ICU-Dauer >= 2 Tage')
print('===========================================================')
display(icu_ge2_vital_lab_summary)

plot_columns = selected_columns.copy()
if lactate_col is not None and ('Labor', lactate_col) not in plot_columns:
    plot_columns.append(('Labor', lactate_col))

n_plots = len(plot_columns)
n_cols = 2
n_rows = math.ceil(n_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows))
axes = np.atleast_1d(axes).ravel()
group_colors = {'Vital': '#4E79A7', 'Labor': '#E15759'}

for ax, (group_label, col) in zip(axes, plot_columns):
    s = pd.to_numeric(analysis_df[col], errors='coerce').dropna().astype(float)
    if s.empty:
        ax.axis('off')
        continue
    clipped = s
    if s.nunique() > 20:
        clipped = s.clip(lower=float(s.quantile(0.01)), upper=float(s.quantile(0.99)))
    ax.hist(clipped, bins=30, color=group_colors[group_label], edgecolor='white', alpha=0.9)
    title_prefix = 'Laktat' if contains_lactate_text(col) else group_label
    ax.set_title(f"{title_prefix}: {prettify_measure_name(col)}")
    ax.set_xlabel(prettify_measure_name(col))
    ax.set_ylabel('Hauefigkeit')
    ax.grid(axis='y', alpha=0.2)

for ax in axes[n_plots:]:
    ax.axis('off')

fig.suptitle('Vital- und Laborparameter bei ANA mit ICU-Dauer >= 2 Tage', fontsize=17, fontweight='bold')
plt.tight_layout()
plt.show()

corr_threshold = max(100, int(len(analysis_df) * 0.1))
corr_candidates = []
for _group_label, col in selected_columns:
    s = pd.to_numeric(analysis_df[col], errors='coerce').astype(float)
    if s.notna().sum() >= corr_threshold:
        corr_candidates.append(col)

lactate_overlap_df = pd.DataFrame()
lactate_overlap_for_corr = 0
if lactate_col is not None:
    lactate_series = pd.to_numeric(analysis_df[lactate_col], errors='coerce').astype(float)
    overlap_rows = []
    for _group_label, col in selected_columns:
        if col == lactate_col:
            continue
        overlap_n = int((lactate_series.notna() & pd.to_numeric(analysis_df[col], errors='coerce').notna()).sum())
        overlap_rows.append({
            'Mit_Parameter': prettify_measure_name(col),
            'Spalte': col,
            'Gemeinsame_Werte_n': overlap_n,
        })
    lactate_overlap_df = pd.DataFrame(overlap_rows).sort_values(['Gemeinsame_Werte_n', 'Mit_Parameter'], ascending=[False, True]).reset_index(drop=True)
    lactate_overlap_for_corr = int(lactate_overlap_df['Gemeinsame_Werte_n'].max()) if not lactate_overlap_df.empty else 0
    if lactate_series.notna().sum() >= 15:
        corr_candidates.append(lactate_col)

corr_candidates = list(dict.fromkeys(corr_candidates))
corr_df = analysis_df[corr_candidates].apply(pd.to_numeric, errors='coerce').astype(float) if corr_candidates else pd.DataFrame()
corr_min_periods = max(15, int(len(analysis_df) * 0.02)) if not corr_df.empty else 15
corr_matrix = corr_df.corr(min_periods=corr_min_periods) if not corr_df.empty else pd.DataFrame()

print('\nKorrelationsmatrix der ausgewaehlten Parameter')
if lactate_col is not None:
    print(f"Minimale gemeinsame Beobachtungen fuer Korrelationen: {corr_min_periods}")
    if lactate_overlap_df.empty:
        print('Keine gemeinsame Belegung zwischen Laktat und den ausgewaehlten Parametern gefunden.')
    else:
        print(f"Maximale gemeinsame Belegung mit Laktat: {lactate_overlap_for_corr}")
        display(lactate_overlap_df.head(10))
else:
    print('Laktat konnte auch per Raw-Fallback nicht in die Analyse aufgenommen werden.')

if corr_matrix.empty or corr_matrix.shape[0] < 2:
    print('Nicht genug ueberlappende Daten fuer eine aussagekraeftige Korrelationsmatrix vorhanden.')
else:
    corr_labels = [prettify_measure_name(col) for col in corr_matrix.columns]
    fig, ax = plt.subplots(figsize=(1.1 * len(corr_labels) + 4, 1.0 * len(corr_labels) + 3))
    im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_labels)))
    ax.set_yticks(range(len(corr_labels)))
    ax.set_xticklabels(corr_labels, rotation=45, ha='right')
    ax.set_yticklabels(corr_labels)
    ax.set_title('Korrelationsmatrix: Vital- und Laborparameter (ICU >= 2 Tage)')

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            value = corr_matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson-Korrelation')
    plt.tight_layout()
    plt.show()


# %% ---- notebook cell [18] ----------------------------------------
if 'ana_desc_df' not in globals() or ana_desc_df.empty:
    raise ValueError("ana_desc_df ist nicht vorhanden. Bitte die vorherige ANA-Zelle ausfuehren.")

clinical_df = ana_desc_df.copy()

core_vars = {
    'alter': 'Alter (Jahre)',
    'icu_duration_h': 'ICU-Dauer (h)',
    'icu_duration_d': 'ICU-Dauer (Tage)',
    'hospital_duration_h': 'Krankenhausdauer (h)',
    'hospital_duration_d': 'Krankenhausdauer (Tage)',
    'zugang_anzahl_gesamt': 'Anzahl Zugaenge',
    'neben_diag_anzahl': 'Anzahl Nebendiagnosen',
}

available_core_vars = [col for col in core_vars if col in clinical_df.columns]
for col in available_core_vars:
    clinical_df[col] = pd.to_numeric(clinical_df[col], errors='coerce')

clinical_rows = []
for col in available_core_vars:
    s = clinical_df[col].dropna()
    if s.empty:
        continue
    clinical_rows.append({
        'Variable': core_vars[col],
        'n': int(s.shape[0]),
        'Mittelwert': round(float(s.mean()), 2),
        'SD': round(float(s.std()), 2),
        'Median': round(float(s.median()), 2),
        'IQR': f"{s.quantile(0.25):.2f} - {s.quantile(0.75):.2f}",
        'Min-Max': f"{s.min():.2f} - {s.max():.2f}",
    })

clinical_overview = pd.DataFrame(clinical_rows)

administrative_overview = pd.DataFrame([
    {'Kennzahl': 'ANA-Definition', 'Wert': 'wardshort = AIN'},
    {'Kennzahl': 'ANA-Aufenthalte', 'Wert': int(len(clinical_df))},
    {'Kennzahl': 'Eindeutige Faelle', 'Wert': int(clinical_df['fallid'].nunique()) if 'fallid' in clinical_df.columns else np.nan},
    {'Kennzahl': 'Eindeutige Patienten', 'Wert': int(clinical_df['pid'].nunique()) if 'pid' in clinical_df.columns else np.nan},
])

print('Klinische Kerntabelle ANA')
print('=========================')
display(administrative_overview)
display(clinical_overview)

if 'oebenekurz' in clinical_df.columns:
    top_depts = (
        clinical_df['oebenekurz']
        .fillna('NA')
        .value_counts(dropna=False)
        .rename_axis('oebenekurz')
        .reset_index(name='n')
    )
    top_depts['pct'] = (top_depts['n'] / len(clinical_df) * 100).round(2)
    print('\nHaeufigste Abteilungen innerhalb AIN')
    display(top_depts.head(10))


# %% ---- notebook cell [21] ----------------------------------------
if 'ana_desc_df' not in globals() or ana_desc_df.empty:
    raise ValueError("ana_desc_df ist nicht vorhanden. Bitte zuerst die ANA-Deskriptivzellen ausfuehren.")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

viz_df = ana_desc_df.copy()
for col in ['alter', 'icu_duration_h', 'hospital_duration_h']:
    if col in viz_df.columns:
        viz_df[col] = pd.to_numeric(viz_df[col], errors='coerce')

if 'icu_duration_d' not in viz_df.columns and 'icu_duration_h' in viz_df.columns:
    viz_df['icu_duration_d'] = viz_df['icu_duration_h'] / 24.0
if 'hospital_duration_d' not in viz_df.columns and 'hospital_duration_h' in viz_df.columns:
    viz_df['hospital_duration_d'] = viz_df['hospital_duration_h'] / 24.0

n_stays = len(viz_df)
n_cases = int(viz_df['fallid'].nunique()) if 'fallid' in viz_df.columns else np.nan
median_age = float(viz_df['alter'].median()) if 'alter' in viz_df.columns else np.nan
median_icu_days = float(viz_df['icu_duration_d'].median()) if 'icu_duration_d' in viz_df.columns else np.nan

icu_groups = None
if 'icu_duration_d' in viz_df.columns:
    bins = [-np.inf, 1, 3, 7, 14, np.inf]
    labels = ['<=1 Tag', '>1-3 Tage', '>3-7 Tage', '>7-14 Tage', '>14 Tage']
    icu_groups = (
        pd.cut(viz_df['icu_duration_d'], bins=bins, labels=labels)
        .value_counts(sort=False, dropna=False)
        .rename_axis('Gruppe')
        .reset_index(name='n')
    )
    icu_groups['pct'] = icu_groups['n'] / n_stays * 100

if 'oebenekurz' in viz_df.columns:
    top_depts_plot = (
        viz_df['oebenekurz']
        .fillna('NA')
        .value_counts()
        .head(8)
        .sort_values(ascending=True)
    )
else:
    top_depts_plot = pd.Series(dtype=float)

fig = plt.figure(figsize=(16, 12), facecolor='white')
gs = GridSpec(3, 4, figure=fig, height_ratios=[0.9, 1.5, 1.2], hspace=0.45, wspace=0.35)

card_specs = [
    ('ANA-Aufenthalte', f'{n_stays:,}'.replace(',', '.')),
    ('Eindeutige Faelle', f'{n_cases:,}'.replace(',', '.') if pd.notna(n_cases) else 'n/a'),
    ('Median Alter', f'{median_age:.1f} J.' if pd.notna(median_age) else 'n/a'),
    ('Median ICU-Dauer', f'{median_icu_days:.2f} Tage' if pd.notna(median_icu_days) else 'n/a'),
]
card_colors = ['#274C77', '#6096BA', '#A3CEF1', '#8B8C89']
text_colors = ['white', 'white', '#1F2933', 'white']

for idx, (title, value) in enumerate(card_specs):
    ax = fig.add_subplot(gs[0, idx])
    ax.set_facecolor(card_colors[idx])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.05, 0.72, title, fontsize=13, fontweight='bold', color=text_colors[idx], transform=ax.transAxes)
    ax.text(0.05, 0.28, value, fontsize=24, fontweight='bold', color=text_colors[idx], transform=ax.transAxes)

ax_hist = fig.add_subplot(gs[1, 0:2])
if 'icu_duration_d' in viz_df.columns:
    hist_data = viz_df['icu_duration_d'].dropna().clip(upper=30)
    ax_hist.hist(hist_data, bins=30, color='#6096BA', edgecolor='white')
    ax_hist.set_title('ICU-Dauer in Tagen (Histogramm, bis 30 Tage gekappt)', fontsize=13)
    ax_hist.set_xlabel('ICU-Dauer (Tage)')
    ax_hist.set_ylabel('Hauefigkeit')
    ax_hist.grid(axis='y', alpha=0.2)

ax_group = fig.add_subplot(gs[1, 2:4])
if icu_groups is not None:
    bars = ax_group.bar(icu_groups['Gruppe'].astype(str), icu_groups['pct'], color=['#274C77', '#6096BA', '#A3CEF1', '#8B8C89', '#D9A441'])
    ax_group.set_title('ICU-Dauer-Gruppen', fontsize=13)
    ax_group.set_ylabel('Anteil (%)')
    ax_group.grid(axis='y', alpha=0.2)
    ax_group.set_ylim(0, max(icu_groups['pct'].max() * 1.15, 10))
    for bar, pct in zip(bars, icu_groups['pct']):
        ax_group.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    ax_group.tick_params(axis='x', rotation=20)

ax_dept = fig.add_subplot(gs[2, 0:2])
if not top_depts_plot.empty:
    ax_dept.barh(top_depts_plot.index.astype(str), top_depts_plot.values, color='#D9A441')
    ax_dept.set_title('Top-Abteilungen innerhalb AIN', fontsize=13)
    ax_dept.set_xlabel('Anzahl Aufenthalte')
    ax_dept.grid(axis='x', alpha=0.2)

ax_box = fig.add_subplot(gs[2, 2:4])
box_data = []
box_labels = []
for col, label in [('alter', 'Alter'), ('icu_duration_d', 'ICU-Tage'), ('hospital_duration_d', 'KH-Tage')]:
    if col in viz_df.columns:
        values = viz_df[col].dropna()
        if col != 'alter':
            values = values.clip(upper=values.quantile(0.95))
        box_data.append(values)
        box_labels.append(label)
if box_data:
    bp = ax_box.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    palette = ['#274C77', '#6096BA', '#D9A441']
    for patch, color in zip(bp['boxes'], palette[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax_box.set_title('Boxplots zentraler Variablen', fontsize=13)
    ax_box.grid(axis='y', alpha=0.2)

fig.suptitle('ANA-Faelle (wardshort = AIN): grafische Uebersicht', fontsize=18, fontweight='bold', y=0.98)
plt.show()


# %% ---- notebook cell [23] ----------------------------------------
if 'los_gt2_df' not in globals() or los_gt2_df.empty:
    if 'ana_desc_df' not in globals() or ana_desc_df.empty:
        raise ValueError("Weder los_gt2_df noch ana_desc_df sind vorhanden. Bitte zuerst die ANA-Zellen ausfuehren.")
    work_df = ana_desc_df.copy()
    if 'icu_duration_d' not in work_df.columns:
        if 'icu_duration_h' not in work_df.columns:
            raise ValueError("Weder icu_duration_d noch icu_duration_h vorhanden.")
        work_df['icu_duration_d'] = pd.to_numeric(work_df['icu_duration_h'], errors='coerce') / 24.0
    los_gt2_df = work_df[work_df['icu_duration_d'] > 2].copy()

if los_gt2_df.empty:
    raise ValueError("Keine ANA-Faelle mit LoS > 2 Tage gefunden.")

import math
import matplotlib.pyplot as plt

analysis_df = los_gt2_df.copy()

for col in analysis_df.columns:
    if col.startswith('vital_') or col.startswith('lab_'):
        analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')


def prettify_feature_name(name: str) -> str:
    label = name
    for prefix in ['diag_main_', 'vital_', 'lab_']:
        if label.startswith(prefix):
            label = label[len(prefix):]
    label = label.replace('_', ' ').strip()
    return label.upper() if label.isalpha() and len(label) <= 5 else label.title()


def split_measure_name(name: str) -> tuple[str, str]:
    short = name
    for prefix in ['vital_', 'lab_']:
        if short.startswith(prefix):
            short = short[len(prefix):]
            break
    for suffix in ['_mean', '_median', '_last', '_first', '_max', '_min', '_std', '_count']:
        if short.endswith(suffix):
            return short[:-len(suffix)], suffix[1:]
    return short, 'value'


def matches_keywords(base_name: str, keywords: list[str]) -> bool:
    tokens = [token for token in base_name.split('_') if token]
    for keyword in keywords:
        for token in tokens:
            if len(keyword) <= 3:
                if token == keyword:
                    return True
            elif token == keyword or token.startswith(keyword):
                return True
    return False


def select_representative_columns(df: pd.DataFrame, columns: list[str], limit: int, keywords: list[str]) -> list[str]:
    rows = []
    suffix_priority = {
        'mean': 0,
        'median': 1,
        'last': 2,
        'first': 3,
        'max': 4,
        'min': 5,
        'std': 6,
        'count': 7,
        'value': 8,
    }

    for col in columns:
        base_name, stat_name = split_measure_name(col)
        if not matches_keywords(base_name, keywords):
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if s.empty or s.nunique() < 5:
            continue
        rows.append({
            'col': col,
            'base_name': base_name,
            'stat_name': stat_name,
            'n': int(s.shape[0]),
            'unique_n': int(s.nunique()),
            'stat_priority': suffix_priority.get(stat_name, 99),
        })

    if not rows:
        return []

    ranking_df = pd.DataFrame(rows).sort_values(
        ['n', 'stat_priority', 'unique_n', 'base_name'],
        ascending=[False, True, False, True],
    )
    ranking_df = ranking_df.drop_duplicates('base_name', keep='first')
    return ranking_df.head(limit)['col'].tolist()


diag_cols = [col for col in analysis_df.columns if col.startswith('diag_main_')]
los_gt2_top_diag = pd.DataFrame()
if diag_cols:
    diag_counts = []
    for col in diag_cols:
        values = pd.to_numeric(analysis_df[col], errors='coerce').fillna(0)
        present_n = int((values > 0).sum())
        if present_n > 0:
            diag_counts.append({
                'Diagnose': prettify_feature_name(col),
                'n': present_n,
                'pct': round(present_n / len(analysis_df) * 100, 2),
                'Spalte': col,
            })
    los_gt2_top_diag = pd.DataFrame(diag_counts).sort_values(['n', 'Diagnose'], ascending=[False, True]).head(5)

vital_cols = [col for col in analysis_df.columns if col.startswith('vital_')]
lab_cols = [col for col in analysis_df.columns if col.startswith('lab_')]

vital_keywords = [
    'herz', 'heart', 'puls', 'pulse', 'rr', 'atem', 'resp', 'spo2', 'sao2',
    'temp', 'temperatur', 'fio2', 'blutdruck', 'syst', 'diast'
]
lab_keywords = [
    'hb', 'hgb', 'krea', 'kreat', 'creat', 'crp', 'leuko', 'thrombo', 'plt',
    'natrium', 'sodium', 'kalium', 'potassium', 'bilirubin', 'laktat',
    'lactat', 'lactate', 'glucose', 'glukose', 'ph', 'pco2', 'po2',
    'bicarbonat', 'baseexcess'
]

selected_vitals = select_representative_columns(analysis_df, vital_cols, limit=4, keywords=vital_keywords)
selected_labs = select_representative_columns(analysis_df, lab_cols + vital_cols, limit=4, keywords=lab_keywords)

selected_metric_rows = []
for group_label, cols in [('Vital', selected_vitals), ('Labor', selected_labs)]:
    for col in cols:
        s = pd.to_numeric(analysis_df[col], errors='coerce').dropna()
        if s.empty:
            continue
        selected_metric_rows.append({
            'Gruppe': group_label,
            'Parameter': prettify_feature_name(col),
            'n': int(s.shape[0]),
            'Mittelwert': round(float(s.mean()), 2),
            'Median': round(float(s.median()), 2),
            'IQR': f"{s.quantile(0.25):.2f} - {s.quantile(0.75):.2f}",
            'Min-Max': f"{s.min():.2f} - {s.max():.2f}",
        })
los_gt2_vital_lab_summary = pd.DataFrame(selected_metric_rows)

print('Top 5 Hauptdiagnosen bei ANA-Faellen mit ICU-LoS > 2 Tage')
print('=========================================================')
if los_gt2_top_diag.empty:
    print('Keine diag_main_-Spalten mit positiven Befunden gefunden.')
else:
    display(los_gt2_top_diag[['Diagnose', 'n', 'pct']])

print('\nAusgewaehlte Vital- und Laborparameter')
if los_gt2_vital_lab_summary.empty:
    print('Keine geeigneten numerischen Vital- oder Laborvariablen fuer Histogramme gefunden.')
else:
    display(los_gt2_vital_lab_summary)

if not selected_labs:
    print('\nHinweis: Im aktuellen Datensatz wurden keine separat erkennbaren Labor-Features gefunden. Es werden daher nur Histogramme fuer Vitalparameter erstellt.')

plot_columns = [('Vital', col) for col in selected_vitals] + [('Labor', col) for col in selected_labs]
if not plot_columns:
    print('\nKeine Histogramme erstellt, da keine geeigneten Spalten gefunden wurden.')
else:
    n_plots = len(plot_columns)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle('Vitals und Labs bei ANA mit ICU-LoS > 2 Tage', fontsize=17, fontweight='bold')

    group_colors = {'Vital': '#4E79A7', 'Labor': '#E15759'}
    for ax, (group_label, col) in zip(axes, plot_columns):
        s = pd.to_numeric(analysis_df[col], errors='coerce').dropna()
        if s.empty:
            ax.axis('off')
            continue
        upper = s.quantile(0.99) if s.nunique() > 10 else s.max()
        lower = s.quantile(0.01) if s.nunique() > 10 else s.min()
        clipped = s.clip(lower=lower, upper=upper)
        ax.hist(clipped, bins=30, color=group_colors[group_label], edgecolor='white', alpha=0.9)
        ax.set_title(f"{group_label}: {prettify_feature_name(col)}")
        ax.set_xlabel(prettify_feature_name(col))
        ax.set_ylabel('Hauefigkeit')
        ax.grid(axis='y', alpha=0.2)

    for ax in axes[n_plots:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# %% ---- notebook cell [25] ----------------------------------------
if 'ana_desc_df' not in globals() or ana_desc_df.empty:
    raise ValueError("ana_desc_df ist nicht vorhanden. Bitte zuerst die ANA-Zellen ausfuehren.")

import matplotlib.pyplot as plt

los_gt2_df = ana_desc_df.copy()

if 'icu_duration_d' not in los_gt2_df.columns:
    if 'icu_duration_h' not in los_gt2_df.columns:
        raise ValueError("Weder icu_duration_d noch icu_duration_h vorhanden.")
    los_gt2_df['icu_duration_d'] = pd.to_numeric(los_gt2_df['icu_duration_h'], errors='coerce') / 24.0

if 'hospital_duration_d' not in los_gt2_df.columns and 'hospital_duration_h' in los_gt2_df.columns:
    los_gt2_df['hospital_duration_d'] = pd.to_numeric(los_gt2_df['hospital_duration_h'], errors='coerce') / 24.0

for col in ['alter', 'icu_duration_d', 'hospital_duration_d', 'zugang_anzahl_gesamt', 'neben_diag_anzahl']:
    if col in los_gt2_df.columns:
        los_gt2_df[col] = pd.to_numeric(los_gt2_df[col], errors='coerce')

los_gt2_df = los_gt2_df[los_gt2_df['icu_duration_d'] > 2].copy()

if los_gt2_df.empty:
    raise ValueError("Keine ANA-Faelle mit LoS > 2 Tage gefunden.")

summary_specs = {
    'alter': 'Alter (Jahre)',
    'icu_duration_d': 'ICU-Dauer (Tage)',
    'hospital_duration_d': 'Krankenhausdauer (Tage)',
    'zugang_anzahl_gesamt': 'Anzahl Zugaenge',
    'neben_diag_anzahl': 'Anzahl Nebendiagnosen',
}

summary_rows = []
for col, label in summary_specs.items():
    if col in los_gt2_df.columns:
        s = los_gt2_df[col].dropna()
        if not s.empty:
            summary_rows.append({
                'Variable': label,
                'n': int(s.shape[0]),
                'Mittelwert': round(float(s.mean()), 2),
                'SD': round(float(s.std()), 2),
                'Median': round(float(s.median()), 2),
                'IQR': f"{s.quantile(0.25):.2f} - {s.quantile(0.75):.2f}",
                'Min-Max': f"{s.min():.2f} - {s.max():.2f}",
            })

los_gt2_summary = pd.DataFrame(summary_rows)

los_gt2_overview = pd.DataFrame([
    {'Kennzahl': 'Filter', 'Wert': 'ANA und ICU-LoS > 2 Tage'},
    {'Kennzahl': 'Anzahl Aufenthalte', 'Wert': int(len(los_gt2_df))},
    {'Kennzahl': 'Eindeutige Faelle', 'Wert': int(los_gt2_df['fallid'].nunique()) if 'fallid' in los_gt2_df.columns else np.nan},
    {'Kennzahl': 'Eindeutige Patienten', 'Wert': int(los_gt2_df['pid'].nunique()) if 'pid' in los_gt2_df.columns else np.nan},
])

if 'oebenekurz' in los_gt2_df.columns:
    los_gt2_top_depts = (
        los_gt2_df['oebenekurz']
        .fillna('NA')
        .value_counts()
        .head(10)
        .rename_axis('oebenekurz')
        .reset_index(name='n')
    )
    los_gt2_top_depts['pct'] = (los_gt2_top_depts['n'] / len(los_gt2_df) * 100).round(2)
else:
    los_gt2_top_depts = pd.DataFrame()

print('ANA-Faelle mit LoS > 2 Tage')
print('===========================')
display(los_gt2_overview)
display(los_gt2_summary)
if not los_gt2_top_depts.empty:
    print('\nTop-Abteilungen')
    display(los_gt2_top_depts)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('ANA-Faelle mit ICU-LoS > 2 Tage', fontsize=18, fontweight='bold')

plot_items = [
    ('alter', 'Alter (Jahre)', '#5B7C99'),
    ('icu_duration_d', 'ICU-Dauer (Tage)', '#6096BA'),
    ('hospital_duration_d', 'Krankenhausdauer (Tage)', '#D9A441'),
]

for ax, (col, label, color) in zip(axes.flat[:3], plot_items):
    if col in los_gt2_df.columns:
        data = los_gt2_df[col].dropna()
        if col in ['icu_duration_d', 'hospital_duration_d'] and not data.empty:
            data = data.clip(upper=data.quantile(0.95))
        ax.hist(data, bins=30, color=color, edgecolor='white')
        ax.set_title(f'{label} - Histogramm')
        ax.set_xlabel(label)
        ax.set_ylabel('Hauefigkeit')
        ax.grid(axis='y', alpha=0.2)

ax_bar = axes[1, 1]
if not los_gt2_top_depts.empty:
    plot_tbl = los_gt2_top_depts.sort_values('n', ascending=True)
    ax_bar.barh(plot_tbl['oebenekurz'].astype(str), plot_tbl['n'], color='#8B8C89')
    ax_bar.set_title('Top-Abteilungen')
    ax_bar.set_xlabel('Anzahl Aufenthalte')
    ax_bar.grid(axis='x', alpha=0.2)
else:
    ax_bar.axis('off')

plt.tight_layout()
plt.show()


# %% ---- notebook cell [27] ----------------------------------------
if 'ana_desc_df' not in globals() or ana_desc_df.empty:
    raise ValueError("ana_desc_df ist nicht vorhanden. Bitte zuerst die ANA-Zellen ausfuehren.")

if 'los_gt2_df' not in globals() or los_gt2_df.empty:
    work_df = ana_desc_df.copy()
    if 'icu_duration_d' not in work_df.columns:
        if 'icu_duration_h' not in work_df.columns:
            raise ValueError("Weder icu_duration_d noch icu_duration_h vorhanden.")
        work_df['icu_duration_d'] = pd.to_numeric(work_df['icu_duration_h'], errors='coerce') / 24.0
    los_gt2_df = work_df[work_df['icu_duration_d'] > 2].copy()

import math
import matplotlib.pyplot as plt

lab_analysis_df = ana_desc_df.copy()
lab_los_gt2_df = los_gt2_df.copy()
lab_cols = [col for col in lab_analysis_df.columns if col.startswith('lab_')]

if not lab_cols:
    raise ValueError("Keine lab_-Spalten im ANA-Datensatz gefunden.")

for col in lab_cols:
    lab_analysis_df[col] = pd.to_numeric(lab_analysis_df[col], errors='coerce')
    if col in lab_los_gt2_df.columns:
        lab_los_gt2_df[col] = pd.to_numeric(lab_los_gt2_df[col], errors='coerce')


def prettify_lab_feature(name: str) -> str:
    label = name
    for prefix in ['lab_', 'vital_', 'diag_main_']:
        if label.startswith(prefix):
            label = label[len(prefix):]
    label = label.replace('_', ' ').strip()
    return label.upper() if label.isalpha() and len(label) <= 5 else label.title()


def split_lab_measure(name: str) -> tuple[str, str]:
    short = name[4:] if name.startswith('lab_') else name
    for suffix in ['_mean', '_median', '_last', '_first', '_max', '_min', '_std', '_count']:
        if short.endswith(suffix):
            return short[:-len(suffix)], suffix[1:]
    return short, 'value'


def token_match(token: str, keyword: str) -> bool:
    if len(keyword) <= 3:
        return token == keyword
    return token == keyword or token.startswith(keyword)


def matches_all_keywords(base_name: str, keywords: list[str]) -> bool:
    tokens = [token for token in base_name.split('_') if token]
    return all(any(token_match(token, keyword) for token in tokens) for keyword in keywords)


def matches_any_keyword(base_name: str, keywords: list[str]) -> bool:
    tokens = [token for token in base_name.split('_') if token]
    return any(token_match(token, keyword) for token in tokens for keyword in keywords)


def choose_best_lab_column(df: pd.DataFrame, columns: list[str], spec: dict):
    suffix_priority = {
        'mean': 0,
        'median': 1,
        'last': 2,
        'first': 3,
        'max': 4,
        'min': 5,
        'std': 6,
        'count': 7,
        'value': 8,
    }
    matches = []
    for col in columns:
        base_name, stat_name = split_lab_measure(col)
        if 'keywords_all' in spec and not matches_all_keywords(base_name, spec['keywords_all']):
            continue
        if 'keywords_any' in spec and not matches_any_keyword(base_name, spec['keywords_any']):
            continue
        if 'forbidden' in spec and matches_any_keyword(base_name, spec['forbidden']):
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if s.empty or s.nunique() < 5:
            continue
        matches.append({
            'column': col,
            'base_name': base_name,
            'stat_name': stat_name,
            'n': int(s.shape[0]),
            'unique_n': int(s.nunique()),
            'stat_priority': suffix_priority.get(stat_name, 99),
        })
    if not matches:
        return None
    ranked = pd.DataFrame(matches).sort_values(
        ['n', 'stat_priority', 'unique_n', 'base_name'],
        ascending=[False, True, False, True],
    )
    return ranked.iloc[0]['column']


def summarize_numeric(df: pd.DataFrame, columns: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for group_label, display_label, col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if s.empty:
            continue
        rows.append({
            'Gruppe': group_label,
            'Parameter': display_label,
            'n': int(s.shape[0]),
            'Mittelwert': round(float(s.mean()), 2),
            'Median': round(float(s.median()), 2),
            'IQR': f"{s.quantile(0.25):.2f} - {s.quantile(0.75):.2f}",
            'Min-Max': f"{s.min():.2f} - {s.max():.2f}",
        })
    return pd.DataFrame(rows)


clinical_lab_specs = [
    {'gruppe': 'Entzuendung', 'label': 'Leukozyten', 'keywords_any': ['leukozyten', 'leuko']},
    {'gruppe': 'Haematologie', 'label': 'Haemoglobin', 'keywords_any': ['haemoglobin', 'hemoglobin', 'hb', 'hgb']},
    {'gruppe': 'Haematologie', 'label': 'Thrombozyten', 'keywords_any': ['thrombozyten', 'thrombo', 'plt']},
    {'gruppe': 'Niere', 'label': 'Kreatinin', 'keywords_any': ['kreatinin', 'krea', 'creatinin', 'creatinin']},
    {'gruppe': 'Leber', 'label': 'Bilirubin gesamt', 'keywords_all': ['bilirubin', 'gesamt']},
    {'gruppe': 'Leber', 'label': 'Bilirubin direkt', 'keywords_all': ['bilirubin', 'direkt']},
    {'gruppe': 'Entzuendung', 'label': 'CRP', 'keywords_any': ['crp']},
    {'gruppe': 'Perfusion', 'label': 'Laktat', 'keywords_any': ['laktat', 'lactat', 'lactate']},
    {'gruppe': 'BGA', 'label': 'pH', 'keywords_any': ['ph']},
    {'gruppe': 'BGA', 'label': 'pCO2', 'keywords_any': ['pco2']},
    {'gruppe': 'BGA', 'label': 'pO2', 'keywords_any': ['po2']},
    {'gruppe': 'Elektrolyte', 'label': 'Natrium', 'keywords_any': ['natrium', 'sodium']},
    {'gruppe': 'Elektrolyte', 'label': 'Kalium', 'keywords_any': ['kalium', 'potassium']},
    {'gruppe': 'Metabolismus', 'label': 'Glukose', 'keywords_any': ['glucose', 'glukose', 'gluk']},
]

clinical_lab_matches = []
for spec in clinical_lab_specs:
    selected_col = choose_best_lab_column(lab_analysis_df, lab_cols, spec)
    clinical_lab_matches.append({
        'Gruppe': spec['gruppe'],
        'Parameter': spec['label'],
        'Spalte': selected_col,
        'Verfuegbar': selected_col is not None,
    })
clinical_lab_selection = pd.DataFrame(clinical_lab_matches)

catalog_rows = []
for col in sorted(lab_cols):
    base_name, stat_name = split_lab_measure(col)
    matched_specs = clinical_lab_selection.loc[clinical_lab_selection['Spalte'] == col, 'Parameter'].tolist()
    all_non_null = int(lab_analysis_df[col].notna().sum()) if col in lab_analysis_df.columns else 0
    los_non_null = int(lab_los_gt2_df[col].notna().sum()) if col in lab_los_gt2_df.columns else 0
    catalog_rows.append({
        'Spalte': col,
        'Klartext': matched_specs[0] if matched_specs else prettify_lab_feature(col),
        'Analyt': prettify_lab_feature(base_name),
        'Statistik': stat_name,
        'ANA_non_null_n': all_non_null,
        'ANA_non_null_pct': round(all_non_null / len(lab_analysis_df) * 100, 2),
        'LoS_gt2_non_null_n': los_non_null,
        'LoS_gt2_non_null_pct': round(los_non_null / len(lab_los_gt2_df) * 100, 2) if len(lab_los_gt2_df) else 0.0,
        'Klinisch_priorisiert': bool(matched_specs),
    })
lab_feature_catalog = pd.DataFrame(catalog_rows).sort_values(
    ['Klinisch_priorisiert', 'LoS_gt2_non_null_n', 'ANA_non_null_n', 'Spalte'],
    ascending=[False, False, False, True],
)

selected_lab_columns = [
    (row['Gruppe'], row['Parameter'], row['Spalte'])
    for _, row in clinical_lab_selection.iterrows()
    if row['Verfuegbar'] and isinstance(row['Spalte'], str)
]
ana_lab_compact_summary = summarize_numeric(lab_analysis_df, selected_lab_columns)
los_gt2_clinical_lab_summary = summarize_numeric(lab_los_gt2_df, selected_lab_columns)

LAB_FEATURE_CATALOG_XLSX = BASE_DIR / 'kisik2_ana_lab_feature_catalog.xlsx'
LAB_FEATURE_CATALOG_CSV = BASE_DIR / 'kisik2_ana_lab_feature_catalog.csv'
with pd.ExcelWriter(LAB_FEATURE_CATALOG_XLSX, engine='openpyxl') as writer:
    lab_feature_catalog.to_excel(writer, sheet_name='Alle_lab_Variablen', index=False)
    clinical_lab_selection.to_excel(writer, sheet_name='Klinische_Auswahl', index=False)
    ana_lab_compact_summary.to_excel(writer, sheet_name='ANA_Laborsummary', index=False)
    los_gt2_clinical_lab_summary.to_excel(writer, sheet_name='LoS_gt2_Laborsummary', index=False)
lab_feature_catalog.to_csv(LAB_FEATURE_CATALOG_CSV, index=False, sep=';')

print('Klinisch priorisierte Laborparameter')
print('===================================')
display(clinical_lab_selection)

print('\nKompakte Laborsektion fuer alle ANA-Faelle')
display(ana_lab_compact_summary)

print('\nKompakte Laborsektion fuer ANA-Faelle mit ICU-LoS > 2 Tage')
display(los_gt2_clinical_lab_summary)

print('\nExportierter Katalog aller lab_-Variablen')
print(str(LAB_FEATURE_CATALOG_XLSX))
print(str(LAB_FEATURE_CATALOG_CSV))
display(lab_feature_catalog.head(25))

plot_specs = selected_lab_columns[:8]
if plot_specs:
    n_plots = len(plot_specs)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle('Klinisch priorisierte Labore bei ANA mit ICU-LoS > 2 Tage', fontsize=17, fontweight='bold')

    group_palette = {
        'Entzuendung': '#C8553D',
        'Haematologie': '#5B8E7D',
        'Niere': '#4E79A7',
        'Leber': '#E09F3E',
        'Perfusion': '#8F5DA2',
        'BGA': '#7A9E9F',
        'Elektrolyte': '#6D597A',
        'Metabolismus': '#D1495B',
    }

    for ax, (group_label, display_label, col) in zip(axes, plot_specs):
        s = pd.to_numeric(lab_los_gt2_df[col], errors='coerce').dropna()
        if s.empty:
            ax.axis('off')
            continue
        lower = s.quantile(0.01) if s.nunique() > 10 else s.min()
        upper = s.quantile(0.99) if s.nunique() > 10 else s.max()
        clipped = s.clip(lower=lower, upper=upper)
        ax.hist(clipped, bins=30, color=group_palette.get(group_label, '#4E79A7'), edgecolor='white', alpha=0.9)
        ax.set_title(f'{display_label} ({group_label})')
        ax.set_xlabel(display_label)
        ax.set_ylabel('Hauefigkeit')
        ax.grid(axis='y', alpha=0.2)

    for ax in axes[n_plots:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print('\nKeine klinisch priorisierten Laborparameter mit ausreichender Belegung gefunden.')