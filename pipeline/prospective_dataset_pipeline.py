# -*- coding: utf-8 -*-
"""
Prospektive KISIK ICU-Daten-Pipeline (OLD-Snapshot-Verarbeitung).
Auto-extrahiert (nur Quellcode, KEINE Notebook-Outputs) aus KISIK-Daten-Pipeline.ipynb (Zellen >=29).

Verarbeitet die taeglichen OLD-Live-Snapshots zu einem prospektiven ML-Datensatz
(kisik2_prospektiv_ml_dataset.parquet): pros_load_day_csv(), pros_detect_open_stay(),
run_prospective_pipeline(). Schluessel ist hier 'fallnr' (vs. 'fallid' retrospektiv);
Datumsfelder im deutschen Format (DD.MM.YYYY HH:MM:SS).

HINWEIS: fest verdrahtete lokale Pfade -> fuer die eigene Umgebung anpassen.
Keine Patientendaten enthalten.
"""

# %% ---- notebook cell [29] ----------------------------------------
from pathlib import Path

if 'pd' not in globals():
    import pandas as pd

feature_priority_rows = [
    {'Feature': 'Alter', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'Rohwert'},
    {'Feature': 'Geschlecht', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'mittel', 'Prioritaet': 'B', 'Empfohlene_Aggregation': 'Rohwert'},
    {'Feature': 'Aufnahmeart (Notfall, elektiv, Verlegung)', 'Derzeit_vorhanden': 'wahrscheinlich ja', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'kategorial'},
    {'Feature': 'Fachbereich / Station / wardshort', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'kategorial'},
    {'Feature': 'Aufnahmezeit, Wochentag, Tageszeit', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'mittel', 'Prioritaet': 'B', 'Empfohlene_Aggregation': 'kategorial oder zyklisch'},
    {'Feature': 'Hauptdiagnosegruppe', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'gruppiert, Top-Kategorien'},
    {'Feature': 'Komorbiditaeten', 'Derzeit_vorhanden': 'teilweise indirekt', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'gruppierte Indikatoren oder Summenscore'},
    {'Feature': 'Prozeduren zu Beginn', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'binaer und Top-Gruppen'},
    {'Feature': 'Zugaenge / invasive Massnahmen', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'binaer und Anzahl'},
    {'Feature': 'Schweregrad-Score', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'erster Wert, max im Fruehfenster'},
    {'Feature': 'Herzfrequenz', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, last, min, max, mean'},
    {'Feature': 'Blutdruck / MAP', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, last, min, max, mean'},
    {'Feature': 'Atemfrequenz', 'Derzeit_vorhanden': 'wahrscheinlich ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, last, min, max, mean'},
    {'Feature': 'Temperatur', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'mittel bis hoch', 'Prioritaet': 'B', 'Empfohlene_Aggregation': 'first, max, mean'},
    {'Feature': 'SpO2', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, min, mean'},
    {'Feature': 'FiO2', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, max, mean'},
    {'Feature': 'ScvO2', 'Derzeit_vorhanden': 'selten oder partiell', 'Klinische_Relevanz': 'mittel', 'Prioritaet': 'C', 'Empfohlene_Aggregation': 'first, last'},
    {'Feature': 'Kreatinin', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, max, last'},
    {'Feature': 'Harnstoff', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, max, last'},
    {'Feature': 'Leukozyten', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, max, mean'},
    {'Feature': 'Haemoglobin', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, min, last'},
    {'Feature': 'Thrombozyten', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, min, last'},
    {'Feature': 'Natrium', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, min, max'},
    {'Feature': 'Kalium', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, min, max'},
    {'Feature': 'Glukose', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, max, mean'},
    {'Feature': 'CRP', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, max'},
    {'Feature': 'Bilirubin', 'Derzeit_vorhanden': 'wahrscheinlich ja', 'Klinische_Relevanz': 'mittel bis hoch', 'Prioritaet': 'B', 'Empfohlene_Aggregation': 'first, max'},
    {'Feature': 'Basenueberschuss', 'Derzeit_vorhanden': 'Ja, als POCT-Surrogat', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, min, max, last'},
    {'Feature': 'Standard-Basenueberschuss', 'Derzeit_vorhanden': 'Ja, als POCT-Surrogat', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, min, max, last'},
    {'Feature': 'Standard-Bicarbonat', 'Derzeit_vorhanden': 'Ja, als POCT-Surrogat', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'A', 'Empfohlene_Aggregation': 'first, min, max, last'},
    {'Feature': 'Anionenluecke', 'Derzeit_vorhanden': 'Ja, als POCT-Surrogat', 'Klinische_Relevanz': 'mittel bis hoch', 'Prioritaet': 'B', 'Empfohlene_Aggregation': 'first, max, mean'},
    {'Feature': 'Gesamt-CO2', 'Derzeit_vorhanden': 'sehr selten', 'Klinische_Relevanz': 'mittel', 'Prioritaet': 'C', 'Empfohlene_Aggregation': 'first oder last'},
    {'Feature': 'Haematokrit POCT', 'Derzeit_vorhanden': 'Ja', 'Klinische_Relevanz': 'mittel', 'Prioritaet': 'B', 'Empfohlene_Aggregation': 'first, min, mean'},
    {'Feature': 'pH', 'Derzeit_vorhanden': 'Nein', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A+', 'Empfohlene_Aggregation': 'falls verfuegbar: first, min, max'},
    {'Feature': 'pCO2', 'Derzeit_vorhanden': 'praktisch nicht nutzbar', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A+', 'Empfohlene_Aggregation': 'falls verfuegbar: first, max'},
    {'Feature': 'pO2', 'Derzeit_vorhanden': 'Nein, nur O2-Surrogate', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A+', 'Empfohlene_Aggregation': 'falls verfuegbar: first, min'},
    {'Feature': 'Laktat', 'Derzeit_vorhanden': 'Nein', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A+', 'Empfohlene_Aggregation': 'falls verfuegbar: first, max, last'},
    {'Feature': 'Beatmung ja oder nein', 'Derzeit_vorhanden': 'unklar', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A+', 'Empfohlene_Aggregation': 'binaer und Dauer'},
    {'Feature': 'Beatmungsparameter (PEEP, Modus)', 'Derzeit_vorhanden': 'nein oder unklar', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A+', 'Empfohlene_Aggregation': 'first, max, mean'},
    {'Feature': 'Katecholamine', 'Derzeit_vorhanden': 'nein oder unklar', 'Klinische_Relevanz': 'sehr hoch', 'Prioritaet': 'A+', 'Empfohlene_Aggregation': 'binaer und Dosis'},
    {'Feature': 'Dialyse / CRRT', 'Derzeit_vorhanden': 'nein oder unklar', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'B', 'Empfohlene_Aggregation': 'binaer'},
    {'Feature': 'Urinausscheidung', 'Derzeit_vorhanden': 'nein oder unklar', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'B', 'Empfohlene_Aggregation': 'Summe 24h'},
    {'Feature': 'Fluessigkeitsbilanz', 'Derzeit_vorhanden': 'nein oder unklar', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'B', 'Empfohlene_Aggregation': 'Netto 24h'},
    {'Feature': 'Antibiotikatherapie frueh', 'Derzeit_vorhanden': 'nein oder unklar', 'Klinische_Relevanz': 'mittel bis hoch', 'Prioritaet': 'B', 'Empfohlene_Aggregation': 'binaer'},
    {'Feature': 'GCS / neurologischer Status', 'Derzeit_vorhanden': 'nein oder unklar', 'Klinische_Relevanz': 'hoch', 'Prioritaet': 'B', 'Empfohlene_Aggregation': 'first, min'},
]

feature_priority_df = pd.DataFrame(feature_priority_rows)
priority_sort = {'A+': 0, 'A': 1, 'B': 2, 'C': 3}
feature_priority_df['Prioritaet_sort'] = feature_priority_df['Prioritaet'].map(priority_sort).fillna(9)
feature_priority_df = feature_priority_df.sort_values(['Prioritaet_sort', 'Feature']).drop(columns='Prioritaet_sort').reset_index(drop=True)

must_have_df = feature_priority_df[feature_priority_df['Prioritaet'].isin(['A+', 'A'])].copy()
nice_to_have_df = feature_priority_df[feature_priority_df['Prioritaet'] == 'B'].copy()
missing_high_value_df = feature_priority_df[feature_priority_df['Prioritaet'] == 'A+'].copy()

output_dir = Path(r'D:/Ausgangsdaten/KISIK Projekt/Eigene Auswertung')
output_dir.mkdir(parents=True, exist_ok=True)
feature_table_xlsx = output_dir / 'kisik2_los_feature_prioritaeten.xlsx'
feature_table_csv = output_dir / 'kisik2_los_feature_prioritaeten.csv'

with pd.ExcelWriter(feature_table_xlsx, engine='openpyxl') as writer:
    feature_priority_df.to_excel(writer, sheet_name='Gesamtuebersicht', index=False)
    must_have_df.to_excel(writer, sheet_name='Must_Have', index=False)
    nice_to_have_df.to_excel(writer, sheet_name='Nice_to_Have', index=False)
    missing_high_value_df.to_excel(writer, sheet_name='Fehlend_hoher_Wert', index=False)

feature_priority_df.to_csv(feature_table_csv, sep=';', index=False)

print('Feature-Prioritaetstabelle fuer das LoS-Modell gespeichert:')
print(str(feature_table_xlsx))
print(str(feature_table_csv))
display(feature_priority_df)


# %% ---- notebook cell [31] ----------------------------------------
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import re

PROS_BASE_DIR = Path(r"D:\Ausgangsdaten\Live-Daten\OLD")
PROS_OUTPUT_DIR = Path(r"D:\Ausgangsdaten\KISIK Projekt\kisik2")

PROS_DATASET_FILE = PROS_OUTPUT_DIR / "kisik2_prospektiv_ml_dataset.parquet"
PROS_SUMMARY_FILE = PROS_OUTPUT_DIR / "kisik2_prospektiv_ml_dataset_summary.xlsx"

PROS_FILE_NAMES: Dict[str, str] = {
    "stays": "fall_aufenthalt.csv",
    "vitals": "vitalzeichen.csv",
    "scores": "score.csv",
    "diagnoses": "diagnose.csv",
    "procedures": "prozeduren.csv",
    "access": "zugaenge.csv",
    "lab": "lab.csv",
    "op_an": "op_an.csv",
    "op_intervals": "op_zeitintervalle.csv",
    "prev_stays": "aufenthalte_vorher_nachher.csv",
    "case_data": "fall_daten.csv",
}

PROS_TOP_FEATURE_LIMITS = {
    "diag_main": 300,
    "proc": 500,
    "zugang": 300,
    "lab": 300,
}

PROS_COLS: Dict[str, str] = {
    "case_id": "fallnr",
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
    "access_text": "text",
    "access_date": "anlegedatum",
    "access_time": "anlegezeit",
    "lab_name": "beschreibung",
    "lab_code": "code",
    "lab_value_primary": "ergebnisf",
    "lab_value_secondary": "ergebnist",
    "lab_time": "erfassdat",
    "lab_analytic": "analytx",
    "admission_type": "aufnahmeart",
    "discharge_type": "entlassart",
}

PROS_REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "stays": [
        PROS_COLS["case_id"],
        PROS_COLS["patient_id"],
        PROS_COLS["icu_start"],
        PROS_COLS["icu_end"],
        PROS_COLS["hospital_start"],
        PROS_COLS["hospital_end"],
    ],
    "vitals": [
        PROS_COLS["case_id"],
        PROS_COLS["vital_name"],
        PROS_COLS["vital_value"],
        PROS_COLS["vital_time"],
    ],
    "scores": [
        PROS_COLS["case_id"],
        PROS_COLS["score_name"],
        PROS_COLS["score_value"],
        PROS_COLS["score_start"],
    ],
    "diagnoses": [
        PROS_COLS["case_id"],
        PROS_COLS["diag_code"],
        PROS_COLS["diag_type"],
    ],
    "procedures": [
        PROS_COLS["case_id"],
        PROS_COLS["proc_code"],
        PROS_COLS["proc_time"],
    ],
    "access": [
        PROS_COLS["case_id"],
        PROS_COLS["access_group"],
        PROS_COLS["access_date"],
        PROS_COLS["access_time"],
    ],
    "lab": [
        PROS_COLS["case_id"],
        PROS_COLS["lab_time"],
        PROS_COLS["lab_value_primary"],
    ],
    "case_data": [
        PROS_COLS["case_id"],
    ],
}


def pros_sanitize_name(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80] if text else "unknown"


def pros_parse_folder_date(folder_name: str) -> Optional[datetime]:
    try:
        return datetime.strptime(folder_name, "%d.%m.%Y")
    except ValueError:
        return None


def pros_get_day_folders(base_dir: Path) -> List[Tuple[datetime, Path]]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Basisordner nicht gefunden: {base_dir}")

    day_folders: List[Tuple[datetime, Path]] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        parsed_date = pros_parse_folder_date(child.name)
        if parsed_date is None:
            continue
        day_folders.append((parsed_date, child))

    day_folders.sort(key=lambda item: item[0])
    if not day_folders:
        raise FileNotFoundError(f"Keine gueltigen Tagesordner unter {base_dir} gefunden.")

    print(
        f"  -> {len(day_folders)} Tagesordner gefunden "
        f"({day_folders[0][0].date()} bis {day_folders[-1][0].date()})"
    )
    return day_folders


def pros_load_day_csv(folder: Path, filename: str, snapshot_date: datetime) -> pd.DataFrame:
    path = folder / filename
    if not path.exists():
        return pd.DataFrame()

    last_error: Optional[Exception] = None
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(path, sep=";", encoding=encoding, low_memory=False)
            df.columns = [str(col).strip().lower() for col in df.columns]
            df["snapshot_date"] = pd.Timestamp(snapshot_date)
            df["source_file"] = path.as_posix()
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Konnte {path} nicht lesen: {last_error}")


def pros_to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def pros_to_numeric_safe(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def pros_detect_open_stay(raw_end_series: pd.Series) -> pd.Series:
    text = raw_end_series.astype(str).str.strip()
    return text.str.contains(r"(^31[./-]12[./-]4000)|(^4000[./-]12[./-]31)", regex=True, na=False)


def pros_optimize_frame(df: pd.DataFrame) -> pd.DataFrame:
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


def pros_find_first_available(day_folders: List[Tuple[datetime, Path]], filename: str) -> Optional[Tuple[datetime, Path]]:
    for snapshot_date, folder in day_folders:
        if (folder / filename).exists():
            return snapshot_date, folder
    return None


def pros_validate_live_schema(day_folders: List[Tuple[datetime, Path]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for file_key, filename in PROS_FILE_NAMES.items():
        first_available = pros_find_first_available(day_folders, filename)
        if first_available is None:
            rows.append({
                "Datei": filename,
                "Status": "fehlt",
                "Fehlende_Spalten": ", ".join(PROS_REQUIRED_COLUMNS.get(file_key, [])),
            })
            continue

        snapshot_date, folder = first_available
        df = pros_load_day_csv(folder, filename, snapshot_date)
        required_cols = PROS_REQUIRED_COLUMNS.get(file_key, [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        rows.append({
            "Datei": filename,
            "Status": "ok" if not missing_cols else "Spalten fehlen",
            "Fehlende_Spalten": ", ".join(missing_cols),
            "Beispielordner": folder.name,
            "Spaltenanzahl": df.shape[1],
        })

    audit_df = pd.DataFrame(rows)
    display(audit_df)
    critical_issues = audit_df[
        audit_df["Status"].isin(["fehlt", "Spalten fehlen"])
        & audit_df["Datei"].isin([PROS_FILE_NAMES[key] for key in ["stays", "vitals", "scores", "diagnoses", "procedures", "access", "lab"]])
    ]
    if not critical_issues.empty:
        raise KeyError("Pflichtdateien oder Pflichtspalten fehlen in den Live-Daten.")
    return audit_df


def pros_load_all_stays(day_folders: List[Tuple[datetime, Path]]) -> pd.DataFrame:
    print(f"\n-> Phase 1: {PROS_FILE_NAMES['stays']} aus {len(day_folders)} Tagen einlesen")
    frames: List[pd.DataFrame] = []
    for snapshot_date, folder in day_folders:
        df = pros_load_day_csv(folder, PROS_FILE_NAMES["stays"], snapshot_date)
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        raise FileNotFoundError("Keine fall_aufenthalt.csv im Live-Daten-Ordner gefunden.")

    raw = pd.concat(frames, ignore_index=True)
    print(f"  -> {len(raw):,} Stay-Rohzeilen aus allen Tagesordnern")

    required_cols = PROS_REQUIRED_COLUMNS["stays"]
    missing_cols = [col for col in required_cols if col not in raw.columns]
    if missing_cols:
        raise KeyError(f"Fehlende Pflichtspalten in fall_aufenthalt.csv: {missing_cols}")

    raw["is_open"] = pros_detect_open_stay(raw[PROS_COLS["icu_end"]])

    for col in [PROS_COLS["icu_start"], PROS_COLS["icu_end"], PROS_COLS["hospital_start"], PROS_COLS["hospital_end"]]:
        raw[col] = pros_to_datetime_safe(raw[col])

    raw[PROS_COLS["icu_start"]] = raw[PROS_COLS["icu_start"]].fillna(raw[PROS_COLS["hospital_start"]])
    raw[PROS_COLS["icu_end"]] = raw[PROS_COLS["icu_end"]].fillna(raw[PROS_COLS["hospital_end"]])

    if PROS_COLS["age"] in raw.columns:
        raw[PROS_COLS["age"]] = pros_to_numeric_safe(raw[PROS_COLS["age"]])

    raw = raw.sort_values([PROS_COLS["case_id"], PROS_COLS["icu_start"], "snapshot_date"]).copy()
    stays = (
        raw
        .groupby([PROS_COLS["case_id"], PROS_COLS["icu_start"]], dropna=True, as_index=False)
        .last()
    )

    open_mask = stays["is_open"].fillna(False)
    stays["last_snapshot_cap"] = stays["snapshot_date"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    stays.loc[open_mask, PROS_COLS["icu_end"]] = stays.loc[open_mask, "last_snapshot_cap"]

    stays = stays.dropna(subset=[PROS_COLS["case_id"], PROS_COLS["icu_start"], PROS_COLS["icu_end"]]).copy()
    stays = stays.sort_values([PROS_COLS["case_id"], PROS_COLS["icu_start"], PROS_COLS["icu_end"]]).copy()

    stays["stay_nr"] = stays.groupby(PROS_COLS["case_id"]).cumcount() + 1
    stays["stay_id"] = stays[PROS_COLS["case_id"]].astype(str) + "_stay" + stays["stay_nr"].astype(str)
    stays["icu_duration_h"] = (
        (stays[PROS_COLS["icu_end"]] - stays[PROS_COLS["icu_start"]]).dt.total_seconds() / 3600
    ).astype("float32")

    if PROS_COLS["hospital_start"] in stays.columns and PROS_COLS["hospital_end"] in stays.columns:
        stays["hospital_duration_h"] = (
            (stays[PROS_COLS["hospital_end"]] - stays[PROS_COLS["hospital_start"]]).dt.total_seconds() / 3600
        ).astype("float32")

    print(f"  -> {len(stays):,} deduplizierte ICU-Stays aus {stays[PROS_COLS['case_id']].nunique():,} Faellen")
    print(f"  -> {int(open_mask.sum()):,} offene Aufenthalte auf letzten Snapshot gecappt")
    return stays


def pros_load_case_data(day_folders: List[Tuple[datetime, Path]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for snapshot_date, folder in day_folders:
        df = pros_load_day_csv(folder, PROS_FILE_NAMES["case_data"], snapshot_date)
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=[PROS_COLS["case_id"]])

    raw = pd.concat(frames, ignore_index=True)
    keep_cols = [
        col for col in [
            PROS_COLS["case_id"],
            PROS_COLS["admission_type"],
            PROS_COLS["discharge_type"],
            "aufnahmeartid",
            "entlassartid",
        ] if col in raw.columns
    ]
    if not keep_cols:
        return pd.DataFrame(columns=[PROS_COLS["case_id"]])

    case_data = (
        raw[keep_cols + ["snapshot_date"]]
        .sort_values([PROS_COLS["case_id"], "snapshot_date"])
        .groupby(PROS_COLS["case_id"], as_index=False)
        .last()
        .drop(columns=["snapshot_date"], errors="ignore")
    )
    return case_data


def pros_register_stays(con: duckdb.DuckDBPyConnection, stays: pd.DataFrame) -> None:
    con.register(
        "pros_stays_df",
        stays[["stay_id", PROS_COLS["case_id"], PROS_COLS["icu_start"], PROS_COLS["icu_end"]]].copy(),
    )
    con.execute("CREATE OR REPLACE TEMP TABLE pros_stays AS SELECT * FROM pros_stays_df")


def pros_build_glob(filename: str) -> str:
    return (PROS_BASE_DIR / "**" / filename).as_posix()


def pros_sql_ts(expr: str) -> str:
    return (
        f"COALESCE(" 
        f"try_strptime(CAST({expr} AS VARCHAR), '%d.%m.%Y %H:%M:%S')," 
        f"try_strptime(CAST({expr} AS VARCHAR), '%d.%m.%Y %H:%M')," 
        f"TRY_CAST({expr} AS TIMESTAMP)" 
        f")"
    )


def pros_sql_num(expr: str) -> str:
    return f"TRY_CAST(REPLACE(CAST({expr} AS VARCHAR), ',', '.') AS DOUBLE)"


def pros_long_metrics_to_wide(long_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["stay_id"])

    local = long_df.copy()
    local["feature_key"] = local["feature_name"].map(pros_sanitize_name)
    metric_map = {
        "mean_value": "mean",
        "median_value": "median",
        "first_value": "first",
        "last_value": "last",
        "min_value": "min",
        "max_value": "max",
    }

    # COUNT-Aggregat: Anzahl Messungen je Stay und Feature
    if "count_value" in local.columns:
        count_part = local.pivot_table(index="stay_id", columns="feature_key", values="count_value", aggfunc="first")
        if not count_part.empty:
            count_part = count_part.astype("float32")
            count_part.columns = [f"{prefix}_{column}_count" for column in count_part.columns]
            parts: List[pd.DataFrame] = [count_part]
        else:
            parts: List[pd.DataFrame] = []
    else:
        parts: List[pd.DataFrame] = []

    for metric_col, suffix in metric_map.items():
        part = local.pivot_table(index="stay_id", columns="feature_key", values=metric_col, aggfunc="first")
        if part.empty:
            continue
        part = part.astype("float32")
        part.columns = [f"{prefix}_{column}_{suffix}" for column in part.columns]
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=["stay_id"])
    return pd.concat(parts, axis=1).reset_index()


def pros_limit_to_top_features(long_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if long_df.empty:
        return long_df

    top_n = PROS_TOP_FEATURE_LIMITS.get(prefix)
    if top_n is None:
        return long_df

    counts = long_df.groupby("feature_name", dropna=False)["stay_id"].nunique().sort_values(ascending=False)
    keep = set(counts.head(top_n).index)
    filtered = long_df[long_df["feature_name"].isin(keep)].copy()
    print(f"  -> {prefix}: {len(counts)} Kategorien gesamt, {len(keep)} beibehalten")
    return filtered


def pros_presence_to_wide(long_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["stay_id"])

    local = pros_limit_to_top_features(long_df[["stay_id", "feature_name"]].dropna().copy(), prefix)
    if local.empty:
        return pd.DataFrame(columns=["stay_id"])

    local["feature_key"] = local["feature_name"].map(pros_sanitize_name)
    wide = pd.crosstab(local["stay_id"], local["feature_key"])
    wide = (wide > 0).astype("uint8")
    wide.columns = [f"{prefix}_{column}" for column in wide.columns]
    return wide.reset_index()

# ---------------------------------------------------------------------------
# Hilfsfunktionen fuer den Ausgabe-Schritt
# (pros_query_* werden in der naechsten Zelle definiert)
# ---------------------------------------------------------------------------

def pros_build_feature_summary(
    base: pd.DataFrame,
    parts: List[Tuple[pd.DataFrame, str]],
) -> pd.DataFrame:
    rows = [{"Feature_Gruppe": "Aufenthalt Meta", "Anzahl_Features": max(len(base.columns) - 1, 0)}]
    for part_df, label in parts:
        rows.append({
            "Feature_Gruppe": label,
            "Anzahl_Features": max(len(part_df.columns) - 1, 0) if not part_df.empty else 0,
        })
    return pd.DataFrame(rows)


def pros_save_outputs(
    base: pd.DataFrame,
    parts: List[Tuple[pd.DataFrame, str]],
    preview_rows: int = 5,
) -> Tuple[int, int, pd.DataFrame, pd.DataFrame]:
    _tmp = PROS_DATASET_FILE.parent / "duckdb_tmp"
    _tmp.mkdir(exist_ok=True)
    con = duckdb.connect(config={"temp_directory": str(_tmp)})
    con.register("base_df", pros_optimize_frame(base))

    select_parts = ["base.*"]
    join_parts = []
    for idx, (part_df, _label) in enumerate(parts):
        if part_df.empty:
            continue
        table_name = f"part_{idx}_df"
        alias = f"p{idx}"
        con.register(table_name, pros_optimize_frame(part_df))
        select_parts.append(f"{alias}.* EXCLUDE (stay_id)")
        join_parts.append(f"LEFT JOIN {table_name} {alias} USING (stay_id)")

    final_sql = "\n".join([
        "CREATE OR REPLACE TABLE pros_final_dataset AS",
        "SELECT",
        "    " + ",\n    ".join(select_parts),
        "FROM base_df base",
        *join_parts,
    ])
    con.execute(final_sql)
    con.execute(
        f"COPY pros_final_dataset TO '{PROS_DATASET_FILE.as_posix()}' "
        f"(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    n_rows = con.execute("SELECT COUNT(*) FROM pros_final_dataset").fetchone()[0]
    n_cols = con.execute(
        "SELECT COUNT(*) FROM pragma_table_info('pros_final_dataset')"
    ).fetchone()[0]
    preview = con.execute(
        f"SELECT * FROM pros_final_dataset LIMIT {preview_rows}"
    ).fetch_df()
    con.close()

    feature_summary = pros_build_feature_summary(base, parts)
    with pd.ExcelWriter(PROS_SUMMARY_FILE, engine="openpyxl") as writer:
        feature_summary.to_excel(writer, sheet_name="Feature_Uebersicht", index=False)
        preview.to_excel(writer, sheet_name="Preview", index=False)
        pd.DataFrame({
            "Kennzahl": ["Zeilen", "Spalten", "Parquet-Datei", "Summary-Datei"],
            "Wert": [n_rows, n_cols, str(PROS_DATASET_FILE), str(PROS_SUMMARY_FILE)],
        }).to_excel(writer, sheet_name="Export", index=False)

    return n_rows, n_cols, preview, feature_summary


# %% ---- notebook cell [32] ----------------------------------------
def pros_load_day_csv(folder: Path, filename: str, snapshot_date: datetime) -> pd.DataFrame:
    path = folder / filename
    if not path.exists():
        return pd.DataFrame()

    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(path, sep=";", encoding=encoding, low_memory=False)
            df.columns = [str(col).strip().lower() for col in df.columns]
            df["snapshot_date"] = pd.Timestamp(snapshot_date)
            df["source_file"] = path.as_posix()
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception:
            continue

    return pd.DataFrame()


def pros_collect_observed_columns(
    day_folders: List[Tuple[datetime, Path]], filename: str, max_samples: int = 12
) -> Tuple[set, List[str], int]:
    observed_columns = set()
    sampled_folders: List[str] = []
    files_seen = 0

    for snapshot_date, folder in day_folders:
        path = folder / filename
        if not path.exists():
            continue
        files_seen += 1
        df = pros_load_day_csv(folder, filename, snapshot_date)
        observed_columns.update(
            col for col in df.columns if col not in {"snapshot_date", "source_file"}
        )
        sampled_folders.append(folder.name)
        if len(sampled_folders) >= max_samples:
            break

    return observed_columns, sampled_folders, files_seen


def pros_validate_live_schema(day_folders: List[Tuple[datetime, Path]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for file_key, filename in PROS_FILE_NAMES.items():
        observed_columns, sampled_folders, files_seen = pros_collect_observed_columns(day_folders, filename)
        required_cols = PROS_REQUIRED_COLUMNS.get(file_key, [])
        missing_cols = [col for col in required_cols if col not in observed_columns]

        if files_seen == 0:
            status = "fehlt"
            usable = False
        elif missing_cols:
            status = "eingeschraenkt"
            usable = False
        else:
            status = "ok"
            usable = True

        rows.append(
            {
                "Datei": filename,
                "Status": status,
                "Verwendbar": usable,
                "Fehlende_Spalten": ", ".join(missing_cols),
                "Gepruefte_Ordner": ", ".join(sampled_folders[:3]),
                "Dateien_im_Sample": files_seen,
            }
        )

    audit_df = pd.DataFrame(rows)
    display(audit_df)

    stays_row = audit_df.loc[audit_df["Datei"] == PROS_FILE_NAMES["stays"]]
    if stays_row.empty or not bool(stays_row["Verwendbar"].iloc[0]):
        raise KeyError("fall_aufenthalt.csv ist im ausgewaehlten Datenbestand nicht sauber nutzbar.")

    return audit_df


def pros_build_file_argument(day_folders: List[Tuple[datetime, Path]], filename: str) -> Optional[str]:
    paths = []
    for _snapshot_date, folder in day_folders:
        path = folder / filename
        if path.exists():
            paths.append(path.as_posix().replace("'", "''"))

    if not paths:
        return None
    return "[" + ", ".join(f"'{path}'" for path in paths) + "]"


def pros_query_vitals(con: duckdb.DuckDBPyConnection, source_arg: Optional[str]) -> pd.DataFrame:
    if not source_arg:
        return pd.DataFrame()
    ts_expr = pros_sql_ts(f"v.{PROS_COLS['vital_time']}")
    sql = f"""
        WITH vitals AS (
            SELECT
                s.stay_id,
                v.{PROS_COLS['vital_name']} AS feature_name,
                {pros_sql_num(f'v.{PROS_COLS["vital_value"]}')} AS value_num,
                {ts_expr} AS ts
            FROM read_csv_auto({source_arg}, delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true, filename=true) v
            JOIN pros_stays s
              ON CAST(v.{PROS_COLS['case_id']} AS VARCHAR) = CAST(s.{PROS_COLS['case_id']} AS VARCHAR)
             AND {ts_expr} BETWEEN s.{PROS_COLS['icu_start']} AND s.{PROS_COLS['icu_end']}
        )
        SELECT
            stay_id,
            feature_name,
            AVG(value_num) AS mean_value,
            MEDIAN(value_num) AS median_value,
            ARG_MIN(value_num, ts) AS first_value,
            ARG_MAX(value_num, ts) AS last_value,
            MIN(value_num) AS min_value,
            MAX(value_num) AS max_value,
            COUNT(value_num) AS count_value
        FROM vitals
        WHERE feature_name IS NOT NULL AND value_num IS NOT NULL AND ts IS NOT NULL
        GROUP BY 1, 2
    """
    return con.execute(sql).fetch_df()


def pros_query_scores(con: duckdb.DuckDBPyConnection, source_arg: Optional[str]) -> pd.DataFrame:
    if not source_arg:
        return pd.DataFrame()
    ts_expr = pros_sql_ts(f"sc.{PROS_COLS['score_start']}")
    sql = f"""
        WITH scores AS (
            SELECT
                s.stay_id,
                sc.{PROS_COLS['score_name']} AS feature_name,
                {pros_sql_num(f'sc.{PROS_COLS["score_value"]}')} AS value_num,
                {ts_expr} AS ts
            FROM read_csv_auto({source_arg}, delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true, filename=true) sc
            JOIN pros_stays s
              ON CAST(sc.{PROS_COLS['case_id']} AS VARCHAR) = CAST(s.{PROS_COLS['case_id']} AS VARCHAR)
             AND {ts_expr} BETWEEN s.{PROS_COLS['icu_start']} AND s.{PROS_COLS['icu_end']}
        )
        SELECT
            stay_id,
            feature_name,
            AVG(value_num) AS mean_value,
            MEDIAN(value_num) AS median_value,
            ARG_MIN(value_num, ts) AS first_value,
            ARG_MAX(value_num, ts) AS last_value,
            MIN(value_num) AS min_value,
            MAX(value_num) AS max_value,
            COUNT(value_num) AS count_value
        FROM scores
        WHERE feature_name IS NOT NULL AND value_num IS NOT NULL AND ts IS NOT NULL
        GROUP BY 1, 2
    """
    return con.execute(sql).fetch_df()


def pros_query_diagnoses(con: duckdb.DuckDBPyConnection, source_arg: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not source_arg:
        return pd.DataFrame(), pd.DataFrame(columns=["stay_id", "neben_diag_anzahl"])
    main_sql = f"""
        SELECT DISTINCT
            s.stay_id,
            d.{PROS_COLS['diag_code']} AS feature_name
        FROM read_csv_auto({source_arg}, delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true, filename=true) d
        JOIN pros_stays s
          ON CAST(d.{PROS_COLS['case_id']} AS VARCHAR) = CAST(s.{PROS_COLS['case_id']} AS VARCHAR)
        WHERE d.{PROS_COLS['diag_code']} IS NOT NULL
          AND lower(coalesce(d.{PROS_COLS['diag_type']}, '')) = '{PROS_COLS['diag_main_value']}'
    """
    side_sql = f"""
        SELECT
            s.stay_id,
            COUNT(*) AS neben_diag_anzahl
        FROM read_csv_auto({source_arg}, delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true, filename=true) d
        JOIN pros_stays s
          ON CAST(d.{PROS_COLS['case_id']} AS VARCHAR) = CAST(s.{PROS_COLS['case_id']} AS VARCHAR)
        WHERE d.{PROS_COLS['diag_code']} IS NOT NULL
          AND lower(coalesce(d.{PROS_COLS['diag_type']}, '')) <> '{PROS_COLS['diag_main_value']}'
        GROUP BY 1
    """
    return con.execute(main_sql).fetch_df(), con.execute(side_sql).fetch_df()


def pros_query_procedures(con: duckdb.DuckDBPyConnection, source_arg: Optional[str]) -> pd.DataFrame:
    if not source_arg:
        return pd.DataFrame()
    ts_expr = pros_sql_ts(f"p.{PROS_COLS['proc_time']}")
    sql = f"""
        SELECT DISTINCT
            s.stay_id,
            p.{PROS_COLS['proc_code']} AS feature_name
        FROM read_csv_auto({source_arg}, delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true, filename=true) p
        JOIN pros_stays s
          ON CAST(p.{PROS_COLS['case_id']} AS VARCHAR) = CAST(s.{PROS_COLS['case_id']} AS VARCHAR)
         AND {ts_expr} BETWEEN s.{PROS_COLS['icu_start']} AND s.{PROS_COLS['icu_end']}
        WHERE p.{PROS_COLS['proc_code']} IS NOT NULL
    """
    return con.execute(sql).fetch_df()


def pros_query_access(con: duckdb.DuckDBPyConnection, source_arg: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not source_arg:
        return pd.DataFrame(), pd.DataFrame(columns=["stay_id", "zugang_anzahl_gesamt"])
    time_component = f"COALESCE(NULLIF(regexp_extract(CAST(a.{PROS_COLS['access_time']} AS VARCHAR), '(\\d{{2}}:\\d{{2}}:\\d{{2}})', 1), ''), '00:00:00')"
    ts_expr = pros_sql_ts(f"a.{PROS_COLS['access_date']} || ' ' || {time_component}")
    sql = f"""
        WITH access_data AS (
            SELECT
                s.stay_id,
                COALESCE(NULLIF(a.{PROS_COLS['access_group']}, ''), NULLIF(a.{PROS_COLS['access_text']}, '')) AS feature_name
            FROM read_csv_auto({source_arg}, delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true, filename=true) a
            JOIN pros_stays s
              ON CAST(a.{PROS_COLS['case_id']} AS VARCHAR) = CAST(s.{PROS_COLS['case_id']} AS VARCHAR)
             AND {ts_expr} BETWEEN s.{PROS_COLS['icu_start']} AND s.{PROS_COLS['icu_end']}
        )
        SELECT stay_id, feature_name
        FROM access_data
        WHERE feature_name IS NOT NULL
    """
    features = con.execute(sql).fetch_df()
    if features.empty:
        counts = pd.DataFrame(columns=["stay_id", "zugang_anzahl_gesamt"])
    else:
        counts = features.groupby("stay_id", dropna=False).size().reset_index(name="zugang_anzahl_gesamt")
    return features, counts


def pros_query_labs(con: duckdb.DuckDBPyConnection, source_arg: Optional[str]) -> pd.DataFrame:
    if not source_arg:
        return pd.DataFrame()
    ts_expr = pros_sql_ts(f"l.{PROS_COLS['lab_time']}")
    sql = f"""
        WITH lab AS (
            SELECT
                s.stay_id,
                COALESCE(
                    NULLIF(trim(l.{PROS_COLS['lab_name']}), ''),
                    NULLIF(trim(l.{PROS_COLS['lab_code']}), ''),
                    NULLIF(trim(l.{PROS_COLS['lab_analytic']}), '')
                ) AS feature_name,
                COALESCE(
                    {pros_sql_num(f'l.{PROS_COLS["lab_value_primary"]}')},
                    {pros_sql_num(f'l.{PROS_COLS["lab_value_secondary"]}')}
                ) AS value_num,
                {ts_expr} AS ts
            FROM read_csv_auto({source_arg}, delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true, filename=true) l
            JOIN pros_stays s
              ON CAST(l.{PROS_COLS['case_id']} AS VARCHAR) = CAST(s.{PROS_COLS['case_id']} AS VARCHAR)
             AND {ts_expr} BETWEEN s.{PROS_COLS['icu_start']} AND s.{PROS_COLS['icu_end']}
        )
        SELECT
            stay_id,
            feature_name,
            AVG(value_num) AS mean_value,
            MEDIAN(value_num) AS median_value,
            ARG_MIN(value_num, ts) AS first_value,
            ARG_MAX(value_num, ts) AS last_value,
            MIN(value_num) AS min_value,
            MAX(value_num) AS max_value,
            COUNT(value_num) AS count_value
        FROM lab
        WHERE feature_name IS NOT NULL AND value_num IS NOT NULL AND ts IS NOT NULL
        GROUP BY 1, 2
    """
    return con.execute(sql).fetch_df()


def run_prospective_pipeline(max_day_folders: Optional[int] = None, write_outputs: bool = True) -> Dict[str, object]:
    print("\n============================================================")
    print(" KISIK2 ICU ML Preprocessing Pipeline – Prospektive Live-Daten")
    print("============================================================\n")

    day_folders = pros_get_day_folders(PROS_BASE_DIR)
    if max_day_folders is not None:
        day_folders = day_folders[:max_day_folders]
        print(f"  -> Testmodus aktiv: nur die ersten {len(day_folders)} Tagesordner werden verarbeitet")

    print("\n-> Schema-Audit")
    schema_audit = pros_validate_live_schema(day_folders)
    schema_lookup = schema_audit.set_index("Datei")

    stays = pros_load_all_stays(day_folders)
    case_data_usable = PROS_FILE_NAMES["case_data"] in schema_lookup.index and bool(schema_lookup.loc[PROS_FILE_NAMES["case_data"], "Verwendbar"])
    case_data = pros_load_case_data(day_folders) if case_data_usable else pd.DataFrame(columns=[PROS_COLS["case_id"]])
    if not case_data.empty:
        stays = stays.merge(case_data, on=PROS_COLS["case_id"], how="left")

    # Zeitmerkmale aus planbegin ableiten (fuer Modell-Kompatibilitaet)
    _ts = pd.to_datetime(stays[PROS_COLS["icu_start"]], errors="coerce")
    stays["admission_hour"]    = _ts.dt.hour.astype("float32")
    stays["admission_weekday"] = _ts.dt.dayofweek.astype("float32")  # 0=Mo, 6=So
    stays["admission_month"]   = _ts.dt.month.astype("float32")
    print("  -> admission_hour / admission_weekday / admission_month aus planbegin berechnet")

    source_args = {key: pros_build_file_argument(day_folders, filename) for key, filename in PROS_FILE_NAMES.items()}

    con = duckdb.connect()
    pros_register_stays(con, stays)

    print("\n-> Vitalzeichen")
    vitals_long = pros_query_vitals(
        con,
        source_args["vitals"] if bool(schema_lookup.loc[PROS_FILE_NAMES["vitals"], "Verwendbar"]) else None,
    )
    vitals_agg = pros_long_metrics_to_wide(vitals_long, "vital")
    print(f"  -> {len(vitals_long):,} Stay-Vital-Kombinationen")

    print("\n-> Scores")
    scores_long = pros_query_scores(
        con,
        source_args["scores"] if bool(schema_lookup.loc[PROS_FILE_NAMES["scores"], "Verwendbar"]) else None,
    )
    scores_agg = pros_long_metrics_to_wide(scores_long, "score")
    print(f"  -> {len(scores_long):,} Stay-Score-Kombinationen")

    print("\n-> Diagnosen")
    diag_source = source_args["diagnoses"] if bool(schema_lookup.loc[PROS_FILE_NAMES["diagnoses"], "Verwendbar"]) else None
    main_diag_long, side_diag_count = pros_query_diagnoses(con, diag_source)
    main_diag_ohe = pros_presence_to_wide(main_diag_long, "diag_main")
    print(f"  -> {len(main_diag_long):,} Stay-Hauptdiagnose-Kombinationen")

    print("\n-> Prozeduren")
    proc_long = pros_query_procedures(
        con,
        source_args["procedures"] if bool(schema_lookup.loc[PROS_FILE_NAMES["procedures"], "Verwendbar"]) else None,
    )
    proc_ohe = pros_presence_to_wide(proc_long, "proc")
    # Prozeduren-Gesamtanzahl je Stay als proc24-kompatible Spalte
    if not proc_long.empty:
        proc_count_df = proc_long.groupby("stay_id", dropna=False).size().reset_index(name="proc24_anzahl_gesamt")
    else:
        proc_count_df = pd.DataFrame(columns=["stay_id", "proc24_anzahl_gesamt"])
    print(f"  -> {len(proc_long):,} Stay-Prozedur-Kombinationen")

    print("\n-> Zugaenge")
    access_source = source_args["access"] if bool(schema_lookup.loc[PROS_FILE_NAMES["access"], "Verwendbar"]) else None
    access_long, access_count = pros_query_access(con, access_source)
    access_ohe = pros_presence_to_wide(access_long, "zugang")
    # Zugang-Gesamtanzahl unter 24h-kompatiblem Spaltennamen
    if not access_count.empty and "zugang_anzahl_gesamt" in access_count.columns:
        access_count_24 = access_count.rename(columns={"zugang_anzahl_gesamt": "zugang24_anzahl_gesamt"})
    else:
        access_count_24 = access_count.copy()
    print(f"  -> {len(access_long):,} Stay-Zugangs-Kombinationen")

    print("\n-> Labor")
    lab_source = source_args["lab"] if bool(schema_lookup.loc[PROS_FILE_NAMES["lab"], "Verwendbar"]) else None
    if lab_source is None:
        print("  -> Labor wird in diesem Lauf uebersprungen, da im ausgewaehlten Snapshot-Set keine matchbare FALLNR-Spalte vorliegt")
    lab_long = pros_query_labs(con, lab_source)
    lab_agg = pros_long_metrics_to_wide(lab_long, "lab")
    print(f"  -> {len(lab_long):,} Stay-Labor-Kombinationen")

    con.close()

    base_cols = [
        "stay_id",
        PROS_COLS["case_id"],
        "stay_nr",
        PROS_COLS["patient_id"],
        PROS_COLS["age"],
        PROS_COLS["icu_start"],
        PROS_COLS["icu_end"],
        PROS_COLS["hospital_start"],
        PROS_COLS["hospital_end"],
        PROS_COLS["department"],
        PROS_COLS["ward"],
        PROS_COLS["admission_type"],
        PROS_COLS["discharge_type"],
        "icu_duration_h",
        "hospital_duration_h",
        "is_open",
        "snapshot_date",
        "admission_hour",
        "admission_weekday",
        "admission_month",
    ]
    base_cols = [column for column in base_cols if column in stays.columns]
    base = pros_optimize_frame(stays[base_cols].copy())

    parts: List[Tuple[pd.DataFrame, str]] = [
        (vitals_agg, "Vitalzeichen"),
        (scores_agg, "Scores"),
        (main_diag_ohe, "Hauptdiagnosen"),
        (pros_optimize_frame(side_diag_count), "Nebendiagnosen"),
        (proc_ohe, "Prozeduren"),
        (pros_optimize_frame(proc_count_df), "Prozeduren-Anzahl"),
        (access_ohe, "Zugaenge"),
        (pros_optimize_frame(access_count_24), "Zugang-Anzahl"),
        (lab_agg, "Labor"),
    ]

    for part_df, label in parts:
        n_features = max(len(part_df.columns) - 1, 0) if not part_df.empty else 0
        print(f"  + {label}: {n_features} Spalten")

    if write_outputs:
        n_rows, n_cols, preview, feature_summary = pros_save_outputs(base, parts)
        print(f"\n  Finales Dataset: {n_rows} Zeilen x {n_cols} Spalten")
        print(f"  Parquet : {PROS_DATASET_FILE}")
        print(f"  Summary : {PROS_SUMMARY_FILE}")
    else:
        preview = base.head().copy()
        feature_summary = pros_build_feature_summary(base, parts)
        n_rows = len(base)
        n_cols = len(base.columns)
        print("\n  Testmodus ohne Export abgeschlossen.")

    print("\nPipeline abgeschlossen.\n")
    return {
        "stays": stays,
        "base": base,
        "preview": preview,
        "feature_summary": feature_summary,
        "schema_audit": schema_audit,
        "rows": n_rows,
        "cols": n_cols,
    }


# %% ---- notebook cell [33] ----------------------------------------
def pros_detect_open_stay(raw_end_series: pd.Series) -> pd.Series:
    text = raw_end_series.astype(str).str.strip()
    return text.str.contains(r"(?:^31[./-]12[./-]4000)|(?:^4000[./-]12[./-]31)", regex=True, na=False)


def pros_validate_live_schema(day_folders: List[Tuple[datetime, Path]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for file_key, filename in PROS_FILE_NAMES.items():
        required_cols = PROS_REQUIRED_COLUMNS.get(file_key, [])
        seen_cols = set()
        sampled_folders: List[str] = []
        files_seen = 0
        non_empty_files = 0

        for snapshot_date, folder in day_folders:
            path = folder / filename
            if not path.exists():
                continue
            files_seen += 1
            df = pros_load_day_csv(folder, filename, snapshot_date)
            sampled_folders.append(folder.name)
            if df.empty:
                continue
            non_empty_files += 1
            seen_cols.update(col for col in df.columns if col not in {"snapshot_date", "source_file"})
            if required_cols and all(col in seen_cols for col in required_cols):
                break

        missing_cols = [col for col in required_cols if col not in seen_cols]
        if files_seen == 0:
            status = "fehlt"
            usable = False
        elif missing_cols:
            status = "eingeschraenkt"
            usable = False
        else:
            status = "ok"
            usable = True

        rows.append(
            {
                "Datei": filename,
                "Status": status,
                "Verwendbar": usable,
                "Fehlende_Spalten": ", ".join(missing_cols),
                "Gepruefte_Ordner": ", ".join(sampled_folders[:3]),
                "Dateien_im_Sample": files_seen,
                "Nichtleere_Dateien": non_empty_files,
            }
        )

    audit_df = pd.DataFrame(rows)
    display(audit_df)

    stays_row = audit_df.loc[audit_df["Datei"] == PROS_FILE_NAMES["stays"]]
    if stays_row.empty or not bool(stays_row["Verwendbar"].iloc[0]):
        raise KeyError("fall_aufenthalt.csv ist im ausgewaehlten Datenbestand nicht sauber nutzbar.")
    return audit_df


def pros_query_access(con: duckdb.DuckDBPyConnection, source_arg: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not source_arg:
        return pd.DataFrame(), pd.DataFrame(columns=["stay_id", "zugang_anzahl_gesamt"])

    date_component = f"NULLIF(regexp_extract(CAST(a.{PROS_COLS['access_date']} AS VARCHAR), '(\\d{{2}}[./]\\d{{2}}[./]\\d{{4}})', 1), '')"
    time_component = f"COALESCE(NULLIF(regexp_extract(CAST(a.{PROS_COLS['access_time']} AS VARCHAR), '(\\d{{2}}:\\d{{2}}:\\d{{2}})', 1), ''), '00:00:00')"
    ts_expr = pros_sql_ts(f"{date_component} || ' ' || {time_component}")

    sql = f"""
        WITH access_data AS (
            SELECT
                s.stay_id,
                COALESCE(NULLIF(a.{PROS_COLS['access_group']}, ''), NULLIF(a.{PROS_COLS['access_text']}, '')) AS feature_name
            FROM read_csv_auto({source_arg}, delim=';', header=true, all_varchar=true, ignore_errors=true, union_by_name=true, filename=true) a
            JOIN pros_stays s
              ON CAST(a.{PROS_COLS['case_id']} AS VARCHAR) = CAST(s.{PROS_COLS['case_id']} AS VARCHAR)
             AND {ts_expr} BETWEEN s.{PROS_COLS['icu_start']} AND s.{PROS_COLS['icu_end']}
        )
        SELECT stay_id, feature_name
        FROM access_data
        WHERE feature_name IS NOT NULL
    """
    features = con.execute(sql).fetch_df()
    if features.empty:
        counts = pd.DataFrame(columns=["stay_id", "zugang_anzahl_gesamt"])
    else:
        counts = features.groupby("stay_id", dropna=False).size().reset_index(name="zugang_anzahl_gesamt")
    return features, counts


# %% ---- notebook cell [34] ----------------------------------------
prospective_smoke_test = run_prospective_pipeline(max_day_folders=3, write_outputs=False)
print(f"Smoke-Test: {prospective_smoke_test['rows']} Stays im Basisdatensatz")
display(prospective_smoke_test['feature_summary'])
display(prospective_smoke_test['preview'].head())