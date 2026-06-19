"""
Data-Leakage-Test fuer KISIK-LoS-Modell
=========================================
Prueft ob die selektierten Features (lab24_, vital24_ etc.)
tatsaechlich auf 24h-Fenster-Daten basieren oder auf Full-Stay-Aggregaten.
"""
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

PARQUET = Path(r"D:\Ausgangsdaten\KISIK Projekt\kisik2\kisik2_icu_ml_dataset.parquet")
FEAT_CSV = Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung\los_selected_features_ain_24h_compact.csv")
MAP_CSV  = Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung\los_retro_markus_icu_los_gt1d_feature_mapping.csv")

# ---- 1. Feature-Namen-Analyse -----------------------------------------
feat_df = pd.read_csv(FEAT_CSV, sep=";")
features = feat_df["Feature"].tolist()

lab24    = [f for f in features if f.startswith("lab24_")]
vital24  = [f for f in features if f.startswith("vital24_")]
proc24   = [f for f in features if f.startswith("proc24_")]
zugang24 = [f for f in features if f.startswith("zugang24_")]
diag     = [f for f in features if f.startswith("diag_")]
basis    = [f for f in features if not any(f.startswith(p) for p in ["lab24_","vital24_","proc24_","zugang24_","diag_"])]

print("=" * 60)
print("1. Feature-Set-Analyse")
print("=" * 60)
print(f"  Basis (Alter, stay_nr, Zeit):  {len(basis)}")
print(f"  lab24_  (Labor 1. 24h):        {len(lab24)}")
print(f"  vital24_ (Vitals 1. 24h):      {len(vital24)}")
print(f"  proc24_ (Prozeduren 1. 24h):   {len(proc24)}")
print(f"  zugang24_ (Zugaenge 1. 24h):   {len(zugang24)}")
print(f"  diag_ (Diagnosen):             {len(diag)}")

# ---- 2. Hat das Parquet 24h-Spalten? -----------------------------------
con = duckdb.connect()
all_cols = con.execute(f"SELECT * FROM read_parquet('{PARQUET.as_posix()}') LIMIT 0").df().columns.tolist()
has_24h = [c for c in all_cols if "24" in c]

print()
print("=" * 60)
print("2. Parquet-Spalten-Check")
print("=" * 60)
print(f"  Parquet gesamt: {len(all_cols)} Spalten")
print(f"  Spalten mit '24' im Namen: {len(has_24h)}")
if has_24h:
    print(f"  -> {has_24h[:10]}")
else:
    print("  -> KEINE 24h-Spalten vorhanden!")
    print("  LEAKAGE BESTAETIGT: Modell mapped lab24_ -> lab_ (Full-Stay-Aggregate)")

# ---- 3. Feature-Mapping pruefen (aus Modelllauf) -----------------------
print()
print("=" * 60)
print("3. Tatsaechliches Feature-Mapping aus dem letzten Modelllauf")
print("=" * 60)
if MAP_CSV.exists():
    map_df = pd.read_csv(MAP_CSV, sep=";")
    print(map_df.groupby("Mapping")["Feature"].count().sort_values(ascending=False).to_string())
    print()
    leak_candidates = map_df[map_df["Mapping"].str.contains("->", na=False)]
    if not leak_candidates.empty:
        print("Gemappte Features (24h-Name -> Full-Stay-Spalte):")
        print(leak_candidates[["Feature","Mapping","Quelle"]].head(20).to_string(index=False))
else:
    print("  Mapping-Datei nicht gefunden")

# ---- 4. Korrelation Feature vs. LoS ------------------------------------
print()
print("=" * 60)
print("4. Korrelation der verwendeten Spalten mit ICU-LoS (Verdacht-Check)")
print("=" * 60)

# Lade Daten (nur AIN-Stays mit LoS > 1 Tag)
allowed = [
    ("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31"),("AIN","IZ01"),
    ("AUG","IZ01"),("AVT","IZ01"),("GCH","IZ01"),("GYN","IZ01"),
    ("HNO","IZ01"),("HTC","IZ01"),("IZPV","IZ01"),("MKG","IZ01"),
    ("NCH","IZ01"),("NUK","IZ01"),("STR","IZ01"),("UCH","IZ01"),("URO","IZ01"),
]
allowed_sql = ", ".join(f"('{w}', '{o}')" for w,o in allowed)
df = con.execute(f"""
    SELECT *
    FROM read_parquet('{PARQUET.as_posix()}')
    WHERE (wardshort, oebenekurz) IN ({allowed_sql})
      AND icu_duration_h / 24.0 > 1
""").df()
print(f"  Kohorte: {len(df):,} Stays")

target = (df["icu_duration_h"] / 24.0).rename("los_days")

# Mapping 24h -> full-stay Spaltennamen
prefix_map = {"lab24_": "lab_", "vital24_": "vital_", "proc24_": "proc_", "zugang24_": "zugang_"}

corr_rows = []
for feat_name in features:
    col_name = feat_name
    mapped_from = None
    for old_p, new_p in prefix_map.items():
        if feat_name.startswith(old_p):
            candidate = new_p + feat_name[len(old_p):]
            if candidate in df.columns:
                col_name = candidate
                mapped_from = old_p + " -> " + new_p
                break
    if col_name not in df.columns:
        continue
    vals = pd.to_numeric(df[col_name], errors="coerce")
    if vals.notna().sum() < 50:
        continue
    corr = float(vals.corr(target))
    corr_rows.append({
        "Feature_Name": feat_name,
        "Parquet_Spalte": col_name,
        "Mapping": mapped_from if mapped_from else "direkt",
        "Pearson_r": round(corr, 3),
        "abs_r": abs(corr),
    })

corr_df = pd.DataFrame(corr_rows).sort_values("abs_r", ascending=False)

print("\nTop-20 korrelierte Features (|r| mit ICU-LoS):")
print(corr_df[["Feature_Name","Parquet_Spalte","Mapping","Pearson_r"]].head(20).to_string(index=False))

# Besonders verdaechtige Statistiken: _last, _max, _min ueber Full-Stay
suspect = corr_df[
    corr_df["Parquet_Spalte"].str.contains(r"_(last|max|min)$", regex=True) &
    (corr_df["Mapping"] != "direkt")
]
print(f"\nVerdaechtig (Full-Stay _last/_max/_min + gemappt): {len(suspect)} Features")
if not suspect.empty:
    print(suspect[["Feature_Name","Parquet_Spalte","Pearson_r"]].head(15).to_string(index=False))

print()
print("=" * 60)
print("FAZIT")
print("=" * 60)
if not has_24h:
    print("LEAKAGE BESTAETIGT:")
    print("  Das Modell wurde mit lab24_/vital24_ Feature-Namen trainiert,")
    print("  aber das Parquet enthaelt NUR Full-Stay-Aggregate (lab_, vital_ etc.).")
    print("  Das Mapping 'lab24_ -> lab_' bedeutet: der Wert ist ueber den")
    print("  GESAMTEN ICU-Aufenthalt berechnet, nicht nur die ersten 24h.")
    print("  Features wie 'last_value', 'max_value' korrelieren direkt mit LoS.")
    print()
    print("Empfehlung: Pipeline auf echte 24h-Fenster-Aggregation erweitern.")
else:
    print("Kein offensichtliches Leakage durch Spalten-Mapping.")
    print("Weitere manuelle Pruefung empfohlen.")

con.close()
