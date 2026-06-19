import duckdb, pandas as pd
from pathlib import Path

pq = Path(r"D:\Ausgangsdaten\KISIK Projekt\kisik2\kisik2_icu_ml_dataset_24h.parquet")
feat_csv = Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung\los_selected_features_ain_24h_compact.csv")

features = pd.read_csv(feat_csv, sep=";")["Feature"].tolist()
cols = set(duckdb.connect().execute(f"SELECT * FROM read_parquet('{pq.as_posix()}') LIMIT 0").df().columns)

direct  = [f for f in features if f in cols]
missing = [f for f in features if f not in cols]

print(f"Direkt gefunden (kein Mapping noetig): {len(direct)}/104")
print(f"Noch fehlend:                          {len(missing)}")

if missing:
    print("\nFehlende Features (brauchen noch Mapping oder sind nicht verfuegbar):")
    for m in missing:
        print(f"  {m}")
