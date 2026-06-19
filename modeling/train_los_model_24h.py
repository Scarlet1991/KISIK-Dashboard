# -*- coding: utf-8 -*-
"""
Retrospektives Modelltraining auf dem leckage-freien 24h-Datensatz +
prospektive Anwendung. Auto-extrahiert aus KISIK-LoS-Modell_v2.ipynb (Zellen 71/73).
Trainiert das LoS-Regressionsmodell (TransformedTargetRegressor, log1p-Ziel) und
wendet es auf die prospektive Kohorte an. Keine Patientendaten enthalten.
"""

# display()-Patch
import builtins

def _dp(*args, **kwargs):
    for a in args:
        print(a.to_string() if hasattr(a, 'to_string') else str(a))

builtins.display = _dp
try:
    import IPython.display as _ipy
    _ipy.display = _dp
except Exception:
    pass

# ============================================================
# LEAKAGE-FREIES TRAINING: 24h-Zeitfenster-Features
# Parquet: kisik2_icu_ml_dataset_24h.parquet
# ============================================================
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

base_dir_markus_train = Path(r"D:\Ausgangsdaten\KISIK Projekt")
analysis_dir_markus_train = base_dir_markus_train / "Eigene Auswertung"
retro_parquet_markus_train = base_dir_markus_train / "kisik2" / "kisik2_icu_ml_dataset_24h.parquet"
selected_features_file_markus_train = analysis_dir_markus_train / "los_selected_features_ain_24h_compact.csv"
output_prefix_markus_train = "los_retro_markus_icu_los_gt1d_24h_clean"
random_state_markus_train = 42

ward_mapping_markus_train = {
    "IZ01": "IMC",
    "310": "IMC Haunstetten",
    "31": "AIN",
    "33": "AIN",
    "34": "AIN",
    "35": "AIN",
    "IZ21": "AIN",
    "IZ31": "AIN",
    "IZ32": "AIN",
    "IZ22": "2MD",
    "32": "2MD",
    "IZ11": "1-3MD IPF",
    "IZ12": "1-3MD IPF",
    "44": "1-3MD IPF",
    "IZ23": "Dialyse",
    "42": "Dialyse",
}
allowed_ward_oebene_markus_train = [
    ("AIN", "IZ32"),
    ("AIN", "IZ21"),
    ("AIN", "IZ31"),
    ("AIN", "IZ01"),
    ("AUG", "IZ01"),
    ("AVT", "IZ01"),
    ("GCH", "IZ01"),
    ("GYN", "IZ01"),
    ("HNO", "IZ01"),
    ("HTC", "IZ01"),
    ("IZPV", "IZ01"),
    ("MKG", "IZ01"),
    ("NCH", "IZ01"),
    ("NUK", "IZ01"),
    ("STR", "IZ01"),
    ("UCH", "IZ01"),
    ("URO", "IZ01"),
]


def normalize_oebene_markus_train(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    return text


def safe_log1p_markus_train(values):
    return np.log1p(np.clip(values, 0, None))


def safe_expm1_markus_train(values):
    return np.expm1(values)


def regression_metrics_markus_train(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "Median_AE": float(np.median(np.abs(y_true - y_pred))),
        "n": int(len(y_true)),
    }


def mapped_feature_candidate_markus_train(feature_name, columns):
    prefix_map = {
        "zugang24_": "zugang_",
        "proc24_": "proc_",
        "lab24_": "lab_",
        "vital24_": "vital_",
        "med24_": "med_",
    }
    for old_prefix, new_prefix in prefix_map.items():
        if feature_name.startswith(old_prefix):
            candidate = new_prefix + feature_name[len(old_prefix):]
            if candidate in columns:
                return candidate, f"{old_prefix} -> {new_prefix}"
    if feature_name.endswith("_24h"):
        candidate = feature_name[:-4]
        if candidate in columns:
            return candidate, "Suffix _24h entfernt"
    return None, None


def build_feature_matrix_markus_train(raw_df, selected_feature_names):
    columns = set(raw_df.columns.astype(str))
    timestamp_values = pd.to_datetime(raw_df["planbegin"], errors="coerce") if "planbegin" in raw_df.columns else pd.Series(pd.NaT, index=raw_df.index)
    derived = pd.DataFrame(index=raw_df.index)
    derived["admission_hour"] = timestamp_values.dt.hour
    derived["admission_weekday"] = timestamp_values.dt.dayofweek
    derived["admission_month"] = timestamp_values.dt.month

    features = pd.DataFrame(index=raw_df.index)
    mapping_rows = []
    for feature_name in selected_feature_names:
        source_column = None
        mapping = None
        if feature_name in columns:
            features[feature_name] = raw_df[feature_name]
            source_column = feature_name
            mapping = "exakter Spaltenname"
        elif feature_name in derived.columns:
            features[feature_name] = derived[feature_name]
            source_column = "planbegin"
            mapping = "aus planbegin abgeleitet"
        else:
            candidate, candidate_mapping = mapped_feature_candidate_markus_train(feature_name, columns)
            if candidate is not None:
                features[feature_name] = raw_df[candidate]
                source_column = candidate
                mapping = candidate_mapping
        mapping_rows.append({
            "Feature": feature_name,
            "verwendet": mapping is not None,
            "Mapping": mapping if mapping is not None else "nicht verwendet",
            "Quelle": source_column,
        })
    feature_mapping = pd.DataFrame(mapping_rows)
    used_features = feature_mapping.loc[feature_mapping["verwendet"], "Feature"].tolist()
    return features[used_features].copy(), feature_mapping


def infer_feature_types_markus_train(feature_df):
    numeric_features = []
    binary_features = []
    categorical_features = []
    for feature_name in feature_df.columns:
        values = feature_df[feature_name]
        numeric_values = pd.to_numeric(values, errors="coerce")
        if numeric_values.notna().sum() >= values.notna().sum() * 0.8:
            feature_df[feature_name] = numeric_values
            unique_values = pd.Series(numeric_values.dropna().unique())
            if len(unique_values) <= 2 and set(unique_values.astype(float).round(8)).issubset({0.0, 1.0}):
                binary_features.append(feature_name)
            else:
                numeric_features.append(feature_name)
        else:
            feature_df[feature_name] = values.astype("string")
            categorical_features.append(feature_name)
    return numeric_features, binary_features, categorical_features


def make_preprocessor_markus_train(numeric_features, binary_features, categorical_features, scale_numeric=False):
    transformers = []
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    if numeric_features:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))
    if binary_features:
        transformers.append(("binary", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), binary_features))
    if categorical_features:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

selected_features_markus_train = pd.read_csv(selected_features_file_markus_train, sep=";")
if "Feature_Set" in selected_features_markus_train.columns:
    selected_features_markus_train = selected_features_markus_train.loc[selected_features_markus_train["Feature_Set"].eq("compact_clinical_24h")].copy()
feature_column_markus_train = "Feature" if "Feature" in selected_features_markus_train.columns else selected_features_markus_train.columns[0]
selected_feature_names_markus_train = selected_features_markus_train[feature_column_markus_train].dropna().astype(str).unique().tolist()

raw_markus_train = duckdb.connect().execute(
    f"select * from read_parquet('{retro_parquet_markus_train.as_posix()}')"
).fetchdf()
raw_markus_train["wardshort"] = raw_markus_train["wardshort"].astype("string").str.strip()
raw_markus_train["oebenekurz"] = raw_markus_train["oebenekurz"].astype("string").str.strip()
raw_markus_train["oebenekurz_norm"] = raw_markus_train["oebenekurz"].map(normalize_oebene_markus_train)
raw_markus_train["mapped_icu_area"] = raw_markus_train["oebenekurz_norm"].map(ward_mapping_markus_train)
allowed_pair_index_markus_train = pd.MultiIndex.from_tuples(allowed_ward_oebene_markus_train, names=["wardshort", "oebenekurz"])
raw_pair_index_markus_train = pd.MultiIndex.from_frame(raw_markus_train[["wardshort", "oebenekurz"]])
raw_markus_train["markus_icu_filter"] = raw_pair_index_markus_train.isin(allowed_pair_index_markus_train)
raw_markus_train["icu_duration_d"] = pd.to_numeric(raw_markus_train["icu_duration_h"], errors="coerce") / 24.0
raw_markus_train["planbegin"] = pd.to_datetime(raw_markus_train["planbegin"], errors="coerce")

period_mask_markus_train = raw_markus_train["planbegin"].between(pd.Timestamp("2017-01-01"), pd.Timestamp("2024-07-20 23:59:59"), inclusive="both")
los_gt1_mask_markus_train = raw_markus_train["icu_duration_d"] > 1.0
model_mask_markus_train = raw_markus_train["markus_icu_filter"] & period_mask_markus_train & los_gt1_mask_markus_train & raw_markus_train["icu_duration_d"].notna()
retro_df_markus_train = raw_markus_train.loc[model_mask_markus_train].copy().reset_index(drop=True)

feature_df_markus_train, feature_mapping_markus_train = build_feature_matrix_markus_train(retro_df_markus_train, selected_feature_names_markus_train)
if feature_df_markus_train.empty:
    raise ValueError("Keine nutzbaren Features nach Mapping gefunden.")
numeric_features_markus_train, binary_features_markus_train, categorical_features_markus_train = infer_feature_types_markus_train(feature_df_markus_train)

x_markus_train = feature_df_markus_train.copy()
y_markus_train = retro_df_markus_train["icu_duration_d"].astype(float).to_numpy()
groups_markus_train = retro_df_markus_train["pid"].astype("string").str.strip().fillna(pd.Series(retro_df_markus_train.index.astype(str), index=retro_df_markus_train.index))

splitter_markus_train = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state_markus_train)
train_idx_markus_train, holdout_idx_markus_train = next(splitter_markus_train.split(x_markus_train, y_markus_train, groups=groups_markus_train))
x_train_markus_train = x_markus_train.iloc[train_idx_markus_train].copy()
x_holdout_markus_train = x_markus_train.iloc[holdout_idx_markus_train].copy()
y_train_markus_train = y_markus_train[train_idx_markus_train]
y_holdout_markus_train = y_markus_train[holdout_idx_markus_train]

models_markus_train = {
    "Ridge_log1p": TransformedTargetRegressor(
        regressor=Pipeline([
            ("preprocessor", make_preprocessor_markus_train(numeric_features_markus_train, binary_features_markus_train, categorical_features_markus_train, scale_numeric=True)),
            ("regressor", Ridge(alpha=10.0)),
        ]),
        func=safe_log1p_markus_train,
        inverse_func=safe_expm1_markus_train,
        check_inverse=False,
    ),
    "RandomForest_log1p": TransformedTargetRegressor(
        regressor=Pipeline([
            ("preprocessor", make_preprocessor_markus_train(numeric_features_markus_train, binary_features_markus_train, categorical_features_markus_train, scale_numeric=False)),
            ("regressor", RandomForestRegressor(n_estimators=220, min_samples_leaf=3, n_jobs=-1, random_state=random_state_markus_train)),
        ]),
        func=safe_log1p_markus_train,
        inverse_func=safe_expm1_markus_train,
        check_inverse=False,
    ),
    "ExtraTrees_log1p": TransformedTargetRegressor(
        regressor=Pipeline([
            ("preprocessor", make_preprocessor_markus_train(numeric_features_markus_train, binary_features_markus_train, categorical_features_markus_train, scale_numeric=False)),
            ("regressor", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=random_state_markus_train)),
        ]),
        func=safe_log1p_markus_train,
        inverse_func=safe_expm1_markus_train,
        check_inverse=False,
    ),
}

metrics_rows_markus_train = []
prediction_frames_markus_train = []
for model_name, model in models_markus_train.items():
    model.fit(x_train_markus_train, y_train_markus_train)
    holdout_pred = np.clip(np.asarray(model.predict(x_holdout_markus_train), dtype=float), 0, None)
    metric_row = regression_metrics_markus_train(y_holdout_markus_train, holdout_pred)
    metric_row.update({"Modell": model_name, "Datensatz": "Retrospektiver Holdout"})
    metrics_rows_markus_train.append(metric_row)
    id_columns_markus_train = [column for column in ["stay_id", "fallid", "pid", "wardshort", "oebenekurz", "mapped_icu_area", "planbegin", "planend"] if column in retro_df_markus_train.columns]
    pred_frame = retro_df_markus_train.iloc[holdout_idx_markus_train][id_columns_markus_train].reset_index(drop=True).copy()
    pred_frame["Modell"] = model_name
    pred_frame["y_true_days"] = y_holdout_markus_train
    pred_frame["y_pred_days"] = holdout_pred
    pred_frame["abs_error_days"] = np.abs(y_holdout_markus_train - holdout_pred)
    prediction_frames_markus_train.append(pred_frame)

metrics_markus_train_df = pd.DataFrame(metrics_rows_markus_train).sort_values(["MAE", "RMSE", "R2"], ascending=[True, True, False]).reset_index(drop=True)
predictions_markus_train_df = pd.concat(prediction_frames_markus_train, ignore_index=True, sort=False)
training_summary_markus_train_df = pd.DataFrame([
    {"Bereich": "Retrospektive Grundgesamtheit nach Markus-ICU-Filter und ICU-LoS > 1 Tag", "Kennzahl": "Stays", "Wert": int(len(retro_df_markus_train))},
    {"Bereich": "Retrospektive Grundgesamtheit nach Markus-ICU-Filter und ICU-LoS > 1 Tag", "Kennzahl": "Patienten", "Wert": int(retro_df_markus_train["pid"].nunique())},
    {"Bereich": "Retrospektive Grundgesamtheit nach Markus-ICU-Filter und ICU-LoS > 1 Tag", "Kennzahl": "Faelle", "Wert": int(retro_df_markus_train["fallid"].nunique())},
    {"Bereich": "Train", "Kennzahl": "Stays", "Wert": int(len(train_idx_markus_train))},
    {"Bereich": "Holdout-Test", "Kennzahl": "Stays", "Wert": int(len(holdout_idx_markus_train))},
    {"Bereich": "LoS-Filter", "Kennzahl": "Min ICU-LoS (Tage)", "Wert": "> 1"},
    {"Bereich": "LoS-Filter", "Kennzahl": "Max ICU-LoS (Tage)", "Wert": "kein oberer Filter"},
    {"Bereich": "ICU-Filter", "Kennzahl": "Definition", "Wert": "Markus Moegelein allowed_ward_oebene"},
    {"Bereich": "Feature-Set", "Kennzahl": "Selektierte Features", "Wert": int(len(selected_feature_names_markus_train))},
    {"Bereich": "Feature-Set", "Kennzahl": "Verwendete Features nach Mapping", "Wert": int(feature_mapping_markus_train["verwendet"].sum())},
    {"Bereich": "Feature-Set", "Kennzahl": "Numerisch", "Wert": int(len(numeric_features_markus_train))},
    {"Bereich": "Feature-Set", "Kennzahl": "Binaer", "Wert": int(len(binary_features_markus_train))},
    {"Bereich": "Feature-Set", "Kennzahl": "Kategorial", "Wert": int(len(categorical_features_markus_train))},
])

metrics_file_markus_train = analysis_dir_markus_train / f"{output_prefix_markus_train}_holdout_metrics.csv"
predictions_file_markus_train = analysis_dir_markus_train / f"{output_prefix_markus_train}_holdout_predictions.csv"
summary_file_markus_train = analysis_dir_markus_train / f"{output_prefix_markus_train}_training_summary.csv"
feature_mapping_file_markus_train = analysis_dir_markus_train / f"{output_prefix_markus_train}_feature_mapping.csv"

metrics_markus_train_df.to_csv(metrics_file_markus_train, sep=";", index=False)
predictions_markus_train_df.to_csv(predictions_file_markus_train, sep=";", index=False)
training_summary_markus_train_df.to_csv(summary_file_markus_train, sep=";", index=False)
feature_mapping_markus_train.to_csv(feature_mapping_file_markus_train, sep=";", index=False)

display(training_summary_markus_train_df)
display(metrics_markus_train_df.round(3))
display(feature_mapping_markus_train.groupby("Mapping", dropna=False).size().rename("Anzahl").reset_index().sort_values("Anzahl", ascending=False))
print("Exporte geschrieben:")
print(metrics_file_markus_train)
print(predictions_file_markus_train)
print(summary_file_markus_train)
print(feature_mapping_file_markus_train)
# ============================================================
# ZELLE 73: Prospektive Anwendung (24h-Modell)
# ============================================================
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

base_dir_pros_benchmark = Path(r"D:\Ausgangsdaten\KISIK Projekt")
analysis_dir_pros_benchmark = base_dir_pros_benchmark / "Eigene Auswertung"
pros_parquet_pros_benchmark = base_dir_pros_benchmark / "kisik2" / "kisik2_prospektiv_ml_dataset.parquet"
senior_benchmark_file_pros_benchmark = analysis_dir_pros_benchmark / "los_senior_estimates_tagesausleitung_stay_level.csv"
output_prefix_pros_benchmark = "los_retro_markus_icu_los_gt1d_24h_clean_randomforest_prospective_senior_benchmark"

required_previous_objects_pros_benchmark = [
    "models_markus_train",
    "build_feature_matrix_markus_train",
    "selected_feature_names_markus_train",
    "x_markus_train",
    "ward_mapping_markus_train",
    "allowed_ward_oebene_markus_train",
    "normalize_oebene_markus_train",
]
missing_previous_objects_pros_benchmark = [name for name in required_previous_objects_pros_benchmark if name not in globals()]
if missing_previous_objects_pros_benchmark:
    raise ValueError(f"Bitte zuerst die Trainingszelle ausfuehren. Fehlend: {missing_previous_objects_pros_benchmark}")

best_model_pros_benchmark = models_markus_train["RandomForest_log1p"]


def normalize_id_pros_benchmark(series):
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def metrics_for_prediction_pros_benchmark(y_true, y_pred):
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    y_pred = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true.loc[valid].to_numpy(dtype=float)
    y_pred = y_pred.loc[valid].to_numpy(dtype=float)
    if len(y_true) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Median_AE": np.nan, "Bias_mean": np.nan}
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "Median_AE": float(np.median(np.abs(y_true - y_pred))),
        "Bias_mean": float(np.mean(y_pred - y_true)),
    }

raw_pros_benchmark = duckdb.connect().execute(
    f"select * from read_parquet('{pros_parquet_pros_benchmark.as_posix()}')"
).fetchdf()
raw_pros_benchmark["wardshort"] = raw_pros_benchmark["wardshort"].astype("string").str.strip()
raw_pros_benchmark["oebenekurz"] = raw_pros_benchmark["oebenekurz"].astype("string").str.strip()
raw_pros_benchmark["oebenekurz_norm"] = raw_pros_benchmark["oebenekurz"].map(normalize_oebene_markus_train)
raw_pros_benchmark["mapped_icu_area"] = raw_pros_benchmark["oebenekurz_norm"].map(ward_mapping_markus_train)
allowed_pair_index_pros_benchmark = pd.MultiIndex.from_tuples(allowed_ward_oebene_markus_train, names=["wardshort", "oebenekurz"])
raw_pair_index_pros_benchmark = pd.MultiIndex.from_frame(raw_pros_benchmark[["wardshort", "oebenekurz"]])
raw_pros_benchmark["markus_icu_filter"] = raw_pair_index_pros_benchmark.isin(allowed_pair_index_pros_benchmark)
raw_pros_benchmark["icu_duration_d"] = pd.to_numeric(raw_pros_benchmark["icu_duration_h"], errors="coerce") / 24.0
raw_pros_benchmark["planbegin"] = pd.to_datetime(raw_pros_benchmark["planbegin"], errors="coerce")

fall_id_source_pros_benchmark = next((column for column in ["fallid", "fallnr", "case_id"] if column in raw_pros_benchmark.columns), None)
if fall_id_source_pros_benchmark is not None:
    raw_pros_benchmark["fallid_like"] = raw_pros_benchmark[fall_id_source_pros_benchmark]
elif "stay_id" in raw_pros_benchmark.columns:
    raw_pros_benchmark["fallid_like"] = raw_pros_benchmark["stay_id"].astype("string").str.replace(r"_stay.*$", "", regex=True)
else:
    raw_pros_benchmark["fallid_like"] = pd.NA

pros_model_mask_pros_benchmark = raw_pros_benchmark["markus_icu_filter"] & raw_pros_benchmark["icu_duration_d"].notna()
pros_model_df_pros_benchmark = raw_pros_benchmark.loc[pros_model_mask_pros_benchmark].copy().reset_index(drop=True)
pros_feature_df_pros_benchmark, pros_feature_mapping_pros_benchmark = build_feature_matrix_markus_train(
    pros_model_df_pros_benchmark,
    selected_feature_names_markus_train,
)
train_feature_columns_pros_benchmark = list(x_markus_train.columns)
for feature_name in train_feature_columns_pros_benchmark:
    if feature_name not in pros_feature_df_pros_benchmark.columns:
        pros_feature_df_pros_benchmark[feature_name] = np.nan
pros_x_pros_benchmark = pros_feature_df_pros_benchmark[train_feature_columns_pros_benchmark].copy()

pros_model_df_pros_benchmark["new_rf_pred_days"] = np.clip(
    np.asarray(best_model_pros_benchmark.predict(pros_x_pros_benchmark), dtype=float),
    0,
    None,
)
pros_model_df_pros_benchmark["new_rf_abs_error_days"] = np.abs(
    pros_model_df_pros_benchmark["icu_duration_d"] - pros_model_df_pros_benchmark["new_rf_pred_days"]
)

id_columns_pros_benchmark = [
    column for column in ["stay_id", "fallid_like", "pid", "wardshort", "oebenekurz", "mapped_icu_area", "planbegin", "planend", "icu_duration_d", "new_rf_pred_days", "new_rf_abs_error_days"]
    if column in pros_model_df_pros_benchmark.columns
]
pros_predictions_export_pros_benchmark = pros_model_df_pros_benchmark[id_columns_pros_benchmark].copy()
pros_predictions_export_pros_benchmark["stay_id_norm"] = normalize_id_pros_benchmark(pros_predictions_export_pros_benchmark["stay_id"])
pros_predictions_export_pros_benchmark["fallid_norm"] = normalize_id_pros_benchmark(pros_predictions_export_pros_benchmark["fallid_like"])

senior_raw_pros_benchmark = pd.read_csv(senior_benchmark_file_pros_benchmark, sep=";")
for column in ["tages_stay_id", "case_id", "tages_fallnr", "tages_pid", "y_true_days_tages", "best_senior_estimate_days", "senior_median_estimate_days", "model_pred_days_tages", "n_senior_estimates"]:
    if column not in senior_raw_pros_benchmark.columns:
        senior_raw_pros_benchmark[column] = pd.NA
senior_raw_pros_benchmark["stay_id_norm"] = normalize_id_pros_benchmark(senior_raw_pros_benchmark["tages_stay_id"])
senior_raw_pros_benchmark["fallid_norm"] = normalize_id_pros_benchmark(senior_raw_pros_benchmark["tages_fallnr"].fillna(senior_raw_pros_benchmark["case_id"]))

senior_join_columns_pros_benchmark = [
    "stay_id_norm",
    "fallid_norm",
    "tages_stay_id",
    "case_id",
    "tages_fallnr",
    "tages_pid",
    "ergebnis_stand",
    "Station",
    "n_senior_estimates",
    "y_true_days_tages",
    "best_senior_estimate_days",
    "best_senior_abs_error_days",
    "senior_median_estimate_days",
    "model_pred_days_tages",
]
senior_stay_level_pros_benchmark = senior_raw_pros_benchmark[senior_join_columns_pros_benchmark].copy()
senior_stay_level_pros_benchmark = senior_stay_level_pros_benchmark.drop_duplicates(subset=["stay_id_norm"], keep="first")

benchmark_by_stay_pros_benchmark = senior_stay_level_pros_benchmark.merge(
    pros_predictions_export_pros_benchmark,
    on="stay_id_norm",
    how="left",
    suffixes=("_senior", "_pros"),
)
benchmark_by_fall_pros_benchmark = senior_stay_level_pros_benchmark.merge(
    pros_predictions_export_pros_benchmark.sort_values("new_rf_abs_error_days").drop_duplicates("fallid_norm"),
    on="fallid_norm",
    how="left",
    suffixes=("_senior", "_pros"),
)
benchmark_df_pros_benchmark = benchmark_by_stay_pros_benchmark.copy()
missing_prediction_mask_pros_benchmark = benchmark_df_pros_benchmark["new_rf_pred_days"].isna()
fill_columns_pros_benchmark = [column for column in benchmark_df_pros_benchmark.columns if column in benchmark_by_fall_pros_benchmark.columns and column not in senior_join_columns_pros_benchmark]
benchmark_df_pros_benchmark.loc[missing_prediction_mask_pros_benchmark, fill_columns_pros_benchmark] = benchmark_by_fall_pros_benchmark.loc[missing_prediction_mask_pros_benchmark, fill_columns_pros_benchmark].to_numpy()
benchmark_df_pros_benchmark["match_level_new_rf"] = np.where(missing_prediction_mask_pros_benchmark & benchmark_df_pros_benchmark["new_rf_pred_days"].notna(), "fallid", "stay_id")
benchmark_df_pros_benchmark.loc[benchmark_df_pros_benchmark["new_rf_pred_days"].isna(), "match_level_new_rf"] = "unmatched"

benchmark_df_pros_benchmark["observed_los_days"] = pd.to_numeric(benchmark_df_pros_benchmark["y_true_days_tages"], errors="coerce").fillna(pd.to_numeric(benchmark_df_pros_benchmark["icu_duration_d"], errors="coerce"))
benchmark_df_pros_benchmark["best_senior_estimate_days"] = pd.to_numeric(benchmark_df_pros_benchmark["best_senior_estimate_days"], errors="coerce")
benchmark_df_pros_benchmark["senior_median_estimate_days"] = pd.to_numeric(benchmark_df_pros_benchmark["senior_median_estimate_days"], errors="coerce")
benchmark_df_pros_benchmark["old_model_pred_days"] = pd.to_numeric(benchmark_df_pros_benchmark["model_pred_days_tages"], errors="coerce")
benchmark_df_pros_benchmark["new_rf_pred_days"] = pd.to_numeric(benchmark_df_pros_benchmark["new_rf_pred_days"], errors="coerce")
benchmark_df_pros_benchmark["senior_abs_error_days"] = np.abs(benchmark_df_pros_benchmark["observed_los_days"] - benchmark_df_pros_benchmark["best_senior_estimate_days"])
benchmark_df_pros_benchmark["new_rf_abs_error_days"] = np.abs(benchmark_df_pros_benchmark["observed_los_days"] - benchmark_df_pros_benchmark["new_rf_pred_days"])
benchmark_df_pros_benchmark["old_model_abs_error_days"] = np.abs(benchmark_df_pros_benchmark["observed_los_days"] - benchmark_df_pros_benchmark["old_model_pred_days"])
benchmark_df_pros_benchmark["new_rf_better_than_senior"] = benchmark_df_pros_benchmark["new_rf_abs_error_days"] < benchmark_df_pros_benchmark["senior_abs_error_days"]
benchmark_df_pros_benchmark["senior_better_than_new_rf"] = benchmark_df_pros_benchmark["senior_abs_error_days"] < benchmark_df_pros_benchmark["new_rf_abs_error_days"]
benchmark_df_pros_benchmark["tie_new_rf_senior"] = np.isclose(benchmark_df_pros_benchmark["senior_abs_error_days"], benchmark_df_pros_benchmark["new_rf_abs_error_days"], equal_nan=False)
benchmark_df_pros_benchmark["los_gt1d_eval"] = benchmark_df_pros_benchmark["observed_los_days"] > 1.0
benchmark_df_pros_benchmark["longstay_ge7_eval"] = benchmark_df_pros_benchmark["observed_los_days"] >= 7.0
benchmark_df_pros_benchmark["has_new_rf_and_senior"] = benchmark_df_pros_benchmark[["observed_los_days", "new_rf_pred_days", "best_senior_estimate_days"]].notna().all(axis=1)

metric_rows_pros_benchmark = []
for subset_label, subset_mask in [
    ("Alle gematchten Oberarzt-Stays", benchmark_df_pros_benchmark["has_new_rf_and_senior"]),
    ("Oberarzt-Stays mit ICU-LoS > 1 Tag", benchmark_df_pros_benchmark["has_new_rf_and_senior"] & benchmark_df_pros_benchmark["los_gt1d_eval"]),
    ("Oberarzt-Stays mit ICU-LoS >= 7 Tage", benchmark_df_pros_benchmark["has_new_rf_and_senior"] & benchmark_df_pros_benchmark["longstay_ge7_eval"]),
    ("Oberarzt-Stays mit ICU-LoS < 7 Tage", benchmark_df_pros_benchmark["has_new_rf_and_senior"] & ~benchmark_df_pros_benchmark["longstay_ge7_eval"]),
]:
    subset = benchmark_df_pros_benchmark.loc[subset_mask].copy()
    for variant_label, pred_col in [
        ("Neues RandomForest_log1p", "new_rf_pred_days"),
        ("Oberarzt beste Schaetzung", "best_senior_estimate_days"),
        ("Alter Tages-ML-Benchmark", "old_model_pred_days"),
    ]:
        metrics = metrics_for_prediction_pros_benchmark(subset["observed_los_days"], subset[pred_col])
        metrics.update({"Subset": subset_label, "Variante": variant_label})
        metric_rows_pros_benchmark.append(metrics)
benchmark_metrics_pros_benchmark_df = pd.DataFrame(metric_rows_pros_benchmark)

head_to_head_rows_pros_benchmark = []
for subset_label, subset in [
    ("Alle gematchten Oberarzt-Stays", benchmark_df_pros_benchmark.loc[benchmark_df_pros_benchmark["has_new_rf_and_senior"]].copy()),
    ("Oberarzt-Stays mit ICU-LoS > 1 Tag", benchmark_df_pros_benchmark.loc[benchmark_df_pros_benchmark["has_new_rf_and_senior"] & benchmark_df_pros_benchmark["los_gt1d_eval"]].copy()),
    ("Oberarzt-Stays mit ICU-LoS >= 7 Tage", benchmark_df_pros_benchmark.loc[benchmark_df_pros_benchmark["has_new_rf_and_senior"] & benchmark_df_pros_benchmark["longstay_ge7_eval"]].copy()),
    ("Oberarzt-Stays mit ICU-LoS < 7 Tage", benchmark_df_pros_benchmark.loc[benchmark_df_pros_benchmark["has_new_rf_and_senior"] & ~benchmark_df_pros_benchmark["longstay_ge7_eval"]].copy()),
]:
    head_to_head_rows_pros_benchmark.append({
        "Subset": subset_label,
        "n": int(len(subset)),
        "Neues_Modell_besser_n": int(subset["new_rf_better_than_senior"].sum()),
        "Oberarzt_besser_n": int(subset["senior_better_than_new_rf"].sum()),
        "Gleichstand_n": int(subset["tie_new_rf_senior"].sum()),
        "Neues_Modell_besser_pct": round(float(subset["new_rf_better_than_senior"].mean() * 100), 1) if len(subset) else np.nan,
        "Oberarzt_besser_pct": round(float(subset["senior_better_than_new_rf"].mean() * 100), 1) if len(subset) else np.nan,
    })
head_to_head_pros_benchmark_df = pd.DataFrame(head_to_head_rows_pros_benchmark)

prospective_predictions_file_pros_benchmark = analysis_dir_pros_benchmark / f"{output_prefix_pros_benchmark}_predictions.csv"
benchmark_cases_file_pros_benchmark = analysis_dir_pros_benchmark / f"{output_prefix_pros_benchmark}_cases.csv"
benchmark_metrics_file_pros_benchmark = analysis_dir_pros_benchmark / f"{output_prefix_pros_benchmark}_metrics.csv"
head_to_head_file_pros_benchmark = analysis_dir_pros_benchmark / f"{output_prefix_pros_benchmark}_head_to_head.csv"
pros_predictions_export_pros_benchmark.to_csv(prospective_predictions_file_pros_benchmark, sep=";", index=False)
benchmark_df_pros_benchmark.to_csv(benchmark_cases_file_pros_benchmark, sep=";", index=False)
benchmark_metrics_pros_benchmark_df.to_csv(benchmark_metrics_file_pros_benchmark, sep=";", index=False)
head_to_head_pros_benchmark_df.to_csv(head_to_head_file_pros_benchmark, sep=";", index=False)

display(pd.DataFrame([
    {"Kennzahl": "Prospektive Markus-ICU-Stays mit beobachteter LoS", "Wert": int(len(pros_model_df_pros_benchmark))},
    {"Kennzahl": "Oberarzt-Stays in Benchmarkdatei", "Wert": int(len(senior_stay_level_pros_benchmark))},
    {"Kennzahl": "Gematcht mit neuer RF-Vorhersage und Oberarzt", "Wert": int(benchmark_df_pros_benchmark["has_new_rf_and_senior"].sum())},
    {"Kennzahl": "Davon ICU-LoS > 1 Tag", "Wert": int((benchmark_df_pros_benchmark["has_new_rf_and_senior"] & benchmark_df_pros_benchmark["los_gt1d_eval"]).sum())},
]))
display(benchmark_metrics_pros_benchmark_df.round(3))
display(head_to_head_pros_benchmark_df)
display(benchmark_df_pros_benchmark["match_level_new_rf"].value_counts(dropna=False).rename_axis("match_level").reset_index(name="n"))

print("Exporte geschrieben:")
print(prospective_predictions_file_pros_benchmark)
print(benchmark_cases_file_pros_benchmark)
print(benchmark_metrics_file_pros_benchmark)
print(head_to_head_file_pros_benchmark)