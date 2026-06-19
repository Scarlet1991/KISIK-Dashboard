"""
Erweiterte Auswertung: Oberarzt vs. ML (RandomForest + XGBoost)
================================================================
- Trainiert RF, ExtraTrees und XGBoost auf 24h-sauberen Features
- Vollstaendiger Head-to-Head: Oberarzt vs. bestes Modell
- Subgruppen, Kalibration, statistischer Test (Wilcoxon)
- Ausgabe: CSV + Konsolenausdruck
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

import builtins
def _dp(*a, **k): [print(x.to_string() if hasattr(x,'to_string') else str(x)) for x in a]
builtins.display = _dp
try:
    import IPython.display as _ipy; _ipy.display = _dp
except Exception: pass

# ---- Pfade ----------------------------------------------------------------
BASE          = Path(r"D:\Ausgangsdaten\KISIK Projekt")
ANALYSIS      = BASE / "Eigene Auswertung"
PARQUET       = BASE / "kisik2" / "kisik2_icu_ml_dataset_24h.parquet"
PROS_PARQUET  = BASE / "kisik2" / "kisik2_prospektiv_ml_dataset.parquet"
SENIOR_CSV    = ANALYSIS / "los_senior_estimates_tagesausleitung_stay_level.csv"
FEAT_CSV      = ANALYSIS / "los_selected_features_ain_24h_compact.csv"
OUT_PREFIX    = ANALYSIS / "los_oberarzt_vs_ml_xgboost"
RANDOM_STATE  = 42

# ---- Ward-Filter (identisch mit Zelle 71) ---------------------------------
allowed_ward_oebene = [
    ("AIN","IZ32"),("AIN","IZ21"),("AIN","IZ31"),("AIN","IZ01"),
    ("AUG","IZ01"),("AVT","IZ01"),("GCH","IZ01"),("GYN","IZ01"),
    ("HNO","IZ01"),("HTC","IZ01"),("IZPV","IZ01"),("MKG","IZ01"),
    ("NCH","IZ01"),("NUK","IZ01"),("STR","IZ01"),("UCH","IZ01"),("URO","IZ01"),
]

prefix_map = {"zugang24_":"zugang_","proc24_":"proc_","lab24_":"lab_",
              "vital24_":"vital_","med24_":"med_"}

def sanitize(v):
    import re
    t = str(v).strip().lower()
    t = re.sub(r"[^a-z0-9]+","_",t)
    t = re.sub(r"_+","_",t).strip("_")
    return t[:80] or "unknown"

def safe_log1p(v):  return np.log1p(np.clip(v, 0, None))
def safe_expm1(v):  return np.expm1(v)

def regression_metrics(y_true, y_pred, label=""):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ae = np.abs(y_true - y_pred)
    return {"Modell": label,
            "n": len(y_true),
            "MAE": round(float(ae.mean()), 3),
            "Median_AE": round(float(np.median(ae)), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
            "R2": round(float(r2_score(y_true, y_pred)), 3),
            "Bias": round(float((y_pred - y_true).mean()), 3)}

def build_feature_matrix(raw_df, selected_features):
    cols = set(raw_df.columns.astype(str))
    ts = pd.to_datetime(raw_df.get("planbegin", pd.Series()), errors="coerce")
    derived = pd.DataFrame({
        "admission_hour":    ts.dt.hour,
        "admission_weekday": ts.dt.dayofweek,
        "admission_month":   ts.dt.month,
    }, index=raw_df.index)

    features = pd.DataFrame(index=raw_df.index)
    rows = []
    for fn in selected_features:
        src, mapping = None, None
        if fn in cols:
            features[fn] = raw_df[fn]; src = fn; mapping = "direkt"
        elif fn in derived.columns:
            features[fn] = derived[fn]; src = "planbegin"; mapping = "abgeleitet"
        else:
            for op, np_ in prefix_map.items():
                if fn.startswith(op):
                    cand = np_ + fn[len(op):]
                    if cand in cols:
                        features[fn] = raw_df[cand]; src = cand; mapping = f"{op}->{np_}"
                        break
        rows.append({"Feature": fn, "Mapping": mapping or "nicht_verfuegbar", "Quelle": src})

    used = [r["Feature"] for r in rows if r["Quelle"] is not None]
    return features[used].copy(), pd.DataFrame(rows)

def infer_types(df):
    num, bin_, cat = [], [], []
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() >= df[c].notna().sum() * 0.8:
            df[c] = v
            uniq = set(v.dropna().round(8).unique())
            (bin_ if len(uniq) <= 2 and uniq <= {0.0, 1.0} else num).append(c)
        else:
            df[c] = df[c].astype("string"); cat.append(c)
    return num, bin_, cat

def make_preprocessor(num, bin_, cat, scale=False):
    t = []
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale: num_steps.append(("sc", StandardScaler()))
    if num:  t.append(("num", Pipeline(num_steps), num))
    if bin_: t.append(("bin", Pipeline([("imp", SimpleImputer(strategy="most_frequent"))]), bin_))
    if cat:  t.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                        ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(t, remainder="drop", verbose_feature_names_out=False)

# ===========================================================================
# 1. Retrospektives Training
# ===========================================================================
print("\n" + "="*65)
print(" SCHRITT 1: Retrospektives Training (24h-clean Parquet)")
print("="*65)

allowed_sql = ", ".join(f"('{w}','{o}')" for w,o in allowed_ward_oebene)
con = duckdb.connect()
df = con.execute(f"""
    SELECT * FROM read_parquet('{PARQUET.as_posix()}')
    WHERE (wardshort, oebenekurz) IN ({allowed_sql})
      AND icu_duration_h / 24.0 > 1
""").df()
con.close()
print(f"Kohorte: {len(df):,} Stays")

target_days = df["icu_duration_h"] / 24.0

# Features laden
feat_df = pd.read_csv(FEAT_CSV, sep=";")
selected = feat_df.loc[feat_df.get("Feature_Set", pd.Series(["compact_clinical_24h"]*len(feat_df))) == "compact_clinical_24h", "Feature"].tolist() \
    if "Feature_Set" in feat_df.columns else feat_df["Feature"].tolist()

X_raw, feat_map = build_feature_matrix(df, selected)
num_f, bin_f, cat_f = infer_types(X_raw.copy())
print(f"Features: {len(X_raw.columns)} verwendet | Direkt: {(feat_map.Mapping=='direkt').sum()} | Mapping: {(feat_map.Mapping.str.contains('->',na=False)).sum()} | Fehlt: {(feat_map.Mapping=='nicht_verfuegbar').sum()}")

# Train/Test Split (Patienten-basiert)
groups = df["pid"].fillna("unknown").astype(str).values if "pid" in df.columns else df.index.astype(str).values
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(X_raw, target_days, groups=groups))

X_train = X_raw.iloc[train_idx]
X_test  = X_raw.iloc[test_idx]
y_train = target_days.iloc[train_idx].values
y_test  = target_days.iloc[test_idx].values
pid_test = df["pid"].iloc[test_idx].values if "pid" in df.columns else None
fid_test = df["fallid"].iloc[test_idx].values if "fallid" in df.columns else None
sid_test = df["stay_id"].iloc[test_idx].values

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

preprocessor = make_preprocessor(num_f, bin_f, cat_f)

# Modelle definieren
MODELS = {
    "RandomForest": TransformedTargetRegressor(
        regressor=Pipeline([
            ("pre", make_preprocessor(num_f, bin_f, cat_f)),
            ("mdl", RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                          min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1))]),
        func=safe_log1p, inverse_func=safe_expm1),

    "ExtraTrees": TransformedTargetRegressor(
        regressor=Pipeline([
            ("pre", make_preprocessor(num_f, bin_f, cat_f)),
            ("mdl", ExtraTreesRegressor(n_estimators=300, max_features="sqrt",
                                         min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1))]),
        func=safe_log1p, inverse_func=safe_expm1),

    "XGBoost": TransformedTargetRegressor(
        regressor=Pipeline([
            ("pre", make_preprocessor(num_f, bin_f, cat_f, scale=False)),
            ("mdl", XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8,
                                  reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=RANDOM_STATE, n_jobs=-1,
                                  verbosity=0))]),
        func=safe_log1p, inverse_func=safe_expm1),

    "Ridge": TransformedTargetRegressor(
        regressor=Pipeline([
            ("pre", make_preprocessor(num_f, bin_f, cat_f, scale=True)),
            ("mdl", Ridge(alpha=1.0))]),
        func=safe_log1p, inverse_func=safe_expm1),
}

holdout_results = []
fitted_models   = {}
preds_test      = {}

for name, model in MODELS.items():
    print(f"\n  Trainiere {name} ...", end=" ", flush=True)
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, None)
    fitted_models[name] = model
    preds_test[name]    = pred
    m = regression_metrics(y_test, pred, name)
    holdout_results.append(m)
    print(f"MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  Bias={m['Bias']:.3f}")

holdout_df = pd.DataFrame(holdout_results).sort_values("MAE")
print("\nHoldout-Ergebnisse:")
print(holdout_df.to_string(index=False))
holdout_df.to_csv(str(OUT_PREFIX) + "_holdout_metrics.csv", sep=";", index=False)

# Bestes Modell
best_name  = holdout_df.iloc[0]["Modell"]
best_model = fitted_models[best_name]
print(f"\nBestes Modell: {best_name}")

# ===========================================================================
# 2. Prospektiver Benchmark mit Oberarzt
# ===========================================================================
print("\n" + "="*65)
print(" SCHRITT 2: Oberarzt vs. ML (prospektive Kohorte)")
print("="*65)

con2 = duckdb.connect()
pros_df = con2.execute(f"""
    SELECT * FROM read_parquet('{PROS_PARQUET.as_posix()}')
    WHERE (wardshort, oebenekurz) IN ({allowed_sql})
      AND icu_duration_h / 24.0 > 0
""").df()
con2.close()
print(f"Prospektive Kohorte: {len(pros_df):,} Stays")

# Alle Modelle auf Prospektiv anwenden
# Fehlende Spalten mit 0 auffuellen damit der ColumnTransformer nicht abbricht
X_pros_raw, _ = build_feature_matrix(pros_df, selected)
for name, model in fitted_models.items():
    X_pros = X_pros_raw.copy()
    # Spalten die im Training vorhanden waren, in Pros aber fehlen -> 0
    train_cols = X_raw.columns.tolist()
    for col in train_cols:
        if col not in X_pros.columns:
            X_pros[col] = 0
    X_pros = X_pros[train_cols]  # gleiche Reihenfolge wie Training
    pros_df[f"pred_{name}"] = np.clip(model.predict(X_pros), 0, None)

# Oberarzt einlesen und matchen
senior_df = pd.read_csv(SENIOR_CSV, sep=";")
print(f"Oberarzt-Datei: {len(senior_df):,} Eintraege")

# Merge ueber stay_id (tages_stay_id im Senior-CSV)
merge_key_senior = "tages_stay_id" if "tages_stay_id" in senior_df.columns else "stay_id"
merge_key_pros   = "stay_id"       if "stay_id"       in pros_df.columns  else "fallnr"

merged = pros_df.merge(
    senior_df, left_on=merge_key_pros, right_on=merge_key_senior, how="inner"
)
print(f"Match: {len(merged):,} Stays")

# Beobachtete LoS: aus tages-Spalte oder icu_duration_h
if "y_true_days_tages" in merged.columns:
    merged["los_obs"] = pd.to_numeric(merged["y_true_days_tages"], errors="coerce")
elif "icu_duration_h" in merged.columns:
    merged["los_obs"] = merged["icu_duration_h"] / 24.0

# Bester Oberarzt-Schaetzer
arzt_col = "best_senior_estimate_days" if "best_senior_estimate_days" in merged.columns \
    else "clinical_estimated_los_in_days"
merged["arzt_pred"] = pd.to_numeric(merged[arzt_col], errors="coerce")
print(f"Oberarzt-Spalte: '{arzt_col}'")
merged = merged.dropna(subset=["arzt_pred", "los_obs"])
print(f"Auswertbare Stays (nach Match + Dropna): {len(merged):,}")

# ===========================================================================
# 3. Head-to-Head Auswertung je Subgruppe
# ===========================================================================
print("\n" + "="*65)
print(" SCHRITT 3: Head-to-Head Auswertung je Subgruppe & Modell")
print("="*65)

subgroups = {
    "Gesamt": merged,
    "LoS > 1 Tag": merged[merged["los_obs"] > 1],
    "LoS 1-7 Tage": merged[(merged["los_obs"] > 1) & (merged["los_obs"] <= 7)],
    "LoS > 7 Tage": merged[merged["los_obs"] > 7],
    "LoS > 14 Tage": merged[merged["los_obs"] > 14],
}

all_rows = []
for sg_name, sg_df in subgroups.items():
    if sg_df.empty:
        continue
    y_true = sg_df["los_obs"].values
    y_arzt = sg_df["arzt_pred"].values

    for model_name in list(fitted_models.keys()) + ["Oberarzt"]:
        if model_name == "Oberarzt":
            y_pred = y_arzt
        else:
            col = f"pred_{model_name}"
            if col not in sg_df.columns:
                continue
            y_pred = sg_df[col].values

        ae = np.abs(y_true - y_pred)
        row = regression_metrics(y_true, y_pred, model_name)
        row["Subgruppe"] = sg_name
        all_rows.append(row)

bench_df = pd.DataFrame(all_rows)[["Subgruppe","Modell","n","MAE","Median_AE","RMSE","R2","Bias"]]
print("\nHead-to-Head Ergebnisse:")
print(bench_df.to_string(index=False))

# ===========================================================================
# 4. Statistischer Test: Wilcoxon Signed-Rank (paarer Vergleich AE)
# ===========================================================================
print("\n" + "="*65)
print(" SCHRITT 4: Wilcoxon Signed-Rank Test (|Fehler| ML vs. Oberarzt)")
print("="*65)

stat_rows = []
for model_name in fitted_models.keys():
    col = f"pred_{model_name}"
    sub = merged.dropna(subset=[col, "arzt_pred", "los_obs"])
    if sub.empty: continue
    ae_ml   = np.abs(sub["los_obs"].values - sub[col].values)
    ae_arzt = np.abs(sub["los_obs"].values - sub["arzt_pred"].values)
    stat, pval = stats.wilcoxon(ae_ml, ae_arzt, alternative="two-sided")
    stat_rows.append({
        "Modell": model_name,
        "Median_AE_ML": round(float(np.median(ae_ml)), 3),
        "Median_AE_Arzt": round(float(np.median(ae_arzt)), 3),
        "Differenz (ML-Arzt)": round(float(np.median(ae_ml) - np.median(ae_arzt)), 3),
        "Wilcoxon-W": round(float(stat), 1),
        "p-Wert": f"{pval:.4f}",
        "Signifikant (p<0.05)": "Ja" if pval < 0.05 else "Nein",
        "Interpretation": "Arzt besser" if np.median(ae_ml) > np.median(ae_arzt) else "ML besser",
    })

stat_df = pd.DataFrame(stat_rows)
print(stat_df.to_string(index=False))

# ===========================================================================
# 5. Stay-Level: Wer ist besser? (Best-of-All-ML vs. Oberarzt)
# ===========================================================================
print("\n" + "="*65)
print(" SCHRITT 5: Stay-Level Head-to-Head (bestes ML vs. Oberarzt)")
print("="*65)

sub = merged.copy()
ml_cols = [f"pred_{m}" for m in fitted_models if f"pred_{m}" in sub.columns]
sub["ae_arzt"] = np.abs(sub["los_obs"] - sub["arzt_pred"])

h2h_rows = []
for col in ml_cols:
    mname = col.replace("pred_","")
    sub[f"ae_{mname}"] = np.abs(sub["los_obs"] - sub[col])
    sub[f"ml_besser_{mname}"] = sub[f"ae_{mname}"] < sub["ae_arzt"]
    n_ml   = int(sub[f"ml_besser_{mname}"].sum())
    n_arzt = int((~sub[f"ml_besser_{mname}"]).sum())
    h2h_rows.append({"Modell": mname,
                     "ML_besser_n": n_ml,
                     "Arzt_besser_n": n_arzt,
                     "Gesamt": len(sub),
                     "ML_besser_pct": round(n_ml/len(sub)*100, 1),
                     "Arzt_besser_pct": round(n_arzt/len(sub)*100, 1)})

h2h_df = pd.DataFrame(h2h_rows).sort_values("ML_besser_pct", ascending=False)
print(h2h_df.to_string(index=False))

# ===========================================================================
# 6. Kalibrierung: mittlere Vorhersage vs. Wahrheit je LoS-Bin
# ===========================================================================
print("\n" + "="*65)
print(" SCHRITT 6: Kalibrierung je LoS-Bin (Oberarzt und Modelle)")
print("="*65)

bins   = [0, 2, 4, 7, 14, 30, 999]
labels = ["0-2","2-4","4-7","7-14","14-30",">30"]
sub["los_bin"] = pd.cut(sub["los_obs"], bins=bins, labels=labels)

cal_rows = []
for sg, grp in sub.groupby("los_bin", observed=False):
    if grp.empty: continue
    n = len(grp)
    row = {"LoS-Bin (Tage)": str(sg), "n": n,
           "Beob. Mittel": round(grp["los_obs"].mean(), 2),
           "Oberarzt Mittel": round(grp["arzt_pred"].mean(), 2)}
    for col in ml_cols:
        row[col.replace("pred_","") + " Mittel"] = round(grp[col].mean(), 2)
    cal_rows.append(row)

cal_df = pd.DataFrame(cal_rows)
print(cal_df.to_string(index=False))

# ===========================================================================
# Exporte
# ===========================================================================
holdout_df.to_csv(  str(OUT_PREFIX)+"_holdout_metrics.csv",    sep=";", index=False)
bench_df.to_csv(    str(OUT_PREFIX)+"_h2h_subgroups.csv",       sep=";", index=False)
stat_df.to_csv(     str(OUT_PREFIX)+"_wilcoxon.csv",            sep=";", index=False)
h2h_df.to_csv(      str(OUT_PREFIX)+"_stay_level_h2h.csv",      sep=";", index=False)
cal_df.to_csv(      str(OUT_PREFIX)+"_calibration_bins.csv",    sep=";", index=False)
merged[[c for c in merged.columns if c in ["stay_id","fallid","los_obs","arzt_pred"]+ml_cols]]\
    .to_csv(str(OUT_PREFIX)+"_predictions.csv", sep=";", index=False)

print("\n" + "="*65)
print("Exporte geschrieben:")
for suffix in ["_holdout_metrics","_h2h_subgroups","_wilcoxon","_stay_level_h2h","_calibration_bins","_predictions"]:
    print(f"  {OUT_PREFIX}{suffix}.csv")

# ===========================================================================
# 7. Publikationsgrafikenn (300 DPI, serifenlos, NEJM-nah)
# ===========================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

FIG_DIR = ANALYSIS
DPI = 300
GREY  = "#4d4d4d"
BLUE  = "#2166ac"
RED   = "#d6604d"
GREEN = "#1a9850"
ORAN  = "#f4a582"
TEAL  = "#4dac26"

MODEL_COLORS = {
    "RandomForest":  BLUE,
    "ExtraTrees":    TEAL,
    "XGBoost":       "#762a83",
    "Ridge":         ORAN,
    "Oberarzt":      RED,
}

MODEL_LABELS = {
    "RandomForest": "Random Forest",
    "ExtraTrees":   "Extra Trees",
    "XGBoost":      "XGBoost",
    "Ridge":        "Ridge Regression",
    "Oberarzt":     "Oberarzt",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": DPI,
})

# ------------------------------------------------------------------
# FIG 1: MAE- und RMSE-Balkendiagramm (Holdout-Testset)
# ------------------------------------------------------------------
fig1, axes = plt.subplots(1, 2, figsize=(10, 4.5))

models_hold = holdout_df["Modell"].tolist()
colors_hold = [MODEL_COLORS.get(m, GREY) for m in models_hold]
labels_hold = [MODEL_LABELS.get(m, m) for m in models_hold]

for ax, metric, ylabel in [
    (axes[0], "MAE",  "MAE (Tage)"),
    (axes[1], "RMSE", "RMSE (Tage)"),
]:
    vals = holdout_df[metric].values
    bars = ax.bar(labels_hold, vals, color=colors_hold, edgecolor="white", linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} – Holdout-Testset", pad=8)
    ax.set_xticks(range(len(labels_hold)))
    ax.set_xticklabels(labels_hold, rotation=25, ha="right")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, color=GREY)

fig1.suptitle("Modellgüte im Holdout-Testset (24h-Zeitfenster-Features)", y=1.01, fontsize=12, weight="bold")
fig1.tight_layout()
fig1_path = FIG_DIR / "fig1_holdout_mae_rmse.png"
fig1.savefig(str(fig1_path), dpi=DPI, bbox_inches="tight")
plt.close(fig1)
print(f"\nFig 1 gespeichert: {fig1_path}")

# ------------------------------------------------------------------
# FIG 2: Head-to-Head Subgruppen (MAE je Modell, gruppierte Balken)
# ------------------------------------------------------------------
gesamt_sub = bench_df[bench_df["Subgruppe"] == "LoS > 1 Tag"].copy()
grp_sub    = bench_df[bench_df["Subgruppe"] != "Gesamt"].copy()

grp_names  = grp_sub["Subgruppe"].unique().tolist()
all_models = grp_sub["Modell"].unique().tolist()
n_grp      = len(grp_names)
n_mod      = len(all_models)
bar_w      = 0.8 / n_mod

fig2, ax2  = plt.subplots(figsize=(12, 5))
for i, mname in enumerate(all_models):
    mdata = grp_sub[grp_sub["Modell"] == mname]
    vals  = [mdata[mdata["Subgruppe"] == g]["MAE"].values[0]
             if not mdata[mdata["Subgruppe"] == g].empty else np.nan
             for g in grp_names]
    xs    = np.arange(n_grp) + i * bar_w
    ax2.bar(xs, vals, width=bar_w*0.9,
            color=MODEL_COLORS.get(mname, GREY),
            label=MODEL_LABELS.get(mname, mname),
            edgecolor="white", linewidth=0.5)

ax2.set_xticks(np.arange(n_grp) + bar_w*(n_mod-1)/2)
ax2.set_xticklabels(grp_names, rotation=15, ha="right")
ax2.set_ylabel("MAE (Tage)")
ax2.set_title("Head-to-Head: MAE nach LoS-Subgruppe\n(Prospektiv, Oberarzt vs. ML-Modelle)", pad=8, weight="bold")
ax2.legend(loc="upper right", framealpha=0.3)
fig2.tight_layout()
fig2_path = FIG_DIR / "fig2_subgroup_mae.png"
fig2.savefig(str(fig2_path), dpi=DPI, bbox_inches="tight")
plt.close(fig2)
print(f"Fig 2 gespeichert: {fig2_path}")

# ------------------------------------------------------------------
# FIG 3: Bland-Altman-Diagramme (XGBoost und Oberarzt)
# ------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

candidates = [m for m in ["XGBoost","RandomForest","ExtraTrees","Ridge"] if f"pred_{m}" in merged.columns]
best_ml = candidates[0] if candidates else None

panels = [(best_ml, BLUE, f"ML: {MODEL_LABELS.get(best_ml, best_ml)}"), ("Oberarzt", RED, "Oberarzt")]
for ax, (mname, col, title) in zip(axes3, panels):
    if mname == "Oberarzt":
        y_pred_ba = merged["arzt_pred"].values
    elif mname and f"pred_{mname}" in merged.columns:
        y_pred_ba = merged[f"pred_{mname}"].values
    else:
        ax.set_visible(False)
        continue
    y_true_ba = merged["los_obs"].values
    diff   = y_pred_ba - y_true_ba
    mean_d = np.nanmean(diff)
    sd_d   = np.nanstd(diff)
    ax.scatter((y_true_ba + y_pred_ba) / 2, diff,
               alpha=0.25, s=12, color=col, linewidths=0)
    ax.axhline(mean_d,          color="black", lw=1.5, linestyle="-")
    ax.axhline(mean_d + 1.96*sd_d, color="black", lw=1, linestyle="--")
    ax.axhline(mean_d - 1.96*sd_d, color="black", lw=1, linestyle="--")
    ax.axhline(0, color=GREY, lw=0.8, linestyle=":")
    ax.set_xlabel("Mittlere LoS (Tage)")
    ax.set_ylabel("Vorhersage – Beobachtung (Tage)")
    ax.set_title(title, pad=6)
    ax.text(0.97, 0.97, f"Bias={mean_d:.2f}d\n±1.96 SD={1.96*sd_d:.2f}d",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

fig3.suptitle("Bland-Altman: Systematische Über-/Unterschätzung der ICU-LoS", weight="bold", y=1.01)
fig3.tight_layout()
fig3_path = FIG_DIR / "fig3_bland_altman.png"
fig3.savefig(str(fig3_path), dpi=DPI, bbox_inches="tight")
plt.close(fig3)
print(f"Fig 3 gespeichert: {fig3_path}")

# ------------------------------------------------------------------
# FIG 4: Scatter (Beobachtet vs. Vorhergesagt) – XGBoost & Oberarzt
# ------------------------------------------------------------------
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True, sharex=True)

for ax, (mname, col, title) in zip(axes4, panels):
    if mname == "Oberarzt":
        y_pred_sc = merged["arzt_pred"].values
    elif mname and f"pred_{mname}" in merged.columns:
        y_pred_sc = merged[f"pred_{mname}"].values
    else:
        ax.set_visible(False)
        continue
    y_true_sc = merged["los_obs"].values
    ax.scatter(y_true_sc, y_pred_sc, alpha=0.2, s=12, color=col, linewidths=0)
    lim = max(np.nanmax(y_true_sc), np.nanmax(y_pred_sc)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6)
    mae_sc = np.nanmean(np.abs(y_true_sc - y_pred_sc))
    r2_sc  = r2_score(y_true_sc[~np.isnan(y_pred_sc)], y_pred_sc[~np.isnan(y_pred_sc)])
    ax.set_xlabel("Beobachtete ICU-LoS (Tage)")
    ax.set_ylabel("Vorhergesagte ICU-LoS (Tage)")
    ax.set_title(title, pad=6)
    ax.text(0.04, 0.96, f"MAE={mae_sc:.2f}d  R²={r2_sc:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

fig4.suptitle("Beobachtete vs. vorhergesagte ICU-Verweildauer", weight="bold", y=1.01)
fig4.tight_layout()
fig4_path = FIG_DIR / "fig4_scatter_observed_predicted.png"
fig4.savefig(str(fig4_path), dpi=DPI, bbox_inches="tight")
plt.close(fig4)
print(f"Fig 4 gespeichert: {fig4_path}")

# ------------------------------------------------------------------
# FIG 5: Kalibrierungskurve – Mittlere Vorhersage vs. Beob. je LoS-Bin
# ------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(9, 5))

los_bins_sorted = cal_df["LoS-Bin (Tage)"].tolist()
x_idx = np.arange(len(los_bins_sorted))
obs_means = cal_df["Beob. Mittel"].values

ax5.plot(x_idx, obs_means, "ko-", lw=2, ms=6, label="Beobachtet", zorder=5)

for mname in list(fitted_models.keys()) + ["Oberarzt"]:
    col_label = MODEL_LABELS.get(mname, mname) + " Mittel"
    if col_label not in cal_df.columns:
        col_label = mname + " Mittel"
    if col_label not in cal_df.columns:
        continue
    vals = cal_df[col_label].values
    ax5.plot(x_idx, vals, "o--", color=MODEL_COLORS.get(mname, GREY),
             lw=1.5, ms=5, label=MODEL_LABELS.get(mname, mname))

ax5.set_xticks(x_idx)
ax5.set_xticklabels(los_bins_sorted, rotation=15, ha="right")
ax5.set_xlabel("LoS-Gruppe (Tage)")
ax5.set_ylabel("Mittlere ICU-LoS (Tage)")
ax5.set_title("Kalibrierungskurve: Mittlere Vorhersage vs. Beobachtung\nje LoS-Stratum (prospektive Kohorte)", pad=8, weight="bold")
ax5.legend(loc="upper left", framealpha=0.3)

fig5.tight_layout()
fig5_path = FIG_DIR / "fig5_calibration.png"
fig5.savefig(str(fig5_path), dpi=DPI, bbox_inches="tight")
plt.close(fig5)
print(f"Fig 5 gespeichert: {fig5_path}")

# ------------------------------------------------------------------
# FIG 6: Stay-Level-Anteil (ML besser vs. Arzt besser) – Stacked Bar
# ------------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(8, 4))

h2h_sorted = h2h_df.sort_values("ML_besser_pct", ascending=True)
y_idx      = np.arange(len(h2h_sorted))
ml_pct     = h2h_sorted["ML_besser_pct"].values
ar_pct     = h2h_sorted["Arzt_besser_pct"].values
mlabels    = [MODEL_LABELS.get(m, m) for m in h2h_sorted["Modell"].values]
mcolors    = [MODEL_COLORS.get(m, GREY) for m in h2h_sorted["Modell"].values]

ax6.barh(y_idx, ml_pct, color=mcolors, label="ML besser", edgecolor="white")
ax6.barh(y_idx, ar_pct, left=ml_pct, color=RED, alpha=0.55, label="Oberarzt besser", edgecolor="white")
ax6.axvline(50, color=GREY, lw=1, linestyle="--")

for i, (mp, ap) in enumerate(zip(ml_pct, ar_pct)):
    ax6.text(mp/2,      i, f"{mp:.0f}%", ha="center", va="center", fontsize=8, color="white", weight="bold")
    ax6.text(mp+ap/2,   i, f"{ap:.0f}%", ha="center", va="center", fontsize=8, color="white", weight="bold")

ax6.set_yticks(y_idx)
ax6.set_yticklabels(mlabels)
ax6.set_xlabel("Anteil Stays (%)")
ax6.set_title("Stay-Level Head-to-Head: ML vs. Oberarzt\n(% Stays mit kleinerem absoluten Fehler)", pad=8, weight="bold")
ax6.legend(loc="lower right", framealpha=0.3)
ax6.set_xlim(0, 100)

fig6.tight_layout()
fig6_path = FIG_DIR / "fig6_staywise_h2h.png"
fig6.savefig(str(fig6_path), dpi=DPI, bbox_inches="tight")
plt.close(fig6)
print(f"Fig 6 gespeichert: {fig6_path}")

print("\n" + "="*65)
print("Alle 6 Publikationsgrafiken gespeichert in:")
print(f"  {FIG_DIR}")
print("="*65)
