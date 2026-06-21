# -*- coding: utf-8 -*-
"""
Zwei Ergaenzungen fuer das Manuskript:
(A) Null-/Naiv-Modell-Baseline + LoS-Verteilungsvergleich (retro vs prospektiv) -> erklaert,
    warum der prospektive MAE nicht einbricht (die Zielgroesse streut prospektiv weniger).
(B) Modell vs. Oberarzt ueber zeitliche Abschnitte (Kalendermonate + Quartale) der
    Oberarzt-Erhebung (Jul-Dez 2025), je mit n, MAE, gepaarter Differenz und Wilcoxon-Test.
Quellen: retro/prospektiv-Parquet (LoS, planbegin) + canonical/metrics_prospective_fair24h_predictions.csv
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN = BASE/"Eigene Auswertung"; CAN = AN/"canonical"
RETRO = BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"
PROS  = BASE/"kisik2"/"kisik2_prospektiv_ml_dataset.parquet"
asql  = "('AIN','IZ32'),('AIN','IZ21'),('AIN','IZ31')"
con = duckdb.connect()

pred = pd.read_csv(CAN/"metrics_prospective_fair24h_predictions.csv", sep=";")
pred["stay_id"] = pred["stay_id"].astype(str)
y  = pred["los_obs"].to_numpy(float)
ob = pred["arzt"].to_numpy(float)
ml = np.clip(pred["pred_ExtraTrees"].to_numpy(float), 0, None)

# ============ (A) Null-Baseline + Verteilung ============
r = con.execute(f"SELECT icu_duration_h/24.0 los FROM read_parquet('{RETRO.as_posix()}') "
                f"WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()["los"].to_numpy()
med_retro = float(np.median(r))                       # deployment-realistic null = retro median
def mae(a,b): return float(np.mean(np.abs(a-b)))
rowsA = []
for nm, los, mdl_mae in [("Retrospective (AIN, full)", r, 2.751), ("Prospective (n=193)", y, mae(y,ml))]:
    rowsA.append({"Cohort":nm, "n":len(los), "LoS_mean":round(los.mean(),2), "LoS_median":round(float(np.median(los)),2),
                  "LoS_std":round(los.std(),2), "pct_gt7d":round(100*(los>7).mean(),1), "pct_gt14d":round(100*(los>14).mean(),1),
                  "Null_MAE_retroMed":round(mae(los,med_retro),2), "Model_MAE":round(mdl_mae,3)})
A = pd.DataFrame(rowsA); A.to_csv(CAN/"prospective_null_baseline.csv", sep=";", index=False)
print("=== (A) Null-/Naiv-Baseline (Konstante = retro Median %.2f d) + Verteilung ===" % med_retro)
print(A.to_string(index=False))
print("Lesart: prospektiv schlaegt das Modell (%.2f) die Konstante (%.2f) NICHT.\n" % (mae(y,ml), mae(y,med_retro)))

# ============ (B) Modell vs. Oberarzt ueber zeitliche Abschnitte ============
d = con.execute(f"SELECT stay_id, CAST(planbegin AS TIMESTAMP) pb FROM read_parquet('{PROS.as_posix()}')").df()
d["stay_id"] = d["stay_id"].astype(str)
m = pred.merge(d, on="stay_id", how="left")
m["los"]=y; m["ae_ob"]=np.abs(ob-y); m["ae_ml"]=np.abs(ml-y)
m["month"]   = m["pb"].dt.to_period("M").astype(str)
m["quarter"] = m["pb"].dt.to_period("Q").astype(str)
print(f"Oberarzt-Erhebungsfenster: {m['pb'].min().date()} bis {m['pb'].max().date()}")

def seg_table(col):
    out=[]
    for g,sub in m.groupby(col):
        ae_o=sub["ae_ob"].to_numpy(); ae_m=sub["ae_ml"].to_numpy(); n=len(sub)
        dmae=ae_o.mean()-ae_m.mean()
        try: wp=stats.wilcoxon(ae_o,ae_m).pvalue
        except Exception: wp=np.nan
        out.append({"Segment":g,"n":n,"LoS_median":round(float(sub['los'].median()),2),
                    "MAE_physician":round(ae_o.mean(),2),"MAE_model":round(ae_m.mean(),2),
                    "dMAE_phys_minus_model":round(dmae,2),"Wilcoxon_p":round(wp,4)})
    return pd.DataFrame(out)

byM = seg_table("month"); byQ = seg_table("quarter")
byM.to_csv(CAN/"prospective_by_month.csv", sep=";", index=False)
byQ.to_csv(CAN/"prospective_by_quarter.csv", sep=";", index=False)
print("\n=== (B) Modell vs. Oberarzt nach Monat ===");   print(byM.to_string(index=False))
print("\n=== (B) Modell vs. Oberarzt nach Quartal ==="); print(byQ.to_string(index=False))

# Figur: MAE-Verlauf Modell vs Oberarzt ueber Monate (gefuellte Luecke + Wertelabels)
PHYC,MODC="#c0392b","#1f5f9e"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
fig, ax = plt.subplots(figsize=(9,5))
x=np.arange(len(byM)); ph=byM["MAE_physician"].values; mo=byM["MAE_model"].values
ax.fill_between(x, ph, mo, where=(mo>=ph), color=MODC, alpha=0.10, interpolate=True, label="model worse than physician")
ax.plot(x, ph, "o-", color=PHYC, lw=2.3, ms=7.5, label="Senior physician", zorder=3)
ax.plot(x, mo, "s-", color=MODC, lw=2.3, ms=7.5, label="Extra Trees (final model)", zorder=3)
for i in range(len(x)):
    ax.annotate(f"{ph[i]:.2f}", (x[i],ph[i]), xytext=(0,-13), textcoords="offset points", ha="center", fontsize=7.6, color=PHYC)
    ax.annotate(f"{mo[i]:.2f}", (x[i],mo[i]), xytext=(0,8),  textcoords="offset points", ha="center", fontsize=7.6, color=MODC)
    ax.annotate(f"n={byM.iloc[i]['n']}", (x[i], max(ph[i],mo[i])+0.34), ha="center", fontsize=8, color="#888")
ax.set_xticks(x); ax.set_xticklabels(byM["Segment"], rotation=0, fontsize=9.5)
ax.set_ylabel("MAE (days) — lower is better"); ax.set_ylim(0, float(max(ph.max(),mo.max()))+0.9)
ax.set_title("Prospective MAE over time: model vs senior physician\n(senior-estimate window, monthly)", weight="bold", fontsize=12.5)
ax.legend(fontsize=9.5, framealpha=.95, loc="upper left")
fig.tight_layout(); fig.savefig(str(CAN/"fig_temporal_mae.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("\nGespeichert: prospective_null_baseline.csv, prospective_by_month.csv, prospective_by_quarter.csv, fig_temporal_mae.png")
