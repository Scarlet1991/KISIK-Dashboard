# -*- coding: utf-8 -*-
"""
Prospektive Unsicherheits- und Kalibrationsanalyse (n=193, dieselben Patienten fuer Modell + Oberarzt).
- Gepaarter, patientenbasierter Bootstrap (B=5000): MAE, RMSE, Bias, R2 je mit 95%-KI
  fuer Oberarzt und finales Modell (ExtraTrees); gepaarte Differenz der absoluten Fehler
  (Oberarzt - ML) mit 95%-KI; Wilcoxon-Test der gepaarten absoluten Fehler (gesamt und 1-7 Tage).
- Kalibration: Calibration-in-the-large (mittlerer Bias), Kalibrationssteigung (OLS obs~pred),
  Observed-vs-Predicted mit geglaetteter (quantils-gebinnter) Kurve, Vorhersageverteilung,
  Auswertung nach LoS-Gruppen.
Quelle: canonical/metrics_prospective_fair24h_predictions.csv (los_obs, arzt, pred_*).
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

AN  = Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung"); CAN = AN/"canonical"
pred = pd.read_csv(CAN/"metrics_prospective_fair24h_predictions.csv", sep=";")
y   = pred["los_obs"].to_numpy(float)
FINAL = "ExtraTrees"
series = {"Senior physician": pred["arzt"].to_numpy(float),
          "Extra Trees (final model)": pred[f"pred_{FINAL}"].to_numpy(float),
          "XGBoost": pred["pred_XGBoost"].to_numpy(float),
          "Random forest": pred["pred_RandomForest"].to_numpy(float),
          "Ridge": pred["pred_Ridge"].to_numpy(float)}
n = len(y); B = 5000; RNG = np.random.default_rng(42)
print(f"n = {n} paired stays | bootstrap B = {B}")

def met(yt, yp):
    e = np.clip(yp,0,None) - yt; ae = np.abs(e)
    return {"MAE":ae.mean(), "RMSE":np.sqrt(np.mean(e**2)), "Bias":e.mean(), "R2":r2_score(yt,yp)}

# ---- paired bootstrap: identical resampled patient indices across all methods ----
idx = RNG.integers(0, n, size=(B, n))
def ci(vals): return np.percentile(vals, [2.5, 97.5])

rows = []
for name, yp in series.items():
    pt = met(y, yp)
    bt = {k:[] for k in ["MAE","RMSE","Bias","R2"]}
    for b in range(B):
        m = met(y[idx[b]], yp[idx[b]])
        for k in bt: bt[k].append(m[k])
    for k in ["MAE","RMSE","Bias","R2"]:
        lo, hi = ci(bt[k])
        rows.append({"Method":name, "Metric":k, "Estimate":round(pt[k],3),
                     "CI_low":round(lo,3), "CI_high":round(hi,3)})
ci_df = pd.DataFrame(rows)
ci_df.to_csv(CAN/"prospective_bootstrap_ci.csv", sep=";", index=False)
print("\n=== Bootstrap 95% CIs ===")
print(ci_df.to_string(index=False))

# ---- paired difference of absolute errors: Oberarzt - ML(final) ----
yp_ob = series["Senior physician"]; yp_ml = series["Extra Trees (final model)"]
ae_ob = np.abs(yp_ob - y); ae_ml = np.abs(np.clip(yp_ml,0,None) - y)
def paired_block(mask, label):
    d = ae_ob[mask] - ae_ml[mask]                 # <0 => physician smaller error (better)
    pt = d.mean()
    bt = [ (ae_ob[idx[b]][mask[idx[b]]] - ae_ml[idx[b]][mask[idx[b]]]).mean()
           if mask[idx[b]].sum()>0 else np.nan for b in range(B) ]
    lo, hi = np.nanpercentile(bt, [2.5, 97.5])
    w_stat, w_p = stats.wilcoxon(ae_ob[mask], ae_ml[mask])
    return {"Subgroup":label, "n":int(mask.sum()),
            "MAE_physician":round(ae_ob[mask].mean(),3), "MAE_ML":round(ae_ml[mask].mean(),3),
            "dAE_phys_minus_ML":round(pt,3), "CI_low":round(lo,3), "CI_high":round(hi,3),
            "Wilcoxon_p":w_p}
diff_rows = [paired_block(np.ones(n,bool), "All (n=193)"),
             paired_block((y>1)&(y<=7), "1-7 days"),
             paired_block(y>7, ">7 days")]
diff_df = pd.DataFrame(diff_rows)
diff_df.to_csv(CAN/"prospective_paired_diff.csv", sep=";", index=False)
print("\n=== Paired difference of absolute errors (physician - ML), 95% CI + Wilcoxon ===")
print(diff_df.to_string(index=False))

# ---- calibration: slope (OLS obs~pred) + CITL (mean bias) with bootstrap CI ----
def slope_intercept(yt, yp):
    b, a = np.polyfit(yp, yt, 1)   # yt = a + b*yp
    return b, a
cal_rows = []
for name in ["Senior physician", "Extra Trees (final model)"]:
    yp = np.clip(series[name],0,None)
    b, a = slope_intercept(y, yp); citl = (yp - y).mean()
    bs = [slope_intercept(y[idx[k]], yp[idx[k]])[0] for k in range(B)]
    lo, hi = ci(bs)
    cal_rows.append({"Method":name, "Calib_slope":round(b,3), "Slope_CI_low":round(lo,3),
                     "Slope_CI_high":round(hi,3), "Intercept":round(a,3),
                     "CITL_mean_bias":round(citl,3)})
cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv(CAN/"calibration_slopes.csv", sep=";", index=False)
print("\n=== Calibration slope (OLS observed~predicted) + calibration-in-the-large ===")
print(cal_df.to_string(index=False))

# ---- analysis by LoS group ----
edges = [1,2,4,7,np.inf]; labels = ["1-2 d","2-4 d","4-7 d",">7 d"]
grp = pd.cut(y, bins=edges, labels=labels, right=True)
gr_rows = []
for g in labels:
    m = np.asarray(grp==g)
    if m.sum()==0: continue
    row = {"LoS group":g, "n":int(m.sum()), "Observed median (d)":round(float(np.median(y[m])),2)}
    for name,key in [("Senior physician","phys"),("Extra Trees (final model)","ML")]:
        yp = np.clip(series[name],0,None)
        row[f"MAE_{key}"]   = round(float(np.abs(yp[m]-y[m]).mean()),2)
        row[f"Bias_{key}"]  = round(float((yp[m]-y[m]).mean()),2)
    gr_rows.append(row)
grp_df = pd.DataFrame(gr_rows)
grp_df.to_csv(CAN/"calibration_by_losgroup.csv", sep=";", index=False)
print("\n=== By LoS group (MAE / bias, days) ===")
print(grp_df.to_string(index=False))

# ============================ FIGURES ============================
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
CAP = 20
def binned_curve(yp, yt, nb=8):
    """quantile-binned calibration curve: mean predicted vs mean observed per bin."""
    qs = np.quantile(yp, np.linspace(0,1,nb+1)); qs[0]-=1e-9
    mp, mo = [], []
    for i in range(nb):
        m = (yp>qs[i]) & (yp<=qs[i+1])
        if m.sum()>=3: mp.append(yp[m].mean()); mo.append(yt[m].mean())
    return np.array(mp), np.array(mo)

# Fig 1: calibration (observed vs predicted) — model + physician, with binned curve + OLS line
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, name in zip(axes, ["Extra Trees (final model)", "Senior physician"]):
    yp = np.clip(series[name],0,None)
    ax.scatter(yp, y, s=16, alpha=.35, color="#185fa5", edgecolor="none")
    ax.plot([0,CAP],[0,CAP],"--",color="#d6604d",lw=1.5,label="Identity")
    b,a = slope_intercept(y, yp); xs = np.linspace(0,CAP,50)
    ax.plot(xs, a+b*xs, "-", color="#1a9850", lw=1.8, label=f"Calibration (slope {b:.2f})")
    mp, mo = binned_curve(yp, y)
    ax.plot(mp, mo, "o-", color="#762a83", lw=1.6, ms=5, label="Binned observed mean")
    citl = (yp-y).mean()
    ax.set_xlim(0,CAP); ax.set_ylim(0,CAP); ax.set_aspect("equal")
    ax.set_xlabel("Predicted ICU LoS (days)"); ax.set_ylabel("Observed ICU LoS (days)")
    ax.set_title(f"{name}\nslope {b:.2f}, CITL {citl:+.2f} d", weight="bold", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=.5)
fig.suptitle("Prospective calibration: observed vs predicted ICU length of stay (n = 193)", weight="bold", fontsize=12.5)
fig.tight_layout(); fig.savefig(str(CAN/"fig_calibration_pros.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# Fig 2: prediction distribution (observed vs model vs physician) — gefuellte Dichten
def _kde(v):
    v=np.clip(v,0,CAP); xs=np.linspace(0,CAP,200)
    try:
        from scipy.stats import gaussian_kde; return xs, gaussian_kde(v)(xs)
    except Exception:
        h,e=np.histogram(v,bins=np.linspace(0,CAP,31),density=True); return (e[:-1]+e[1:])/2, h
fig, ax = plt.subplots(figsize=(8, 4.8))
for v,lab,col in [(y,"Observed LoS","#7f8c8d"),
                  (series["Extra Trees (final model)"],"Extra Trees predicted","#1f5f9e"),
                  (series["Senior physician"],"Senior physician estimate","#c0392b")]:
    xs,dn=_kde(v); ax.fill_between(xs,dn,alpha=.22,color=col); ax.plot(xs,dn,lw=2.2,color=col,label=lab)
ax.set_xlabel("ICU length of stay (days)"); ax.set_ylabel("Density"); ax.set_xlim(0,CAP)
ax.set_title("Prospective prediction distributions vs observed LoS (n = 193)", weight="bold", fontsize=12.5)
ax.text(0.97, 0.55, "model predictions\ncompressed to ~3–6 d", transform=ax.transAxes,
        fontsize=8.5, color="#1f5f9e", ha="right", va="top")
ax.legend(fontsize=9.5, framealpha=.95)
fig.tight_layout(); fig.savefig(str(CAN/"fig_prediction_distribution.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# Fig 3: bootstrap distribution of paired MAE difference (physician - ML)
dvals = np.array([ (np.abs(yp_ob[idx[b]]-y[idx[b]]).mean() - np.abs(np.clip(yp_ml[idx[b]],0,None)-y[idx[b]]).mean()) for b in range(B) ])
lo, hi = ci(dvals); pt = ae_ob.mean()-ae_ml.mean()
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.hist(dvals, bins=40, color="#b5d4f4", edgecolor="#185fa5")
ax.axvline(0, color="#444", lw=1.2, ls="--", label="No difference")
ax.axvline(pt, color="#d6604d", lw=2, label=f"Observed ΔMAE = {pt:.2f} d")
ax.axvspan(lo, hi, color="#d6604d", alpha=.12, label=f"95% CI [{lo:.2f}, {hi:.2f}]")
ax.set_xlabel("ΔMAE = MAE(physician) − MAE(Extra Trees), days"); ax.set_ylabel("Bootstrap frequency")
ax.set_title("Paired bootstrap: physician − model MAE difference (n = 193)", weight="bold", fontsize=12)
ax.legend(fontsize=9, framealpha=.5)
fig.tight_layout(); fig.savefig(str(CAN/"fig_bootstrap_diff_mae.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

print("\nGespeichert: prospective_bootstrap_ci.csv, prospective_paired_diff.csv, calibration_slopes.csv,")
print("            calibration_by_losgroup.csv, fig_calibration_pros.png, fig_prediction_distribution.png, fig_bootstrap_diff_mae.png")
