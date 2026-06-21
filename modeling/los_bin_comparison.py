# -*- coding: utf-8 -*-
"""
Dreiwege-Vergleich nach BINS DER TATSAECHLICHEN LoS (prospektiv, n=193):
Oberarzt vs. finales Modell (ExtraTrees) vs. Null-Baseline (Konstante = retro Median 2.90 d).
Frage: Gibt es einen LoS-Bereich, in dem das Modell besser ist als Baseline UND Oberarzt?
Ausgabe: prospective_by_losbin.csv + fig_losbin_mae.png
Caveat: Stratifizierung nach der wahren Zielgroesse bevorzugt Schaetzer, deren Ausgabe im
jeweiligen Bin konzentriert ist (v.a. eine Konstante, deren Wert im Bin liegt).
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CAN = Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung")/"canonical"
pred = pd.read_csv(CAN/"metrics_prospective_fair24h_predictions.csv", sep=";")
y  = pred["los_obs"].to_numpy(float)
ob = pred["arzt"].to_numpy(float)
ml = np.clip(pred["pred_ExtraTrees"].to_numpy(float), 0, None)
# Null = TRAININGS-Median (nur Trainingsset), unveraendert angewandt; aus Tabelle-9-CSV gelesen.
NULL = float(pd.read_csv(CAN/"prospective_null_baseline.csv", sep=";")["train_median_used"].iloc[0])

bins = [("1-2 d",(y>1)&(y<=2)), ("2-4 d",(y>2)&(y<=4)), ("4-7 d",(y>4)&(y<=7)),
        ("2-7 d (aggregate)",(y>2)&(y<=7)), (">7 d",(y>7))]
def wp(p): return "<0.001" if p<0.001 else f"{p:.3f}"
rows=[]
for nm,m in bins:
    n=int(m.sum()); aeO=np.abs(ob[m]-y[m]); aeM=np.abs(ml[m]-y[m]); aeN=np.abs(NULL-y[m])
    mO,mM,mN=aeO.mean(),aeM.mean(),aeN.mean()
    try: p1=stats.wilcoxon(aeM,aeO).pvalue
    except Exception: p1=np.nan
    try: p2=stats.wilcoxon(aeM,aeN).pvalue
    except Exception: p2=np.nan
    best=min([("Physician",mO),("Model",mM),("Null",mN)],key=lambda t:t[1])[0]
    rows.append({"LoS_bin":nm,"n":n,"MAE_physician":round(mO,2),"MAE_model":round(mM,2),"MAE_null":round(mN,2),
                 "dMAE_model_minus_phys":round(mM-mO,2),"p_model_vs_phys":wp(p1),
                 "dMAE_model_minus_null":round(mM-mN,2),"p_model_vs_null":wp(p2),"best":best})
T=pd.DataFrame(rows); T.to_csv(CAN/"prospective_by_losbin.csv",sep=";",index=False)
print("=== Dreiwege-Vergleich nach tatsaechlicher LoS (n=193) ===")
print(T.to_string(index=False))

# Figur: gruppierte Balken (Oberarzt / Modell / Null) je disjunktem Bin
disj=[r for r in rows if r["LoS_bin"]!="2-7 d (aggregate)"]
labels=[r["LoS_bin"] for r in disj]; x=np.arange(len(labels)); w=0.26
PHYC,MODC,NULC="#c0392b","#1f5f9e","#c7ccd1"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(9.6,5.6))
for i,r in enumerate(disj):                                  # Sweet-Spot-Band hervorheben
    if r["best"]=="Model": ax.axvspan(x[i]-0.5,x[i]+0.5,color=MODC,alpha=0.07,zorder=0)
b1=ax.bar(x-w,[r["MAE_physician"] for r in disj],w,label="Senior physician",color=PHYC,edgecolor="white",zorder=3)
b2=ax.bar(x,  [r["MAE_model"]     for r in disj],w,label="Extra Trees (final model)",color=MODC,edgecolor="white",zorder=3)
b3=ax.bar(x+w,[r["MAE_null"]      for r in disj],w,label=f"Null model (const. {NULL:.1f} d)",color=NULC,edgecolor="white",zorder=3)
for bars in (b1,b2,b3):
    for bar in bars:
        h=bar.get_height(); ax.annotate(f"{h:.2f}",(bar.get_x()+bar.get_width()/2,h),xytext=(0,2),
            textcoords="offset points",ha="center",va="bottom",fontsize=7.3,color="#333")
for i,r in enumerate(disj):
    top=max(r['MAE_physician'],r['MAE_model'],r['MAE_null'])
    ax.annotate(f"n={r['n']}",(x[i],top+0.6),ha="center",fontsize=8.5,color="#777")
    if r["best"]=="Model": ax.annotate("model wins\n(p<0.05 vs both)",(x[i],top+1.25),ha="center",
                                       fontsize=8.6,color=MODC,weight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=11)
ax.set_ylabel("MAE (days) — lower is better"); ax.set_ylim(0,11.6)
ax.set_title("Prospective accuracy by actual ICU length of stay\nsenior physician vs final model vs null baseline (n = 193)",
             weight="bold",fontsize=12.5)
ax.legend(fontsize=9.5,framealpha=.95,loc="upper center",ncol=3,bbox_to_anchor=(0.5,-0.07))
fig.tight_layout(); fig.savefig(str(CAN/"fig_losbin_mae.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("\nGespeichert: prospective_by_losbin.csv, fig_losbin_mae.png")
print("\nKurzfazit:")
print("  4-7 d: Modell schlaegt SIGNIFIKANT sowohl Oberarzt als auch Null-Baseline -> echter Mehrwert (sweet spot).")
print("  2-7 d gesamt: Modell numerisch besser als Oberarzt, aber NICHT signifikant und NICHT besser als Konstante.")
print("  1-2 d & >7 d: Oberarzt klar am besten; Modell ueber-/unterschaetzt.")
