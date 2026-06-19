# -*- coding: utf-8 -*-
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path

AN = Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung")
retro = pd.read_csv(AN/"los_tweedie_hazard_retro.csv", sep=";")
pros  = pd.read_csv(AN/"los_tweedie_hazard_prospektiv.csv", sep=";")
DPI=300

MODELS = ["log1p_Mean","Tweedie_1.3","Gamma","Hazard_E","Hazard_Median"]
LAB = {"log1p_Mean":"log1p (Ref.)","Tweedie_1.3":"Tweedie p=1.3","Gamma":"Gamma",
       "Hazard_E":"Hazard (E[LoS])","Hazard_Median":"Hazard (Median)","Oberarzt":"Oberarzt"}
COL = {"log1p_Mean":"#888780","Tweedie_1.3":"#ef9f27","Gamma":"#1a9850",
       "Hazard_E":"#762a83","Hazard_Median":"#1d9e75","Oberarzt":"#d6604d"}

def val(dfr, model, sg, metric):
    r = dfr[(dfr["Modell"]==model)&(dfr["Subgruppe"]==sg)]
    return float(r[metric].values[0]) if len(r) else np.nan

plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                     "axes.spines.right":False,"figure.dpi":DPI})
fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))

# ---- Panel A: Retro MAE nach Subgruppe -------------------------------------
sgs=["1-7d",">7d",">14d"]; x=np.arange(len(sgs)); w=0.16
for i,m in enumerate(MODELS):
    vals=[val(retro,m,sg,"MAE") for sg in sgs]
    b=ax[0,0].bar(x+(i-2)*w, vals, w, label=LAB[m], color=COL[m], edgecolor="white", linewidth=.4)
ax[0,0].set_xticks(x); ax[0,0].set_xticklabels(["1–7 d (n=2727)",">7 d (n=702)",">14 d (n=270)"])
ax[0,0].set_ylabel("MAE (Tage)")
ax[0,0].set_title("A  Retrospektiv: MAE nach LoS-Subgruppe", weight="bold", fontsize=11, loc="left")
ax[0,0].legend(fontsize=8, framealpha=.3, ncol=2)
ax[0,0].annotate("Tail-Methoden\nbesser", xy=(1-0*w,7.04), xytext=(1.3,10),
                 fontsize=8.5, color="#762a83", weight="bold",
                 arrowprops=dict(arrowstyle="->", color="#762a83", lw=1.1))

# ---- Panel B: Retro overall R2 + Bias --------------------------------------
r2=[val(retro,m,"gesamt","R2") for m in MODELS]
bias=[val(retro,m,"gesamt","Bias") for m in MODELS]
xb=np.arange(len(MODELS))
bars=ax[0,1].bar(xb, r2, .55, color=[COL[m] for m in MODELS], edgecolor="white")
ax[0,1].axhline(r2[0], color="#888780", ls="--", lw=1)
ax[0,1].text(len(MODELS)-1, r2[0]+.004, "log1p-Referenz", fontsize=7.5, color="#888780", ha="right")
for xi,(v,bi) in enumerate(zip(r2,bias)):
    ax[0,1].text(xi, v+.004, f"R²={v:.3f}", ha="center", fontsize=7.5)
    ax[0,1].text(xi, .01, f"Bias\n{bi:+.2f}", ha="center", fontsize=7.5, color="white", weight="bold")
ax[0,1].set_xticks(xb); ax[0,1].set_xticklabels([LAB[m] for m in MODELS], rotation=18, ha="right")
ax[0,1].set_ylabel("R² (gesamt)"); ax[0,1].set_ylim(0,.42)
ax[0,1].set_title("B  Retrospektiv: Gesamt-R² & Bias", weight="bold", fontsize=11, loc="left")

# ---- Panel C: Prospektiv MAE nach Subgruppe (vs Oberarzt) ------------------
pm=["Oberarzt","log1p_Mean","Tweedie_1.3","Hazard_E","Hazard_Median"]; w=0.16
for i,m in enumerate(pm):
    vals=[val(pros,m,sg,"MAE") for sg in sgs]
    ax[1,0].bar(x+(i-2)*w, vals, w, label=LAB[m], color=COL[m], edgecolor="white", linewidth=.4)
ax[1,0].set_xticks(x); ax[1,0].set_xticklabels(["1–7 d (n=217)",">7 d (n=70)",">14 d (n=30)"])
ax[1,0].set_ylabel("MAE (Tage)")
ax[1,0].set_title("C  Prospektiv: MAE vs. Oberarzt", weight="bold", fontsize=11, loc="left")
ax[1,0].legend(fontsize=8, framealpha=.3)
ax[1,0].annotate("Hazard-Median\nschlägt Oberarzt", xy=(0+2*w,1.249), xytext=(0.25,6.5),
                 fontsize=8.5, color="#1d9e75", weight="bold",
                 arrowprops=dict(arrowstyle="->", color="#1d9e75", lw=1.1))

# ---- Panel D: Bias (Unterschätzung) retro vs prospektiv -------------------
mods2=["log1p_Mean","Tweedie_1.3","Gamma","Hazard_E","Hazard_Median"]
br=[val(retro,m,"gesamt","Bias") for m in mods2]
bp=[val(pros,m,"gesamt","Bias") for m in mods2]
xb=np.arange(len(mods2)); w2=0.36
ax[1,1].bar(xb-w2/2, br, w2, label="retrospektiv", color="#b5d4f4", edgecolor="white")
ax[1,1].bar(xb+w2/2, bp, w2, label="prospektiv", color="#185fa5", edgecolor="white")
ax[1,1].axhline(0, color="black", lw=.8)
ax[1,1].axhline(val(pros,"Oberarzt","gesamt","Bias"), color="#d6604d", ls="--", lw=1.2)
ax[1,1].text(len(mods2)-1, val(pros,"Oberarzt","gesamt","Bias")-0.18, "Oberarzt (prosp.)",
             fontsize=7.5, color="#d6604d", ha="right")
ax[1,1].set_xticks(xb); ax[1,1].set_xticklabels([LAB[m] for m in mods2], rotation=18, ha="right")
ax[1,1].set_ylabel("Mittlerer Bias (Tage)")
ax[1,1].set_title("D  Systematischer Bias (← Unterschätzung)", weight="bold", fontsize=11, loc="left")
ax[1,1].legend(fontsize=8, framealpha=.3)

fig.suptitle("Tail-Methoden für ICU-LoS: Tweedie/Gamma & diskrete Hazard-Modellierung",
             y=1.005, fontsize=13.5, weight="bold")
fig.tight_layout()
p = AN/"fig8_tweedie_hazard.png"
fig.savefig(str(p), dpi=DPI, bbox_inches="tight")
print(f"Gespeichert: {p}")
