# -*- coding: utf-8 -*-
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung")
DPI = 300
RED="#d6604d"; BLUE="#2166ac"; PURP="#762a83"; GREY="#888780"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                     "axes.spines.right":False,"figure.dpi":DPI})

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# --- Panel A: Subgruppen-MAE (prospektiv) ---
groups = ["1–7 d\n(n=217)", ">7 d\n(n=70)", ">14 d\n(n=30)"]
oberarzt = [1.396, 7.740, 11.767]
mean     = [1.521, 11.246, 17.894]
p50      = [1.214, 11.868, 18.558]
x = np.arange(len(groups)); w = 0.26
ax[0].bar(x-w, oberarzt, w, label="Oberarzt", color=RED, edgecolor="white")
ax[0].bar(x,   mean,     w, label="ML Mean",  color=BLUE, edgecolor="white")
ax[0].bar(x+w, p50,      w, label="ML P50 (Median)", color=PURP, edgecolor="white")
for xi,(o,m,p) in enumerate(zip(oberarzt,mean,p50)):
    for dx,v in [(-w,o),(0,m),(w,p)]:
        ax[0].text(xi+dx, v+0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=GREY)
ax[0].set_xticks(x); ax[0].set_xticklabels(groups)
ax[0].set_ylabel("MAE (Tage)")
ax[0].set_title("Prospektive Genauigkeit nach LoS-Subgruppe", fontsize=11, weight="bold")
ax[0].legend(loc="upper left", framealpha=0.3, fontsize=9)
ax[0].annotate("P50 schlägt\nOberarzt", xy=(0+w,1.214), xytext=(0.45,5.0),
               fontsize=8.5, color=PURP, weight="bold",
               arrowprops=dict(arrowstyle="->", color=PURP, lw=1.2))

# --- Panel B: P80-Coverage ---
groups2 = ["gesamt\n(n=360)", "1–7 d", ">7 d", ">14 d"]
cov_arzt = [51.1, 52.1, 21.4, 6.7]
cov_p80  = [76.9, 93.5, 1.4, 0.0]
x2 = np.arange(len(groups2)); w2 = 0.36
ax[1].bar(x2-w2/2, cov_arzt, w2, label="Oberarzt (Punktschätzung)", color=RED, edgecolor="white")
ax[1].bar(x2+w2/2, cov_p80,  w2, label="ML P80-Quantil", color="#1a9850", edgecolor="white")
ax[1].axhline(80, color=GREY, lw=1.2, ls="--")
ax[1].text(3.35, 81.5, "Ziel 80%", fontsize=8, color=GREY, ha="right")
for xi,(a,p) in enumerate(zip(cov_arzt,cov_p80)):
    ax[1].text(xi-w2/2, a+1.5, f"{a:.0f}", ha="center", fontsize=7.5, color=GREY)
    ax[1].text(xi+w2/2, p+1.5, f"{p:.0f}", ha="center", fontsize=7.5, color=GREY)
ax[1].set_xticks(x2); ax[1].set_xticklabels(groups2)
ax[1].set_ylabel("Coverage: beob. LoS ≤ Vorhersage (%)")
ax[1].set_ylim(0, 105)
ax[1].set_title("P80-Coverage für Kapazitätsplanung", fontsize=11, weight="bold")
ax[1].legend(loc="upper right", framealpha=0.3, fontsize=9)

fig.suptitle("Quantilregression + OP-Features im prospektiven Oberarzt-Vergleich",
             y=1.01, fontsize=12.5, weight="bold")
fig.tight_layout()
p = OUT / "fig7_quantile_op_prospective.png"
fig.savefig(str(p), dpi=DPI, bbox_inches="tight")
print(f"Gespeichert: {p}")
