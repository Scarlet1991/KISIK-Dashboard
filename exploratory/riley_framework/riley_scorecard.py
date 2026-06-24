# -*- coding: utf-8 -*-
"""Saubere Uebersichtsgrafik: Riley-Kombimodell vs Oberarzt (prospektiv, leckfrei).
Liest vsphys_continuous.csv + vsphys_longstay.csv -> fig_scorecard_vs_physician.png"""
import sys, io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
OUT=Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung\exploratory_riley")
C=pd.read_csv(OUT/"vsphys_continuous.csv",sep=";"); L=pd.read_csv(OUT/"vsphys_longstay.csv",sep=";")
MOD="#1f5f9e"; PHY="#c0392b"
m=C[C.estimator.str.startswith("Combined")].iloc[0]; p=C[C.estimator.str.startswith("Senior")].iloc[0]
def lrow(tgt,who): return L[(L.target==tgt)&(L.estimator.str.startswith(who))].iloc[0]

fig,ax=plt.subplots(1,3,figsize=(15.5,5.4))
def grouped(a,labels,mv,pv,title,ylab,fmt="%.2f",ymax=None,note=None):
    x=np.arange(len(labels)); w=0.38
    b1=a.bar(x-w/2,mv,w,label="Combined model",color=MOD); b2=a.bar(x+w/2,pv,w,label="Senior physician",color=PHY)
    a.bar_label(b1,fmt=fmt,fontsize=8.5,padding=2); a.bar_label(b2,fmt=fmt,fontsize=8.5,padding=2)
    a.set_xticks(x); a.set_xticklabels(labels,fontsize=10); a.set_ylabel(ylab); a.set_title(title,weight="bold",fontsize=11.5)
    if ymax: a.set_ylim(0,ymax)
    a.legend(fontsize=9,loc="upper right")
    if note: a.text(0.5,-0.16,note,transform=a.transAxes,ha="center",fontsize=8.5,color="#555")
# (A) continuous accuracy (lower better)
grouped(ax[0],["MAE","RMSE","MAE >7 d"],[m.MAE,m.RMSE,m.MAE_gt7],[p.MAE,p.RMSE,p.MAE_gt7],
        "(A) Continuous accuracy — lower is better","days",ymax=11,
        note=f"R²: model {m.R2:.2f} vs physician {p.R2:.2f}  ·  calib. slope {m.calib_slope:.2f} vs {p.calib_slope:.2f}")
# (B) AUROC (higher better)
m7,p7=lrow(">=7d","Model"),lrow(">=7d","Phys"); m10,p10=lrow(">=10d","Model"),lrow(">=10d","Phys")
grouped(ax[1],["LOS ≥ 7 d","LOS ≥ 10 d"],[m7.AUROC,m10.AUROC],[p7.AUROC,p10.AUROC],
        "(B) Long-stay discrimination — AUROC (higher better)","AUROC",fmt="%.3f",ymax=1.05)
ax[1].axhline(0.5,color="#999",ls=":",lw=1)
# (C) operating point >=7d
grouped(ax[2],["Sensitivity","Specificity","PPV"],[m7.sens,m7.spec,m7.PPV],[p7.sens,p7.spec,p7.PPV],
        "(C) Long-stay ≥7 d at operating point","proportion",ymax=1.05,
        note=f"model flags {int(m7.flagged)} pts (thr 0.30) · physician flags {int(p7.flagged)} (est ≥7 d) · {int(p7.obs_count)} truly ≥7 d")
fig.suptitle("Riley-informed combined approach vs senior physician — prospective (n=286, leak-free)\n"
             "the physician is superior on every axis: accuracy, calibration and long-stay discrimination",
             weight="bold",fontsize=13)
fig.tight_layout(rect=[0,0.02,1,0.95]); fig.savefig(str(OUT/"fig_scorecard_vs_physician.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Gespeichert: fig_scorecard_vs_physician.png")
