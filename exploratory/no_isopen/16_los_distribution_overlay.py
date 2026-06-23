# -*- coding: utf-8 -*-
"""Observed ICU-LoS distribution — retrospective vs prospective in ONE figure,
normalised to % of stays (comparable despite very different n). English, manuscript subgroups."""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd, duckdb
from scipy.stats import gaussian_kde
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; NOISO=AN/"exploratory_no_isopen"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"
asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"
con=duckdb.connect()
retro=con.execute(f"SELECT icu_duration_h/24.0 los FROM read_parquet('{RETRO.as_posix()}') "
                  f"WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()["los"].to_numpy(float)
pros=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")["__los__"].to_numpy(float)

XMAX=40; BW=0.5
R_C="#185fa5"; P_C="#d9663b"   # retro blue, prospective orange
BOUND=[2,4,7]; BANDLAB=[(1.5,"1–2 d"),(3,"2–4 d"),(5.5,"4–7 d"),((7+XMAX)/2,">7 d")]

fig,ax=plt.subplots(figsize=(13,6.5))
# faint subgroup bands + boundaries
band_fc=["#eef3fb","#eef6ef","#fdf4e6","#fbeeee"]
edges_b=[1]+BOUND+[XMAX]
for i in range(4):
    ax.axvspan(edges_b[i],edges_b[i+1],color=band_fc[i],zorder=0)
for b in BOUND: ax.axvline(b,color="#999",ls="--",lw=1.1,zorder=1)

bins=np.arange(0,XMAX+BW,BW); ctr=(bins[:-1]+bins[1:])/2
def pct_hist(d):
    c,_=np.histogram(d[d<=XMAX],bins=bins); return c/len(d)*100
hr=pct_hist(retro); hp=pct_hist(pros)
ax.bar(bins[:-1],hr,width=BW,align="edge",color=R_C,alpha=0.30,edgecolor="none",zorder=2)
ax.bar(bins[:-1],hp,width=BW,align="edge",color=P_C,alpha=0.30,edgecolor="none",zorder=2)
xs=np.linspace(1,XMAX,400)
ax.plot(xs,gaussian_kde(retro[retro<=XMAX])(xs)*BW*100,color=R_C,lw=2.6,zorder=4)
ax.plot(xs,gaussian_kde(pros[pros<=XMAX])(xs)*BW*100,color=P_C,lw=2.6,zorder=4)

ymax=max(hr.max(),hp.max())*1.16; ax.set_ylim(0,ymax); ax.set_xlim(0,XMAX)
for cx,lab in BANDLAB:
    ax.text(cx,ymax*0.95,lab,fontsize=9.5,weight="bold",color="#555",ha="center",va="center",
            bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="#bbb",lw=1.0),zorder=5)
ax.set_xlabel("Observed ICU length of stay (days)",fontsize=12,weight="bold")
ax.set_ylabel("Share of stays (%, per 0.5-day bin)",fontsize=12,weight="bold")
ax.set_title("Observed ICU length-of-stay distribution — retrospective vs prospective cohort",fontsize=14,weight="bold",pad=14)

def shares(d): return [100*(((d>1)&(d<=2)).mean()),100*(((d>2)&(d<=4)).mean()),100*(((d>4)&(d<=7)).mean()),100*((d>7).mean())]
sr=shares(retro); sp=shares(pros)
leg=[Line2D([0],[0],color=R_C,lw=2.6,label=f"Retrospective — development (n={len(retro):,}, median {np.median(retro):.1f} d)"),
     Line2D([0],[0],color=P_C,lw=2.6,label=f"Prospective — evaluation (n={len(pros)}, median {np.median(pros):.1f} d)"),
     Patch(fc=R_C,alpha=0.30,label="histogram (retrospective)"),Patch(fc=P_C,alpha=0.30,label="histogram (prospective)")]
ax.legend(handles=leg,fontsize=9.5,loc="upper right",framealpha=0.95)
sub=("subgroup share  1–2 / 2–4 / 4–7 / >7 d:   "
     f"retro {sr[0]:.0f}/{sr[1]:.0f}/{sr[2]:.0f}/{sr[3]:.0f}%   ·   "
     f"prosp {sp[0]:.0f}/{sp[1]:.0f}/{sp[2]:.0f}/{sp[3]:.0f}%")
ax.text(0.5,1.005,sub,transform=ax.transAxes,ha="center",fontsize=9.5,color="#555")
ax.text(0.995,-0.13,f"Axis truncated at {XMAX} d (retro max {retro.max():.0f} d, prosp max {pros.max():.0f} d)",
        transform=ax.transAxes,ha="right",fontsize=8.5,style="italic",color="#777")
fig.tight_layout(); fig.savefig(str(NOISO/"fig_los_distribution_overlay.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Gespeichert: fig_los_distribution_overlay.png")
print(f"  retro shares {['%.0f'%s for s in sr]} | pros shares {['%.0f'%s for s in sp]}")
