# -*- coding: utf-8 -*-
"""Observed ICU-LoS distribution — separate full-width figures for the retrospective and
prospective cohorts, in English, with the MANUSCRIPT subgroups (1-2 / 2-4 / 4-7 / >7 days)."""
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
pp=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
pros=pp["__los__"].to_numpy(float)

# manuscript subgroups: 1-2, 2-4, 4-7, >7 days  (boundaries 2,4,7)
BANDS=[(1,2,"#e9eefb","#2f6fd0","1–2 d"),
       (2,4,"#e8f4e9","#2e9e4f","2–4 d"),
       (4,7,"#fdf0db","#d99316","4–7 d"),
       (7,None,"#fbeaea","#d23b3b",">7 d")]
GREY="#b3b3bd"; CURVE="#3a3a3a"; XMAX=40; BW=0.5

def build(data, title, fname, paccent):
    d=data[data<=XMAX]; n=len(data); nc=len(d)
    bins=np.arange(0,XMAX+BW,BW); counts,edges=np.histogram(d,bins=bins)
    fig,ax=plt.subplots(figsize=(13,6))
    for lo,hi,fc,lc,_ in BANDS:
        ax.axvspan(lo,(hi if hi else XMAX),color=fc,zorder=0)
        if hi: ax.axvline(hi,color=lc,ls="--",lw=1.4,zorder=2)
    ax.axvline(BANDS[0][0],color=BANDS[0][3],ls="--",lw=1.4,zorder=2)
    ax.bar(edges[:-1],counts,width=BW,align="edge",color=GREY,alpha=0.65,edgecolor="white",linewidth=0.3,zorder=3)
    xs=np.linspace(d.min(),XMAX,400); ys=gaussian_kde(d)(xs)*nc*BW
    ax.plot(xs,ys,color=CURVE,lw=2.3,zorder=4)
    ymax=counts.max()*1.16; ax.set_ylim(0,ymax); ax.set_xlim(0,XMAX)
    centers=[1.5,3.0,5.5,(7+XMAX)/2]
    for (lo,hi,fc,lc,lab),cx in zip(BANDS,centers):
        ax.text(cx,ymax*0.93,lab,fontsize=10,weight="bold",color=lc,ha="center",va="center",
                bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=lc,lw=1.2),zorder=5)
    ax.set_xlabel("Observed ICU length of stay (days)",fontsize=12,weight="bold")
    ax.set_ylabel("Number of ICU stays",fontsize=12,weight="bold")
    med=np.median(data)
    sub=" · ".join(f"{lab}: {pct:.0f}%" for (lo,hi,_,_,lab),pct in
        [(b,100*(((data>b[0])&(data<=b[1])).mean() if b[1] else (data>b[0]).mean())) for b in BANDS])
    ax.set_title(f"{title}",fontsize=14,weight="bold",pad=30)
    ax.text(0.5,1.035,f"n = {n:,} · median {med:.1f} d · share by subgroup — {sub}",transform=ax.transAxes,
            ha="center",fontsize=10,color="#555")
    leg=[Patch(fc=GREY,alpha=0.65,label="Histogram"),Line2D([0],[0],color=CURVE,lw=2.3,label="Density curve")]+\
        [Patch(fc=b[2],ec=b[3],label=b[4]) for b in BANDS]
    ax.legend(handles=leg,title="Legend",fontsize=9.5,title_fontsize=10,loc="upper right",framealpha=0.95)
    ax.text(0.995,-0.13,f"Axis truncated at {XMAX} d (max {data.max():.0f} d; {100*(data>XMAX).mean():.1f}% of stays > {XMAX} d)",
            transform=ax.transAxes,ha="right",fontsize=8.5,style="italic",color="#777")
    fig.tight_layout(); fig.savefig(str(NOISO/fname),dpi=300,bbox_inches="tight"); plt.close(fig)
    print(f"Gespeichert: {fname}  (n={n:,}, <= {XMAX}d: {nc:,})")

build(retro,"Observed ICU length-of-stay distribution — retrospective development cohort",
      "fig_los_distribution_retro.png", "#2f6fd0")
build(pros, "Observed ICU length-of-stay distribution — prospective evaluation cohort",
      "fig_los_distribution_pros.png", "#d23b3b")
