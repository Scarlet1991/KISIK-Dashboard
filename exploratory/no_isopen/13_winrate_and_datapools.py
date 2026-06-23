# -*- coding: utf-8 -*-
"""(1) Win-rate-Grafik Oberarzt vs Extra Trees (no_isopen, n=286): diverging bars + Fehler-Scatter.
   (2) Datenpool-Uebersicht: retrospektiv (Entwicklung) vs prospektiv (Evaluierung + Oberarzt-Prognose)."""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

import duckdb
BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; NOISO=AN/"exploratory_no_isopen"
S=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))
pp=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
obs=pp["__los__"].to_numpy(float); arzt=pp["__arzt__"].to_numpy(float)
et=np.clip(pp["__pred_ExtraTrees__"].to_numpy(float),0,None); iso=pp["__is_open__"].to_numpy(int)
# retrospektive LoS fuer den LoS-Mix-Streifen
_asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"
retro_los=duckdb.connect().execute(f"SELECT icu_duration_h/24.0 los FROM read_parquet('{(BASE/'kisik2'/'kisik2_icu_ml_dataset_24h.parquet').as_posix()}') "
    f"WHERE (wardshort,oebenekurz) IN ({_asql}) AND icu_duration_h/24.0>1").df()["los"].to_numpy(float)
def _shares(d): return [100*(((d>1)&(d<=2)).mean()),100*(((d>2)&(d<=4)).mean()),100*(((d>4)&(d<=7)).mean()),100*((d>7).mean())]
ea=np.abs(arzt-obs); ee=np.abs(et-obs); TIE=0.25; diff=ea-ee
et_win=diff>TIE; oa_win=diff<-TIE; tie=~(et_win|oa_win)
RED="#c0392b"; BLUE="#185fa5"; GREY="#b4b2a9"

# ============ (1) WIN-RATE: diverging stacked bars + error scatter ============
subs=[("Overall",np.ones(len(obs),bool)),(">7 d",obs>7),("4–7 d",(obs>4)&(obs<=7)),
      ("2–4 d",(obs>2)&(obs<=4)),("1–2 d",(obs>=1)&(obs<=2))]
fig,(axL,axR)=plt.subplots(1,2,figsize=(13.5,5.6),gridspec_kw={"width_ratios":[1.15,1]})

# Panel A: diverging bars centred on the tie band
labels=[]; y=np.arange(len(subs))
for i,(lab,m) in enumerate(subs):
    k=m.sum(); po=100*oa_win[m].mean(); pt=100*tie[m].mean(); pe=100*et_win[m].mean()
    labels.append(f"{lab}\n(n={int(k)})")
    axL.barh(i,-po,color=RED,edgecolor="white"); axL.barh(i,-pt/2,left=0,color=GREY,edgecolor="white") if False else None
    axL.barh(i,pt,left=-pt/2,color=GREY,edgecolor="white")          # tie centred at 0
    axL.barh(i,-po,left=-pt/2,color=RED,edgecolor="white")          # physician to the left
    axL.barh(i, pe,left= pt/2,color=BLUE,edgecolor="white")         # Extra Trees to the right
    axL.text(-pt/2-po-1,i,f"{po:.0f}%",va="center",ha="right",fontsize=9,color=RED,weight="bold")
    axL.text( pt/2+pe+1,i,f"{pe:.0f}%",va="center",ha="left",fontsize=9,color=BLUE,weight="bold")
axL.axvline(0,color="#555",lw=0.8); axL.set_yticks(y); axL.set_yticklabels(labels,fontsize=9.5)
axL.set_xlim(-100,100); axL.set_xlabel("← physician closer        share of stays (%)        Extra Trees closer →")
axL.set_title("(A) Per-case winner by length-of-stay subgroup",weight="bold",fontsize=11)
axL.spines["left"].set_visible(False)
from matplotlib.patches import Patch
axL.legend(handles=[Patch(color=RED,label="Physician closer"),Patch(color=GREY,label="Tie (±0.25 d)"),
                    Patch(color=BLUE,label="Extra Trees closer")],fontsize=8.3,loc="lower center",
           bbox_to_anchor=(0.5,-0.30),ncol=3,frameon=False)

# Panel B: |error| scatter, colour by winner, diagonal
CAP=20
ax=axR
cols=np.where(et_win,BLUE,np.where(oa_win,RED,GREY))
ax.scatter(np.clip(ea,0,CAP),np.clip(ee,0,CAP),c=cols,s=26,alpha=.75,edgecolors="white",linewidths=.4)
ax.plot([0,CAP],[0,CAP],"--",color="#555",lw=1.3)
ax.fill_between([0,CAP],[0,CAP],CAP,color=BLUE,alpha=.05)   # ET better region (below diag)
ax.fill_between([0,CAP],0,[0,CAP],color=RED,alpha=.05)      # physician better region (above diag)
ax.text(CAP*0.96,CAP*0.06,"Extra Trees\nbetter",ha="right",va="bottom",fontsize=9,color=BLUE,weight="bold")
ax.text(CAP*0.06,CAP*0.96,"Physician\nbetter",ha="left",va="top",fontsize=9,color=RED,weight="bold")
ax.set_xlim(0,CAP); ax.set_ylim(0,CAP); ax.set_aspect("equal","box")
ax.set_xlabel("Senior-physician absolute error (days)"); ax.set_ylabel("Extra Trees absolute error (days)")
ax.set_title("(B) Per-stay absolute error (clipped at 20 d)",weight="bold",fontsize=11)
fig.suptitle(f"Per-case agreement, senior physician vs final model (Extra Trees), prospective cohort (n={len(obs)})\n"
             "overall the physician is closer in 62% of stays, but Extra Trees wins the 4–7 day band (65%)",
             weight="bold",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(str(NOISO/"fig_winrate_no_isopen.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Gespeichert: fig_winrate_no_isopen.png")

# ============ (2) DATENPOOL-UEBERSICHT ============
fig,ax=plt.subplots(figsize=(13,7.2)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,100)
def card(x,y,w,h,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.6,rounding_size=2.2",
                 linewidth=1.6,edgecolor=ec,facecolor=fc,mutation_aspect=1))
def txt(x,y,s,fs=10,w="normal",c="#222",ha="left",va="top",it=False):
    ax.text(x,y,s,fontsize=fs,weight=w,color=c,ha=ha,va=va,style=("italic" if it else "normal"))

ax.text(50,98,"KISIK ICU length-of-stay — data pools",fontsize=16,weight="bold",ha="center",color="#1f4e79")
ax.text(50,94,"three anaesthesiology-run intensive-care units (AIN: IZ21 / IZ31 / IZ32); target = ICU LoS in days",
        fontsize=10,ha="center",color="#555",style="italic")

# --- Retrospective pool (left) ---
card(3,40,44,49,"#eaf1fb","#185fa5")
txt(6,86,"RETROSPECTIVE  ·  model development",12.5,"bold","#0c447c")
txt(6,81.5,f"{S['n_stays']:,} ICU stays  —  {S['n_patients']:,} patients",11.5,"bold","#222")
txt(6,77.5,"routine clinical-information-system data\n(labs · vitals · procedures · access · diagnoses · admin)",9.5,c="#444")
txt(6,71,"Outcome (LoS, days):",10,"bold","#222")
txt(6,67.8,f"median {S['los_days_median']} d   ·   p90 {S['los_days_p90']:.1f} d   ·   max {S['los_days_max']:.0f} d",9.8,c="#444")
txt(6,63,"Predictors:",10,"bold","#222")
txt(6,59.8,f"{S['n_features_used_leakagefree']} features from the first 24 h after admission",9.8,c="#444")
txt(6,54.5,"Patient-grouped split:",10,"bold","#222")
txt(6,51.3,f"train {S['n_train']:,}   →   held-out test {S['n_test']:,}",9.8,c="#444")
txt(6,46,"used to: tune 5 models, select & freeze the\nfinal model (Extra Trees), interpret predictors",9.5,c="#185fa5",it=True)

# --- Prospective pool (right) ---
card(53,40,44,49,"#fdeee9","#c0392b")
txt(56,86,"PROSPECTIVE  ·  real-world evaluation",12.5,"bold","#922")
txt(56,81.5,f"{len(obs)} ICU stays with a matched\nsenior-physician LoS estimate",11.5,"bold","#222")
txt(56,75,f"{int((iso==0).sum())} discharged (final LoS known)",9.8,c="#444")
txt(56,71.8,f"  +  {int((iso==1).sum())} still in the ICU at estimation",9.8,c="#444")
txt(56,68.6,"      (censored lower-bound LoS)",9.2,c="#999",it=True)
txt(56,63.5,"Outcome (LoS, days):",10,"bold","#222")
txt(56,60.3,f"median {np.median(obs):.1f} d   ·   mean {obs.mean():.1f} d   ·   max {obs.max():.0f} d",9.8,c="#444")
txt(56,55.5,"Senior-physician prognosis:",10,"bold","#c0392b")
txt(56,52.3,f"median {np.median(arzt):.0f} d  ·  mean {arzt.mean():.1f} d   (the benchmark)",9.8,c="#444")
txt(56,47,"same 24 h features reconstructed identically;\nphysician vs model scored on the same outcomes",9.5,c="#c0392b",it=True)

# arrow: frozen model from retro -> prospective
ax.add_patch(FancyArrowPatch((47.3,52),(52.6,52),arrowstyle="-|>",mutation_scale=22,lw=2.2,color="#1f4e79"))
txt(49.9,55.6,"frozen\nmodel",8.6,"bold","#1f4e79",ha="center")

# --- Subgroup distribution strip (bottom) ---
txt(50,35,"Length-of-stay mix (share of stays)",11,"bold","#222",ha="center")
_sr=_shares(retro_los); _sp=_shares(obs)
bins=[(lab,round(_sr[i]),round(_sp[i])) for i,lab in enumerate(["1–2 d","2–4 d","4–7 d",">7 d"])]  # (label, retro%, pros%)
x0=14; bw=18; gap=4
for i,(lab,rp,ppc) in enumerate(bins):
    bx=x0+i*(bw+gap)
    ax.add_patch(FancyBboxPatch((bx,18),bw*rp/35,8,boxstyle="square,pad=0",fc="#85b7eb",ec="white",lw=1))
    ax.add_patch(FancyBboxPatch((bx,8),bw*ppc/35,8,boxstyle="square,pad=0",fc="#f0997b",ec="white",lw=1))
    txt(bx,28.5,lab,9.5,"bold","#222")
    txt(bx+bw*rp/35+0.6,22,f"{rp}%",8.6,c="#0c447c",va="center")
    txt(bx+bw*ppc/35+0.6,12,f"{ppc}%",8.6,c="#922",va="center")
txt(x0,3.5,"retrospective",9,"bold","#185fa5"); txt(x0+22,3.5,"prospective",9,"bold","#c0392b")
fig.savefig(str(NOISO/"fig_data_pools_overview.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Gespeichert: fig_data_pools_overview.png")
