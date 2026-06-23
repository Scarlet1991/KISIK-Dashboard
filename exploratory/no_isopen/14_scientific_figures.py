# -*- coding: utf-8 -*-
"""Zwei wissenschaftliche Grafiken (no_isopen):
   (A) Studienfluss-/Datenpool-Diagramm (TRIPOD participant flow, Item 20a)
   (B) Forest-Plot: Ueberlegenheit Extra Trees vs Oberarzt je LoS-Subgruppe (paired bootstrap 95%-CI)."""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","mathtext.default":"regular"})

AN=Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung"); CAN=AN/"canonical"; NOISO=AN/"exploratory_no_isopen"
S=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))
_pc=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
NP=len(_pc); ND=int((_pc["__is_open__"]==0).sum()); NO=int((_pc["__is_open__"]==1).sum())
MEDP=float(_pc["__los__"].median()); MEDA=float(_pc["__arzt__"].median())

# ============ (A) STUDY-FLOW / DATA-POOL DIAGRAM ============
fig,ax=plt.subplots(figsize=(11.5,9.6)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,100)
def box(cx,cy,w,h,lines,fc="#ffffff",ec="#333333",lw=1.3,fs=9.2,tc="#222"):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.3,rounding_size=1.4",
                 linewidth=lw,edgecolor=ec,facecolor=fc))
    ls=3.25; n=len(lines); top=cy+(n-1)*ls/2
    for i,(s,b) in enumerate(lines):
        ax.text(cx,top-i*ls,s,ha="center",va="center",fontsize=fs+(0.6 if b else 0),
                weight=("bold" if b else "normal"),color=tc)
def arrow(x1,y1,x2,y2,c="#333",lw=1.4):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=15,lw=lw,color=c))

ax.text(50,99,"Study flow and data pools",fontsize=15,weight="bold",ha="center",color="#1f4e79")
ax.text(50,95.8,"ICU length-of-stay prediction — three anaesthesiology-run units (AIN: IZ21 / IZ31 / IZ32)",
        fontsize=9.5,ha="center",color="#555",style="italic")

# source
box(50,90.5,90,6.5,[("Routine clinical-information-system records, anaesthesiology ICU department",True),
                    ("labs · vital signs · procedures (OPS) · vascular access · diagnoses (ICD-10) · admission data",False)],fc="#f4f4f2")
arrow(41,87.3,27,82.0); arrow(59,87.3,73,82.0)

# --- LEFT: retrospective development ---
ax.text(4,84.3,"RETROSPECTIVE — development",fontsize=10.5,weight="bold",ha="left",color="#0c447c")
box(27,76,42,9.5,[("Eligible ICU stays",True),("AIN units · ICU LoS > 2 days",False),
                  ("first-24h features computable",False)],fc="#eaf1fb",ec="#185fa5")
arrow(27,71.25,27,66.0)
box(27,60,42,9.5,[("Development cohort",True),(f"{S['n_stays']:,} stays · {S['n_patients']:,} patients",False),
                  (f"median LoS {S['los_days_median']} d · {S['n_features_used_leakagefree']} predictors",False)],fc="#eaf1fb",ec="#185fa5")
ax.text(27,53.6,"patient-grouped 80/20 split",fontsize=8.4,style="italic",color="#185fa5",ha="center",va="center")
arrow(24,52.0,16.5,47.7); arrow(30,52.0,37.5,47.7)
box(16.5,43,20,9,[("Training set",True),(f"{S['n_train']:,} stays",False),("5 models tuned (CV)",False)],fc="#dbe7f7",ec="#185fa5",fs=8.4)
box(37.5,43,20,9,[("Held-out test",True),(f"{S['n_test']:,} stays",False),("internal performance",False)],fc="#dbe7f7",ec="#185fa5",fs=8.4)
arrow(16.5,38.5,23.5,34.0); arrow(37.5,38.5,30.5,34.0)
box(27,30,42,6.8,[("Final model selected & frozen",True),("Extra Trees — lowest CV-MAE",False)],fc="#185fa5",ec="#0c447c",fs=9.2,tc="white")

# --- RIGHT: prospective evaluation ---
ax.text(96,84.3,"PROSPECTIVE — evaluation",fontsize=10.5,weight="bold",ha="right",color="#922")
box(73,76,42,9.5,[("Prospective ICU stays (live snapshots)",True),("AIN units · same eligibility",False),
                  ("matched to senior-physician estimate",False)],fc="#fdeee9",ec="#c0392b")
arrow(73,71.25,73,66.0)
box(73,60,42,9.5,[("Prospective cohort",True),(f"{NP} stays with senior-physician LoS estimate",False),
                  (f"median LoS {MEDP:.1f} d · physician median {MEDA:.0f} d",False)],fc="#fdeee9",ec="#c0392b")
arrow(70,55.25,62.5,47.7); arrow(76,55.25,83.5,47.7)
box(62.5,43,20,9,[("Discharged",True),(f"{ND} stays",False),("final LoS known",False)],fc="#fbe0d6",ec="#c0392b",fs=8.4)
box(83.5,43,20,9,[("Still in ICU",True),(f"{NO} stays",False),("censored bound",False)],fc="#f3ece6",ec="#c79a86",fs=8.4)
arrow(62.5,38.5,70,34.0); arrow(83.5,38.5,76,34.0)
box(73,30,42,6.8,[("Prospective evaluation set",True),(f"n = {NP} stays",False)],fc="#fdeee9",ec="#c0392b",fs=9.2)

# bottom: head-to-head (both arms feed in)
arrow(27,26.6,43,20.6); arrow(73,26.6,57,20.6)
box(50,14.5,72,9.5,[(f"Head-to-head prospective comparison (n = {NP})",True),
                    ("frozen model vs senior physician — features reconstructed identically",False),
                    ("MAE · R² · bias · calibration · paired bootstrap 95% CI + one-sided Wilcoxon",False)],
    fc="#f4f4f2",ec="#333",fs=8.9)
fig.savefig(str(NOISO/"fig_flow_diagram.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Gespeichert: fig_flow_diagram.png")

# ============ (B) FOREST PLOT — superiority Extra Trees vs physician ============
SUP=pd.read_csv(NOISO/"superiority_vs_oberarzt.csv",sep=";")
D=SUP[(SUP["Kohorte"]=="no_isopen")&(SUP["Modell"]=="ExtraTrees")].copy()
ordr=["overall","2-4 d","4-7 d",">7 d"]
disp={"overall":"Overall","2-4 d":"2–4 d","4-7 d":"4–7 d",">7 d":">7 d"}
D=D.set_index("Subgruppe").loc[ordr].reset_index()

fig,ax=plt.subplots(figsize=(9.2,5.2))
y=np.arange(len(D))[::-1]
for i,(_,r) in enumerate(D.iterrows()):
    yy=y[i]; lo,hi,d=float(r["CI_low"]),float(r["CI_high"]),float(r["dMAE"])
    c="#1b7f3b" if lo>0 else ("#c0392b" if hi<0 else "#777777")
    ax.plot([lo,hi],[yy,yy],color=c,lw=2.4,solid_capstyle="round")
    ax.plot([lo,lo],[yy-0.12,yy+0.12],color=c,lw=2.0); ax.plot([hi,hi],[yy-0.12,yy+0.12],color=c,lw=2.0)
    ax.scatter([d],[yy],s=95,color=c,zorder=5,edgecolors="white",linewidths=1.1)
    n=int(r["n"]); p=str(r["p_one_sided"])
    star=" *" if (lo>0 and (("<" in p) or float(p)<0.05)) else ""
    ax.text(hi+0.12,yy,f"{d:+.2f} [{lo:.2f}, {hi:.2f}]{star}",va="center",ha="left",fontsize=9,color=c,weight="bold")
    ax.text(-3.55,yy,f"{disp[r['Subgruppe']]}",va="center",ha="left",fontsize=10,weight="bold",color="#222")
    ax.text(-3.55,yy-0.28,f"n={n} · physician {r['MAE_Arzt']:.2f} vs ET {r['MAE_ML']:.2f} d",va="center",ha="left",fontsize=7.8,color="#777")
ax.axvline(0,color="#333",lw=1.0,ls="-")
ax.axvspan(-3.6,0,color="#c0392b",alpha=0.04); ax.axvspan(0,3.6,color="#1b7f3b",alpha=0.05)
ax.text(0.02,len(D)-0.55,"Extra Trees better →",fontsize=8.6,color="#1b7f3b",ha="left",weight="bold")
ax.text(-0.02,len(D)-0.55,"← physician better",fontsize=8.6,color="#c0392b",ha="right",weight="bold")
ax.set_yticks([]); ax.set_ylim(-0.8,len(D)-0.2); ax.set_xlim(-3.6,2.4)
ax.set_xlabel("ΔMAE = MAE(physician) − MAE(Extra Trees), days   (>0 ⇒ model more accurate)",fontsize=10)
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
ax.set_title("Superiority of the final model over the senior physician by length-of-stay subgroup\n"
             "point = ΔMAE, whiskers = paired bootstrap 95% CI (B=5000); * one-sided Wilcoxon p<0.05",
             weight="bold",fontsize=11)
fig.tight_layout(); fig.savefig(str(NOISO/"fig_forest_superiority.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Gespeichert: fig_forest_superiority.png")
