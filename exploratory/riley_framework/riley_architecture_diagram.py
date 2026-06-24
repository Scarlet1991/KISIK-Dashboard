# -*- coding: utf-8 -*-
"""Architektur-/Datenfluss-Diagramm: welche Daten (welcher Filter) -> welche Funktion trainiert,
dann prospektiv kombiniert angewandt (arzt-gesteuerter Soft-Gate-Hybrid)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mp
from pathlib import Path
OUT=Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung\exploratory_riley")
fig,ax=plt.subplots(figsize=(13,9.6)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,100)
def box(cx,cy,w,h,lines,fc,ec,fs=9.0,tc="#222",lw=1.4):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.3,rounding_size=1.4",lw=lw,edgecolor=ec,facecolor=fc))
    ls=3.3; n=len(lines); top=cy+(n-1)*ls/2
    for i,(s,b) in enumerate(lines):
        ax.text(cx,top-i*ls,s,ha="center",va="center",fontsize=fs+(0.8 if b else 0),weight=("bold" if b else "normal"),color=tc)
def ar(x1,y1,x2,y2,c="#333",lw=1.6,style="-|>",ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=15,lw=lw,color=c,linestyle=ls))

ax.text(50,98,"ICU-LoS prediction — physician-gated soft-routing hybrid: data, filters & functions",fontsize=13.5,weight="bold",ha="center",color="#1f4e79")
ax.text(50,94.5,"leakage-free (OPS 8-98f removed) · first-24-hour features · AIN intensive-care units",fontsize=9.5,ha="center",color="#555",style="italic")

# ===== TRAINING zone (retrospective) =====
ax.add_patch(FancyBboxPatch((2,57),96,34,boxstyle="round,pad=0.4,rounding_size=2",lw=1.1,edgecolor="#9bbbe0",facecolor="#f3f7fc"))
ax.text(5.5,88.5,"TRAINING — retrospective development data",fontsize=10.5,weight="bold",ha="left",color="#0c447c")
box(50,83.5,82,6.2,[("Retrospective cohort · AIN units · ICU LoS > 1 day · 81 leak-free 24-h features · training n = 10,283",True)],"#eaf1fb","#185fa5",fs=9.2)
ax.text(50,77.4,"split the training set by observed LoS",fontsize=8.4,style="italic",color="#185fa5",ha="center")
ar(40,80.4,28,74.6); ar(60,80.4,72,74.6)
box(27,69.5,44,9.5,[("Filter:  1 < LoS ≤ 7 d   (n ≈ 7,993)",True),("SHORT expert  =  log1p-ExtraTrees",False),("+ linear recalibration",False)],"#dbe7f7","#185fa5",fs=8.8)
box(73,69.5,44,9.5,[("Filter:  LoS > 7 d   (n ≈ 2,290)",True),("LONG expert  =  Tweedie / log1p-ExtraTrees",False),("+ linear recalibration",False)],"#dbe7f7","#185fa5",fs=8.8)
ax.text(50,61.5,"recalibration coefficients (a, b) per expert via grouped cross-validation (out-of-fold), then frozen",fontsize=8.2,style="italic",color="#555",ha="center")

# ===== APPLICATION zone (prospective), left -> right =====
ax.add_patch(FancyBboxPatch((2,3.5),96,49,boxstyle="round,pad=0.4,rounding_size=2",lw=1.1,edgecolor="#e3a79a",facecolor="#fdf1ee"))
ax.text(5.5,50.2,"APPLICATION — prospective evaluation (all stays)",fontsize=10.5,weight="bold",ha="left",color="#922")
box(14,26,21,18,[("PROSPECTIVE",True),("stay",True),("",False),("n = 286",False),("first-24-h features",False),("+ senior estimate",False)],"#fdeee9","#c0392b",fs=8.6)
# middle column: applied functions
box(49,40,40,6.0,[("short_pred = SHORT_expert(features)",False)],"#eaf1fb","#185fa5",fs=8.8)
box(49,30,40,6.0,[("long_pred  = LONG_expert(features)",False)],"#eaf1fb","#185fa5",fs=8.8)
box(49,17.5,40,8.0,[("regime weight  w = σ((senior est − c)/s)",True),("c = 7 d, s = 1.0   (tuned by nested CV)",False)],"#fdf2e0","#d99316",fs=8.6)
# final
box(83,28,26,12.5,[("FINAL  per stay",True),("LoS =",False),("(1−w) · short_pred",False),("+  w · long_pred",False)],"#e7f5ec","#1b7f3b",fs=8.8)
# arrows prospective -> middle
ar(24.6,30,29,40,c="#c0392b"); ar(24.6,27,29,30.5,c="#c0392b"); ar(24.6,22,29,18.5,c="#c0392b")
# middle -> final
ar(69,40,70.5,31,c="#185fa5"); ar(69,30,70.5,29,c="#185fa5"); ar(69,17.5,70.5,25,c="#d99316")
# frozen experts feed the applied functions (dashed top-down)
ar(27,64.7,46,43.2,c="#185fa5",lw=1.4,ls=(0,(4,3))); ax.text(34,55,"frozen → applied",fontsize=7.6,style="italic",color="#185fa5",ha="center",rotation=-32)
ar(73,64.7,55,33.2,c="#185fa5",lw=1.4,ls=(0,(4,3))); ax.text(67,55,"frozen → applied",fontsize=7.6,style="italic",color="#185fa5",ha="center",rotation=38)

leg=[mp.Patch(fc="#dbe7f7",ec="#185fa5",label="trained on retrospective data, then frozen"),
     mp.Patch(fc="#fdf2e0",ec="#d99316",label="gate from senior-physician estimate (prospective)"),
     mp.Patch(fc="#e7f5ec",ec="#1b7f3b",label="combined prospective prediction")]
ax.legend(handles=leg,loc="lower center",bbox_to_anchor=(0.5,-0.01),ncol=3,fontsize=8.2,frameon=False)
fig.savefig(str(OUT/"fig_architecture.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Gespeichert: fig_architecture.png")
