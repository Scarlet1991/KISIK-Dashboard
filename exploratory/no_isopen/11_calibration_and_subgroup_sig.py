# -*- coding: utf-8 -*-
"""
(1) Kalibrierungs-Grafik (no_isopen, n=286): finales Modell Extra Trees vs Oberarzt.
(2) Subgruppen-MAE-Grafik (no_isopen) MIT statistischer Ueberlegenheit des finalen
    Modells (Extra Trees) gegenueber dem Oberarzt pro LoS-Bin (paired bootstrap 95%-CI).

Liest: alt_matrices_no_isopen/prospective_rebuilt_286.parquet (obs, arzt, ExtraTrees-pred),
       metrics_subgroups_no_isopen.csv (Balken), superiority_vs_oberarzt.csv (Signifikanz).
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

AN=Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung"); CAN=AN/"canonical"
NOISO=AN/"exploratory_no_isopen"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

pp=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
obs=pp["__los__"].to_numpy(float); arzt=pp["__arzt__"].to_numpy(float)
et =np.clip(pp["__pred_ExtraTrees__"].to_numpy(float),0,None)
n=len(obs)

# ============================================================
# (1) KALIBRIERUNG — predicted (x) vs observed (y), Dezil-Mittel + 95% CI, Identitaet
# ============================================================
def calib_points(pred, q=10):
    df=pd.DataFrame({"p":pred,"o":obs})
    df["bin"]=pd.qcut(df["p"],q,duplicates="drop")
    g=df.groupby("bin",observed=True)
    mp=g["p"].mean().to_numpy(); mo=g["o"].mean().to_numpy()
    se=g["o"].std(ddof=1).to_numpy()/np.sqrt(g["o"].size().to_numpy())
    return mp,mo,1.96*se
def calib_fit(pred):  # observed ~ a + b*pred  (b=1,a=0 ideal)
    b,a=np.polyfit(pred,obs,1); return b,a

CAP=20
fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharex=True,sharey=True)
for ax,(pred,name,col) in zip(axes,[(et,"Extra Trees (final model)","#1f5f9e"),(arzt,"Senior physician","#c0392b")]):
    mp,mo,ci=calib_points(pred)
    b,a=calib_fit(pred)
    ax.plot([0,CAP],[0,CAP],"--",color="#888",lw=1.4,label="ideal (perfect calibration)")
    ax.errorbar(mp,mo,yerr=ci,fmt="o",color=col,ms=7,capsize=3,lw=1.4,label="observed mean per decile (95% CI)")
    xs=np.linspace(0,CAP,50); ax.plot(xs,a+b*xs,"-",color=col,lw=2,alpha=.8,label=f"fit: slope {b:.2f}, intercept {a:+.2f}")
    ax.set_xlim(0,CAP); ax.set_ylim(0,CAP); ax.set_aspect("equal","box")
    ax.set_xlabel("Predicted ICU LoS (days)"); ax.set_title(name,weight="bold",fontsize=12)
    ax.legend(fontsize=8.5,loc="upper left",framealpha=.5)
axes[0].set_ylabel("Observed ICU LoS (days)")
fig.suptitle(f"Calibration — prospective cohort (n={n}); deciles of predicted LoS vs observed mean\n"
             "points below the dashed line indicate over-prediction; open (censored) stays bias long-stay points downward",
             weight="bold",fontsize=11.5)
fig.tight_layout(); fig.savefig(str(NOISO/"fig_calibration_no_isopen.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Gespeichert: fig_calibration_no_isopen.png")
print(f"  Extra Trees: slope {calib_fit(et)[0]:.2f}, intercept {calib_fit(et)[1]:+.2f}")
print(f"  Oberarzt:    slope {calib_fit(arzt)[0]:.2f}, intercept {calib_fit(arzt)[1]:+.2f}")

# ============================================================
# (2) SUBGRUPPEN-MAE + Signifikanz Extra Trees vs Oberarzt
# ============================================================
SUB=pd.read_csv(NOISO/"metrics_subgroups_no_isopen.csv",sep=";")
SUP=pd.read_csv(NOISO/"superiority_vs_oberarzt.csv",sep=";")
SUPC=SUP[SUP["Kohorte"]=="no_isopen"].copy()
# Label-Mapping: SUB nutzt en-dash, SUPC nutzt hyphen
bins=["2–4 d","4–7 d",">7 d"]; bins_hy=["2-4 d","4-7 d",">7 d"]
ns={b:int(SUB[SUB["Subgroup"]==b]["n"].iloc[0]) for b in bins}

models=["Oberarzt","Ridge","RandomForest","ExtraTrees","XGBoost","Tweedie","Null"]
disp={"Oberarzt":"Senior physician","RandomForest":"Random Forest","ExtraTrees":"Extra Trees (final)","Null":"Null (mean)"}
col={"Oberarzt":"#c0392b","Ridge":"#7f8c8d","RandomForest":"#3498db","ExtraTrees":"#1a6ea3",
     "XGBoost":"#27ae60","Tweedie":"#8e44ad","Null":"#bdc3c7"}
def mae(m,b):
    r=SUB[(SUB["Modell"]==m)&(SUB["Subgroup"]==b)]; return float(r["MAE"].iloc[0]) if len(r) else 0.0

xi=np.arange(len(bins)); nm=len(models); tw=0.84; bw=tw/nm
off=np.linspace(-(tw-bw)/2,(tw-bw)/2,nm)
fig,ax=plt.subplots(figsize=(13.5,6.4))
for i,m in enumerate(models):
    vals=[mae(m,b) for b in bins]
    bars=ax.bar(xi+off[i],vals,bw,label=disp.get(m,m),color=col[m],edgecolor="white",lw=0.4)
    ax.bar_label(bars,fmt="%.1f",fontsize=6.6,padding=1)
ax.set_xticks(xi); ax.set_xticklabels([f"{b}\n(n={ns[b]})" for b in bins],fontsize=10.5)
ax.set_ylabel("MAE (days) — lower is better"); ymax=max(SUB["MAE"])*1.30; ax.set_ylim(0,ymax)
ax.legend(fontsize=8.8,ncol=7,loc="upper center",bbox_to_anchor=(0.5,-0.07))

# Signifikanz Extra Trees vs Oberarzt pro Bin (paired bootstrap 95%-CI; dMAE = MAE_Arzt - MAE_ET)
io_a=models.index("Oberarzt"); io_e=models.index("ExtraTrees")
def siginfo(b_hy):
    r=SUPC[(SUPC["Subgruppe"]==b_hy)&(SUPC["Modell"]=="ExtraTrees")]
    if not len(r): return None
    r=r.iloc[0]; lo,hi,d,p=float(r["CI_low"]),float(r["CI_high"]),float(r["dMAE"]),str(r["p_one_sided"])
    if lo>0:   return ("model",d,lo,hi,p,"#1b7f3b")    # Extra Trees signifikant besser
    if hi<0:   return ("phys", d,lo,hi,p,"#b03030")    # Oberarzt signifikant besser
    return     ("ns",   d,lo,hi,p,"#777777")
def stars(p):
    try: pv=0.0005 if "<" in p else float(p)
    except: pv=1.0
    return "***" if pv<0.001 else "**" if pv<0.01 else "*" if pv<0.05 else "n.s."

for k,(b,bh) in enumerate(zip(bins,bins_hy)):
    info=siginfo(bh)
    if info is None: continue
    kind,d,lo,hi,p,c=info
    xa=xi[k]+off[io_a]; xe=xi[k]+off[io_e]
    h=max(mae("Oberarzt",b),mae("ExtraTrees",b))
    ytop=h+ymax*0.085
    # Klammer zwischen Oberarzt- und ExtraTrees-Balken
    ax.plot([xa,xa,xe,xe],[ytop-ymax*0.02,ytop,ytop,ytop-ymax*0.02],lw=1.3,color=c)
    if kind=="model":
        lab=f"Extra Trees superior {stars(p)}\nΔMAE +{d:.2f} d  [{lo:.2f}, {hi:.2f}]"
    elif kind=="phys":
        lab=f"Physician superior {stars('<0.001' if hi<0 else p)}\nΔMAE {d:.2f} d  [{lo:.2f}, {hi:.2f}]"
    else:
        lab=f"n.s.\nΔMAE {d:+.2f} d  [{lo:.2f}, {hi:.2f}]"
    ax.text((xa+xe)/2,ytop+ymax*0.012,lab,ha="center",va="bottom",fontsize=8.2,weight="bold",color=c,linespacing=1.15)

ax.set_title(f"Prospective MAE by LoS subgroup (n={n}) with Extra-Trees-vs-physician superiority test\n"
             "bracket = final model vs senior physician (paired bootstrap 95% CI; ΔMAE = MAE_physician − MAE_model, >0 ⇒ model better)",
             weight="bold",fontsize=11.5,pad=12)
fig.tight_layout(); fig.savefig(str(NOISO/"fig_subgroup_mae_no_isopen.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Gespeichert (ueberschrieben): fig_subgroup_mae_no_isopen.png  (mit Signifikanz)")
for b,bh in zip(bins,bins_hy):
    info=siginfo(bh); print(f"  {b}: {info[0]:<5} dMAE={info[1]:+.2f} CI[{info[2]:.2f},{info[3]:.2f}] p={info[4]}")
