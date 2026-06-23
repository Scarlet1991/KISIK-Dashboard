# -*- coding: utf-8 -*-
"""Table 1 — patient/stay characteristics for the retrospective (development) and
prospective (evaluation) cohorts, rendered as a styled figure (LoS>2, AIN units)."""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; NOISO=AN/"exploratory_no_isopen"; CAN=AN/"canonical"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"
PROS =BASE/"kisik2"/"kisik2_prospektiv_ml_dataset.parquet"
SENIOR=AN/"los_senior_estimates_tagesausleitung_stay_level.csv"
asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"
con=duckdb.connect()

dr=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>2").df()
dp=con.execute(f"SELECT * FROM read_parquet('{PROS.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>2").df()
sen=pd.read_csv(SENIOR,sep=";"); dp["stay_id"]=dp["stay_id"].astype(str); sen["tages_stay_id"]=sen["tages_stay_id"].astype(str)
dp=dp.merge(sen[["tages_stay_id","best_senior_estimate_days"]],left_on="stay_id",right_on="tages_stay_id",how="inner")
dp["arzt"]=pd.to_numeric(dp["best_senior_estimate_days"],errors="coerce"); dp=dp.dropna(subset=["arzt"]).reset_index(drop=True)
for d in (dr,dp): d["los"]=d["icu_duration_h"]/24.0
print(f"Retro n={len(dr)} (pid {dr['pid'].nunique()}) | Prospektiv n={len(dp)}")

def num(s): return pd.to_numeric(s,errors="coerce")
def msd(s): s=num(s).dropna(); return f"{s.mean():.1f} ± {s.std():.1f}"
def medi(s): s=num(s).dropna(); q1,q3=np.percentile(s,[25,75]); return f"{np.median(s):.1f} [{q1:.1f}–{q3:.1f}]"
def npct(mask,n): k=int(mask.sum()); return f"{k} ({100*k/n:.1f}%)"
def score_msd(d,prefix):
    cols=[c for c in d.columns if c.startswith(prefix)]
    first=[c for c in cols if c.endswith("_first")] or [c for c in cols if c.endswith("_mean")]
    if not first: return None
    s=num(d[first[0]]).dropna()
    return f"{s.mean():.1f} ± {s.std():.1f}" if len(s)>10 else None

nR,nP=len(dr),len(dp)
R=[("section","Stays / patients","",""),
   ("row","ICU stays, n", f"{nR}", f"{nP}"),
   ("row","Unique patients, n", f"{dr['pid'].nunique()}", f"{dp['fallnr'].nunique() if 'fallnr' in dp else dp['stay_id'].nunique()}"),
   ("section","Age, years","",""),
   ("row","Mean ± SD", msd(dr['alter']), msd(dp['alter'])),
   ("row","Median [IQR]", medi(dr['alter']), medi(dp['alter'])),
   ("section","ICU care unit, n (%)","",""),
  ]
for u in ["IZ21","IZ31","IZ32"]:
    R.append(("row",f"  {u}", npct(dr['oebenekurz'].astype(str)==u,nR), npct(dp['oebenekurz'].astype(str)==u,nP)))
R+=[("section","ICU stay number, n (%)","",""),
    ("row","  First ICU stay", npct(num(dr['stay_nr'])==1,nR), npct(num(dp['stay_nr'])==1,nP)),
    ("row","  Repeat ICU stay", npct(num(dr['stay_nr'])>1,nR), npct(num(dp['stay_nr'])>1,nP)),
    ("section","Severity score, first 24 h (mean ± SD)","",""),
   ]
for lab,pref in [("SAPS II","score_saps_ii"),("SOFA","score_sofa"),("TISS-28","score_tiss_28")]:
    vr=score_msd(dr,pref); vp=score_msd(dp,pref)
    R.append(("row",f"  {lab}", vr or "—", vp or "— (not reconstructed)"))
R+=[("section","Length of stay, days","",""),
    ("row","Mean ± SD", msd(dr['los']), msd(dp['los'])),
    ("row","Median [IQR]", medi(dr['los']), medi(dp['los'])),
    ("section","LoS subgroup, n (%)","",""),
    ("row","  2–4 days", npct((dr['los']>2)&(dr['los']<=4),nR), npct((dp['los']>2)&(dp['los']<=4),nP)),
    ("row","  4–7 days", npct((dr['los']>4)&(dr['los']<=7),nR), npct((dp['los']>4)&(dp['los']<=7),nP)),
    ("row","  >7 days",  npct(dr['los']>7,nR), npct(dp['los']>7,nP)),
   ]
# reconstructed first-24h features for the prospective cohort (same definitions as retro)
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
def pros_prev(c): return npct(num(PR[c]).fillna(0)>0,nP) if c in PR.columns else "—"
# early ICU complex treatment (the suffix-coded 8-98f family; flagged as a leak — descriptive only)
f98r=[c for c in dr.columns if c.startswith("proc24_8_98f")]
f98p=[c for c in PR.columns if c.startswith("proc24_8_98f")]
anyR=(dr[f98r].apply(lambda s:num(s).fillna(0)>0).any(axis=1)) if f98r else pd.Series(False,index=dr.index)
anyP=(PR[f98p].apply(lambda s:num(s).fillna(0)>0).any(axis=1)) if f98p else pd.Series(False,index=PR.index)
R.append(("section","Early ICU treatment, first 24 h, n (%)","",""))
R.append(("row","  Complex intensive-care treatment (OPS 8-98f)", npct(anyR,nR), npct(anyP,nP)))
# top main diagnoses comparable across cohorts (restricted to reconstructed selected features)
diagcols=[c for c in dr.columns if c.startswith("diag_main_") and c in PR.columns]
prev=[(c,(num(dr[c]).fillna(0)>0).mean()) for c in diagcols]
top=[c for c,_ in sorted(prev,key=lambda x:-x[1])[:5]]
LAB={"diag_main_z99_1":"Ventilator dependence (Z99.1)","diag_main_g91_8":"Hydrocephalus, other (G91.8)",
     "diag_main_g91_0":"Communicating hydrocephalus (G91.0)","diag_main_j12_8":"Viral pneumonia (J12.8)",
     "diag_main_i61_0":"Intracerebral haemorrhage (I61.0)","diag_main_i60_1":"Subarachnoid haemorrhage (I60.1)"}
R.append(("section","Most frequent main diagnosis, first 24 h, n (%)","",""))
for c in top:
    lab=LAB.get(c, c.replace("diag_main_","ICD ").replace("_",".").upper())
    R.append(("row",f"  {lab}", npct(num(dr[c]).fillna(0)>0,nR), pros_prev(c)))
# prospective-specific
R+=[("section","Prospective evaluation specifics","",""),
    ("row","Senior-physician estimate, median [IQR] d", "—", medi(dp['arzt'])),
    ("row","Still in ICU at estimation (censored), n (%)", "—", npct(num(dp['is_open'])==1,nP)),
   ]

pd.DataFrame([r for r in R if r[0]=="row"],columns=["t","Characteristic","Retrospective","Prospective"]).drop(columns="t")\
  .to_csv(AN/"table1_characteristics.csv",sep=";",index=False)

# ---------------- render styled table ----------------
plt.rcParams.update({"font.family":"DejaVu Sans"})
rowh=0.42; H=len(R)*rowh+1.6
fig,ax=plt.subplots(figsize=(9.6,H*0.49)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,len(R)+3.6)
ax.text(0,len(R)+3.0,"Table 1.  Characteristics of the development and evaluation cohorts",
        fontsize=12.5,weight="bold",ha="left",color="#1f4e79")
xC,xR,xP=2,62,84
ax.text(xC,len(R)+1.2,"Characteristic",fontsize=9.5,weight="bold",ha="left",va="center")
ax.text(xR,len(R)+1.25,"Retrospective",fontsize=9.3,weight="bold",ha="center",va="center")
ax.text(xR,len(R)+0.75,f"(development, n = {nR:,})",fontsize=8.2,ha="center",va="center",color="#444")
ax.text(xP,len(R)+1.25,"Prospective",fontsize=9.3,weight="bold",ha="center",va="center")
ax.text(xP,len(R)+0.75,f"(evaluation, n = {nP})",fontsize=8.2,ha="center",va="center",color="#444")
ax.plot([0,100],[len(R)+0.4,len(R)+0.4],color="#1f4e79",lw=1.4)
for i,(typ,lab,vr,vp) in enumerate(R):
    yy=len(R)-1-i
    if typ=="section":
        ax.add_patch(plt.Rectangle((0,yy-0.16),100,rowh+0.12,facecolor="#dce6f2",edgecolor="none"))
        ax.text(xC,yy+0.04,lab,fontsize=9.2,weight="bold",ha="left",color="#0c447c",va="center")
    else:
        ax.text(xC,yy+0.04,lab,fontsize=8.8,ha="left",va="center",color="#222")
        ax.text(xR,yy+0.04,vr,fontsize=8.8,ha="center",va="center",color="#222")
        ax.text(xP,yy+0.04,vp,fontsize=8.8,ha="center",va="center",color="#222")
        ax.plot([0,100],[yy-0.18,yy-0.18],color="#e8e8e8",lw=0.5)
ax.text(0,-1.0,"SD, standard deviation; IQR, interquartile range; SAPS, Simplified Acute Physiology Score; "
        "SOFA, Sequential Organ Failure Assessment; TISS, Therapeutic Intervention Scoring System; "
        "AIN, anaesthesiology intensive-care units (IZ21/IZ31/IZ32). Cohort restricted to ICU stays > 2 days. "
        "Sex was not available in the data extract.",fontsize=6.6,ha="left",va="top",color="#666",wrap=True)
fig.savefig(str(NOISO/"fig_table1_characteristics.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print("Gespeichert: fig_table1_characteristics.png + table1_characteristics.csv")
