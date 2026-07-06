# -*- coding: utf-8 -*-
"""(A) Bestes Hybridmodell bestaetigen (prospektiv n=286): Oberarzt vs Voll-3-Komp vs Long-only(c_hi7) vs Long-only(c_hi10,s2).
(B) Verbessert FiO2 das Hybridmodell? Das einzige ML-Element des Hybrids ist der LONG-Experte (retro >7d).
    Daher ehrlicher Test: GroupKFold(5)-OOF des LONG-Experten auf retro >7d, MIT vs OHNE FiO2.
    FiO2-Spalten: vital24_fio2_m_{mean,last,max,min,count} + Praesenz-Flag. Recal-invariante Spearman + roher MAE/R2.
    Gepaarter Bootstrap (B=5000) auf |Fehler| + Wilcoxon. Entscheidung: behalten nur wenn sig. besser.
Ausgabe: exploratory_riley/best_hybrid.csv , exploratory_riley/fio2_test.csv
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
from collections import Counter
import duckdb, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score
BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"; K=BASE/"kisik2"; OUT=AN/"exploratory_riley"
RETRO=K/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"; RS=42; B=5000; rng=np.random.default_rng(RS)
asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; bpet=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["ExtraTrees"]
con=duckdb.connect()
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
allcols=list(con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') LIMIT 0").df().columns)
present=[f for f in feat if f in allcols and not f.startswith(("lab_","vital_","proc_","zugang_")) and not f.startswith("proc24_8_98f")]
FIO2=[c for c in ["vital24_fio2_m_mean","vital24_fio2_m_last","vital24_fio2_m_max","vital24_fio2_m_min","vital24_fio2_m_count"] if c in allcols]
meta=["icu_duration_h","oebenekurz","wardshort","pid"]
selc=list(dict.fromkeys(meta+present+FIO2)); colstr=", ".join('"'+c+'"' for c in selc)
df=con.execute(f"SELECT {colstr} FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet"); los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float); N=len(los)
y=(df["icu_duration_h"]/24.0).values; groups=df["pid"].fillna("unknown").astype(str).values
cat=["oebenekurz"]
def Xf(frame,cols):
    X=frame.reindex(columns=cols).copy()
    for c in cols: X[c]=(X[c].astype(str) if c=="oebenekurz" else pd.to_numeric(X[c],errors="coerce"))
    return X
def split_train():
    Xtmp=Xf(df,present); tr,_=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(Xtmp,y,groups)); return tr
TR=split_train(); gtr=groups[TR]; ytr=y[TR]
def pre(cols):
    numc=[c for c in cols if c!="oebenekurz"]; cc=[c for c in cat if c in cols]
    parts=[("num",SimpleImputer(strategy="median"),numc)]
    if cc: parts.append(("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cc))
    return ColumnTransformer(parts)
def et(cols): return TransformedTargetRegressor(Pipeline([("pre",pre(cols)),("mdl",ExtraTreesRegressor(**bpet,random_state=RS,n_jobs=2))]),func=np.log1p,inverse_func=np.expm1)
def fit_recal(cols,mask):
    Xt=Xf(df,cols).iloc[TR].iloc[mask]; yt=ytr[mask]; gt=gtr[mask]
    m=et(cols); m.fit(Xt,yt); oof=np.clip(cross_val_predict(et(cols),Xt,yt,groups=gt,cv=GroupKFold(3),n_jobs=1),0,None)
    b,a=np.polyfit(oof,yt,1); return m,a,b
def applyP(m,a,b,cols):
    Xp=pd.DataFrame(index=PR.index)
    for c in cols: Xp[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
    return np.clip(a+b*np.clip(m.predict(Xp[cols]),0,None),0,None)

# ===== (A) bestes Hybridmodell (prospektiv) =====
ms=(ytr>1)&(ytr<=7); ml=ytr>7
Ms,as_,bs=fit_recal(present,ms.nonzero()[0]); SHORT=applyP(Ms,as_,bs,present)
Ml,al,bl=fit_recal(present,ml.nonzero()[0]); LONG =applyP(Ml,al,bl,present)
def gates(a,c_lo,s,c_hi=7.0):
    pl=1/(1+np.exp(-(a-c_hi)/s)); psh=1/(1+np.exp(-(c_lo-a)/s)); pm=np.clip(1-pl-psh,0,None); tot=pl+psh+pm
    return psh/tot,pm/tot,pl/tot
def blend_full(c_lo,s): psh,pm,pl=gates(arzt,c_lo,s); return np.clip(psh*SHORT+pm*arzt+pl*LONG,0,None)
def blend_long(c_hi,s): pl=1/(1+np.exp(-(arzt-c_hi)/s)); return np.clip((1-pl)*arzt+pl*LONG,0,None)
def nested_full():
    oof=np.full(N,np.nan); picks=[]
    for trn,tst in KFold(5,shuffle=True,random_state=RS).split(np.arange(N)):
        best=None
        for c_lo in [3,4,5]:
            for s in [1.0,1.5,2.0]:
                p=blend_full(c_lo,s); v=mean_absolute_error(los[trn],p[trn])
                if best is None or v<best[0]: best=(v,c_lo,s)
        _,c_lo,s=best; picks.append((c_lo,s)); p=blend_full(c_lo,s); oof[tst]=p[tst]
    return oof
HFULL=nested_full(); HL7=blend_long(7,1.0); HL10=blend_long(10,2.0)
SG=[("1-2 d",1,2),("2-4 d",2,4),("4-7 d",4,7)]
def met(p):
    p=np.clip(p,0,None); m=los>7; d={}
    for l,lo,hi in SG: mm=(los>lo)&(los<=hi); d[l]=round(float(np.abs(los[mm]-p[mm]).mean()),2)
    return dict(MAE=round(float(mean_absolute_error(los,p)),3),R2=round(float(r2_score(los,p)),3),
                slope=round(float(np.polyfit(p,los,1)[0]),3),gt7=round(float(np.abs(los[m]-p[m]).mean()),2),**d)
cands={"Oberarzt":arzt,"Voll 3-Komp":HFULL,"Long-only (c_hi7)":HL7,"Long-only (c_hi10,s2)":HL10}
rowsA=[{"Modell":k,**met(v)} for k,v in cands.items()]
def supr(a_pred,b_pred,mask):  # dMAE=MAE_b-MAE_a; >0 => a besser
    yt=los[mask]; ea=np.abs(yt-a_pred[mask]); eb=np.abs(yt-b_pred[mask]); n=len(yt)
    idx=rng.integers(0,n,size=(B,n)); dd=eb[idx].mean(1)-ea[idx].mean(1); lo,hi=np.percentile(dd,[2.5,97.5])
    try: p=float(wilcoxon(ea,eb).pvalue)
    except: p=float("nan")
    return round(float(ea.mean()-eb.mean()),3),round(float(lo),2),round(float(hi),2),round(p,4)
print("=== (A) Hybrid-Varianten prospektiv (n=%d) ==="%N)
print(pd.DataFrame(rowsA)[["Modell","MAE","R2","slope","gt7","1-2 d","2-4 d","4-7 d"]].to_string(index=False))
m7=los>7
print("\nVs Oberarzt (dMAE=MAE_Arzt-MAE_Hybrid; >0 = Hybrid besser):")
for k,v in cands.items():
    if k=="Oberarzt": continue
    d,lo,hi,p=supr(v,arzt,np.ones(N,bool)); d7,lo7,hi7,p7=supr(v,arzt,m7)
    print(f"  {k:<22} gesamt dMAE {d:+.3f}[{lo:+.2f},{hi:+.2f}]p={p:.3f} | >7d dMAE {d7:+.3f}[{lo7:+.2f},{hi7:+.2f}]p={p7:.4f}")
pd.DataFrame(rowsA).to_csv(OUT/"best_hybrid.csv",sep=";",index=False)

# ===== (B) FiO2-Test: LONG-Experte retro >7d, MIT vs OHNE FiO2 (GroupKFold(5)-OOF) =====
print("\n=== (B) FiO2-Test am LONG-Experten (retro >7d, GroupKFold(5)-OOF) ===")
print("FiO2-Spalten:",FIO2)
Xl_idx=ml.nonzero()[0]; yL=ytr[ml]; gL=gtr[ml]
fio_cov=float(np.isfinite(pd.to_numeric(df.iloc[TR].iloc[Xl_idx]["vital24_fio2_m_mean"],errors="coerce")).mean())
print("FiO2-Abdeckung im LONG-Trainingsset: %.1f%%  (n_long=%d)"%(100*fio_cov,len(yL)))
def oof_long(cols):
    X=Xf(df,cols).iloc[TR].iloc[Xl_idx]
    return np.clip(cross_val_predict(et(cols),X,yL,groups=gL,cv=GroupKFold(5),n_jobs=1),0,None)
base_cols=present
fio_cols=present+FIO2
oof_base=oof_long(base_cols); oof_fio=oof_long(fio_cols)
def recal(oof,yt): b,a=np.polyfit(oof,yt,1); return np.clip(a+b*oof,0,None)
rb=recal(oof_base,yL); rf=recal(oof_fio,yL)
def evalp(name,p):
    return dict(Variante=name,MAE=round(float(mean_absolute_error(yL,p)),3),R2=round(float(r2_score(yL,p)),3),
               Spearman=round(float(spearmanr(p,yL).correlation),3))
rowsB=[evalp("LONG ohne FiO2",rb),evalp("LONG mit FiO2",rf)]
print(pd.DataFrame(rowsB).to_string(index=False))
ea=np.abs(yL-rb); eb=np.abs(yL-rf); n=len(yL)  # ea=ohne, eb=mit
idx=rng.integers(0,n,size=(B,n)); dd=ea[idx].mean(1)-eb[idx].mean(1)  # >0 => FiO2 besser (kleinerer Fehler)
lo,hi=np.percentile(dd,[2.5,97.5]); pw=float(wilcoxon(ea,eb).pvalue)
dmae=float(ea.mean()-eb.mean())
print(f"\nDelta-MAE (ohne - mit) = {dmae:+.3f} d  95%CI [{lo:+.3f},{hi:+.3f}]  Wilcoxon p={pw:.4f}")
better = lo>0 and pw<0.05 and dmae>0
print("FiO2 verbessert LONG signifikant:" , better)
# auch: Praesenz-Flag allein (Leak-Check)
df2=df.copy(); df2["fio2_present"]=np.isfinite(pd.to_numeric(df["vital24_fio2_m_mean"],errors="coerce")).astype(float)
Xfl=Xf(df2,present+["fio2_present"]).iloc[TR].iloc[Xl_idx]
oof_flag=np.clip(cross_val_predict(et(present+["fio2_present"]),Xfl,yL,groups=gL,cv=GroupKFold(5),n_jobs=1),0,None)
rfl=recal(oof_flag,yL)
print("Nur Praesenz-Flag:", evalp("LONG + fio2_present",rfl))
verdict = "BEHALTEN" if better else "VERWERFEN"
print("\n>>> ENTSCHEIDUNG FiO2:", verdict)
print("    Grund:", ("retro >7d CV signifikant besser" if better else
       "kein signifikanter retro-CV-Vorteil; zudem prospektiv nicht verfuegbar (<50%% Abdeckung, nicht im Deployment-Stream)"))
pd.DataFrame(rowsB+[dict(Variante="dMAE_ohne_minus_mit",MAE=round(dmae,3),R2="",Spearman=""),
                    dict(Variante="CI_lo",MAE=round(lo,3)),dict(Variante="CI_hi",MAE=round(hi,3)),
                    dict(Variante="wilcoxon_p",MAE=round(pw,4)),dict(Variante="verdict",MAE=verdict),
                    dict(Variante="fio2_coverage_long",MAE=round(fio_cov,3))]).to_csv(OUT/"fio2_test.csv",sep=";",index=False)
print("\nGespeichert: best_hybrid.csv , fio2_test.csv")
