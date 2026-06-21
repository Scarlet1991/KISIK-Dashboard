# -*- coding: utf-8 -*-
"""
Gated-Ensemble / Routing-Experiment: Bringt "pro Region das beste ML-Tool" die beste Gesamtgenauigkeit?
WICHTIG: Routing-Variable muss zum Vorhersagezeitpunkt verfuegbar sein -> NICHT die wahre LoS.
Getestet werden:
  (1) Bestes Einzeltool (Baseline)
  (2) Routing per REGRESSOR-Vorhersage (1 und 2 gelernte Schwellen)
  (3) Routing per REGIME-KLASSIFIKATOR (XGBoost multiclass; "findet Grenzen")
  (4) Oracle: Routing per WAHRER LoS-Gruppe (Obergrenze, nicht erreichbar)
  (5) Per-Stay-Oracle (absolute Obergrenze)
Sauberer 3-Wege-Split (Train=Tools fitten, Dev=Routing lernen, Test=auswerten), patientengruppiert.
Retro AIN-Kohorte, 84 First-24h-Features, log1p/Tweedie/Gamma/Hazard/Quantil als Spezialtools.
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier
import time
BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
OUT=AN/"los_routing_experiment"; RS=42; TMAX=90
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
asql="('AIN','IZ32'),('AIN','IZ21'),('AIN','IZ31')"
con=duckdb.connect()
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
df["los_days"]=df["icu_duration_h"]/24.0
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns and not f.startswith(("lab_","vital_","proc_","zugang_"))]
cat=[c for c in present if c=="oebenekurz"]; numc=[c for c in present if c not in cat]
def design(frame):
    parts=[frame[numc].apply(pd.to_numeric,errors="coerce")]
    if cat: parts.append(pd.get_dummies(frame[cat].astype(str),prefix=cat).astype(float))
    X=pd.concat(parts,axis=1); X.columns=[str(c) for c in X.columns]; return X
y=df["los_days"].to_numpy(float); groups=df["pid"].fillna("unknown").astype(str).to_numpy()
Xall=design(df); COLS=Xall.columns.tolist()
# 3-Wege-Split: erst Test(20%) abtrennen, Rest -> Train(75%)/Dev(25%)
idx=np.arange(len(df))
rest,test=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(idx,y,groups))
g_rest=groups[rest]
tr_rel,dev_rel=next(GroupShuffleSplit(1,test_size=0.25,random_state=RS).split(rest,y[rest],g_rest))
train,dev=rest[tr_rel],rest[dev_rel]
def mat(ix): return Xall.iloc[ix].to_numpy(dtype=np.float64,na_value=np.nan)
Xtr,Xdev,Xte=mat(train),mat(dev),mat(test); ytr,ydev,yte=y[train],y[dev],y[test]
log(f"Split: Train {len(train):,} | Dev {len(dev):,} | Test {len(test):,}")

COMMON=dict(n_estimators=600,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,random_state=RS,n_jobs=-1)
def clip(p): return np.clip(p,0,None)
tools={}
log("Tools trainieren ...")
m=XGBRegressor(objective="reg:squarederror",**COMMON); m.fit(Xtr,np.log1p(ytr)); tools["Mean"]=("log",m)
m=XGBRegressor(objective="reg:tweedie",tweedie_variance_power=1.5,**COMMON); m.fit(Xtr,ytr); tools["Tweedie"]=("raw",m)
m=XGBRegressor(objective="reg:gamma",**COMMON); m.fit(Xtr,np.clip(ytr,0.01,None)); tools["Gamma"]=("raw",m)
for a,nm in [(0.5,"P50"),(0.8,"P80")]:
    m=XGBRegressor(objective="reg:quantileerror",quantile_alpha=a,**COMMON); m.fit(Xtr,np.log1p(ytr)); tools[nm]=("log",m)
def tpred(tag,X): k,m=tools[tag]; p=m.predict(X); return clip(np.expm1(p) if k=="log" else p)
# Hazard
T=np.clip(np.ceil(ytr).astype(int),1,TMAX); Xrep=np.repeat(Xtr,T,axis=0)
dd=np.concatenate([np.arange(1,c+1) for c in T]).astype(float); yh=(dd==np.repeat(T,T)).astype(int)
clf=XGBClassifier(objective="binary:logistic",n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,random_state=RS,n_jobs=-1,eval_metric="logloss")
clf.fit(np.column_stack([Xrep,dd]).astype(np.float32),yh)
def hazardE(X):
    n=X.shape[0]; S=np.ones(n); E=np.zeros(n)
    for t in range(1,TMAX+1):
        h=np.clip(clf.predict_proba(np.column_stack([X,np.full(n,t)]).astype(np.float32))[:,1],1e-6,1-1e-6); E+=S; S=S*(1-h)
    return E
tools["HazardE"]=("fn",None)
def predict_all(X):
    P={t:tpred(t,X) for t in tools if t!="HazardE"}; P["HazardE"]=hazardE(X); return P
log("Vorhersagen Dev/Test ..."); Pdev=predict_all(Xdev); Pte=predict_all(Xte)
TOOLS=list(Pte.keys())

# Regime-Klassifikator (findet Grenzen): Klassen nach LoS-Gruppe
edges=[1,2,4,7,np.inf]; labels=["1-2","2-4","4-7",">7"]
def binlab(v): return np.array(pd.cut(v,bins=edges,labels=labels,right=True).astype(str))
cl=XGBClassifier(objective="multi:softprob",num_class=4,n_estimators=500,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=RS,n_jobs=-1,eval_metric="mlogloss")
ytr_lab=binlab(ytr); lab2i={l:i for i,l in enumerate(labels)}
cl.fit(Xtr,np.array([lab2i[l] for l in ytr_lab]))
def clf_bin(X): return np.array([labels[i] for i in cl.predict(X)])

def MAE(mask,pred,yt): return float(np.mean(np.abs(pred[mask]-yt[mask]))) if mask.sum()>0 else np.nan
# ---- (1) bestes Einzeltool (auf Test) ----
single={t:mean_absolute_error(yte,Pte[t]) for t in TOOLS}
best_single=min(single,key=single.get)
# ---- (2) Routing per Regressor-Vorhersage (Router=Mean), Schwellen auf Dev gelernt ----
router_dev,router_te=Pdev["Mean"],Pte["Mean"]
def assign_regions_by_threshold(thr):
    edges_=[-np.inf]+list(thr)+[np.inf];
    reg_dev=np.digitize(router_dev,thr); reg_te=np.digitize(router_te,thr)
    pred=np.empty_like(yte); choice={}
    for r in range(len(thr)+1):
        md=reg_dev==r
        best=min(TOOLS,key=lambda t: MAE(md,Pdev[t],ydev)) if md.sum()>0 else best_single
        choice[r]=best; mt=reg_te==r; pred[mt]=Pte[best][mt]
    return mean_absolute_error(yte,pred),choice
# Grid-Suche 1 Schwelle
grid=np.round(np.arange(2.5,7.1,0.25),2)
one=[(t,)+ (assign_regions_by_threshold([t])) for t in grid]
best1=min(one,key=lambda z:z[1]);
two=[]
for i in range(len(grid)):
    for j in range(i+1,len(grid)):
        mae,ch=assign_regions_by_threshold([grid[i],grid[j]]); two.append(((grid[i],grid[j]),mae,ch))
best2=min(two,key=lambda z:z[1])
# ---- (3) Routing per Klassifikator-vorhergesagtem Bin (Zuordnung auf Dev) ----
pbin_dev,pbin_te=clf_bin(Xdev),clf_bin(Xte)
pred_cl=np.empty_like(yte); choice_cl={}
for b in labels:
    md=pbin_dev==b
    best=min(TOOLS,key=lambda t: MAE(md,Pdev[t],ydev)) if md.sum()>0 else best_single
    choice_cl[b]=best; mt=pbin_te==b; pred_cl[mt]=Pte[best][mt]
mae_cl=mean_absolute_error(yte,pred_cl)
# Klassifikator-Trennguete
acc=float((pbin_te==binlab(yte)).mean()); cm=confusion_matrix(binlab(yte),pbin_te,labels=labels)
# ---- (3b) Variante: erzwinge vorhergesagtes >7 -> P80 (Oracle-Praeferenz) ----
choice_force=dict(choice_cl); choice_force[">7"]="P80"
pred_force=np.empty_like(yte)
for b in labels: pred_force[pbin_te==b]=Pte[choice_force[b]][pbin_te==b]
mae_force=mean_absolute_error(yte,pred_force)
m7=pbin_te==">7"; prec7=float((binlab(yte)[m7]==">7").mean())
mae_haz7=MAE(m7,Pte["HazardE"],yte); mae_p807=MAE(m7,Pte["P80"],yte)
print(f"\nVorhergesagte >7-Gruppe: n={int(m7.sum())} | Praezision (wirklich >7) {prec7*100:.0f}% | "
      f"davon wahre Gruppen: "+", ".join(f"{b}:{int((binlab(yte)[m7]==b).sum())}" for b in labels))
print(f"  MAE auf der vorhergesagten >7-Gruppe: HazardE {mae_haz7:.3f} vs P80 {mae_p807:.3f}")
print(f"  -> Klassifikator-Routing mit >7->P80: Gesamt-Test-MAE {mae_force:.3f}  (vs >7->HazardE {mae_cl:.3f})")

# ---- (3c) P80 nur fuer SEHR sichere Langlieger: Wahrscheinlichkeits-Schwelle (Recall opfern, Praezision gewinnen) ----
pl_dev=cl.predict_proba(Xdev)[:,lab2i[">7"]]; pl_te=cl.predict_proba(Xte)[:,lab2i[">7"]]
base_dev=np.empty_like(ydev)
for b in labels: base_dev[pbin_dev==b]=Pdev[choice_cl[b]][pbin_dev==b]
best=(None,np.inf,None)
print("\nP80 nur fuer P(>7)>tau (tau auf Dev gewaehlt):")
for tau in np.round(np.arange(0.40,0.96,0.05),2):
    pdv=base_dev.copy(); s=pl_dev>tau; pdv[s]=Pdev["P80"][s]; m=mean_absolute_error(ydev,pdv)
    prc=float((binlab(ydev)[s]==">7").mean()) if s.sum()>0 else np.nan
    if tau in (0.5,0.7,0.9): print(f"  tau={tau}: Dev n_sel={int(s.sum())} prec={prc*100:.0f}% Dev-MAE {m:.3f}")
    if m<best[1]: best=(tau,m,None)
TAU=best[0]
pt=pred_cl.copy(); selte=pl_te>TAU; pt[selte]=Pte["P80"][selte]
mae_conf=mean_absolute_error(yte,pt)
prec=float((binlab(yte)[selte]==">7").mean()) if selte.sum()>0 else np.nan
mae_p80_sel=MAE(selte,Pte["P80"],yte); mae_base_sel=MAE(selte,pred_cl,yte)
print(f"  -> bestes tau={TAU}: Test ausgewaehlt n={int(selte.sum())}/{len(yte)} | Praezision {prec*100:.0f}% wirklich >7")
print(f"     auf dieser Teilmenge: Basis-Routing {mae_base_sel:.3f} vs P80 {mae_p80_sel:.3f} | Gesamt-Test-MAE {mae_conf:.3f}")
# ---- (4) Oracle: Routing per WAHRER Gruppe ----
tb=binlab(yte); pred_or=np.empty_like(yte); choice_or={}
for b in labels:
    m=tb==b; best=min(TOOLS,key=lambda t: MAE(m,Pte[t],yte)); choice_or[b]=best; pred_or[m]=Pte[best][m]
mae_or=mean_absolute_error(yte,pred_or)
# ---- (5) Per-Stay-Oracle ----
stack=np.vstack([Pte[t] for t in TOOLS]); per_stay=float(np.mean(np.min(np.abs(stack-yte),axis=0)))

print("\n=== EINZELTOOLS (Test-MAE) ===");
for t in sorted(single,key=single.get): print(f"  {t:9s} {single[t]:.3f}")
print(f"\nKlassifikator-Bin-Genauigkeit: {acc*100:.1f}%  (Confusion rows=wahr {labels})"); print(cm)
print("\n=== ROUTING-STRATEGIEN (Test-MAE) ===")
rows=[("Bestes Einzeltool",single[best_single],best_single),
      (f"Router Regressor, 1 Schwelle @{best1[0]}",best1[1],str(best1[2])),
      (f"Router Regressor, 2 Schwellen @{best2[0]}",best2[1],str(best2[2])),
      ("Router Klassifikator (pred. Bin)",mae_cl,str(choice_cl)),
      ("Router Klassifikator, >7->P80 erzwungen",mae_force,str(choice_force)),
      (f"Router + P80 nur fuer sehr sichere Langlieger (P(>7)>{TAU})",mae_conf,f"n_sel={int(selte.sum())}, prec={prec*100:.0f}%"),
      ("Oracle: wahre Gruppe (Obergrenze)",mae_or,str(choice_or)),
      ("Per-Stay-Oracle (absolute Grenze)",per_stay,"-")]
res=pd.DataFrame(rows,columns=["Strategie","Test_MAE","Tool_je_Region"])
res["Test_MAE"]=res["Test_MAE"].round(3)
print(res[["Strategie","Test_MAE"]].to_string(index=False))
print("\nTool-Zuordnung Klassifikator-Routing:",choice_cl)
print("Tool-Zuordnung Oracle (wahre Gruppe):",choice_or)
res.to_csv(f"{OUT}.csv",sep=";",index=False)
gain=single[best_single]-mae_cl
print(f"\nFazit: bestes Einzeltool {single[best_single]:.3f} | erreichbares Routing (Klassif.) {mae_cl:.3f} "
      f"(Gewinn {gain:+.3f} d) | Oracle wahre Gruppe {mae_or:.3f} | Per-Stay-Oracle {per_stay:.3f}")
log("fertig")
