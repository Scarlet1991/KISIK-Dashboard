# -*- coding: utf-8 -*-
"""EXPLORATIV: Gesamt-Performance des Riley-Kombimodells (leckfrei) + Oberarzt auf DENSELBEN Rahmen
umgerechnet. Kontinuierlich (Tweedie+Rekalibrierung vs Arzt) und Langlieger-Klassifikation
(Modell-Wahrscheinlichkeit vs Arzt-Schaetzung als Score / Arzt>=K als Entscheidung). Mit Grafiken.
Prospektiv no_isopen (n=286). Ausgabe: Eigene Auswertung/exploratory_riley/
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score, brier_score_loss,
                             roc_curve, confusion_matrix, mean_squared_error)
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

BASE=Path(r"D:\Ausgangsdaten\KISIK Projekt"); AN=BASE/"Eigene Auswertung"; CAN=AN/"canonical"
OUT=AN/"exploratory_riley"; OUT.mkdir(parents=True,exist_ok=True)
RETRO=BASE/"kisik2"/"kisik2_icu_ml_dataset_24h.parquet"; FEAT=AN/"los_selected_features_ain_24h_compact.csv"
RS=42; asql="('AIN','IZ32'), ('AIN','IZ21'), ('AIN','IZ31')"; bptw=json.loads((CAN/"summary.json").read_text(encoding="utf-8"))["best_params"]["Tweedie"]

con=duckdb.connect()
df=con.execute(f"SELECT * FROM read_parquet('{RETRO.as_posix()}') WHERE (wardshort,oebenekurz) IN ({asql}) AND icu_duration_h/24.0>1").df()
y=(df["icu_duration_h"]/24.0).values; groups=df["pid"].fillna("unknown").astype(str).values
feat=pd.read_csv(FEAT,sep=";")["Feature"].tolist()
present=[f for f in feat if f in df.columns and not f.startswith(("lab_","vital_","proc_","zugang_")) and not f.startswith("proc24_8_98f")]
cat=[c for c in present if c=="oebenekurz"]; numc=[c for c in present if c not in cat]
def Xf(frame):
    X=frame.reindex(columns=present).copy()
    for c in present: X[c]=(X[c].astype(str) if c=="oebenekurz" else pd.to_numeric(X[c],errors="coerce"))
    return X
X=Xf(df); tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=RS).split(X,y,groups))
Xtr,ytr,gtr=X.iloc[tr],y[tr],groups[tr]
def pre(): return ColumnTransformer([("num",SimpleImputer(strategy="median"),numc),
    ("cat",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat)])
PR=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float)
Xp=pd.DataFrame(index=PR.index)
for c in present: Xp[c]=(PR[c].astype(str) if c=="oebenekurz" else (pd.to_numeric(PR[c],errors="coerce") if c in PR.columns else np.nan))
Xp=Xp[present]; gkf=GroupKFold(4)

# ===== kontinuierlich: Tweedie + eingefrorene Rekalibrierung =====
TW=Pipeline([("pre",pre()),("mdl",XGBRegressor(objective="reg:tweedie",**bptw,random_state=RS,n_jobs=-1,tree_method="hist"))]); TW.fit(Xtr,ytr)
oof=np.clip(cross_val_predict(TW,Xtr,ytr,groups=gtr,cv=gkf,n_jobs=-1),0,None); b,a=np.polyfit(oof,ytr,1)
model_cont=np.clip(a+b*np.clip(TW.predict(Xp),0,None),0,None)
def cont(obs,p):
    p=np.clip(p,0,None); sb,_=np.polyfit(p,obs,1); m=obs>7
    return dict(MAE=round(float(mean_absolute_error(obs,p)),3),RMSE=round(float(np.sqrt(mean_squared_error(obs,p))),3),
                R2=round(float(r2_score(obs,p)),3),calib_slope=round(float(sb),3),
                bias=round(float((p-obs).mean()),3),MAE_gt7=round(float(np.abs(obs[m]-p[m]).mean()),3))
print("="*78,"\nGESAMT-PERFORMANCE (prospektiv n=%d) — kontinuierlich\n"%len(los),"="*78)
cc=pd.DataFrame([{"estimator":"Combined model (Tweedie+recal)",**cont(los,model_cont)},
                 {"estimator":"Senior physician",**cont(los,arzt)}])
print(cc.to_string(index=False)); cc.to_csv(OUT/"vsphys_continuous.csv",sep=";",index=False)

# ===== Langlieger-Klassifikation: Modell-Wahrscheinlichkeit vs Arzt =====
def make_clf(): return Pipeline([("pre",pre()),("mdl",ExtraTreesClassifier(n_estimators=600,min_samples_leaf=5,max_features=0.5,random_state=RS,n_jobs=-1,class_weight="balanced"))])
def oppoint(yt,score,decision):
    tn,fp,fn,tp=confusion_matrix(yt,decision,labels=[0,1]).ravel()
    return dict(AUROC=round(float(roc_auc_score(yt,score)),3),flagged=int(decision.sum()),
                sens=round(tp/max(tp+fn,1),3),spec=round(tn/max(tn+fp,1),3),
                PPV=round(tp/max(tp+fp,1),3),NPV=round(tn/max(tn+fn,1),3))
rows=[]; roc={}
THR={7:0.30,10:0.30}   # operative Modell-Schwellen (siehe Schwellen-Tabelle)
for K in (7,10):
    ytrK=(ytr>=K).astype(int); ypK=(los>=K).astype(int)
    clf=make_clf(); clf.fit(Xtr,ytrK); pp=clf.predict_proba(Xp)[:,1]; roc[("model",K)]=(ypK,pp)
    rM=oppoint(ypK,pp,(pp>=THR[K]).astype(int)); rM.update(estimator=f"Model P(LOS>={K})",target=f">={K}d",
              op=f"thr={THR[K]}",Brier=round(float(brier_score_loss(ypK,pp)),3),exp_count=round(float(pp.sum()),1),obs_count=int(ypK.sum()))
    rows.append(rM)
    roc[("phys",K)]=(ypK,arzt)
    rP=oppoint(ypK,arzt,(arzt>=K).astype(int)); rP.update(estimator=f"Physician (est>={K})",target=f">={K}d",
              op=f"arzt>={K}",Brier=np.nan,exp_count=int((arzt>=K).sum()),obs_count=int(ypK.sum()))
    rows.append(rP)
cl=pd.DataFrame(rows)[["target","estimator","op","AUROC","flagged","sens","spec","PPV","NPV","Brier","exp_count","obs_count"]]
print("\n"+"="*78,"\nLanglieger-Klassifikation: Modell vs Oberarzt\n","="*78); print(cl.to_string(index=False))
cl.to_csv(OUT/"vsphys_longstay.csv",sep=";",index=False)

# ===== Grafiken =====
fig,ax=plt.subplots(2,2,figsize=(13.5,11))
CAP=25
def cv(p,q=8):
    d=pd.DataFrame({"p":np.clip(p,0,None),"o":los}); d["b"]=pd.qcut(d["p"],q,duplicates="drop"); g=d.groupby("b",observed=True)
    return g["p"].mean().to_numpy(),g["o"].mean().to_numpy()
ax[0,0].plot([0,CAP],[0,CAP],"--",color="#888",label="ideal")
for p,lab,c in [(model_cont,"Combined model","#1f5f9e"),(arzt,"Senior physician","#c0392b")]:
    mp,mo=cv(p); ax[0,0].plot(mp,mo,"o-",color=c,lw=1.9,ms=6,label=lab)
ax[0,0].set_xlim(0,CAP); ax[0,0].set_ylim(0,CAP); ax[0,0].set_aspect("equal","box")
ax[0,0].set_xlabel("Predicted LoS (days)"); ax[0,0].set_ylabel("Observed LoS (days)")
ax[0,0].set_title("(A) Continuous calibration — model vs physician",weight="bold",fontsize=11); ax[0,0].legend(fontsize=9,loc="upper left")
# ROC panels
for K,axi in [(7,ax[0,1]),(10,ax[1,0])]:
    for who,c in [("model","#1f5f9e"),("phys","#c0392b")]:
        yk,sc=roc[(who,K)]; fpr,tpr,_=roc_curve(yk,sc); auc=roc_auc_score(yk,sc)
        axi.plot(fpr,tpr,color=c,lw=2,label=f"{'Model' if who=='model' else 'Physician'} (AUROC {auc:.2f})")
    # Operating points
    ykm,ppm=roc[("model",K)]; dm=(ppm>=THR[K]).astype(int); tn,fp,fn,tp=confusion_matrix(ykm,dm,labels=[0,1]).ravel()
    axi.plot(fp/max(fp+tn,1),tp/max(tp+fn,1),"o",color="#1f5f9e",ms=10,mec="white",label=f"model @{THR[K]}")
    dp=(arzt>=K).astype(int); tn,fp,fn,tp=confusion_matrix(ykm,dp,labels=[0,1]).ravel()
    axi.plot(fp/max(fp+tn,1),tp/max(tp+fn,1),"s",color="#c0392b",ms=10,mec="white",label=f"physician ≥{K}d")
    axi.plot([0,1],[0,1],"--",color="#888"); axi.set_xlabel("1 − specificity"); axi.set_ylabel("sensitivity")
    axi.set_title(f"({'B' if K==7 else 'C'}) ROC LOS≥{K}d — model vs physician",weight="bold",fontsize=11); axi.legend(fontsize=8.5,loc="lower right")
# operating-point bars (>=7): sens/spec/PPV
m7=cl[(cl.target==">=7d")&(cl.estimator.str.startswith("Model"))].iloc[0]; p7=cl[(cl.target==">=7d")&(cl.estimator.str.startswith("Phys"))].iloc[0]
met=["sens","spec","PPV"]; xx=np.arange(3); w=0.36
ax[1,1].bar(xx-w/2,[m7[k] for k in met],w,label=f"Model @{THR[7]}",color="#1f5f9e")
ax[1,1].bar(xx+w/2,[p7[k] for k in met],w,label="Physician ≥7d",color="#c0392b")
for i,k in enumerate(met):
    ax[1,1].text(i-w/2,m7[k]+.01,f"{m7[k]:.2f}",ha="center",fontsize=8); ax[1,1].text(i+w/2,p7[k]+.01,f"{p7[k]:.2f}",ha="center",fontsize=8)
ax[1,1].set_xticks(xx); ax[1,1].set_xticklabels(["Sensitivity","Specificity","PPV"]); ax[1,1].set_ylim(0,1.05)
ax[1,1].set_title("(D) Long-stay (≥7d) operating point — model vs physician",weight="bold",fontsize=11); ax[1,1].legend(fontsize=9)
fig.suptitle(f"Combined Riley-informed approach vs senior physician (prospective, n={len(los)}, leak-free)",weight="bold",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.975]); fig.savefig(str(OUT/"fig_combined_vs_physician.png"),dpi=300,bbox_inches="tight"); plt.close(fig)
print(f"\nGespeichert: fig_combined_vs_physician.png + vsphys_continuous.csv + vsphys_longstay.csv")
