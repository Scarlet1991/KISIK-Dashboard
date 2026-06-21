# -*- coding: utf-8 -*-
"""
Teil B: Ist die OBERARZTPROGNOSE ein besserer Langlieger-DETEKTOR als die Modellvorhersage?
Hypothese: Die Arztprognose ist nicht komprimiert -> trennt das Regime (kurz/lang) besser als das
kompressionsbehaftete Modell. Falls ja, waere sie genau das Routing-Signal, das dem Modell fehlt.
Daten: canonical/metrics_prospective_fair24h_predictions.csv (los_obs, arzt, pred_ExtraTrees, ...), n=193.
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from scipy import stats
CAN=Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung")/"canonical"
d=pd.read_csv(CAN/"metrics_prospective_fair24h_predictions.csv",sep=";")
y=d["los_obs"].to_numpy(float); arzt=d["arzt"].to_numpy(float); et=d["pred_ExtraTrees"].to_numpy(float)
xgb=d["pred_XGBoost"].to_numpy(float) if "pred_XGBoost" in d else et
print(f"n={len(y)} | Spannweiten: arzt [{arzt.min():.1f},{arzt.max():.1f}] vs ExtraTrees [{et.min():.1f},{et.max():.1f}] (Kompression sichtbar)")
print(f"SD der Schaetzungen: arzt {arzt.std():.2f} vs ExtraTrees {et.std():.2f} (beobachtet {y.std():.2f})")
print(f"Spearman mit wahrer LoS: arzt rho={stats.spearmanr(arzt,y)[0]:.3f} | ExtraTrees rho={stats.spearmanr(et,y)[0]:.3f}\n")

def auc(score,yb):
    try: return roc_auc_score(yb,score)
    except Exception: return np.nan
rows=[]
for thr in [4,7,14]:
    yb=(y>thr).astype(int)
    if yb.sum()<5:
        print(f">{thr}d: nur {int(yb.sum())} Faelle - uebersprungen"); continue
    # kombiniertes Modell (logistische Regression, patientenunabhaengige 5-fold CV)
    Xc=np.column_stack([arzt,et])
    try:
        pc=cross_val_predict(LogisticRegression(max_iter=1000),Xc,yb,cv=5,method="predict_proba")[:,1]
        auc_comb=roc_auc_score(yb,pc)
    except Exception: auc_comb=np.nan
    rows.append({"Schwelle":f">{thr}d","n_pos":int(yb.sum()),"Praevalenz_%":round(100*yb.mean(),1),
                 "AUC_Oberarzt":round(auc(arzt,yb),3),"AUC_ExtraTrees":round(auc(et,yb),3),
                 "AUC_XGBoost":round(auc(xgb,yb),3),"AUC_kombiniert":round(auc_comb,3)})
res=pd.DataFrame(rows)
print("=== DISKRIMINATION: Langlieger erkennen (ROC-AUC) ===")
print(res.to_string(index=False))

# Operating point: Arzt-Schaetzung > 7 als Langlieger-Flag
print("\n=== Arzt-Flag 'Schaetzung > 7 d' als Langlieger-Detektor (wahr >7d) ===")
for flag_name,flag in [("arzt>7",arzt>7),("ExtraTrees>7",et>7),("arzt>5",arzt>5)]:
    yb=(y>7).astype(int); tp=int((flag&(yb==1)).sum()); fp=int((flag&(yb==0)).sum())
    fn=int((~flag&(yb==1)).sum()); tn=int((~flag&(yb==0)).sum())
    sens=tp/(tp+fn) if tp+fn else np.nan; spec=tn/(tn+fp) if tn+fp else np.nan; ppv=tp/(tp+fp) if tp+fp else np.nan
    print(f"  {flag_name:14s}: geflaggt={tp+fp:3d} | Sensitivitaet {sens*100:4.0f}% | Spezifitaet {spec*100:4.0f}% | PPV {ppv*100:4.0f}%")
res.to_csv(CAN.parent/"los_physician_detector.csv",sep=";",index=False)
print(f"\nGespeichert: los_physician_detector.csv")
print("\nLesart: hoehere AUC = besserer Regime-Detektor. Wenn Oberarzt >> ExtraTrees, liefert die Arztprognose")
print("genau das Trennsignal fuer Langlieger, das dem komprimierten Modell fehlt (vgl. Routing-Limit in 06).")
