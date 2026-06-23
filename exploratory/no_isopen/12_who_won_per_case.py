# -*- coding: utf-8 -*-
"""Fall-fuer-Fall: in welchen Stays war der Oberarzt vs. Extra Trees naeher an der wahren LoS?
no_isopen-Kohorte (n=286). Gewinner = kleinerer |Vorhersage - Beobachtung|."""
import sys, io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd

AN=Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung"); CAN=AN/"canonical"
pp=pd.read_parquet(CAN/"alt_matrices_no_isopen"/"prospective_rebuilt_286.parquet")
obs=pp["__los__"].to_numpy(float); arzt=pp["__arzt__"].to_numpy(float)
et=np.clip(pp["__pred_ExtraTrees__"].to_numpy(float),0,None)
isopen=pp["__is_open__"].to_numpy(int)

ea=np.abs(arzt-obs); ee=np.abs(et-obs)
TIE=0.25  # |Differenz der Fehler| <= 0.25 d => "Gleichstand"
diff=ea-ee  # >0 => ET besser (kleinerer Fehler), <0 => Arzt besser
et_win=diff>TIE; oa_win=diff<-TIE; tie=~(et_win|oa_win)

def line(mask,label):
    k=int(mask.sum())
    return f"{label:<26} {k:>4}  ({100*k/len(obs):5.1f}%)"
print(f"=== FALL-FUER-FALL: Oberarzt vs. Extra Trees (n={len(obs)}, Gleichstand-Schwelle ±{TIE} d) ===")
print(line(oa_win,"Oberarzt naeher"))
print(line(et_win,"Extra Trees naeher"))
print(line(tie,"Gleichstand (±0.25 d)"))
print(f"\nMittlerer |Fehler|: Oberarzt {ea.mean():.2f} d | Extra Trees {ee.mean():.2f} d")
print(f"Median |Fehler|:    Oberarzt {np.median(ea):.2f} d | Extra Trees {np.median(ee):.2f} d")

# nach LoS-Subgruppe
print("\n=== Gewinnquote nach LoS-Subgruppe ===")
print(f"{'Subgruppe':<10}{'n':>5}{'Arzt %':>9}{'ET %':>8}{'Gleich %':>10}{'Arzt MAE':>10}{'ET MAE':>9}")
subs=[("1-2 d",(obs>=1)&(obs<=2)),("2-4 d",(obs>2)&(obs<=4)),("4-7 d",(obs>4)&(obs<=7)),(">7 d",obs>7)]
for lab,m in subs:
    k=int(m.sum())
    print(f"{lab:<10}{k:>5}{100*oa_win[m].mean():>8.0f}%{100*et_win[m].mean():>7.0f}%{100*tie[m].mean():>9.0f}%{ea[m].mean():>10.2f}{ee[m].mean():>9.2f}")

# nach is_open-Status
print("\n=== Gewinnquote nach is_open-Status ===")
for lab,m in [("is_open=0 (entlassen)",isopen==0),("is_open=1 (offen/zensiert)",isopen==1)]:
    k=int(m.sum())
    print(f"{lab:<28} n={k:>3}  Arzt {100*oa_win[m].mean():4.0f}% | ET {100*et_win[m].mean():4.0f}% | Gleich {100*tie[m].mean():4.0f}%")

# wo gewinnt ET am deutlichsten / wo verliert er am deutlichsten
df=pd.DataFrame({"stay":pp["__stay_id__"].astype(str),"obs":obs,"arzt":arzt,"ET":et,
                 "err_arzt":ea,"err_ET":ee,"vorteil_ET":diff,"is_open":isopen})
df.to_csv(AN/"exploratory_no_isopen"/"who_won_per_case.csv",sep=";",index=False)
print("\n--- 5 Stays, in denen Extra Trees am deutlichsten besser war (groesster Vorteil) ---")
print(df.sort_values("vorteil_ET",ascending=False).head(5)[["obs","arzt","ET","err_arzt","err_ET"]].round(1).to_string(index=False))
print("\n--- 5 Stays, in denen der Oberarzt am deutlichsten besser war ---")
print(df.sort_values("vorteil_ET").head(5)[["obs","arzt","ET","err_arzt","err_ET"]].round(1).to_string(index=False))
print(f"\nGespeichert: {AN/'exploratory_no_isopen'/'who_won_per_case.csv'}")
