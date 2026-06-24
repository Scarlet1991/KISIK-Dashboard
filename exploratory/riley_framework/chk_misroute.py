# -*- coding: utf-8 -*-
import sys, io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")
import pandas as pd, numpy as np
from pathlib import Path
PR=pd.read_parquet(Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung\canonical\alt_matrices_no_isopen\prospective_rebuilt_286.parquet"))
los=PR["__los__"].to_numpy(float); arzt=PR["__arzt__"].to_numpy(float)
for K in (5,7):
    print(f"\n=== Arzt-Gate K={K}: Fehlrouting bei KURZEN Stays (obs<=4 d) ===")
    for lab,lo,hi in [("1-2 d",1,2),("2-4 d",2,4)]:
        m=(los>lo)&(los<=hi); mis=m&(arzt>=K)
        print(f"  {lab}: n={int(m.sum())}, davon arzt>={K} (-> Lang-Experte): {int(mis.sum())} ({100*mis.sum()/max(m.sum(),1):.0f}%)")
    print(f"  -> diese Faelle bekommen die hohe Lang-Prognose statt der guten Kurz-Prognose")
