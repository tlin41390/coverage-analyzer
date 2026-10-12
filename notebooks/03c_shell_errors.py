from pathlib import Path 
import sys 
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent 
sys.path.insert(0, str(ROOT))
from src.load import KEYS, PROC 
from src.shells import shell_at 

DY, W = 10, 15 #thresholds the tuning picked

trk = pd.read_parquet(PROC/"tracking.parquet")
p = pd.read_parquet(PROC/"shells.parquet")
train = p[p.week <=14].copy() # analyze errors on train weeks only

# Which coverages does the detector get wrong?
train["ok"] = train.mid_shell == train.post_shell
print("Agreement by family at t=10:")
print(train.groupby("family").ok.mean().sort_values().map("{:.1%}".format), "\n")

# What does it measure instead? (row = true family, col = measured shell)
print("Measured shell at t=10 by family, row %:")
print((pd.crosstab(train.family, train.mid_shell, normalize="index") * 100).round(1), "\n")

# Does timing matter?
print("Agreement by measurement frame:")
index = train.set_index(KEYS)
for t in [3, 5, 10, 15, 20]:
    s = shell_at(trk, t, DY, W).reindex(index.index)
    print(f" t={t:>2}: {(s == index.post_shell).mean():.1%}")