from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.load import KEYS, PROC 
from src.shells import CENTER 

WIDTH = 15 # use the width the tuning picked

trk = pd.read_parquet(PROC / "tracking.parquet")
d0 = trk[(trk.side == "def") & (trk.t_rel == 0)]
mid = d0[(d0.y - CENTER).abs() <=WIDTH]
second = mid.groupby(KEYS).x_rel.apply(lambda s: s.nlargest(2).iloc[-1] if len(s) >= 2 else None).dropna()

fig, ax = plt.subplots(figsize=(8 ,4))
ax.hist(second, bins=60, range= (0, 25))
ax.set(xlabel="depth of 2nd-deepest middle defender at snap (yds)", ylabel="plays")
out = ROOT / "outputs" / "snap_second_depth.png"
fig.savefig(out, dpi = 100, bbox_inches = "tight")
print("saved", out)