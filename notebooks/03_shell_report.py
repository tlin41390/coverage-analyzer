from pathlib import Path 
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.load import KEYS, PROC
from src.shells import shell_at, tune

T_POST = 10

trk = pd.read_parquet(PROC / "tracking.parquet")
plays = pd.read_parquet(PROC / "plays.parquet")

# Tune on weeks 1-14
train_plays = plays[plays.week <= 14]
train_trk = trk.merge(train_plays[KEYS], on=KEYS)
grid = tune(train_trk, train_plays, T_POST)
print("Top threshold settings (train weeks): \n", grid.head(5), "\n")
best = grid.iloc[0]
dy, w = best.deep_yds, best.width

# Measure shells at the snap and 1 second in, for every play
p = plays.set_index(KEYS)
PRE_DEEP = dy 
p["pre_shell"] = shell_at(trk, 0 , PRE_DEEP, w)
p["mid_shell"] = shell_at(trk, T_POST, dy, w)
p = p.reset_index()

# Validate on held-out weeks 15-18
test = p[p.week > 14]
print(f"Held-out agreement, measured shell at t={T_POST} vs label: "
      f"{(test.mid_shell == test.post_shell).mean():.1%}\n")

# Pre-snap look vs coverage actually run
print("Shell at snap (rows) vs label shels (cols), row %:")
print((pd.crosstab(p.pre_shell, p.post_shell, normalize="index") *100).round(1), "\n")
print("Shell at snap by coverage family, row %:")
print((pd.crosstab(p.family, p.pre_shell, normalize="index") * 100).round(1), "\n")

# Rotations: the shell at snap differs from the shell of the coverage run
p["rotated"] = p.pre_shell != p.post_shell
print(f"Rotation rate: {p.rotated.mean():.1%}")
print("\nMost common rotations:")
print(p[p.rotated].groupby(["pre_shell", "post_shell"]).size()
    .sort_values(ascending=False).head(6))
print("\nRotation rate by defense (top 10):")
print(p.groupby("defensive_team").rotated.mean().sort_values(ascending=False)
      .head(10).map("{:.1%}".format))

p.to_parquet(PROC / "shells.parquet", index=False)
print("\nsaved", PROC / "shells.parquet")