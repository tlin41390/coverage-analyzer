from pathlib import Path
import pandas as pd 

raw = Path("data/raw/bdb2026")
files = sorted(p.relative_to(raw) for p in raw.rglob("*.csv"))
print("Files:", *files, sep="\n ")

#Look for coverage labels in any non-tracking file
for p in raw.rglob("*.csv"):
    if "input" in p.name or "output" in p.name:
        continue
    df = pd.read_csv(p, nrows=2000)
    cov_cols = [c for c in df.columns if "cover" in c.lower() or "man" in c.lower()]
    print("f\n{p.name}: {df.shape[1]} columns")
    print(" columns:", df.columns.tolist())
    for c in cov_cols:
        print(f" {c}:", df[c].value_counts(dropna=False).head(15).to_dict())

#Inspect one week of input tracking
inp = next(raw.rglob("input_2023_w01.csv"))
trk = pd.read_csv(inp)
print("\nInput columns:", trk.columns.tolist())
print("Players per play:",
        trk.groupby(["game_id", "play_id"])["nfl_id"].nunique().describe())
print("Frames per play:",
      trk.groupby(["game_id", "play_id"])["frame_id"].max().describe())