import numpy as np
import pandas as pd
from src.load import KEYS, FIELD_W 

CENTER = FIELD_W / 2

def shell_at(trk, t, deep_yds=10.0, width=15.0):
    """Count deep middle-of field defenders (capped at 2) at frame t of each play.
    If a play ends before frame t, its last frame is used instead."""
    d = trk[trk["side"] == "def"]
    last = d.groupby(KEYS)["t_rel"].transform("max")
    f = d[d["t_rel"] == np.minimum(t, last)]
    deep = (f["x_rel"] >= deep_yds) & ((f["y"] - CENTER).abs() <= width)
    return deep.groupby([f["game_id"], f["play_id"]]).sum().clip(upper=2)

def tune(trk, plays, t_post, depths=(8,9,10,11,12,13), widths=(10,12,15,18)):
    """Grid-search thresholds that best match the labeled shell at t_post."""
    target = plays.set_index(KEYS)["post_shell"]
    rows = []
    for dy in depths:
        for w in widths:
            s = shell_at(trk, t_post, dy,w).reindex(target.index)
            rows.append({"deep_yds": dy, "width": w,
                        "agreement": (s == target).mean()})
    return pd.DataFrame(rows).sort_values("agreement", ascending=False)