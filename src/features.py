import numpy as np
import pandas as pd
from src.load import KEYS
from src.shells import CENTER, shell_at 

LB_POS = ["ILB","MLB","OLB","LB"]
SITUATION = ["down", "yards_to_go", "offense_formation",
             "receiver_alignment", "defenders_in_the_box"]

def at_frame(trk, t):
    """One frame per play: frame t, or the play's last frame if itends sooner."""
    last = trk.groupby(KEYS)["t_rel"].transform("max")
    return trk[trk["t_rel"] == np.minimum(t, last)]

def depth_geometry(f, tag):
    """Depth and lateral position of the three deepest defenders, plus
    the geometry between the top two."""
    d = f[f["side"] == "def"].copy()
    d["yc"] = d["y"] - CENTER
    d = d.sort_values(KEYS + ["x_rel"], ascending=[True, True, False])
    d["rank"] = d.groupby(KEYS).cumcount() 
    top = d[d["rank"] < 3].pivot_table(index=KEYS, columns="rank", values=["x_rel","yc"])
    top.columns = [f"{tag}_{'depth' if v == "x_rel" else 'yc'}{r}" for v, r in top.columns]
    top[f"{tag}_gap01"] = top[f"{tag}_depth0"] - top[f"{tag}_depth1"]
    top[f"{tag}_sep01"] = (top[f"{tag}_yc0"] - top[f"{tag}_yc1"]).abs()
    top[f"{tag}_split01"] = (np.sign(top[f"{tag}_yc0"]) != np.sign(top[f"{tag}_yc1"])).astype(int)
    top[f"{tag}_offcenter0"] = top[f"{tag}_yc0"].abs()
    return top 

def match_receivers(f):
    """Pair each defender with the nearest receiver at this frame."""
    d = f.loc[f["side"] == "def", KEYS + ["nfl_id","x_rel", "y", "player_position"]]
    o = f.loc[(f["side"] == "off") & (f["player_role"] != "Passer"),
            KEYS + ["nfl_id", "x_rel", "y"]]
    pairs = d.merge(o, on = KEYS, suffixes=("", "_r"))
    pairs["dist"] = np.hypot(pairs["x_rel"] - pairs["x_rel_r"], pairs["y"] - pairs["y_r"])
    best = pairs.loc[pairs.groupby(KEYS + ["nfl_id"])["dist"].idxmin()].copy()
    best["cushion"] = best["x_rel"] - best["x_rel_r"]
    # positive = defender is farther from field center than his receiver (outside leverage)
    best["leverage"] = (best["y"] - CENTER).abs() - (best["y_r"] - CENTER).abs()
    return best 

def defender_count(trk):
    f0 = at_frame(trk,0)
    return f0[f0["side"] == "def"].groupby(KEYS).size().rename("n_def")

def snap_features(trk, deep_yds = 10, width = 15):
    f0 = at_frame(trk,0)
    feats = depth_geometry(f0, "s")
    m = match_receivers(f0)
    cb = m[m["player_position"] == "CB"].groupby(KEYS).agg(
        cb_cushion_mean = ("cushion", "mean"),
        cb_cushion_min = ("cushion", "min"),
        cb_leverage_mean = ("leverage", "mean"))
    d0 = f0[f0["side"] == "def"]
    lb = d0[d0["player_position"].isin(LB_POS)].groupby(KEYS)["x_rel"].mean().rename("lb_depth")
    feats = feats.join([cb, lb], how = "left")
    feats["pre_shell"] = shell_at(trk, 0, deep_yds, width)
    return feats

def post_features(trk, t_post = 10, deep_yds = 10, width = 15):
    f0, f1 = at_frame(trk, 0), at_frame(trk, t_post)
    feats = depth_geometry(f1, "p")
    feats["midshell"] = shell_at(trk, t_post, deep_yds, width)
    
    # Rotation: movement of two deepest defenders at snap
    d0 = f0[f0["side"] == "def"].sort_values(KEYS + ["x_rel"], ascending = [True, True, False])
    top2 = d0.groupby(KEYS).head(2)[KEYS + ["nfl_id", "x_rel", "y"]]
    later = f1.loc[f1["side"] == "def", KEYS + ["nfl_id", "x_rel", "y"]]
    mv = top2.merge(later, on = KEYS + ["nfl_id"], suffixes = ("_0", "_1"))
    mv["d_depth"] = mv["x_rel_1"] - mv["x_rel_0"]
    mv["d_lat"] = (mv["y_1"] - mv["y_0"]).abs()
    feats = feats.join(mv.groupby(KEYS).agg(
        rot_depth_min=("d_depth", "min"),
        rot_depth_max=("d_depth", "max"),
        rot_lat_max=("d_lat", "max")), how = "left")
    
    # Man vs zone: does each defender move with the receiver he lined up on?
    m = match_receivers(f0)[KEYS + ["nfl_id","nfl_id_r"]]
    w = trk[trk["t_rel"].between(1, t_post)].copy()
    rad = np.deg2rad(w["dir"])
    w["vx"], w["vy"] = w["s"] * np.sin(rad), w["s"] * np.cos(rad)
    v = w[KEYS + ["nfl_id", "t_rel", "vx", "vy", "x_rel", "y"]]
    v_r = v.rename(columns={"nfl_id": "nfl_id_r", "vx": "vx_r","vy": "vy_r", "x_rel":"x_rel_r","y": "y_r"})
    pair = (m.merge(v, on = KEYS + ["nfl_id"]).merge(v_r, on = KEYS + ["nfl_id_r","t_rel"]).sort_values("t_rel"))
    dot = pair["vx"] * pair["vx_r"] + pair["vy"] * pair["vy_r"]
    norm = np.hypot(pair["vx"], pair["vy"]) * np.hypot(pair["vx_r"], pair["vy_r"])
    pair["cos"] = dot / (norm + 1e-6)             # 1 = moving the same direction
    pair["sep"] = np.hypot(pair["x_rel"] - pair["x_rel_r"], pair["y"] - pair["y_r"])
    per_def = pair.groupby(KEYS + ["nfl_id"]).agg(cos=("cos", "mean"), sep_end = ("sep", "last"))
    per_play = per_def.groupby(KEYS).agg(
        mirror_mean = ("cos", "mean"), mirror_max=("cos", "max"),
        sep_end_mean = ("sep_end", "mean"), sep_end_min = ("sep_end", "min"))
    return feats.join(per_play, how = "left")

def build_features(trk, plays):
    out = plays.set_index(KEYS).join(
        [snap_features(trk), post_features(trk), defender_count(trk)], how = "left")
    return out.reset_index()
    