from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = (ROOT/"data"/"raw"/ "bdb2026"/"114239_nfl_competition_files_published_analytics_final")
PROC = ROOT/"data"/"processed"
FIELD_W = 160 / 3
KEYS = ["game_id", "play_id"]
DROP = ["player_birth_date", "player_height", "player_weight"]

#Coverage family and safety shell (number of deep safeties the coverage uses)
COVERAGE_MAP = {
    "COVER_0_MAN": ("Cover 0", 0),
    "COVER_1_MAN": ("Cover 1", 1),
    "COVER_2_MAN": ("2-Man", 2),
    "COVER_2_ZONE": ("Cover 2", 2),
    "COVER_3_ZONE": ("Cover 3", 1),
    "COVER_4_ZONE": ("Cover 4", 2),
    "COVER_6_ZONE": ("Cover 6", 2),
}

def load_inputs(weeks=range(1, 19)):
    dfs = []
    for w in weeks:
        df = pd.read_csv(RAW / "train"/ f"input_2023_w{w:02d}.csv")
        dfs.append(df.drop(columns=DROP, errors="ignore"))
        print(f"loaded weeks {w}")
    return pd.concat(dfs, ignore_index=True)

def normalize_direction(df):
    """Flip left-moving plays so the offense always moves toward increasing x."""
    df = df.copy()
    left = df["play_direction"] == "left"
    df.loc[left, "x"] = 120 - df.loc[left, "x"]
    df.loc[left, "y"] = FIELD_W - df.loc[left, "y"]
    for c in ["o", "dir"]:
        df.loc[left, c] = (df.loc[left, c] + 180) % 360
    df["ball_land_x"] = np.where(left, 120 - df["ball_land_x"], df["ball_land_x"])
    df["ball_land_y"] = np.where(left, FIELD_W - df["ball_land_y"], df["ball_land_y"])
    # Assumes absolute_yardline_number is in the same coordinate system as raw x.
    # The sanity report checks this.
    df["los_x"] = np.where(left, 120 - df["absolute_yardline_number"],
                            df["absolute_yardline_number"])
    return df

def add_relative_coords(df):
    df["t_rel"] = df["frame_id"] - df.groupby(KEYS)["frame_id"].transform("min")
    qb0 = df[(df["t_rel"] ==0) & (df["player_role"] == "Passer")]
    ball_y = qb0.groupby(KEYS)["y"].first().rename("ball_y").reset_index()
    df = df.merge(ball_y, on=KEYS, how="inner")
    df["x_rel"] = df["x"] - df["los_x"] #dept past the line of scrimmage
    df["y_rel"] = df["y"] - df["ball_y"] #width relative to the passer
    df["side"] = df["player_side"].str.lower().str[:3]  # "off" / "def"
    return df
    
def load_labels():
    sup = pd.read_csv(RAW/"supplementary_data.csv", low_memory=False)
    nullified = (sup["play_nullified_by_penalty"].astype(str).str.upper().isin(["Y","TRUE"]))
    sup = sup[~nullified & sup["team_coverage_type"].isin(COVERAGE_MAP)].copy()
    sup["family"] = sup["team_coverage_type"].map(lambda c: COVERAGE_MAP[c][0])
    sup["post_shell"] = sup["team_coverage_type"].map(lambda c: COVERAGE_MAP[c][1])
    sup["is_man"] = (sup["team_coverage_man_zone"] == "MAN_COVERAGE").astype(int)
    keep = KEYS + ["week", "family", "post_shell", "is_man", "team_coverage_type",
                   "possession_team", "defensive_team", "pass_result",
                   "expected_points_added", "yards_gained",
                   "down", "yards_to_go", "offense_formation",
                   "receiver_alignment", "defenders_in_the_box"]
    return sup[keep]

def sanity_report(trk, plays):
    players = trk.drop_duplicates(KEYS + ["nfl_id"])
    print(f"\nPlays with tracking + labels: {len(plays):,}")
    print("\nCoverage families:\n", plays["family"].value_counts())
    print("\nPlayer roles:\n", players["player_role"].value_counts())
    per_side = players.groupby(KEYS)["side"].value_counts().unstack(fill_value=0)
    print("\nPlayers per play by side:\n", per_side.describe().loc[["mean","min","max"]])
    print("\nDefensive positions: \n",
          players.loc[players["side"] == "def", "player_position"].value_counts())
    
    f0 = trk[trk["t_rel"] == 0]
    print(f"\nFrame 0 mean speed: {f0['s'].mean():.2f} yd/s "
        "(near 0 means tracking starts at the snap)")
    print(f"Defenders past LOS at frame 0: "
          f"{(f0.loc[f0['side'] == 'def', 'x_rel'] > 0).mean():.1%} (expect ~100%)")
    print(f"Offense at/behind LOS at frame 0: "
          f"{(f0.loc[f0['side'] == 'off', 'x_rel'] <=1).mean():.1%} (expect ~100%)")
    
def build():
    PROC.mkdir(parents=True, exist_ok=True)
    trk = add_relative_coords(normalize_direction(load_inputs()))
    plays = load_labels()
    trk = trk.merge(plays[KEYS], on=KEYS, how="inner")
    plays = plays.merge(trk[KEYS].drop_duplicates(), on=KEYS, how="inner")
    trk.to_parquet(PROC/"tracking.parquet", index=False)
    plays.to_parquet(PROC/"plays.parquet", index = False)
    sanity_report(trk, plays)
    return trk, plays

if __name__ == "__main__":
    build()