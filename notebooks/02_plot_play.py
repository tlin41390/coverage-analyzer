from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
trk = pd.read_parquet(ROOT / "data/processed/tracking.parquet")
shells = pd.read_parquet(ROOT / "data/processed/shells.parquet")

FRAMES = [0, 10]   # the snap, and the frame the detector measured
FAMILY = "Cover 3"

misses = shells[(shells.week <= 14) & (shells.mid_shell != shells.post_shell)
                & (shells.family == FAMILY)]
print(f"{len(misses)} {FAMILY} misses in train weeks")
sample = misses.sample(4, random_state=1).reset_index()

for i, row in sample.iterrows():
    for frame in FRAMES:
        f = trk[(trk.game_id == row.game_id) & (trk.play_id == row.play_id) & (trk.t_rel == frame)]
        fig, ax = plt.subplots(figsize=(9, 6))
        for side, color in [("off", "tab:blue"), ("def", "tab:red")]:
            s = f[f.side == side]
            ax.scatter(s.x_rel, s.y_rel, c=color, s=80, label=side)
            for _, p in s.iterrows():
                ax.annotate(p.player_position, (p.x_rel, p.y_rel), fontsize=7, ha="center", va="bottom")
        ax.axvline(0, color="gray", ls="--")
        ax.set(xlim=(-15, 30), ylim=(-30, 30), xlabel="yards past LOS", ylabel="yards from passer",
               title=(f"{row.family} ({'man' if row.is_man else 'zone'}) - frame {frame}\n"
                      f"label shell={row.post_shell}, measured at t=10: {row.mid_shell}, at snap: {row.pre_shell}"))
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        out = ROOT / "outputs" / f"miss_{FAMILY.replace(' ', '')}_{i}_f{frame}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
        print("saved", out)