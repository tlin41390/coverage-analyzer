# Coverage Analyzer

Identifies NFL pass coverages from player tracking data. Given where the defenders line up at the snap and how they move in the first second of the play, it measures the safety shell, detects rotations (disguises), and predicts the coverage the defense is running.

The project uses the [NFL Big Data Bowl 2026 (Analytics)](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics) dataset: 2023 regular-season tracking for weeks 1–18, plus per-play coverage labels.

## Status

| Stage | State |
| --- | --- |
| Data loading, normalisation and label cleaning | Working |
| Rule-based safety shell detector and rotation analysis | Working |
| Feature engineering (snap alignment and first-second movement) | Working |
| LightGBM baseline classifiers | Working |
| Trained model saving, disguise scoring, app, video (CV) pipeline | Not started (empty placeholder files) |

There is no single entry point yet. The program is a data build step followed by a set of analysis scripts that you run in order.

## What it does so far

### 1. Builds a clean dataset (`src/load.py`)

- Reads the 18 weekly `input_2023_wXX.csv` tracking files and `supplementary_data.csv`.
- Flips plays moving left so the offense always moves toward increasing `x`. Orientation, direction, ball landing spot and line of scrimmage are flipped to match.
- Adds coordinates relative to the play:
  - `t_rel`: frames since the first tracked frame, which is treated as the snap. Tracking is 10 frames per second, so `t_rel = 10` is one second in.
  - `x_rel`: yards past the line of scrimmage.
  - `y_rel`: yards left or right of the passer.
  - `side`: `off` or `def`.
- Keeps plays that have one of seven coverage labels and were not nullified by a penalty, then maps each label to a coverage family and the number of deep safeties that coverage uses:

  | Label | Family | Deep safeties (`post_shell`) |
  | --- | --- | --- |
  | `COVER_0_MAN` | Cover 0 | 0 |
  | `COVER_1_MAN` | Cover 1 | 1 |
  | `COVER_2_MAN` | 2-Man | 2 |
  | `COVER_2_ZONE` | Cover 2 | 2 |
  | `COVER_3_ZONE` | Cover 3 | 1 |
  | `COVER_4_ZONE` | Cover 4 | 2 |
  | `COVER_6_ZONE` | Cover 6 | 2 |

- Writes `data/processed/tracking.parquet` (about 4.9 million rows) and `data/processed/plays.parquet` (14,064 plays), then prints a sanity report: plays per family, players per side, and checks that frame 0 really is the snap.

### 2. Measures the safety shell (`src/shells.py`)

`shell_at(trk, t)` counts the defenders who are deep (at least `deep_yds` past the line of scrimmage) and in the middle of the field (within `width` yards of the field's centre line) at frame `t`. The count is capped at 2, so the result is a 0-, 1- or 2-high shell. If a play ends before frame `t`, its last frame is used.

`tune()` grid-searches the depth and width thresholds for the best match against the labelled shell. The thresholds it picked are 10 yards deep and 15 yards wide.

Comparing the shell at the snap with the shell of the coverage actually run gives a rotation flag: the defense showed one look and played another.

### 3. Builds features (`src/features.py`)

One row per play, in two groups:

- **Snap alignment** (available before the play starts)
  - Depth and lateral position of the three deepest defenders, the depth gap and width between the top two, and whether they sit on opposite sides of the field.
  - Cornerback cushion and leverage against the nearest receiver.
  - Mean linebacker depth.
  - The measured shell at the snap (`pre_shell`).
- **First second** (frames 1–10 after the snap)
  - The same depth geometry one second in, plus the measured shell at that point (`midshell`).
  - Rotation: how far the two deepest defenders at the snap moved in depth and laterally.
  - Man versus zone signal: whether each defender moves in the same direction as the receiver he lined up on (`mirror_*`), and how close he is to that receiver at the end (`sep_end_*`).

Play situation columns from the labels file (`down`, `yards_to_go`, `offense_formation`, `receiver_alignment`, `defenders_in_the_box`) are also used.

### 4. Trains baseline classifiers (`notebooks/04_baseline.py`)

Trains LightGBM models on weeks 1–14 and tests on weeks 15–18, for two targets: the coverage family (seven classes) and man versus zone. Each target is trained on four feature sets so you can see what each group of features adds.

## Current results

From a run on 13 October 2026: 10,824 training plays and 3,240 test plays.

**Shell detector.** Measured at one second after the snap, the rule-based shell agrees with the labelled shell on 72.3% of held-out plays. It is weakest on Cover 3 (70.3% on training weeks) and strongest on 2-Man (95.5%). Agreement peaks when measured between frames 10 and 15. The shell at the snap differs from the shell of the coverage run on 34.6% of plays.

**Classifiers** (test weeks 15–18):

| Features | Coverage family accuracy | Coverage family macro F1 | Man/zone accuracy | Man/zone macro F1 |
| --- | --- | --- | --- | --- |
| Majority class | 0.320 | 0.069 | 0.721 | 0.419 |
| Defender count only | 0.198 | 0.163 | 0.729 | 0.666 |
| Snap alignment | 0.524 | 0.476 | 0.815 | 0.775 |
| Snap + situation | 0.532 | 0.479 | 0.823 | 0.784 |
| First second | 0.706 | 0.655 | 0.874 | 0.843 |

The "defender count only" row is a leakage check. The dataset tracks only some of the players on each play, so the number of tracked defenders could give the answer away by itself. It does not for coverage family, and helps only slightly for man versus zone.

The largest remaining confusions are Cover 1 versus Cover 3 and Cover 4 versus Cover 6. The snap-only model leans most on the depth of the two deepest defenders, linebacker depth and cornerback cushion.

## Requirements

- Python 3.12 or newer. `src/features.py` uses f-string syntax that older versions reject.
- About 1 GB of disk space for the raw data.
- A Kaggle account that has accepted the competition rules, if you download the data with the Kaggle CLI.

## Setup

Run every command from the repository root.

### 1. Create a virtual environment and install dependencies

Linux, macOS or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The `.venv` currently in this folder was created inside WSL, so it only works from WSL. To run from native Windows, create a separate environment.

`requirements-cv.txt` is for the video pipeline, which is not written yet. You do not need it.

### 2. Get the data

Download the competition files and unzip them into `data/raw/bdb2026/`:

```bash
kaggle competitions download -c nfl-big-data-bowl-2026-analytics -p data/raw/bdb2026
unzip data/raw/bdb2026/nfl-big-data-bowl-2026-analytics.zip -d data/raw/bdb2026
```

You can also download the zip from the Kaggle website and extract it by hand. The loader expects this exact layout:

```
data/raw/bdb2026/
└── 114239_nfl_competition_files_published_analytics_final/
    ├── supplementary_data.csv
    └── train/
        ├── input_2023_w01.csv
        ├── ...
        └── input_2023_w18.csv
```

If your extracted folder has a different name, change `RAW` at the top of `src/load.py`.

## Running

Run the steps in this order. Later steps read files that earlier steps write.

| Step | Command | Reads | Writes |
| --- | --- | --- | --- |
| 1 | `python notebooks/01_inspect_2026.py` | raw CSVs | nothing (optional: prints file list, columns and label counts) |
| 2 | `python -m src.load` | raw CSVs | `tracking.parquet`, `plays.parquet` |
| 3 | `python notebooks/03_shell_report.py` | step 2 output | `shells.parquet` |
| 4 | `python notebooks/03c_shell_errors.py` | steps 2 and 3 output | nothing (prints error breakdown) |
| 5 | `python notebooks/03b_snap_depth.py` | step 2 output | `outputs/snap_second_depth.png` |
| 6 | `python notebooks/02_plot_play.py` | steps 2 and 3 output | `outputs/miss_Cover3_*.png` |
| 7 | `python notebooks/04_baseline.py` | step 2 output | `features.parquet` |

All parquet files go to `data/processed/`. Step 2 loads all 18 weeks into memory and is the slowest step.

What each analysis script reports:

- **`03_shell_report.py`** tunes the shell thresholds on weeks 1–14, reports agreement on weeks 15–18, cross-tabulates the shell at the snap against the coverage run, and lists the most common rotations and the defenses that rotate most.
- **`03c_shell_errors.py`** shows which coverage families the shell detector gets wrong, what it measures instead, and how agreement changes with the frame it measures at.
- **`03b_snap_depth.py`** plots a histogram of the depth of the second-deepest middle defender at the snap.
- **`02_plot_play.py`** draws four Cover 3 plays that the shell detector got wrong, at the snap and at frame 10, with each player labelled by position. Change `FAMILY` and `FRAMES` at the top of the file to look at other cases.
- **`04_baseline.py`** prints the accuracy table, a confusion matrix for the best coverage model, and the top features of the snap-only model.

### Things to know

- `04_baseline.py` caches features in `data/processed/features.parquet`. Delete that file after you change `src/features.py`, or the script will keep using the old features.
- `03c_shell_errors.py` and `03b_snap_depth.py` have the tuned thresholds (10 and 15) written in as constants. Update them if the tuning in `03_shell_report.py` picks different values.
- `data/`, `models/` and `outputs/` are in `.gitignore`, so a fresh clone has none of them. Create `outputs/` before running the plotting scripts.

## Project layout

```
coverage-analyzer/
├── src/
│   ├── load.py          # raw CSVs -> normalised tracking + labelled plays
│   ├── shells.py        # rule-based deep safety count and threshold tuning
│   ├── features.py      # per-play snap and first-second features
│   ├── labels.py        # empty
│   ├── disguise.py      # empty
│   ├── model.py         # empty
│   ├── train.py         # empty
│   └── cv/              # empty (detect.py, homography.py)
├── notebooks/           # analysis scripts, run as plain Python files
├── app.py               # empty
├── data/                # raw and processed data (not in git)
├── models/              # empty (not in git)
├── outputs/             # generated plots (not in git)
├── requirements.txt
└── requirements-cv.txt
```
