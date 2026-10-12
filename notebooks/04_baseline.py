from pathlib import Path
import sys 
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix 

ROOT = Path(__file__).resolve().parent.parent 
sys.path.insert(0, str(ROOT))
from src.load import PROC 
from src.features import build_features, SITUATION 

# Build features once and cache them. Delete features.parquet after changing features.py.
feats_path = PROC / "features.parquet"
if feats_path.exists():
    df = pd.read_parquet(feats_path)
else:
    trk = pd.read_parquet(PROC/"tracking.parquet")
    plays = pd.read_parquet(PROC/"plays.parquet")
    df = build_features(trk, plays)
    df.to_parquet(feats_path, index = False)
    print("sved", feats_path)
    
for c in ["offense_formation", "receiver_alignment"]:
    df[c] = df[c].astype("category")
    
snap_cols = ([c for c in df.columns if c.startswith("s_")]
              + ["cb_cushion_mean", "cb_cushion_min", "cb_leverage_mean", "lb_depth", "pre_shell"])
post_cols = ([c for c in df.columns if c.startswith("p_") or c.startswith("rot_")]
             + ["midshell", "mirror_mean", "mirror_max", "sep_end_mean", "sep_end_min"])
FEATURE_SETS = {
    "count only (leakage floor)": ["n_def"],
    "snap alignment": snap_cols,
    "snap + situation": snap_cols + SITUATION,
    "first second": snap_cols + SITUATION + post_cols,
}

train, test = df[df.week <= 14], df[df.week > 14]
print(f"train plays: {len(train):,} test plays: {len(test):,}\n")

def make_model():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              class_weight="balanced", subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=0, verbose=-1)

rows, models = [], {}
for target in ["family", "is_man"]:
    maj = train[target].mode()[0]
    rows.append({"target": target, "features": "majority class",
                 "accuracy": (test[target] == maj).mean(),
                 "macro_f1": f1_score(test[target], [maj] * len(test), average = "macro")})
    for name, cols in FEATURE_SETS.items():
        model = make_model().fit(train[cols], train[target])
        pred = model.predict(test[cols])
        models[(target, name)] = (model, cols, pred)
        rows.append({"target": target, "features": name,
                     "accuracy": accuracy_score(test[target], pred),
                     "macro_f1": f1_score(test[target], pred, average = "macro")})
print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

# Where does the best coverage model go wrong?
model, cols, pred = models[("family", "first second")]
labels = sorted(test.family.unique())
cm = pd.DataFrame(confusion_matrix(test.family, pred, labels=labels), index=labels, columns=labels)
print("Confusion matrix, first-second model (rows = true, cols = predicted):")
print(cm, "\n")

# What does the snap-only model rely on?
model, cols, _ = models[("family", "snap + situation")]
imp = pd.Series(model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
print("Top features, snap + situation model:")
print((imp / imp.sum()).head(12).round(3))
        