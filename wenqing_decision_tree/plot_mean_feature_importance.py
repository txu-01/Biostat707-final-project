"""Plot saved tree importances; no models are fitted."""
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(HERE / "results" / "feature_importances.csv")
if data.duplicated(["repeat", "fold", "feature"]).any():
    raise ValueError("Duplicate feature records within a fold.")
summary = data.groupby("feature")["importance"].agg(["mean", "std", "count"])
if not summary["count"].eq(50).all() or data[["repeat", "fold"]].drop_duplicates().shape[0] != 50:
    raise ValueError("Expected each feature to have values from 50 outer folds.")
top = summary.sort_values("mean", ascending=False).head(10).iloc[::-1]
destination = HERE / "figures" / "mean_feature_importance.png"
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top.index, top["mean"], xerr=top["std"], capsize=3, color="steelblue")
ax.set(xlabel="Mean impurity-based importance (error bars: ±1 SD)",
       title="Decision tree feature importance across 50 outer CV folds")
fig.tight_layout()
with destination.open("xb") as output:
    fig.savefig(output, format="png", dpi=200)
plt.close(fig)
print(f"Saved {destination}")
