"""Paired comparison: for each seed, how much better is a method than random?

Same seed = same split and same starting labels, so we can subtract per seed.
The label counts below were fixed BEFORE looking at this check.
"""
import pandas as pd

from src.config import RESULTS_DIR

COMPARE_AT = (300, 500, 1000)
METHODS = ["least_confidence", "margin", "entropy", "semi_supervised"]

if __name__ == "__main__":
    random_df = pd.read_csv(RESULTS_DIR / "random.csv")
    print("Accuracy difference vs random (method minus random), per seed:\n")
    for n in COMPARE_AT:
        base = random_df[random_df["n_labels"] == n].set_index("seed")["accuracy"]
        for m in METHODS:
            d = pd.read_csv(RESULTS_DIR / f"{m}.csv")
            d = d[d["n_labels"] == n].set_index("seed")["accuracy"]
            diff = d - base                       # aligned by seed
            print(f"n={n:5d}  {m:17s} mean diff {diff.mean():+.4f}  "
                  f"std {diff.std():.4f}  better in {(diff > 0).sum()}/10 seeds")
        print()