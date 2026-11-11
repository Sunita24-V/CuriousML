"""Summary for a custom dataset: error bars, ceiling, majority baseline, paired check.
Usage: python -m src.report_custom --csv data/emotion.csv --text-col text --label-col label --name emotion
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classifier import evaluate, train
from src.config import FIGURES_DIR, RESULTS_DIR, ensure_dirs
from src.data import make_splits
from src.pipeline import auto_budget, embed_cached, load_csv


def ceiling(X, y, seeds, test_frac=0.2):
    """Best case: train on ALL non-test samples, same splits as the experiment."""
    accs = []
    for seed in seeds:
        n_init = auto_budget(len(np.unique(y)))["n_init"]
        labeled, pool, test = make_splits(y, seed=seed, test_size=int(test_frac * len(y)),
                                          n_labeled=n_init)
        tr = np.concatenate([labeled, pool])
        accs.append(evaluate(train(X[tr], y[tr], seed=seed), X[test], y[test]))
    return np.mean(accs), np.std(accs, ddof=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--text-col", required=True)
    p.add_argument("--label-col", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--max-rows", type=int, default=10000)
    a = p.parse_args()
    ensure_dirs()

    texts, y, classes = load_csv(a.csv, a.text_col, a.label_col)
    texts, y = texts[:a.max_rows], y[:a.max_rows]
    X = embed_cached(texts, a.name)                 # loads the saved file, no recomputing
    df = pd.read_csv(RESULTS_DIR / f"custom_{a.name}.csv")
    seeds = sorted(df["seed"].unique())

    # Reference points
    majority = np.bincount(y).max() / len(y)
    full_mean, full_std = ceiling(X, y, seeds)
    print(f"Majority-class share (accuracy of always guessing the biggest class): {majority:.4f}")
    print(f"Full-supervised ceiling: {full_mean:.4f} (std {full_std:.4f})")

    # Mean and std over seeds, saved and plotted
    summary = df.groupby(["method", "n_labels"])["accuracy"].agg(["mean", "std"]).reset_index()
    summary.round(4).to_csv(RESULTS_DIR / f"custom_{a.name}_summary.csv", index=False)
    plt.figure(figsize=(8, 5))
    for m in summary["method"].unique():
        s = summary[summary["method"] == m]
        plt.errorbar(s["n_labels"], s["mean"], yerr=s["std"], marker="o",
                     markersize=4, capsize=3, label=m)
    plt.axhline(full_mean, linestyle="--", color="black", label=f"full supervised ({full_mean:.4f})")
    plt.xlabel("Number of human-labeled samples")
    plt.ylabel(f"Test accuracy (mean ± std over {len(seeds)} seeds)")
    plt.title(f"CuriousML on '{a.name}'")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    path = FIGURES_DIR / f"custom_{a.name}.png"
    plt.savefig(path, dpi=150)
    print("Saved figure:", path)

    # Paired check vs random at two label counts fixed by rule: the middle and the last round
    counts = sorted(df["n_labels"].unique())
    print("\nDifference vs random (method minus random, per seed):")
    for n in (counts[len(counts) // 2], counts[-1]):
        base = df[(df.method == "random") & (df.n_labels == n)].set_index("seed")["accuracy"]
        for m in ("least_confidence", "margin", "entropy"):
            d = df[(df.method == m) & (df.n_labels == n)].set_index("seed")["accuracy"]
            diff = d - base
            print(f"n={n:4d}  {m:17s} mean diff {diff.mean():+.4f}  std {diff.std():.4f}  "
                  f"better in {(diff > 0).sum()}/{len(diff)} seeds")