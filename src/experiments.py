"""Runs the experiments and saves results as CSV files in results/."""
import numpy as np
import pandas as pd

from src.active_learning import run_loop
from src.classifier import evaluate, train
from src.config import RESULTS_DIR, SEEDS, ensure_dirs
from src.data import make_splits
from src.embeddings import load_embeddings
from src.strategies import STRATEGIES


def run_full_supervised(seeds=SEEDS):
    """Best-case reference: train on ALL training labels (labeled + pool)."""
    ensure_dirs()
    X, y = load_embeddings()
    rows = []
    for seed in seeds:
        labeled_idx, pool_idx, test_idx = make_splits(y, seed=seed)
        train_idx = np.concatenate([labeled_idx, pool_idx])
        clf = train(X[train_idx], y[train_idx], seed=seed)
        acc = evaluate(clf, X[test_idx], y[test_idx])   # test set used ONLY here
        rows.append({"method": "full_supervised", "seed": seed,
                     "n_labels": len(train_idx), "accuracy": acc})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "full_supervised.csv", index=False)
    return df


def run_strategy(name, seeds=SEEDS):
    """Run one selection strategy over all seeds and save results/<name>.csv."""
    ensure_dirs()
    X, y = load_embeddings()
    rows = []
    for seed in seeds:
        # Same seed -> same split for every method, so comparisons are fair
        labeled_idx, pool_idx, test_idx = make_splits(y, seed=seed)
        records = run_loop(X, y, labeled_idx, pool_idx, test_idx,
                           query_fn=STRATEGIES[name], seed=seed)
        for n_labels, acc in records:
            rows.append({"method": name, "seed": seed,
                         "n_labels": n_labels, "accuracy": acc})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return df


if __name__ == "__main__":
    # Run every strategy (random already done, but re-running is cheap)
    all_results = [run_strategy(name) for name in STRATEGIES]
    df = pd.concat(all_results)

    # Mean accuracy per label count, one column per method
    table = df.pivot_table(index="n_labels", columns="method",
                           values="accuracy", aggfunc="mean").round(4)
    print("Mean test accuracy over", df["seed"].nunique(), "seeds:")
    print(table.to_string())   # to_string() prints ALL columns without "..."