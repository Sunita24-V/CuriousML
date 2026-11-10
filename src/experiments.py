"""Runs the experiments and saves results as CSV files in results/."""
import numpy as np
import pandas as pd

from src.classifier import evaluate, train
from src.config import RESULTS_DIR, SEEDS, ensure_dirs
from src.data import make_splits
from src.embeddings import load_embeddings


def run_full_supervised(seeds=SEEDS):
    """Best-case reference: train on ALL training labels (labeled + pool)."""
    ensure_dirs()
    X, y = load_embeddings()
    rows = []
    for seed in seeds:
        labeled_idx, pool_idx, test_idx = make_splits(y, seed=seed)

        # Every label except the locked test set
        train_idx = np.concatenate([labeled_idx, pool_idx])

        clf = train(X[train_idx], y[train_idx], seed=seed)
        acc = evaluate(clf, X[test_idx], y[test_idx])   # test set used ONLY here
        rows.append({"method": "full_supervised", "seed": seed,
                     "n_labels": len(train_idx), "accuracy": acc})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "full_supervised.csv", index=False)
    return df


if __name__ == "__main__":
    df = run_full_supervised()
    print(df.to_string(index=False))
    print(f"\nMean accuracy: {df['accuracy'].mean():.4f}")
    print(f"Std deviation: {df['accuracy'].std():.4f}")