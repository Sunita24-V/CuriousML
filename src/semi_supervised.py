"""Confidence-based pseudo-labeling (semi-supervised learning).

Idea: train on human labels, let the model label the pool itself, keep only
the predictions it is very sure about, and retrain on both.
"""
import numpy as np
import pandas as pd

from src.classifier import evaluate, train
from src.config import RESULTS_DIR, SEEDS, ensure_dirs
from src.data import make_splits
from src.embeddings import load_embeddings

THRESHOLD = 0.90   # only trust predictions with at least 90% confidence


def pseudo_label_train(X, y, labeled_idx, pool_idx, seed, threshold=THRESHOLD):
    """Train with human labels + confident pseudo-labels. Pool labels are NOT used here."""
    # 1) First model: human labels only
    clf = train(X[labeled_idx], y[labeled_idx], seed=seed)

    # 2) Predict the pool and keep only the confident predictions
    probs = clf.predict_proba(X[pool_idx])
    confident = probs.max(axis=1) >= threshold
    pseudo_idx = pool_idx[confident]
    pseudo_y = clf.classes_[probs[confident].argmax(axis=1)]   # the model's own guesses

    # 3) Second model: human labels + pseudo-labels
    X_new = np.vstack([X[labeled_idx], X[pseudo_idx]])
    y_new = np.concatenate([y[labeled_idx], pseudo_y])
    return train(X_new, y_new, seed=seed), pseudo_idx, pseudo_y


def run_semi_supervised(seeds=SEEDS, batch_size=100, n_rounds=9):
    """Same budget steps as random sampling, with pseudo-labels added on top.

    Human labels are chosen RANDOMLY with the same random generator as the random
    baseline, so both methods have identical human-labeled sets. The only
    difference is the extra pseudo-labeled data.
    """
    ensure_dirs()
    X, y = load_embeddings()
    rows = []
    for seed in seeds:
        labeled_idx, pool_idx, test_idx = make_splits(y, seed=seed)
        rng = np.random.default_rng(seed)
        labeled = list(labeled_idx)
        pool = np.array(pool_idx)

        for round_no in range(n_rounds + 1):
            clf, p_idx, p_y = pseudo_label_train(X, y, np.array(labeled), pool, seed)
            acc = evaluate(clf, X[test_idx], y[test_idx])   # test set: scoring only

            # Diagnostic only (never used for training): how many pseudo-labels were right?
            pseudo_acc = (y[p_idx] == p_y).mean() if len(p_idx) > 0 else np.nan
            rows.append({"method": "semi_supervised", "seed": seed,
                         "n_labels": len(labeled), "accuracy": acc,
                         "n_pseudo": len(p_idx), "pseudo_acc": pseudo_acc})
            if round_no == n_rounds:
                break

            picked = rng.choice(len(pool), size=batch_size, replace=False)
            labeled.extend(pool[picked].tolist())
            pool = np.delete(pool, picked)

        assert not (set(labeled) & set(test_idx)), "LEAKAGE: test samples in labeled set!"

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "semi_supervised.csv", index=False)
    return df


if __name__ == "__main__":
    df = run_semi_supervised()
    summary = df.groupby("n_labels").agg(
        semi_acc=("accuracy", "mean"),
        n_pseudo=("n_pseudo", "mean"),
        pseudo_acc=("pseudo_acc", "mean")).round(4)
    # Add the random baseline (same human labels) for comparison
    random_df = pd.read_csv(RESULTS_DIR / "random.csv")
    summary["random_acc"] = random_df.groupby("n_labels")["accuracy"].mean().round(4)
    print(summary.to_string())