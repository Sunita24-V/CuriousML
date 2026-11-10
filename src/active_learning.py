"""The label-budget loop: train -> score on test -> pick pool samples -> add labels -> repeat.

Random sampling and active learning use THIS SAME loop.
Only the query function (the picking rule) is different.
"""
import numpy as np

from src.classifier import evaluate, train


def run_loop(X, y, labeled_idx, pool_idx, test_idx, query_fn,
             seed, batch_size=100, n_rounds=9):
    """Return a list of (n_labels, test_accuracy), one entry per round."""
    rng = np.random.default_rng(seed)
    labeled = list(labeled_idx)
    pool = np.array(pool_idx)
    records = []

    for round_no in range(n_rounds + 1):
        # 1) Train on the currently labeled samples only
        clf = train(X[labeled], y[labeled], seed=seed)

        # 2) Measure on the locked test set (scoring only, never for choosing)
        records.append((len(labeled), evaluate(clf, X[test_idx], y[test_idx])))
        if round_no == n_rounds:
            break

        # 3) The strategy sees only the POOL FEATURES (probabilities), no labels
        probs = clf.predict_proba(X[pool])
        picked = query_fn(probs, batch_size, rng)

        # 4) "Ask the human": the picked samples move from pool to labeled
        labeled.extend(pool[picked].tolist())
        pool = np.delete(pool, picked)

    # Safety check: the test set must never have entered the labeled set
    assert not (set(labeled) & set(test_idx)), "LEAKAGE: test samples in labeled set!"
    return records