"""Run the whole comparison on ANY text classification data (a CSV with text + label)."""
import hashlib

import numpy as np
import pandas as pd

from src.active_learning import run_loop
from src.config import EMBEDDINGS_DIR, RESULTS_DIR, ensure_dirs
from src.data import make_splits
from src.embeddings import embed_texts
from src.strategies import STRATEGIES


def load_csv(path, text_col, label_col):
    """Read a CSV and return clean texts and integer labels (plus the class names)."""
    df = pd.read_csv(path)[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    classes = sorted(df[label_col].astype(str).unique())
    label_to_id = {c: i for i, c in enumerate(classes)}
    y = df[label_col].astype(str).map(label_to_id).values
    return df[text_col].tolist(), y, classes


def embed_cached(texts, name):
    """Embed once per dataset, then reuse the saved file (name + text fingerprint)."""
    ensure_dirs()
    fingerprint = hashlib.md5("".join(texts[:50]).encode()).hexdigest()[:8]
    path = EMBEDDINGS_DIR / f"{name}_{len(texts)}_{fingerprint}.npy"
    if path.exists():
        return np.load(path)
    X = embed_texts(texts)
    np.save(path, X)
    return X


def auto_budget(n_classes):
    """Budget rules that depend ONLY on the number of classes (fixed before any results)."""
    return {"n_init": 10 * n_classes,      # starting labels
            "batch_size": 5 * n_classes,   # labels added per round
            "n_rounds": 10}                # rounds of adding labels


def run_comparison(X, y, name, seeds=range(10), methods=("random", "least_confidence", "margin", "entropy"),
                   test_frac=0.2):
    """Run every method over all seeds. Returns a long table and saves it to results/."""
    ensure_dirs()
    n_classes = len(np.unique(y))
    b = auto_budget(n_classes)
    test_size = int(test_frac * len(y))
    rows = []
    for seed in seeds:
        labeled, pool, test = make_splits(y, seed=seed, test_size=test_size, n_labeled=b["n_init"])
        for m in methods:
            records = run_loop(X, y, labeled, pool, test, query_fn=STRATEGIES[m], seed=seed,
                               batch_size=b["batch_size"], n_rounds=b["n_rounds"])
            for n_labels, acc in records:
                rows.append({"dataset": name, "method": m, "seed": seed,
                             "n_labels": n_labels, "accuracy": acc})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"custom_{name}.csv", index=False)
    return df