"""Load the text dataset (AG News) and split it into test / labeled / pool."""
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_SEED

# Class id -> class name for AG News
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
SUBSET_SIZE = 10_000   # total samples used in the project
TEST_SIZE = 2_000      # locked test set
INIT_LABELED = 100     # small starting labeled set


def load_subset(n=SUBSET_SIZE, seed=DEFAULT_SEED):
    """Return a DataFrame with columns 'text' and 'label', n rows in total."""
    df = load_dataset("fancyzhx/ag_news", split="train").to_pandas()
    per_class = n // len(LABEL_NAMES)          # same number from each class
    df = df.groupby("label").sample(n=per_class, random_state=seed)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_splits(labels, seed=DEFAULT_SEED, test_size=TEST_SIZE, n_labeled=INIT_LABELED):
    """Return row indices (labeled_idx, pool_idx, test_idx).

    We return INDICES, not copies of the data, so embeddings (computed once)
    can be looked up by index later.
    """
    labels = np.asarray(labels)
    all_idx = np.arange(len(labels))

    # Step 1: lock the TEST set first (stratified = same class mix as the whole data)
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, stratify=labels, random_state=seed)

    # Step 2: from the rest, take a small LABELED set; everything else is the POOL
    pool_idx, labeled_idx = train_test_split(
        train_idx, test_size=n_labeled, stratify=labels[train_idx], random_state=seed)

    return labeled_idx, pool_idx, test_idx


def check_splits(labeled_idx, pool_idx, test_idx, n_total):
    """Leakage check: the three sets must not overlap and must cover everything."""
    L, P, T = set(labeled_idx), set(pool_idx), set(test_idx)
    assert not (L & P), "labeled and pool overlap!"
    assert not (L & T), "labeled and test overlap!"
    assert not (P & T), "pool and test overlap!"
    assert len(L) + len(P) + len(T) == n_total, "some samples are missing!"
    print("Leakage check passed: no overlap between test, labeled and pool.")


if __name__ == "__main__":
    df = load_subset()
    labeled_idx, pool_idx, test_idx = make_splits(df["label"].values)
    check_splits(labeled_idx, pool_idx, test_idx, len(df))

    print("\nSplit sizes:")
    print("  test   :", len(test_idx))
    print("  labeled:", len(labeled_idx))
    print("  pool   :", len(pool_idx))

    print("\nClass counts in each set:")
    for name, idx in [("test", test_idx), ("labeled", labeled_idx), ("pool", pool_idx)]:
        counts = df.loc[idx, "label"].map(LABEL_NAMES).value_counts()
        print(f"  {name:8s}", counts.to_dict())