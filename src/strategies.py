"""Sample-selection strategies.

Every strategy has the same inputs and output:
  probs : predicted class probabilities for the POOL samples (n_pool, n_classes)
  n     : how many samples to pick
  rng   : a seeded random generator
It returns POSITIONS inside the pool array. It never sees any labels.
"""
import numpy as np


def random_query(probs, n, rng):
    """Baseline: pick n pool samples completely at random."""
    return rng.choice(len(probs), size=n, replace=False)


def least_confidence(probs, n, rng):
    """Pick samples where the model's top probability is LOWEST."""
    uncertainty = 1 - probs.max(axis=1)          # high = unsure
    return np.argsort(-uncertainty)[:n]


def margin_sampling(probs, n, rng):
    """Pick samples where the top two probabilities are CLOSEST."""
    sorted_p = np.sort(probs, axis=1)            # each row from small to large
    margin = sorted_p[:, -1] - sorted_p[:, -2]   # best minus second best
    return np.argsort(margin)[:n]                # smallest margin first


def entropy_sampling(probs, n, rng):
    """Pick samples where the probabilities are most spread out (highest entropy)."""
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return np.argsort(-entropy)[:n]


STRATEGIES = {
    "random": random_query,
    "least_confidence": least_confidence,
    "margin": margin_sampling,
    "entropy": entropy_sampling,
}