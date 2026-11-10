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


# Phase 7 will add: least_confidence, margin, entropy
STRATEGIES = {"random": random_query}