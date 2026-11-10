"""The ONE classifier used by every experiment: Logistic Regression.

Using the same classifier and settings everywhere keeps the comparison fair.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.config import DEFAULT_SEED


def make_classifier(seed=DEFAULT_SEED):
    """Create an untrained Logistic Regression (same settings every time)."""
    return LogisticRegression(max_iter=1000, random_state=seed)


def train(X_train, y_train, seed=DEFAULT_SEED):
    """Train a new classifier on the given embeddings and labels."""
    clf = make_classifier(seed)
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test):
    """Return accuracy on the given data (we only pass the TEST set here)."""
    return accuracy_score(y_test, clf.predict(X_test))


if __name__ == "__main__":
    from src.data import make_splits, LABEL_NAMES
    from src.embeddings import load_embeddings

    X, y = load_embeddings()
    labeled_idx, pool_idx, test_idx = make_splits(y)

    # Train ONLY on the 100 labeled samples
    clf = train(X[labeled_idx], y[labeled_idx])

    # Measure on the locked test set (used only for scoring)
    acc = evaluate(clf, X[test_idx], y[test_idx])
    print(f"Trained on {len(labeled_idx)} labels -> test accuracy: {acc:.4f}")

    # Probabilities for 3 pool samples (active learning will use these later)
    probs = clf.predict_proba(X[pool_idx[:3]])
    print("\nClass order:", [LABEL_NAMES[c] for c in clf.classes_])
    for i, p in enumerate(probs):
        print(f"Pool sample {i}: ", [round(float(v), 3) for v in p])