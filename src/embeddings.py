"""Turn text into vectors with a pre-trained Sentence Transformer.

The model is NOT trained here. We only use it to convert text to numbers.
Embeddings are computed once and saved, so later steps just load them.
"""
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DATA_DIR, EMBEDDINGS_DIR, ensure_dirs
from src.data import load_subset

MODEL_NAME = "all-MiniLM-L6-v2"   # small and fast; gives 384 numbers per text
EMB_PATH = EMBEDDINGS_DIR / "ag_news_embeddings.npy"
LABELS_PATH = EMBEDDINGS_DIR / "ag_news_labels.npy"
TEXTS_PATH = DATA_DIR / "ag_news_subset.csv"   # kept for the Streamlit demo


def embed_texts(texts, batch_size=64):
    """Convert a list of strings into a (n_texts, 384) NumPy array."""
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(texts, batch_size=batch_size,
                        show_progress_bar=True, convert_to_numpy=True)


def build_and_save():
    """Load the subset, embed every text once, and save everything to disk."""
    ensure_dirs()
    df = load_subset()                       # same 10,000 rows every time
    start = time.time()
    X = embed_texts(df["text"].tolist())
    print(f"Embedding took {time.time() - start:.1f} seconds")

    np.save(EMB_PATH, X)                     # the vectors
    np.save(LABELS_PATH, df["label"].values) # the true labels (same row order)
    df.to_csv(TEXTS_PATH, index=False)       # the raw texts (same row order)
    return X


def load_embeddings():
    """Load the saved embeddings and labels. Row i here = row i of the CSV."""
    return np.load(EMB_PATH), np.load(LABELS_PATH)


if __name__ == "__main__":
    if EMB_PATH.exists():
        print("Embeddings already saved, loading them instead of recomputing.")
    else:
        build_and_save()

    X, y = load_embeddings()
    print("Embeddings shape:", X.shape)
    print("Labels shape    :", y.shape)
    print("First text, first 5 numbers:", np.round(X[0][:5], 3))