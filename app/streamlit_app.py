"""CuriousML demo: you are the human labeler, the model picks what to ask.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# Let this file import from src/ (Streamlit adds only app/ to the path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.classifier import evaluate, train
from src.config import DEFAULT_SEED
from src.data import LABEL_NAMES, make_splits
from src.embeddings import TEXTS_PATH, load_embeddings
from src.strategies import margin_sampling

st.set_page_config(page_title="CuriousML", page_icon="🔍")


@st.cache_data
def load_all():
    """Load embeddings, true labels and raw texts once."""
    X, y = load_embeddings()
    texts = pd.read_csv(TEXTS_PATH)["text"].tolist()
    return X, y, texts


def retrain():
    """Train on the current labeled set, score on the locked test set, pick next sample."""
    s = st.session_state
    X, y, _ = load_all()
    clf = train(X[s.labeled], s.y_work[s.labeled], seed=DEFAULT_SEED)
    acc = evaluate(clf, X[s.test], y[s.test])        # test set: scoring only
    probs = clf.predict_proba(X[s.pool])
    s.query_pos = int(margin_sampling(probs, 1, None)[0])   # most uncertain pool sample
    s.query_probs = probs[s.query_pos]
    s.history.append({"n_labels": len(s.labeled), "accuracy": acc})


def init_state():
    """Start a fresh session: 100 labeled samples, the rest is the pool."""
    s = st.session_state
    X, y, _ = load_all()
    labeled, pool, test = make_splits(y, seed=DEFAULT_SEED)
    s.labeled = list(labeled)
    s.pool = np.array(pool)
    s.test = test
    s.y_work = y.copy()     # labels the model may use; pool entries are never read until YOU label them
    s.history = []
    retrain()


def give_label(class_id):
    """The human's click: move the asked sample from the pool to the labeled set."""
    s = st.session_state
    idx = s.pool[s.query_pos]
    s.y_work[idx] = class_id
    s.labeled.append(int(idx))
    s.pool = np.delete(s.pool, s.query_pos)
    retrain()


if "labeled" not in st.session_state:
    init_state()
s = st.session_state
_, _, texts = load_all()

st.title("CuriousML")
st.caption("Learning More from Fewer Labels")

c1, c2 = st.columns(2)
c1.metric("Human-labeled samples", len(s.labeled))
c2.metric("Test accuracy", f"{s.history[-1]['accuracy']:.4f}")

st.subheader("The model is most unsure about this text:")
st.info(texts[s.pool[s.query_pos]])

st.write("Model's probabilities:")
st.bar_chart(pd.Series(s.query_probs, index=list(LABEL_NAMES.values())))

st.write("**What is the correct label?**")
cols = st.columns(4)
for col, (cid, name) in zip(cols, LABEL_NAMES.items()):
    if col.button(name, key=f"label_{cid}", use_container_width=True):
        give_label(cid)
        st.rerun()

st.subheader("Accuracy as you label")
st.line_chart(pd.DataFrame(s.history).set_index("n_labels"))

if st.sidebar.button("Reset demo"):
    init_state()
    st.rerun()
st.sidebar.write("Strategy: margin sampling")
st.sidebar.write("Test set: locked, used only to show accuracy.")