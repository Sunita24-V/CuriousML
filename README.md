# CuriousML: Reducing Data Labeling Effort Using Active and Semi-Supervised Learning

**Learning More from Fewer Labels**

## Project objective
Labeling data by hand is slow and costly. This project measures how much
labeling effort we can save when a text classifier chooses which samples
a human should label (active learning) or learns from its own confident
guesses (semi-supervised learning).

## Main research question
How few human-labeled text samples are needed for a classifier to reach
good performance when the model intelligently chooses which samples
should be labeled?

## Methods compared
1. Fully supervised learning (all training labels, best-case reference)
2. Random sampling (baseline)
3. Active learning: least confidence, margin sampling, entropy sampling
4. Semi-supervised learning: confidence-based pseudo-labeling

All methods use the same classifier (Logistic Regression), the same
embeddings, the same seeds and the same test set. Only the way labels
are chosen or obtained changes.

## Planned workflow
1. Project setup (this phase)
2. Load one text dataset and split it into test / labeled / pool
3. Create sentence embeddings once and save them
4. Train Logistic Regression on embeddings
5. Fully supervised baseline
6. Random sampling baseline
7. Active learning (3 strategies)
8. Semi-supervised pseudo-labeling
9. Compare: test accuracy vs number of human labels
10. Repeat with several seeds, report mean and standard deviation
11. Streamlit demo of the active-learning loop
12. Final results and documentation

The test set is locked: it is never used for training, for choosing
samples, or for pseudo-labeling. It is used only to measure accuracy.

## Technology stack
Python, pandas, numpy, scikit-learn, sentence-transformers (pre-trained,
used only to create embeddings), matplotlib, streamlit, joblib.

## Folder structure
```
app/         Streamlit demo
notebooks/   Optional exploration notebooks
report/      Report and slides
results/     CSV results and figures (committed)
src/         Reusable project code (a Python package)
data/        Local data, git-ignored
embeddings/  Saved embeddings, git-ignored
models/      Saved models, git-ignored
```

## How to run (VS Code or terminal)
```
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.config        # prints the project paths (setup check)
```
Run all commands from the project root folder.

## Status
Phase 1 (setup) complete. No ML code yet.
