"""Download a public text dataset from Hugging Face and save it as data/<name>.csv"""
import sys

from datasets import load_dataset

from src.config import DATA_DIR, ensure_dirs

SOURCES = {"emotion": "dair-ai/emotion", "imdb": "stanfordnlp/imdb"}

if __name__ == "__main__":
    name = sys.argv[1]
    ensure_dirs()
    df = load_dataset(SOURCES[name], split="train").to_pandas()
    # Shuffle and keep at most 8,000 rows (some files are sorted by class)
    df = df.sample(n=min(8000, len(df)), random_state=42)
    df[["text", "label"]].to_csv(DATA_DIR / f"{name}.csv", index=False)
    print(f"Saved data/{name}.csv with {len(df)} rows. Class counts:")
    print(df["label"].value_counts().sort_index())