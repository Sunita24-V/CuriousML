"""Command-line tool: python -m src.run_custom --csv data.csv --text-col text --label-col label --name mydata"""
import argparse

from src.pipeline import embed_cached, load_csv, run_comparison

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--text-col", required=True)
    p.add_argument("--label-col", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--max-rows", type=int, default=10000)   # keeps runs light
    args = p.parse_args()

    texts, y, classes = load_csv(args.csv, args.text_col, args.label_col)
    texts, y = texts[:args.max_rows], y[:args.max_rows]
    print(f"{len(texts)} texts, {len(classes)} classes: {classes}")

    X = embed_cached(texts, args.name)
    df = run_comparison(X, y, args.name)
    print(df.pivot_table(index="n_labels", columns="method", values="accuracy").round(4).to_string())