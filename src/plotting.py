"""Combine all results and draw accuracy vs number of human labels (mean ± std)."""
import matplotlib
matplotlib.use("Agg")            # draw to a file, no pop-up window needed
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR, ensure_dirs

METHODS = ["random", "least_confidence", "margin", "entropy", "semi_supervised"]


def load_results():
    """Read every method's CSV and stack them into one table."""
    frames = [pd.read_csv(RESULTS_DIR / f"{m}.csv") for m in METHODS]
    return pd.concat(frames)[["method", "seed", "n_labels", "accuracy"]]


def summarize(df):
    """Mean and standard deviation over seeds, per method and label count."""
    return (df.groupby(["method", "n_labels"])["accuracy"]
              .agg(["mean", "std"]).reset_index())


def labels_needed(summary, target):
    """First label count where a method's MEAN accuracy reaches the target."""
    out = {}
    for m in METHODS:
        s = summary[(summary["method"] == m) & (summary["mean"] >= target)]
        out[m] = int(s["n_labels"].min()) if len(s) else "not reached"
    return out


def plot_accuracy_vs_labels(summary, full_mean):
    plt.figure(figsize=(8, 5))
    for m in METHODS:
        s = summary[summary["method"] == m]
        plt.errorbar(s["n_labels"], s["mean"], yerr=s["std"],
                     marker="o", markersize=4, capsize=3, linewidth=1.5, label=m)
    # Best-case reference: all 8,000 labels
    plt.axhline(full_mean, linestyle="--", color="black",
                label=f"full supervised ({full_mean:.4f})")
    plt.xlabel("Number of human-labeled samples")
    plt.ylabel("Test accuracy (mean ± std over 10 seeds)")
    plt.title("CuriousML: accuracy vs number of human labels")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "accuracy_vs_labels.png"
    plt.savefig(path, dpi=150)
    return path


if __name__ == "__main__":
    ensure_dirs()
    df = load_results()
    summary = summarize(df)
    summary.round(4).to_csv(RESULTS_DIR / "summary.csv", index=False)

    full_mean = pd.read_csv(RESULTS_DIR / "full_supervised.csv")["accuracy"].mean()
    print("Saved figure:", plot_accuracy_vs_labels(summary, full_mean))

    # Labels needed to reach a fixed share of the full-supervised accuracy.
    # The shares (97%, 98%, 99%) are fixed here, not picked from the results.
    print(f"\nFull-supervised mean accuracy: {full_mean:.4f}")
    for share in (0.97, 0.98, 0.99):
        target = share * full_mean
        print(f"\nLabels needed to reach {int(share*100)}% of full accuracy "
              f"(mean accuracy >= {target:.4f}):")
        for m, n in labels_needed(summary, target).items():
            print(f"  {m:18s} {n}")