"""
Learning curves for a train run: runs/<run>/history.json -> runs/<run>/learning_curves.png

- Left: train vs val loss. Val loss rising while train keeps falling = overfitting.
- Right: val accuracy + macro-F1, with the kept (best) epoch marked.
train.py calls plot() itself; run this by hand to redraw.

Usage:
    python code/plot_curves.py                      newest train run
    python code/plot_curves.py runs/train-...       a specific run
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt  # noqa: E402

from rundir import latest_run  # noqa: E402


def plot(run):
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    metrics_file = run / "metrics.json"
    best = json.loads(metrics_file.read_text(encoding="utf-8")).get("best_epoch") if metrics_file.exists() else None
    epochs = [h["epoch"] for h in history]

    fig, (loss_ax, score_ax) = plt.subplots(1, 2, figsize=(11, 4))
    loss_ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train (smoothed CE)")
    loss_ax.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val (plain CE)")
    loss_ax.set(xlabel="epoch", ylabel="loss", title="Loss")

    score_ax.plot(epochs, [h["val_accuracy"] for h in history], marker="o", label="val accuracy")
    score_ax.plot(epochs, [h["val_macro_f1"] for h in history], marker="o", label="val macro-F1")
    score_ax.set(xlabel="epoch", ylabel="score", title="Validation score")

    for ax in (loss_ax, score_ax):
        if best:
            ax.axvline(best, linestyle="--", color="gray", label=f"kept epoch {best}")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(run.name)
    fig.tight_layout()
    fig.savefig(run / "learning_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run("train", "history.json")
    plot(run)
    print(f"saved {run / 'learning_curves.png'}")
