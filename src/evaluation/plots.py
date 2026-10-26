from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402


def roc_plot(y, s, path: Path, title: str, auroc: float | None = None) -> None:
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUROC {auroc:.3f}" if auroc is not None else None)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="1 - specificity", ylabel="sensitivity", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def pr_plot(y, s, path: Path, title: str) -> None:
    p, r, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(r, p)
    ax.axhline(float(np.mean(y)), color="k", ls="--", lw=0.8, label="prevalence")
    ax.set(xlabel="recall (sensitivity)", ylabel="precision", title=title, ylim=(0, 1.02))
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def confusion_plot(m: dict, path: Path, title: str) -> None:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["below thr.", "above thr."], yticklabels=["non-PD", "PD"],
           xlabel="model output", ylabel="dataset label", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def score_hist(y, s, thr: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.6))
    y, s = np.asarray(y), np.asarray(s)
    ax.hist([s[y == 0], s[y == 1]], bins=15, label=["non-PD", "PD"], stacked=False)
    ax.axvline(thr, color="k", ls="--", label="threshold")
    ax.set(xlabel="model score (uncalibrated)", ylabel="count", title=title)
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
