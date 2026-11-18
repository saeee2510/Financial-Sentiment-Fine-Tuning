"""Metrics, confusion matrices and error analysis."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import LABELS


def metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(LABELS)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                  "support": int(s[i])} for i in range(len(LABELS))},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(LABELS))).tolist(),
    }


def top_confusions(cm, k=3):
    cm = np.array(cm)
    pairs = [(cm[i, j], LABELS[i], LABELS[j]) for i in range(len(LABELS)) for j in range(len(LABELS)) if i != j]
    return [{"true": t, "predicted": p, "count": int(c)} for c, t, p in sorted(pairs, reverse=True)[:k]]


def errors_table(texts, y_true, y_pred, conf, n=25):
    df = pd.DataFrame({"text": texts, "true": [LABELS[i] for i in y_true],
                       "predicted": [LABELS[i] for i in y_pred], "confidence": conf})
    wrong = df[df["true"] != df["predicted"]].sort_values("confidence", ascending=False)
    return wrong.head(n)


def plot_confusions(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (name, m) in zip(axes, results.items()):
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)), xticklabels=LABELS, yticklabels=LABELS,
               xlabel="Predicted", ylabel="True", title=f"{name}\nmacro-F1 {m['macro_f1']:.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
