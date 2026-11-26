"""Generate figures from the measured results. Solid colors, bar charts only.

Run per dataset (uses config.DATASET) for per-class / distribution charts, e.g.:
    DATASET=bbc     ./.venv/bin/python -m src.visualize
    DATASET=tickets ./.venv/bin/python -m src.visualize
The cross-dataset comparison chart reads both metrics.json files and is regenerated
on every run.
"""

import os
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import config
from src import data

FIG_DIR = os.path.join(config.HERE, "figures")

# Solid, colorblind-friendly palette (no gradients).
BLUE = "#1f77b4"
GRAY = "#8c8c8c"
GREEN = "#2ca02c"
RED = "#d62728"


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, config.HERE))


def per_class_f1():
    """Bar chart of per-class F1 for each target of the current dataset."""
    bundle = joblib.load(os.path.join(config.ARTIFACT_DIR, "model.joblib"))
    df = data.load()
    for target in config.TARGETS:
        X_tr, X_te, y_tr, y_te = data.split(df, target)
        Xv = bundle["vectorizer"].transform(X_te)
        y_pred = bundle["models"][target].predict(Xv)
        classes = sorted(y_te.unique())
        f1s = f1_score(y_te, y_pred, labels=classes, average=None, zero_division=0)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(classes)), f1s, color=BLUE)
        ax.axhline(1.0 / len(classes), color=RED, linestyle="--", linewidth=1,
                   label=f"random guess ({1.0/len(classes):.2f})")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("F1 score")
        ax.set_title(f"{config.DATASET} — per-class F1 ({target})")
        ax.legend()
        safe = target.replace(" ", "_")
        _save(fig, f"{config.DATASET}_{safe}_per_class_f1.png")


def class_distribution():
    """Bar chart of class counts for each target of the current dataset."""
    df = data.load()
    for target in config.TARGETS:
        vc = df[target].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(vc)), vc.values, color=GREEN)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=20, ha="right")
        ax.set_ylabel("count")
        ax.set_title(f"{config.DATASET} — class distribution ({target})")
        safe = target.replace(" ", "_")
        _save(fig, f"{config.DATASET}_{safe}_class_distribution.png")


def comparison():
    """Grouped bar chart: model accuracy vs chance floor across all targets/datasets."""
    rows = []
    for ds in ("tickets", "bbc"):
        p = os.path.join(config.HERE, "artifacts", ds, "metrics.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            m = json.load(f)
        for target, r in m.items():
            label = f"{ds}\n{target}"
            rows.append((label, r["test_acc"], r["acc_floor"]))
    if not rows:
        return
    labels = [r[0] for r in rows]
    acc = [r[1] for r in rows]
    floor = [r[2] for r in rows]
    x = range(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - w / 2 for i in x], acc, width=w, color=BLUE, label="model accuracy")
    ax.bar([i + w / 2 for i in x], floor, width=w, color=GRAY, label="chance floor")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Same pipeline, different data: model accuracy vs chance")
    for i, v in zip(x, acc):
        ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.legend()
    _save(fig, "accuracy_vs_chance.png")


if __name__ == "__main__":
    per_class_f1()
    class_distribution()
    comparison()
