"""Honest evaluation: CV + held-out metrics for each target, next to a chance baseline."""

import os
import json

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import config
from src import data
from src.train import build_vectorizer, build_classifier


def _model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("clf", build_classifier()),
        ]
    )


def evaluate():
    df = data.load()
    results = {}
    md = ["# Results (measured)\n",
          "All numbers below are measured in the sandbox on CPU. Each model score is shown "
          "next to a **chance baseline** (`most_frequent`). A model is only meaningful if it "
          "clearly beats that baseline.\n"]

    for target in config.TARGETS:
        X_tr, X_te, y_tr, y_te = data.split(df, target)
        n_classes = df[target].nunique()

        # Chance baseline (most frequent class).
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_tr, y_tr)
        dummy_pred = dummy.predict(X_te)
        dummy_acc = accuracy_score(y_te, dummy_pred)
        dummy_f1 = f1_score(y_te, dummy_pred, average="macro")

        # Cross-validated model (vectorizer fit inside each fold -> no leakage).
        # Single pass computing both metrics per fold.
        skf = StratifiedKFold(
            n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
        )
        pipe = _model_pipeline()
        cv = cross_validate(
            pipe, X_tr, y_tr, cv=skf,
            scoring=["f1_macro", "accuracy"], n_jobs=-1,
        )
        cv_f1 = cv["test_f1_macro"]
        cv_acc = cv["test_accuracy"]

        # Held-out test.
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        test_acc = accuracy_score(y_te, y_pred)
        test_f1 = f1_score(y_te, y_pred, average="macro")

        results[target] = {
            "n_classes": int(n_classes),
            "chance_acc": round(float(dummy_acc), 4),
            "chance_macro_f1": round(float(dummy_f1), 4),
            "cv_macro_f1_mean": round(float(cv_f1.mean()), 4),
            "cv_macro_f1_std": round(float(cv_f1.std()), 4),
            "cv_acc_mean": round(float(cv_acc.mean()), 4),
            "test_acc": round(float(test_acc), 4),
            "test_macro_f1": round(float(test_f1), 4),
        }

        md.append(f"\n## {target}  ({n_classes} classes)\n")
        md.append("| metric | chance (most_frequent) | TF-IDF + LinearSVC |")
        md.append("|--------|------------------------|-----------------|")
        md.append(f"| accuracy | {dummy_acc:.3f} | {test_acc:.3f} |")
        md.append(f"| macro-F1 | {dummy_f1:.3f} | {test_f1:.3f} |")
        md.append(
            f"\nCross-val (train, {config.CV_FOLDS}-fold): "
            f"macro-F1 {cv_f1.mean():.3f} ± {cv_f1.std():.3f}, "
            f"accuracy {cv_acc.mean():.3f}\n"
        )
        md.append("```")
        md.append(classification_report(y_te, y_pred, zero_division=0))
        md.append("confusion matrix:")
        md.append(np.array2string(confusion_matrix(y_te, y_pred)))
        md.append("```")

    # Verdict line. Honest floor = the better of (most-frequent accuracy) and
    # (1/n_classes, the random-guess rate on balanced classes). A model only shows
    # real signal if it clearly beats that floor.
    md.append("\n## Reading these numbers\n")
    for target, r in results.items():
        acc_floor = max(r["chance_acc"], 1.0 / r["n_classes"])
        lift = r["test_acc"] - acc_floor
        verdict = (
            "beats chance — real signal"
            if lift > 0.05
            else "at/near chance (no learnable signal in this public data)"
        )
        results[target]["acc_floor"] = round(float(acc_floor), 4)
        results[target]["lift_over_floor"] = round(float(lift), 4)
        md.append(
            f"- **{target}**: test accuracy {r['test_acc']:.3f} vs chance floor "
            f"{acc_floor:.3f} (lift {lift:+.3f}) → {verdict}."
        )

    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(config.ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(config.RESULTS_PATH, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {os.path.basename(config.RESULTS_PATH)} and "
          f"artifacts/{config.DATASET}/metrics.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate()
