"""Train one TF-IDF + linear classifier per target and save the artifact."""

import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import config
from src import data


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )


def build_classifier() -> LinearSVC:
    # LinearSVC is fast on high-dim sparse text and handles multiclass (OvR) natively.
    return LinearSVC(class_weight="balanced", random_state=config.RANDOM_SEED)


def train():
    df = data.load()
    print(f"Loaded {len(df)} rows")

    vectorizer = build_vectorizer()
    X_all = vectorizer.fit_transform(df["text"])

    models = {}
    for target in config.TARGETS:
        clf = build_classifier()
        clf.fit(X_all, df[target])
        models[target] = clf
        print(f"Trained head: {target} ({len(clf.classes_)} classes)")

    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(config.ARTIFACT_DIR, "model.joblib")
    joblib.dump(
        {"vectorizer": vectorizer, "models": models, "targets": config.TARGETS},
        path,
    )
    size_mb = os.path.getsize(path) / 1048576
    print(f"Saved artifact -> {path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    train()
