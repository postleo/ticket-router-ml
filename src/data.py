"""Download, load, clean, and split the ticket dataset."""

import os
import re
import zipfile
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def download() -> str:
    """Fetch + unzip the CC0 dataset if not already present. Returns CSV path."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    csv_path = os.path.join(config.DATA_DIR, config.CSV_NAME)
    if os.path.exists(csv_path):
        return csv_path
    zip_path = os.path.join(config.DATA_DIR, "raw.zip")
    print(f"Downloading dataset -> {zip_path}")
    urllib.request.urlretrieve(config.DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(config.DATA_DIR)
    print(f"Extracted {config.CSV_NAME}")
    return csv_path


_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = str(text).lower().strip()
    return _WS.sub(" ", text)


def load() -> pd.DataFrame:
    """Load CSV, build the combined text field, drop rows missing text/targets."""
    csv_path = download()
    df = pd.read_csv(csv_path)
    df["text"] = (
        df[config.TEXT_COLS].fillna("").agg(" ".join, axis=1).map(_clean)
    )
    df = df.dropna(subset=config.TARGETS)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def split(df: pd.DataFrame, target: str):
    """Stratified train/test split for one target."""
    return train_test_split(
        df["text"],
        df[target],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=df[target],
    )


if __name__ == "__main__":
    d = load()
    print("rows:", len(d), "| columns:", list(d.columns))
    for t in config.TARGETS:
        print(f"\n{t} distribution:\n{d[t].value_counts().to_string()}")
