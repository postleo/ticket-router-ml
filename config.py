"""Data-specific configuration.

The SAME pipeline (src/) runs on different datasets selected by the DATASET env var.
This is the "config swap" that lets Option 1's implementation be reused on a
real-signal dataset (BBC News) with no code changes.

    DATASET=tickets  (default)  -> synthetic support tickets (no learnable signal)
    DATASET=bbc                 -> BBC News topic classification (real signal)

To add real ticket data: add an entry here pointing at your columns and rerun.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.environ.get("DATASET", "tickets")

_DATASETS = {
    "tickets": {
        "url": (
            "https://www.kaggle.com/api/v1/datasets/download/"
            "suraj520/customer-support-ticket-dataset"
        ),
        "csv": "customer_support_tickets.csv",
        "text_cols": ["Ticket Subject", "Ticket Description"],
        "targets": ["Ticket Type", "Ticket Priority"],
    },
    "bbc": {
        "url": (
            "https://www.kaggle.com/api/v1/datasets/download/"
            "jacopoferretti/bbc-articles-dataset"
        ),
        "csv": "archive/bbc_text_cls.csv",
        "text_cols": ["text"],
        "targets": ["labels"],
    },
}

if DATASET not in _DATASETS:
    raise ValueError(f"Unknown DATASET={DATASET!r}; choose from {list(_DATASETS)}")

_cfg = _DATASETS[DATASET]
DATA_URL = _cfg["url"]
CSV_NAME = _cfg["csv"]
TEXT_COLS = _cfg["text_cols"]
TARGETS = _cfg["targets"]

DATA_DIR = os.path.join(HERE, "data", DATASET)
ARTIFACT_DIR = os.path.join(HERE, "artifacts", DATASET)
RESULTS_PATH = os.path.join(HERE, f"RESULTS-{DATASET}.md")

TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_SEED = 42
