"""Shared helpers for Lab 1 (seeds, data split, model factories, small utilities)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
N_TRAIN_RUNS = 5      # timed training runs (after 1 warm-up)
N_INFER_RUNS = 100    # single-sample inference repeats
N_MEM_RUNS = 5        # fresh-process repeats for each peak-memory measurement

SRC_DIR = Path(__file__).resolve().parent
LAB_DIR = SRC_DIR.parent
RESULTS_DIR = LAB_DIR / "results"

# Hyper-parameters are fixed by the manual - do not change them.
MODEL_FACTORIES = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED),
}


def set_seeds(include_torch: bool = True) -> None:
    """Seed everything the manual lists (Global rules -> Seeds)."""
    random.seed(SEED)
    np.random.seed(SEED)
    if include_torch:
        try:
            import torch

            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        except ImportError:
            print("WARNING: torch is not installed, torch seeds were not set.")


def load_split():
    """Breast Cancer Wisconsin, 70/30 stratified split, random_state=42."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)


def model_path(name: str) -> Path:
    return RESULTS_DIR / f"model_{name}.joblib"


def md_table(rows: list[dict]) -> str:
    """Tiny markdown-table writer (avoids needing the `tabulate` package)."""
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)
