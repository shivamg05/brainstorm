#!/usr/bin/env python3
"""
Diagnose prediction distribution and per-class recall.

Usage:
    uv run python examples/diagnose_predictions.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print as rprint

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.ml.cnn_gru import CNNGRU


DATA_PATH = Path("./data")


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[float, float]:
    recalls = {}
    labels = np.unique(y_true)
    for label in labels:
        mask = y_true == label
        denom = mask.sum()
        recalls[float(label)] = float((y_pred[mask] == label).sum() / max(denom, 1))
    return recalls


def main() -> None:
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("\n[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()

    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    val_features, val_labels = load_raw_data(DATA_PATH, step="validation")
    y_val = val_labels["label"].to_numpy()
    model = CNNGRU.load()

    preds = []
    for sample in val_features.to_numpy():
        preds.append(model.predict(sample))
    y_pred = np.array(preds)

    rprint("[bold cyan]Prediction distribution (val)[/]")
    pred_counts = Counter(y_pred.tolist())
    for label, count in sorted(pred_counts.items()):
        rprint(f"- {label}: {count} ({count / len(y_pred):.2%})")

    rprint("\n[bold cyan]Per-class recall (val)[/]")
    recalls = per_class_recall(y_val, y_pred)
    for label, score in sorted(recalls.items()):
        rprint(f"- {label}: {score:.3f}")

    true_counts = Counter(y_val.tolist())
    rprint("\n[bold cyan]True label distribution (val)[/]")
    for label, count in sorted(true_counts.items()):
        rprint(f"- {label}: {count} ({count / len(y_val):.2%})")


if __name__ == "__main__":
    main()
