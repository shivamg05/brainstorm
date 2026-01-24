#!/usr/bin/env python3
"""
Check stratified window sampling behavior without training.

Usage:
    uv run python examples/check_stratification.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from rich import print as rprint

from brainstorm.loading import load_raw_data


DATA_PATH = Path("./data")

SEQ_LEN = 128
STRIDE = SEQ_LEN * 3 // 4
STRATIFY_RATIO = 0.45


def main() -> None:
    _, train_labels = load_raw_data(DATA_PATH, step="train")
    y = train_labels["label"].to_numpy()

    n_samples = len(y)
    n_sequences = (n_samples - SEQ_LEN) // STRIDE
    starts_all = np.arange(0, n_samples - SEQ_LEN, STRIDE)

    nonzero = (y != 0).astype(np.int32)
    cumsum = np.concatenate([[0], nonzero.cumsum()])
    window_sum = cumsum[SEQ_LEN:] - cumsum[:-SEQ_LEN]
    starts_nonzero = starts_all[window_sum[starts_all] > 0]
    starts_zero = starts_all[window_sum[starts_all] == 0]

    classes = np.unique(y)
    nonzero_classes = classes[classes != 0]
    n_balanced = max(1, int(n_sequences * STRATIFY_RATIO))
    per_class = max(1, n_balanced // max(len(nonzero_classes), 1))

    balanced_starts = []
    for cls in nonzero_classes:
        idx = np.where(y == cls)[0]
        if idx.size == 0:
            continue
        starts = idx - (SEQ_LEN // 2)
        starts = np.clip(starts, 0, n_samples - SEQ_LEN - 1)
        starts = (starts // STRIDE) * STRIDE
        starts = starts[(starts >= 0) & (starts <= (n_samples - SEQ_LEN - 1))]
        if starts.size == 0:
            continue
        starts = np.unique(starts)
        take = min(per_class, starts.size)
        balanced_starts.append(np.random.permutation(starts)[:take])

    if balanced_starts:
        balanced_starts = np.concatenate(balanced_starts)
    else:
        balanced_starts = np.array([], dtype=int)

    n_random = n_sequences - len(balanced_starts)
    if n_random > 0:
        random_starts = np.random.permutation(starts_all)[:n_random]
        starts = np.concatenate([balanced_starts, random_starts])
    else:
        starts = balanced_starts[:n_sequences]

    rprint("[bold cyan]Stratification sampling check[/]")
    rprint(f"- total windows: {n_sequences}")
    rprint(f"- balanced windows: {len(balanced_starts)}")
    rprint(f"- random windows: {n_sequences - len(balanced_starts)}")
    rprint(f"- nonzero windows available: {len(starts_nonzero)}")
    rprint(f"- zero-only windows available: {len(starts_zero)}")

    # Estimate per-timestep label distribution in sampled windows
    idx = starts[:, None] + np.arange(SEQ_LEN)[None, :]
    y_window = y[idx].ravel()
    counts = Counter(y_window.tolist())

    rprint("\n[bold cyan]Sampled timestep label distribution[/]")
    total = len(y_window)
    for label in sorted(counts.keys()):
        count = counts[label]
        rprint(f"- {label}: {count} ({count / total:.2%})")


if __name__ == "__main__":
    main()
