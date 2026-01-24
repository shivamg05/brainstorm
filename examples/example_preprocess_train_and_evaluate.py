#!/usr/bin/env python3
"""
Example script: Preprocess -> Train -> Evaluate (MLP)

Usage:
    uv run python examples/example_preprocess_train_and_evaluate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.loading import load_raw_data
from brainstorm.ml.mlp import MLP
from brainstorm.preprocessing import design_bandpass_sos, preprocess_features


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("./data")

EPOCHS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

FS_HZ = 1000
TAU_S = 0.05
BANDS_HZ = [
    (1, 4),
    (4, 8),
    (8, 12),
    (13, 30),
    (30, 55),
    (65, 100),  # skip 55–65 to avoid 60 Hz noise
    (100, 150),
    (150, 250),
]


def _print_dataset_table(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    val_features: pd.DataFrame,
    val_labels: pd.DataFrame,
) -> None:
    console = Console()
    table = Table(title="Dataset Overview", show_header=True, header_style="bold magenta")
    table.add_column("Split", style="cyan", width=10)
    table.add_column("Features Shape", style="green")
    table.add_column("Labels Shape", style="green")
    table.add_column("Time Range (s)", style="yellow")
    table.add_column("Unique Labels", style="blue")

    table.add_row(
        "Train",
        str(train_features.shape),
        str(train_labels.shape),
        f"{train_features.index[0]:.2f} → {train_features.index[-1]:.2f}",
        str(sorted(train_labels["label"].unique().tolist())),
    )
    table.add_row(
        "Val",
        str(val_features.shape),
        str(val_labels.shape),
        f"{val_features.index[0]:.2f} → {val_features.index[-1]:.2f}",
        str(sorted(val_labels["label"].unique().tolist())),
    )
    console.print(table)
    print()


def main() -> None:
    rprint("\n[bold green]Evaluating model with preprocessing...[/]\n")

    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("\n[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()
        rprint("[bold green]✓ Data downloaded successfully![/]\n")

    rprint(f"\n[bold cyan]Loading data from:[/] {DATA_PATH}\n")
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    val_features, val_labels = load_raw_data(DATA_PATH, step="validation")
    _print_dataset_table(train_features, train_labels, val_features, val_labels)

    X_train = train_features.to_numpy(dtype=np.float32)
    X_val = val_features.to_numpy(dtype=np.float32)
    y_train = train_labels["label"].to_numpy()

    sos_bank = design_bandpass_sos(BANDS_HZ, fs_hz=FS_HZ)
    train_feat, stats = preprocess_features(X_train, sos_bank, FS_HZ, TAU_S)
    val_feat, _ = preprocess_features(X_val, sos_bank, FS_HZ, TAU_S, stats=stats)

    train_flat = train_feat.reshape(train_feat.shape[0], -1)
    val_flat = val_feat.reshape(val_feat.shape[0], -1)
    train_df = pd.DataFrame(train_flat, index=train_features.index)
    val_df = pd.DataFrame(val_flat, index=val_features.index)

    rprint("\n[bold green]Training model...[/]\n")
    model = MLP(
        input_size=train_df.shape[1],
        hidden_size=HIDDEN_SIZE,
    )
    model.fit(
        X=train_df.values,
        y=y_train,  # type: ignore[union-attr]
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        verbose=True,
    )

    rprint("\n[bold green]Evaluating model on validation set...[/]\n")
    evaluator = ModelEvaluator(
        test_features=val_df,
        test_labels=val_labels[["label"]],  # type: ignore[union-attr]
    )
    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)
    rprint("\n[bold green]Evaluation complete![/]\n")


if __name__ == "__main__":
    main()
